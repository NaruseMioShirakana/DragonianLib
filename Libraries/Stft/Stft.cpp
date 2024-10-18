#include "stft.hpp"
#include "Base.h"
#include "cblas.h"
#include "Util/Logger.h"
#include "fftw3.h"
#include "Tensor/TensorBase.h"
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace DlCodecStft
{
    double HZ2Mel(const double frequency)
    {
        constexpr auto f_min = 0.0;
        constexpr auto f_sp = 200.0 / 3;
        auto mel = (frequency - f_min) / f_sp;
        constexpr auto min_log_hz = 1000.0;
        constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;
        const auto logstep = log(6.4) / 27.0;
        if (frequency >= min_log_hz)
            mel = min_log_mel + log(frequency / min_log_hz) / logstep;
        return mel;
    }

    double Mel2HZ(const double mel)
    {
        constexpr auto f_min = 0.0;
        constexpr auto f_sp = 200.0 / 3;
        auto freqs = f_min + f_sp * mel;
        constexpr auto min_log_hz = 1000.0;
        constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;
        const auto logstep = log(6.4) / 27.0;
        if (mel >= min_log_mel)
            freqs = min_log_hz * exp(logstep * (mel - min_log_mel));
        return freqs;
    }

    void HannWindow(double* data, int size) {
        for (int i = 0; i < size; i++) {
            const double windowValue = 0.5 * (1 - cos(2 * STFT::PI * i / (size - 1)));
            data[i] *= windowValue;
        }
    }

    void ConvertDoubleToFloat(const DragonianLibSTL::Vector<double>& input, float* output)
    {
        for (size_t i = 0; i < input.Size(); i++) {
            output[i] = static_cast<float>(input[i]);
        }
    }

    double CalculatePowerSpectrum(fftw_complex fc) {
        return sqrt(fc[0] * fc[0] + fc[1] * fc[1]);
    }

    void CalculatePowerSpectrum(double* real, const double* imag, int size) {
        for (int i = 0; i < size; i++) {
            real[i] = real[i] * real[i] + imag[i] * imag[i];
        }
    }

    void ConvertPowerSpectrumToDecibels(double* data, int size) {
        for (int i = 0; i < size; i++) {
            data[i] = 10 * log10(data[i]);
        }
    }

    STFT::STFT(int WindowSize, int HopSize, int FFTSize)
    {
        WINDOW_SIZE = WindowSize;
        HOP_SIZE = HopSize;
        if (FFTSize > 0)
            FFT_SIZE = FFTSize;
        else
            FFT_SIZE = WINDOW_SIZE / 2 + 1;
    }

    STFT::~STFT() = default;

    std::pair<DragonianLibSTL::Vector<float>, int64_t> Mel::GetMel(const DragonianLibSTL::Vector<int16_t>& audioData) const
    {
        DragonianLibSTL::Vector<double> floatAudio(audioData.Size());
        for (size_t i = 0; i < audioData.Size(); ++i)
            floatAudio[i] = double(audioData[i]) / 32768.;
        return operator()(floatAudio);
    }

    std::pair<DragonianLibSTL::Vector<float>, int64_t> STFT::operator()(const DragonianLibSTL::Vector<double>& audioData) const
    {
        const int NUM_FRAMES = (int(audioData.Size()) - WINDOW_SIZE) / HOP_SIZE + 1;
        DragonianLibSTL::Vector hannWindow(WINDOW_SIZE, 0.0);
        const auto fftOut = (fftw_complex*)(fftw_malloc(sizeof(fftw_complex) * FFT_SIZE));
        const fftw_plan plan = fftw_plan_dft_r2c_1d(WINDOW_SIZE, hannWindow.Data(), fftOut, FFTW_ESTIMATE);
        DragonianLibSTL::Vector spectrogram(size_t(NUM_FRAMES) * FFT_SIZE, 0.f);
        for (int i = 0; i < NUM_FRAMES; i++) {
            std::memcpy(hannWindow.Data(), &audioData[size_t(i) * HOP_SIZE], size_t(sizeof(double)) * WINDOW_SIZE);
            HannWindow(hannWindow.Data(), WINDOW_SIZE);
            fftw_execute(plan);
            const auto BgnPtn = size_t(unsigned(i * FFT_SIZE));
            for (int j = 0; j < FFT_SIZE; j++)
                spectrogram[BgnPtn + j] = float(CalculatePowerSpectrum(fftOut[j]));
        }
        fftw_destroy_plan(plan);
        fftw_free(fftOut);
        return { std::move(spectrogram), int64_t(NUM_FRAMES) };
    }

    std::pair<DragonianLibSTL::Vector<float>, int64_t> Mel::GetMel(const DragonianLibSTL::Vector<double>& audioData) const
    {
        auto BgnTime = clock();
        const auto Spec = stft(audioData);  //[frame, nfft] * [nfft, mel_bins]  |  [mel_bins, nfft] * [nfft, frame]
        DragonianLib::LogInfo(L"Stft Use Time " + std::to_wstring(clock() - BgnTime) + L"ms");
        const auto NFrames = Spec.second;
        DragonianLibSTL::Vector Mel(MEL_SIZE * NFrames, 0.f);
        BgnTime = clock();
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            MEL_SIZE,
            blasint(NFrames),
            FFT_SIZE,
            1.f,
            MelBasis.Data(),
            FFT_SIZE,
            Spec.first.Data(),
            blasint(FFT_SIZE),
            0.f,
            Mel.Data(),
            blasint(NFrames)
        );
        for (auto& it : Mel)
            it = log(std::max(1e-5f, it));
        DragonianLib::LogInfo(L"Mel Transform Use Time " + std::to_wstring(clock() - BgnTime) + L"ms");
        return { std::move(Mel), (int64_t)NFrames };
    }

    std::pair<DragonianLibSTL::Vector<float>, int64_t> Mel::operator()(const DragonianLibSTL::Vector<double>& audioData) const
    {
        return GetMel(audioData);
    }

    Mel::Mel(int WindowSize, int HopSize, int SamplingRate, int MelSize, double FreqMin, double FreqMax) :
        stft(WindowSize, HopSize, WindowSize / 2 + 1)
    {
    	double mel_min = HZ2Mel(FreqMin);
    	double mel_max = HZ2Mel(FreqMax);

        if (MelSize > 0)
            MEL_SIZE = MelSize;
        FFT_SIZE = WindowSize / 2 + 1;
        sr = SamplingRate;

        const int nfft = (FFT_SIZE - 1) * 2;
        const double fftfreqval = 1. / (double(nfft) / double(SamplingRate));
        auto fftfreqs = DragonianLibSTL::Arange<double>(0, FFT_SIZE + 2);
        fftfreqs.Resize(FFT_SIZE, 0.f);
        for (auto& i : fftfreqs)
            i *= fftfreqval;

        auto mel_f = DragonianLibSTL::Arange<double>(mel_min, mel_max + 1., (mel_max - mel_min) / (MEL_SIZE + 1));
        mel_f.Resize(MEL_SIZE + 2, 0.f); //[MEL_SIZE + 2]

        std::vector<double> fdiff;
        std::vector<std::vector<double>> ramps; //[MEL_SIZE + 2, FFTSize]

        ramps.reserve(MEL_SIZE + 2);
        for (auto& i : mel_f)
        {
            i = Mel2HZ(i);
	        ramps.emplace_back(FFT_SIZE, i);
        }
        for (auto& i : ramps)
            for (int j = 0; j < FFT_SIZE; ++j)
                i[j] -= fftfreqs[j];

        fdiff.reserve(MEL_SIZE + 2); //[MEL_SIZE + 1]
        for (size_t i = 1; i < mel_f.Size(); ++i)
            fdiff.emplace_back(mel_f[i] - mel_f[i - 1]);

        MelBasis = DragonianLibSTL::Vector(size_t(FFT_SIZE) * MelSize, 0.f);

        for (int i = 0; i < MelSize; ++i)
        {
            const auto enorm = 2. / (mel_f[i + 2] - mel_f[i]);
            for (int j = 0; j < FFT_SIZE; ++j)
                MelBasis[i * FFT_SIZE + j] = (float)(std::max(0., std::min(-ramps[i][j] / fdiff[i], ramps[i + 2][j] / fdiff[i + 1])) * enorm);
        }
    }

    DragonianLibSTL::Vector<float> CQT(
        const DragonianLibSTL::Vector<float>& AudioData,
        int SamplingRate, 
        int HopSize, 
        float FreqMin, 
        int CQTBins, 
        int BinsPerOctave, 
        float Tuning, 
        float FilterScale, 
        float Norm, 
        float Sparsity, 
        const char* Window, 
        bool Scale, 
        const char* PaddingMode, 
        const char* ResourceType
    )
    {
        return VQT(
            AudioData,
            SamplingRate,
            HopSize,
            FreqMin,
            CQTBins,
            "Equal",
            0,
            BinsPerOctave,
            Tuning,
            FilterScale,
            Norm,
            Sparsity,
            Window,
            Scale,
            PaddingMode,
            ResourceType
        );
    }

    DragonianLibSTL::Vector<float> VQT(
        const DragonianLibSTL::Vector<float>& AudioData,
        int SamplingRate,
        int HopSize,
        float FreqMin,
        int CQTBins,
        const char* Intervals,
        float Gamma,
        int BinsPerOctave,
        float Tuning,
        float FilterScale,
        float Norm,
        float Sparsity,
        const char* Window,
        bool Scale,
        const char* PaddingMode,
        const char* ResourceType
    )
    {
        DragonianLibNotImplementedError;
    }

}