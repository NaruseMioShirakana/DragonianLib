#include "../stft.hpp"
#include "cblas.h"
#include "fftw3.h"

_D_Dragonian_Lib_Space_Begin

namespace FunctionTransform
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
			const double windowValue = 0.5 * (1 - cos(2 * StftKernel::PI * i / (size - 1)));
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

	StftKernel::StftKernel(int WindowSize, int HopSize, int FFTSize)
	{
		WINDOW_SIZE = WindowSize;
		HOP_SIZE = HopSize;
		if (FFTSize > 0)
			FFT_SIZE = FFTSize;
		else
			FFT_SIZE = WINDOW_SIZE / 2 + 1;
	}

	StftKernel::~StftKernel() = default;

	struct _M_MCPX
	{
		fftw_complex Complex;
		operator fftw_complex& () { return Complex; }
		operator const fftw_complex& () const { return Complex; }
	};

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		return operator()(Signal.Cast<Float64>());
	}

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		return operator()(Signal.Cast<Float64>());
	}

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const
	{
		const auto BatchSize = Signal.Size(0);
		const auto Channel = Signal.Size(1);
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		const auto NUM_FRAMES = (SignalSize - WINDOW_SIZE) / HOP_SIZE + 1;
		const auto Shape = Dimensions<4>{ BatchSize, Channel, NUM_FRAMES, FFT_SIZE };
		auto Output = Tensor<Float32, 4, Device::CPU>::New(Shape);
		auto SignalCont = Signal.Continuous().Evaluate();
		const auto& SignalDataBegin = SignalCont.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < Channel; c++)
			{
				const auto SignalData = SignalDataBegin + (b * Channel + c) * SignalSize;
				auto SpectrogramData = Output.Data() + (b * Channel + c) * NUM_FRAMES * FFT_SIZE;
				Output.AppendTask(
					[this, NUM_FRAMES, SignalData, SpectrogramData]
					{
						DragonianLibSTL::Vector hannWindow(WINDOW_SIZE, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_SIZE)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_r2c_1d(WINDOW_SIZE, hannWindow.Data(), (fftw_complex*)fftOut.get(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < NUM_FRAMES; i++) {
							std::memcpy(hannWindow.Data(), &SignalData[size_t(i) * HOP_SIZE], size_t(sizeof(double)) * WINDOW_SIZE);
							HannWindow(hannWindow.Data(), WINDOW_SIZE);
							fftw_execute(plan.get());
							const auto BgnPtn = size_t(unsigned(i * FFT_SIZE));
							for (int j = 0; j < FFT_SIZE; j++)
								SpectrogramData[BgnPtn + j] = float(CalculatePowerSpectrum(fftOut.get()[j]));
						}
					}
				);
			}
		}
		return std::move(Output.Evaluate());
	}

	Tensor<Complex32, 4, Device::CPU> StftKernel::Execute(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		const auto BatchSize = Signal.Size(0);
		const auto Channel = Signal.Size(1);
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		const auto NUM_FRAMES = (SignalSize - WINDOW_SIZE) / HOP_SIZE + 1;
		const auto Shape = Dimensions<4>{ BatchSize, Channel, NUM_FRAMES, FFT_SIZE };
		auto Output = Tensor<Complex32, 4, Device::CPU>::New(Shape);
		auto SignalCont = Signal.Continuous().Evaluate();
		const auto& SignalDataBegin = SignalCont.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < Channel; c++)
			{
				const auto SignalData = SignalDataBegin + (b * Channel + c) * SignalSize;
				auto SpectrogramData = Output.Data() + (b * Channel + c) * NUM_FRAMES * FFT_SIZE;
				Output.AppendTask(
					[this, NUM_FRAMES, SignalData, SpectrogramData]
					{
						DragonianLibSTL::Vector hannWindow(WINDOW_SIZE, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_SIZE)),
							fftw_free
						);
						auto ComplexPtr = (fftw_complex*)fftOut.get();
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_r2c_1d(WINDOW_SIZE, hannWindow.Data(), ComplexPtr, FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < NUM_FRAMES; i++) {
							std::memcpy(hannWindow.Data(), &SignalData[size_t(i) * HOP_SIZE], size_t(sizeof(double)) * WINDOW_SIZE);
							HannWindow(hannWindow.Data(), WINDOW_SIZE);
							fftw_execute(plan.get());
							const auto BgnPtn = size_t(unsigned(i * FFT_SIZE));
							for (int j = 0; j < FFT_SIZE; j++)
								SpectrogramData[BgnPtn + j] = { (Float32)ComplexPtr[j][0], (Float32)ComplexPtr[j][1] };
						}
					}
				);
			}
		}
		return std::move(Output.Evaluate());
	}

	Tensor<Float32, 3, Device::CPU> StftKernel::Inverse(const Tensor<Float32, 4, Device::CPU>& Spectrogram) const
	{
		const auto [BatchSize, ChannelCount, FrameCount, FFTSize] =
			Spectrogram.Shape().RawArray();
		const auto SignalSize = FrameCount * HOP_SIZE + WINDOW_SIZE;
		auto Output = Tensor<Float32, 3, Device::CPU>::New(
			{ BatchSize, ChannelCount, SignalSize }
		);
		auto SpectrogramCont = Spectrogram.Continuous().Evaluate();
		const auto& SpectrogramDataBegin = SpectrogramCont.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < ChannelCount; c++)
			{
				const auto SpectrogramData = SpectrogramDataBegin + (b * ChannelCount + c) * FrameCount * FFTSize;
				auto SignalData = Output.Data() + (b * ChannelCount + c) * SignalSize;
				Output.AppendTask(
					[this, FrameCount, SpectrogramData, SignalData]
					{
						DragonianLibSTL::Vector hannWindow(WINDOW_SIZE, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_SIZE)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_c2r_1d(WINDOW_SIZE, (fftw_complex*)fftOut.get(), hannWindow.Data(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < FrameCount; i++) {
							const auto BgnPtn = size_t(unsigned(i * FFT_SIZE));
							for (int j = 0; j < FFT_SIZE; j++)
							{
								fftOut.get()[j][0] = static_cast<double>(SpectrogramData[BgnPtn + j]);
								fftOut.get()[j][1] = 0.;
							}
							fftw_execute(plan.get());
							for (int j = 0; j < WINDOW_SIZE; j++)
								SignalData[size_t(i) * HOP_SIZE + j] += float(hannWindow[j]);
						}
					}
				);
			}
		}
		return std::move(Output.Evaluate());
	}

	MFCCKernel::MFCCKernel(
		int WindowSize, int HopSize, int SamplingRate, int MelBins,
		double FreqMin, double FreqMax, DLogger _Logger
	) : _MyStftKernel(WindowSize, HopSize, WindowSize / 2 + 1), _MyLogger(std::move(_Logger))
	{
		double mel_min = HZ2Mel(FreqMin);
		double mel_max = HZ2Mel(FreqMax);

		if (MelBins > 0)
			_MyMelBins = MelBins;
		_MyFFTSize = WindowSize / 2 + 1;
		_MySamplingRate = SamplingRate;

		const int nfft = (_MyFFTSize - 1) * 2;
		const double fftfreqval = 1. / (double(nfft) / double(SamplingRate));
		auto fftfreqs = DragonianLibSTL::Arange<double>(0, _MyFFTSize + 2);
		fftfreqs.Resize(_MyFFTSize, 0.f);
		for (auto& i : fftfreqs)
			i *= fftfreqval;

		auto mel_f = DragonianLibSTL::Arange<double>(mel_min, mel_max + 1., (mel_max - mel_min) / (_MyMelBins + 1));
		mel_f.Resize(_MyMelBins + 2, 0.f); //[_MyMelBins + 2]

		std::vector<double> fdiff;
		std::vector<std::vector<double>> ramps; //[_MyMelBins + 2, FFTSize]

		ramps.reserve(_MyMelBins + 2);
		for (auto& i : mel_f)
		{
			i = Mel2HZ(i);
			ramps.emplace_back(_MyFFTSize, i);
		}
		for (auto& i : ramps)
			for (int j = 0; j < _MyFFTSize; ++j)
				i[j] -= fftfreqs[j];

		fdiff.reserve(_MyMelBins + 2); //[_MyMelBins + 1]
		for (size_t i = 1; i < mel_f.Size(); ++i)
			fdiff.emplace_back(mel_f[i] - mel_f[i - 1]);

		_MyMelBasis = DragonianLibSTL::Vector(size_t(_MyFFTSize) * MelBins, 0.f);

		for (int i = 0; i < MelBins; ++i)
		{
			const auto enorm = 2. / (mel_f[i + 2] - mel_f[i]);
			for (int j = 0; j < _MyFFTSize; ++j)
				_MyMelBasis[i * _MyFFTSize + j] = (float)(std::max(0., std::min(-ramps[i][j] / fdiff[i], ramps[i + 2][j] / fdiff[i + 1])) * enorm);
		}
	}

	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < _MyStftKernel.WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");

		auto BgnTime = clock();
		const auto Spec = _MyStftKernel(Signal);
		if (_MyLogger)
			_MyLogger->Log((L"Stft Use Time " + std::to_wstring(clock() - BgnTime) + L"ms"), Logger::LogLevel::Info);

		const auto [BatchSize, ChannelCount, FrameCount, FFTSize] = Spec.Shape().RawArray();
		const auto MelShape = Dimensions<4>{ BatchSize, ChannelCount, _MyMelBins, FrameCount };
		auto Result = Tensor<Float32, 4, Device::CPU>::New(MelShape);
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < ChannelCount; c++)
			{
				const auto SpecData = Spec.Data() + (b * ChannelCount + c) * FrameCount * FFTSize;
				auto MelData = Result.Data() + (b * ChannelCount + c) * _MyMelBins * FrameCount;
				Result.AppendTask(
					[this, SpecData, MelData, FrameCount]
					{
						cblas_sgemm(
							CblasRowMajor,
							CblasNoTrans,
							CblasTrans,
							_MyMelBins,
							blasint(FrameCount),
							_MyFFTSize,
							1.f,
							_MyMelBasis.Data(),
							_MyFFTSize,
							SpecData,
							blasint(_MyFFTSize),
							0.f,
							MelData,
							blasint(FrameCount)
						);
						for (int i = 0; i < _MyMelBins * FrameCount; i++)
							MelData[i] = log(std::max(1e-5f, MelData[i]));
					}
				);
			}
		}

		BgnTime = clock();
		Result.Evaluate();
		if (_MyLogger)
			_MyLogger->Log((L"Mel Transform Use Time " + std::to_wstring(clock() - BgnTime) + L"ms"), Logger::LogLevel::Info);
		return Result;
	}

	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < _MyStftKernel.WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		return operator()(Signal.Cast<Float64>());
	}

	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < _MyStftKernel.WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		return operator()(Signal.Cast<Float64>());
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
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

}

_D_Dragonian_Lib_Space_End