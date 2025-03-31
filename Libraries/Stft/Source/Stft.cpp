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

	void HannWindow(double* data, const float* srcdata, int size) {
		for (int i = 0; i < size; i++) {
			const double windowValue = 0.5 * (1 - cos(2 * StftKernel::PI * i / (size - 1)));
			data[i] = srcdata[i] * windowValue;
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

	struct _M_MCPX
	{
		fftw_complex Complex;
		operator fftw_complex& () { return Complex; }
		operator const fftw_complex& () const { return Complex; }
		operator Complex32() const { return { static_cast<float>(Complex[0]), static_cast<float>(Complex[1]) }; }
		operator Complex64() const { return { Complex[0], Complex[1] }; }
	};

	StftKernel::StftKernel(
		int NumFFT, int HopSize, int WindowSize,
		bool Center, PaddingType Padding
	)
	{
		if (NumFFT <= 2)
			_D_Dragonian_Lib_Throw_Exception("Invalid FFT size.");
		NUM_FFT = NumFFT;
		FFT_BINS = NumFFT / 2 + 1;
		if (HopSize <= 0)
			HOP_SIZE = NumFFT / 4;
		else
			HOP_SIZE = HopSize;
		if (WindowSize <= 0)
			WINDOW_SIZE = NumFFT;
		else
		{
			if (WindowSize > NumFFT)
				_D_Dragonian_Lib_Throw_Exception("Window size is too large.");
			WINDOW_SIZE = WindowSize;
			PADDING = (NumFFT - WindowSize) / 2;
		}
		CENTER = Center;
		if (CENTER)
			CENTER_PADDING_SIZE = HOP_SIZE / 2;
		else
			CENTER_PADDING_SIZE = 0;
		PADDING_TYPE = Padding;
	}

	StftKernel::~StftKernel() = default;

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
		auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		auto SignalCont = Signal.Continuous().Evaluate();
		SignalCont = SignalCont.Padding(
			{
				None,
				None,
				{ CENTER_PADDING_SIZE + WINDOW_SIZE / 2 , WINDOW_SIZE / 2 }
			},
			PADDING_TYPE
		).Evaluate();
		SignalSize = SignalCont.Size(2);
		const auto NUM_FRAMES = (SignalSize - WINDOW_SIZE) / HOP_SIZE + 1;
		const auto Shape = Dimensions<4>{ BatchSize, Channel, NUM_FRAMES, FFT_BINS };
		auto Output = Tensor<Float32, 4, Device::CPU>::New(Shape);
		const auto& SignalDataBegin = SignalCont.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < Channel; c++)
			{
				const auto SignalData = SignalDataBegin + (b * Channel + c) * SignalSize;
				auto SpectrogramData = Output.Data() + (b * Channel + c) * NUM_FRAMES * FFT_BINS;
				Output.AppendTask(
					[this, NUM_FRAMES, SignalData, SpectrogramData]
					{
						DragonianLibSTL::Vector hannWindow(NUM_FFT, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_BINS)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_r2c_1d(NUM_FFT, hannWindow.Data(), (fftw_complex*)fftOut.get(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < NUM_FRAMES; i++) {
							std::memcpy(
								hannWindow.Data() + PADDING,
								&SignalData[size_t(i) * HOP_SIZE],
								size_t(sizeof(double)) * WINDOW_SIZE
							);
							HannWindow(
								hannWindow.Data() + PADDING,
								WINDOW_SIZE
							);
							fftw_execute(plan.get());
							const auto BgnPtn = size_t(unsigned(i * FFT_BINS));
							for (int j = 0; j < FFT_BINS; j++)
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
		auto SignalSize = Signal.Size(2);
		if (SignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		auto SignalCont = Signal.Continuous().Evaluate();
		SignalCont = SignalCont.Padding(
			{
				None,
				None,
				{ CENTER_PADDING_SIZE + WINDOW_SIZE / 2 , WINDOW_SIZE / 2 }
			},
			PADDING_TYPE
		).Evaluate();
		SignalSize = SignalCont.Size(2);
		const auto NUM_FRAMES = (SignalSize - WINDOW_SIZE) / HOP_SIZE + 1;
		const auto Shape = Dimensions<4>{ BatchSize, Channel, NUM_FRAMES, FFT_BINS };
		auto Output = Tensor<Complex32, 4, Device::CPU>::New(Shape);
		const auto& SignalDataBegin = SignalCont.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < Channel; c++)
			{
				const auto SignalData = SignalDataBegin + (b * Channel + c) * SignalSize;
				auto SpectrogramData = Output.Data() + (b * Channel + c) * NUM_FRAMES * FFT_BINS;
				Output.AppendTask(
					[this, NUM_FRAMES, SignalData, SpectrogramData]
					{
						DragonianLibSTL::Vector hannWindow(NUM_FFT, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_BINS)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_r2c_1d(NUM_FFT, hannWindow.Data(), (fftw_complex*)fftOut.get(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < NUM_FRAMES; i++) {
							HannWindow(
								hannWindow.Data() + PADDING,
								&SignalData[size_t(i) * HOP_SIZE],
								WINDOW_SIZE
							);
							fftw_execute(plan.get());
							const auto BgnPtn = size_t(unsigned(i * FFT_BINS));
							for (int j = 0; j < FFT_BINS; j++)
								SpectrogramData[BgnPtn + j] = fftOut.get()[j];
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
		auto Output = Tensor<Float32, 3, Device::CPU>::Zeros(
			{ BatchSize, ChannelCount, SignalSize }
		).Evaluate();
		auto SpectrogramCont = Spectrogram.Continuous().Evaluate();
		auto SpectrogramBins = Spectrogram.Size(3);
		if (SpectrogramBins != FFT_BINS)
			_D_Dragonian_Lib_Throw_Exception("Invalid FFT size.");
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
						DragonianLibSTL::Vector hannWindow(NUM_FFT, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_BINS)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_c2r_1d(NUM_FFT, (fftw_complex*)fftOut.get(), hannWindow.Data(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < FrameCount; i++) {
							const auto BgnPtn = size_t(unsigned(i * FFT_BINS));
							for (int j = 0; j < FFT_BINS; j++)
							{
								fftOut.get()[j][0] = static_cast<double>(SpectrogramData[BgnPtn + j]);
								fftOut.get()[j][1] = 0.f;
							}
							fftw_execute(plan.get());
							for (int j = 0; j < WINDOW_SIZE; j++)
								SignalData[size_t(i) * HOP_SIZE + j] += float(hannWindow[j + PADDING]) / float(NUM_FFT);
						}
					}
				);
			}
		}
		return std::move(Output.Evaluate());
	}

	Tensor<Float32, 3, Device::CPU> StftKernel::Inverse(const Tensor<Complex32, 4, Device::CPU>& Spectrogram) const
	{
		const auto [BatchSize, ChannelCount, FrameCount, FFTSize] =
			Spectrogram.Shape().RawArray();
		const auto SignalSize = FrameCount * HOP_SIZE + WINDOW_SIZE;
		auto Output = Tensor<Float32, 3, Device::CPU>::Zeros(
			{ BatchSize, ChannelCount, SignalSize }
		).Evaluate();
		auto SpectrogramCont = Spectrogram.Continuous().Evaluate();
		auto SpectrogramBins = Spectrogram.Size(3);
		if (SpectrogramBins != FFT_BINS)
			_D_Dragonian_Lib_Throw_Exception("Invalid FFT size.");
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
						DragonianLibSTL::Vector hannWindow(NUM_FFT, 0.0);
						const auto fftOut = std::shared_ptr<_M_MCPX>(
							(_M_MCPX*)(fftw_malloc(sizeof(_M_MCPX) * FFT_BINS)),
							fftw_free
						);
						const auto plan = std::shared_ptr<fftw_plan_s>(
							fftw_plan_dft_c2r_1d(NUM_FFT, (fftw_complex*)fftOut.get(), hannWindow.Data(), FFTW_ESTIMATE),
							fftw_destroy_plan
						);
						for (int i = 0; i < FrameCount; i++) {
							const auto BgnPtn = size_t(unsigned(i * FFT_BINS));
							for (int j = 0; j < FFT_BINS; j++)
							{
								fftOut.get()[j][0] = static_cast<double>(SpectrogramData[BgnPtn + j].real());
								fftOut.get()[j][1] = static_cast<double>(SpectrogramData[BgnPtn + j].imag());
							}
							fftw_execute(plan.get());
							for (int j = 0; j < WINDOW_SIZE; j++)
								SignalData[size_t(i) * HOP_SIZE + j] += float(hannWindow[j + PADDING]) / float(NUM_FFT);
						}
					}
				);
			}
		}
		return std::move(Output.Evaluate());
	}

	MFCCKernel::MFCCKernel(
		int SamplingRate, int NumFFT, int HopSize, int WindowSize, int MelBins,
		double FreqMin, double FreqMax, bool Center, PaddingType Padding, DLogger _Logger
	) : _MyStftKernel(NumFFT, HopSize, WindowSize, Center, Padding), _MyLogger(std::move(_Logger))
	{
		double mel_min = HZ2Mel(FreqMin);
		double mel_max = HZ2Mel(FreqMax);

		if (MelBins > 0)
			_MyMelBins = MelBins;
		_MyFFTSize = NumFFT / 2 + 1;
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

		_MyMelBasis = DragonianLibSTL::Vector(size_t(_MyFFTSize) * _MyMelBins, 0.f);

		for (int i = 0; i < _MyMelBins; ++i)
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
						//MelBasis[MelBins, FFTSize]
						//SpecData[FrameCount, FFTSize]
						//MelData[MelBins, FrameCount]
						//MelBasis * SpecData.T
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