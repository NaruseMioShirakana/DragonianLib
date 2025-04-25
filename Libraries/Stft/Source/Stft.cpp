#include <omp.h>
#include "cblas.h"
#include "fftw3.h"
#include "../stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

_D_Dragonian_Lib_Space_Begin

namespace FunctionTransform
{
	static inline std::mutex FFTW_MUTEX;

	static double HZ2Mel(const double frequency)
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

	static double Mel2HZ(const double mel)
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

	static void HannWindowFn(double* Window, int WindowSize)
	{
		const auto Step = 2 * StftKernel::PI / ((double)WindowSize - 1.);
		for (int i = 0; i < WindowSize; i++)
			Window[i] = 0.5 * (1 - cos(Step * i));
	}

	template <typename Type>
	static void WindowFn(double* Output, const Type* Signal, const double* Window, int WindowSize)
	{
		for (int i = 0; i < WindowSize; i++)
			if constexpr (TypeTraits::IsIntegerValue<Type>)
				Output[i] = double(Signal[i]) / double(std::numeric_limits<Type>::max()) * Window[i];
			else
				Output[i] = double(Signal[i]) * Window[i];
	}

	static void InverseWindowFn(double* Output, const double* Signal, const double* Window, double WindowPowSum, int WindowSize)
	{
		for (int i = 0; i < WindowSize; i++)
			Output[i] = double(Signal[i]) * Window[i] / WindowPowSum;
	}

	struct FFTW_COMPLEX
	{
		_D_Dragonian_Lib_Constexpr_Force_Inline FFTW_COMPLEX(
			double Real = 0., double Imaginary = 0.
		) : Complex{ Real, Imaginary }
		{
			return;
		}
		_D_Dragonian_Lib_Constexpr_Force_Inline FFTW_COMPLEX(
			const Complex32& Cpx
		) : Complex{ static_cast<double>(Cpx.real()), static_cast<double>(Cpx.imag()) }
		{

		}
		_D_Dragonian_Lib_Constexpr_Force_Inline FFTW_COMPLEX(
			const Complex64& Cpx
		) : Complex{ Cpx.real(), Cpx.imag() }
		{

		}

		double Abs() const
		{
			return sqrt(Complex[0] * Complex[0] + Complex[1] * Complex[1]);
		}

		operator fftw_complex& ()
		{
			return Complex;
		}
		operator const fftw_complex& () const
		{
			return Complex;
		}
		operator Complex32() const
		{
			return { static_cast<float>(Complex[0]), static_cast<float>(Complex[1]) };
		}
		operator Complex64() const
		{
			return { Complex[0], Complex[1] };
		}
		operator Float32() const
		{
			return static_cast<float>(Abs());
		}
		operator Float64() const
		{
			return Abs();
		}

		fftw_complex Complex;
	};

	static fftw_plan CreateDftReal2ComplexPlan1D(int N, double* In, FFTW_COMPLEX* Out, int Flags)
	{
		std::lock_guard lg(FFTW_MUTEX);
		return fftw_plan_dft_r2c_1d(N, In, (fftw_complex*)Out, Flags);
	}

	static fftw_plan CreateDftComplex2RealPlan1D(int N, FFTW_COMPLEX* In, double* Out, int Flags)
	{
		std::lock_guard lg(FFTW_MUTEX);
		return fftw_plan_dft_c2r_1d(N, (fftw_complex*)In, Out, Flags);
	}

	static void DestoryDftPlan(fftw_plan plan)
	{
		std::lock_guard lg(FFTW_MUTEX);
		fftw_destroy_plan(plan);
	}

	template <typename TypeInput, typename TypeOutput>
	static void ExecuteStft(
		Tensor<TypeOutput, 4, Device::CPU>& View, TypeOutput* Spectrogram, const TypeInput* Signal, const double* WINDOW,
		int WINDOW_SIZE, int NUM_FFT, int HOP_SIZE, int PADDING, int FFT_BINS, int NUM_FRAMES
	)
	{
		constexpr int Stride = 100;
		for (int Thread = 0; Thread < NUM_FRAMES; Thread += Stride)
		{
			View.AppendTask(
				[NUM_FFT, FFT_BINS, NUM_FRAMES, PADDING, HOP_SIZE, Signal, WINDOW, WINDOW_SIZE, Spectrogram, Thread]
				{
					const auto FrameEnd = std::min(Thread + Stride, NUM_FRAMES);
					DragonianLibSTL::Vector VInputBuffer(NUM_FFT, 0.0);
					const auto ComplexTensor = std::shared_ptr<FFTW_COMPLEX>(
						(FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * FFT_BINS),
						fftw_free
					);
					auto SignalData = VInputBuffer.Data();
					auto ComplexData = ComplexTensor.get();

					const auto PlanDftReal2Complex = std::shared_ptr<fftw_plan_s>(
						CreateDftReal2ComplexPlan1D(NUM_FFT, SignalData, ComplexData, FFTW_ESTIMATE),
						DestoryDftPlan
					);
					for (int CurFrame = Thread; CurFrame < FrameEnd; ++CurFrame)
					{
						WindowFn(
							SignalData + PADDING,
							&Signal[size_t(CurFrame) * HOP_SIZE],
							WINDOW,
							WINDOW_SIZE
						);
						fftw_execute(PlanDftReal2Complex.get());
						const auto BeginIdx = UInt64(CurFrame) * FFT_BINS;
						for (int j = 0; j < FFT_BINS; j++)
						{
							auto Cpx = ComplexData[j];
							Spectrogram[BeginIdx + j] = Cpx;
						}
					}
				}
			);
		}
	}

	template <typename TypeInput, typename TypeOutput>
	static void ExecuteIStft(
		Tensor<TypeOutput, 3, Device::CPU>& View, TypeOutput* Signal, const TypeInput* Spectrogram, const double* WINDOW,
		double WINDOW_POW_SUM, int WINDOW_SIZE, int NUM_FFT, int HOP_SIZE, int PADDING, int FFT_BINS, int NUM_FRAMES
	)
	{
		constexpr int Stride = 100;
		for (int Thread = 0; Thread < NUM_FRAMES; Thread += Stride)
		{
			View.AppendTask(
				[NUM_FFT, FFT_BINS, NUM_FRAMES, PADDING, HOP_SIZE, Signal, WINDOW, WINDOW_SIZE, WINDOW_POW_SUM, Spectrogram, Thread]
				{
					const auto FrameEnd = std::min(Thread + Stride, NUM_FRAMES);
					DragonianLibSTL::Vector VOutputBuffer(NUM_FFT, 0.0);
					const auto ComplexTensor = std::shared_ptr<FFTW_COMPLEX>(
						(FFTW_COMPLEX*)fftw_malloc(sizeof(FFTW_COMPLEX) * FFT_BINS),
						fftw_free
					);
					auto SignalData = VOutputBuffer.Data();
					auto ComplexData = ComplexTensor.get();

					const auto PlanDftComplex2Real = std::shared_ptr<fftw_plan_s>(
						CreateDftComplex2RealPlan1D(NUM_FFT, ComplexData, SignalData, FFTW_ESTIMATE),
						DestoryDftPlan
					);
					for (int CurFrame = Thread; CurFrame < FrameEnd; ++CurFrame)
					{
						const auto BeginIdx = UInt64(CurFrame) * FFT_BINS;
						for (int j = 0; j < FFT_BINS; j++)
							ComplexData[j] = Spectrogram[BeginIdx + j];
						fftw_execute(PlanDftComplex2Real.get());
						InverseWindowFn(
							SignalData + PADDING,
							SignalData + PADDING,
							WINDOW,
							WINDOW_POW_SUM,
							WINDOW_SIZE
						);
						for (int j = 0; j < WINDOW_SIZE; j++)
							Signal[size_t(CurFrame) * HOP_SIZE + j] += TypeOutput(SignalData[j + PADDING]);
					}
				}
			);
		}
	}

	template <typename TypeOutput, typename TypeInput>
	static Tensor<TypeOutput, 4, Device::CPU> ExecuteStft(
		const Tensor<TypeInput, 3, Device::CPU>& RawSignal, const double* WINDOW, int WINDOW_SIZE,
		int NUM_FFT, int HOP_SIZE, int PADDING, int FFT_BINS, int CENTER_PADDING_SIZE, PaddingType PADDING_TYPE
	)
	{
		const auto [BatchSize, Channel, RawSignalSize] = RawSignal.Size().RawArray();

		if (RawSignalSize < WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");

		auto SignalCont = RawSignal.Padding(
			{
				None,
				None,
				{ CENTER_PADDING_SIZE + WINDOW_SIZE / 2 , WINDOW_SIZE / 2 }
			},
			PADDING_TYPE
		).Evaluate();

		const auto SignalSize = SignalCont.Size(2);

		const auto NUM_FRAMES = (SignalSize - WINDOW_SIZE) / HOP_SIZE + 1;
		
		auto SpectrogramTensor = Tensor<TypeOutput, 4, Device::CPU>::New(
			Dimensions<4>{ BatchSize, Channel, NUM_FRAMES, FFT_BINS }
		);

		const auto SignalDataBegin = SignalCont.Data();
		const auto SpecDataBegin = SpectrogramTensor.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < Channel; c++)
			{
				const auto SignalData = SignalDataBegin + (b * Channel + c) * SignalSize;
				const auto SpectrogramData = SpecDataBegin + (b * Channel + c) * NUM_FRAMES * FFT_BINS;
				ExecuteStft(
					SpectrogramTensor, SpectrogramData, SignalData, WINDOW,
					WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, static_cast<int>(NUM_FRAMES)
				);
			}
		}
		return std::move(SpectrogramTensor.Evaluate());
	}

	template <typename TypeOutput, typename TypeInput>
	static Tensor<TypeOutput, 3, Device::CPU> ExecuteIStft(
		const Tensor<TypeInput, 4, Device::CPU>& RawSpectrogram, const double* WINDOW, int WINDOW_SIZE, double WINDOW_POW_SUM,
		int NUM_FFT, int HOP_SIZE, int PADDING, int FFT_BINS
	)
	{
		const auto [BatchSize, ChannelCount, FrameCount, StftBins] = RawSpectrogram.Size().RawArray();
		if (StftBins != FFT_BINS)
			_D_Dragonian_Lib_Throw_Exception("Invalid FFT_BINS.");
		const auto SignalSize = FrameCount * HOP_SIZE + WINDOW_SIZE;
		auto Signal = Tensor<TypeOutput, 3, Device::CPU>::Zeros(
			{ BatchSize, ChannelCount, SignalSize }
		);
		auto Spectrogram = RawSpectrogram.Continuous().Evaluate();
		Signal.Evaluate();
		const auto SpectrogramDataBegin = Spectrogram.Data();
		const auto SignalBegin = Signal.Data();
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < ChannelCount; c++)
			{
				const auto SignalData = SignalBegin + (b * ChannelCount + c) * SignalSize;
				const auto SpectrogramData = SpectrogramDataBegin + (b * ChannelCount + c) * FrameCount * FFT_BINS;
				ExecuteIStft(
					Signal, SignalData, SpectrogramData, WINDOW, WINDOW_POW_SUM,
					WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, static_cast<int>(FrameCount)
				);
			}
		}
		return std::move(Signal.Evaluate());
	}

	StftKernel::StftKernel(
		int NumFFT, int HopSize, int WindowSize, const double* Window,
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

		if (Window)
			WINDOW = { Window, Window + WINDOW_SIZE };
		else
		{
			WINDOW.Resize(WINDOW_SIZE);
			HannWindowFn(WINDOW.Data(), WINDOW_SIZE);
		}
		WINDOW_POWER_SUM = 0.;
		for (int i = 0; i < WINDOW_SIZE; i++)
			WINDOW_POWER_SUM += WINDOW[i] * WINDOW[i];
	}

	StftKernel::~StftKernel() = default;

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const
	{
		return ExecuteStft<Float32>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		return ExecuteStft<Float32>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Float64, 4, Device::CPU> StftKernel::operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const
	{
		return ExecuteStft<Float64>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Complex32, 4, Device::CPU> StftKernel::Execute(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		return ExecuteStft<Complex32>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Complex64, 4, Device::CPU> StftKernel::Execute(const Tensor<Float64, 3, Device::CPU>& Signal) const
	{
		return ExecuteStft<Complex64>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Float32, 3, Device::CPU> StftKernel::Inverse(const Tensor<Float32, 4, Device::CPU>& Spectrogram) const
	{
		return ExecuteIStft<Float32>(
			Spectrogram, WINDOW.Data(), WINDOW_SIZE, WINDOW_POWER_SUM, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS
		);
	}

	Tensor<Float32, 3, Device::CPU> StftKernel::Inverse(const Tensor<Complex32, 4, Device::CPU>& Spectrogram) const
	{
		return ExecuteIStft<Float32>(
			Spectrogram, WINDOW.Data(), WINDOW_SIZE, WINDOW_POWER_SUM, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS
		);
	}

	Tensor<Float64, 3, Device::CPU> StftKernel::Inverse(const Tensor<Complex64, 4, Device::CPU>& Spectrogram) const
	{
		return ExecuteIStft<Float64>(
			Spectrogram, WINDOW.Data(), WINDOW_SIZE, WINDOW_POWER_SUM, NUM_FFT, HOP_SIZE, PADDING, FFT_BINS
		);
	}

	Tensor<Float32, 4, Device::CPU> StftKernel::operator()(const Tensor<Float32, 3, Device::CPU>& Signal, Int64 HopSize) const
	{
		return ExecuteStft<Float32>(
			Signal, WINDOW.Data(),
			WINDOW_SIZE, NUM_FFT, static_cast<int>(HopSize), PADDING, FFT_BINS, CENTER_PADDING_SIZE, PADDING_TYPE
		);
	}

	Tensor<Float32, 3, Device::CPU> StftKernel::Inverse(const Tensor<Complex32, 4, Device::CPU>& Spectrogram, Int64 HopSize)
	{
		const auto [BatchSize, ChannelCount, FrameCount, FFTBin] =
			Spectrogram.Shape().RawArray();
		const auto FFTSize = (FFTBin - 1) * 2;
		const auto SignalSize = FrameCount * HopSize;
		return StftKernel(static_cast<int>(FFTSize), static_cast<int>(HopSize)).Inverse(Spectrogram)[{None, None, { None, SignalSize }}].Contiguous().Evaluate();
	}

	MFCCKernel::MFCCKernel(
		int SamplingRate, int NumFFT, int HopSize, int WindowSize, int MelBins,
		double FreqMin, double FreqMax, const double* Window, bool Center, PaddingType Padding, DLogger _Logger
	) : STFT_KERNEL(NumFFT, HopSize, WindowSize, Window, Center, Padding), _MyLogger(std::move(_Logger))
	{
		double MEL_MIN = HZ2Mel(FreqMin);
		double MEL_MAX = HZ2Mel(FreqMax);

		if (MelBins > 0)
			MEL_BINS = MelBins;
		FFT_SIZE = NumFFT;
		FFT_BINS = NumFFT / 2 + 1;
		SAMPLING_RATE = SamplingRate;

		auto Weight = Tensor<Float32, 2, Device::CPU>::Empty(
			Dimensions{ MEL_BINS, FFT_BINS }
		);
		const auto DSR = 1.f / float(SAMPLING_RATE);
		const auto VAl = 1.f / (DSR * float(FFT_SIZE));
		const auto N = float(FFT_BINS);
		auto FFT_FREQS = Tensor<Float32, 1, Device::CPU>::Arange(
			0.f, N, 1.f
		);
		auto MEL_F = Tensor<Float32, 1, Device::CPU>::Linspace(
			float(MEL_MIN), float(MEL_MAX), MEL_BINS + 2, true
		).Evaluate();
		for (auto& POINT : MEL_F.GetRng())
			POINT = (float)Mel2HZ((double)POINT);
		FFT_FREQS *= VAl;
		auto F_DIFF = MEL_F.Diff(0).Evaluate();
		auto RAMPS = Functional::Outer(
			MEL_F,
			FFT_FREQS,
			Functional::InnerOuterType::SUB
		).Evaluate();
		for (auto i : TemplateLibrary::Ranges(MEL_BINS))
		{
			auto UPPER = RAMPS[i + 2].Ignore() / F_DIFF[i + 1];
			auto LOWER = -RAMPS[i].Ignore() / F_DIFF[i];
			Weight[i].Ignore().TensorAssign(Functional::Min(LOWER, UPPER).Max(0.f));
		}
		auto ENORM = MEL_F[{Range{ 2, None }}].Ignore() -
			MEL_F[{Range{ None, MEL_BINS }}].Ignore();
		(Weight *= 2) /= ENORM.UnSqueeze(-1);
		WEIGHT = std::move(Weight.Evaluate());
		WEIGHTDBL = WEIGHT.Cast<Float64>().Evaluate();
	}
	
	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float32, 4, Device::CPU>& Spectrogram) const
	{
		const auto Spec = Spectrogram.Continuous().Evaluate();
		const auto [BatchSize, ChannelCount, FrameCount, FFTSize] = Spec.Shape().RawArray();
		const auto MelShape = Dimensions<4>{ BatchSize, ChannelCount, MEL_BINS, FrameCount };
		auto Result = Tensor<Float32, 4, Device::CPU>::New(MelShape);
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < ChannelCount; c++)
			{
				const auto SpecData = Spec.Data() + (b * ChannelCount + c) * FrameCount * FFTSize;
				auto MelData = Result.Data() + (b * ChannelCount + c) * MEL_BINS * FrameCount;
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
							MEL_BINS,
							blasint(FrameCount),
							FFT_BINS,
							1.f,
							WEIGHT.Data(),
							FFT_BINS,
							SpecData,
							blasint(FFT_BINS),
							0.f,
							MelData,
							blasint(FrameCount)
						);
						for (int i = 0; i < MEL_BINS * FrameCount; i++)
							MelData[i] = log(std::max(1e-5f, MelData[i]));
					}
				);
			}
		}

		const auto BgnTime = clock();
		Result.Evaluate();
		if (_MyLogger)
			_MyLogger->Log((L"Mel Transform Use Time " + std::to_wstring(clock() - BgnTime) + L"ms"), Logger::LogLevel::Info);
		return std::move(Result.Evaluate());
	}

	Tensor<Float64, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float64, 4, Device::CPU>& Spectrogram) const
	{
		const auto Spec = Spectrogram.Continuous().Evaluate();
		const auto [BatchSize, ChannelCount, FrameCount, FFTSize] = Spec.Shape().RawArray();
		const auto MelShape = Dimensions<4>{ BatchSize, ChannelCount, MEL_BINS, FrameCount };
		auto Result = Tensor<Float64, 4, Device::CPU>::New(MelShape);
		for (SizeType b = 0; b < BatchSize; b++)
		{
			for (SizeType c = 0; c < ChannelCount; c++)
			{
				const auto SpecData = Spec.Data() + (b * ChannelCount + c) * FrameCount * FFTSize;
				auto MelData = Result.Data() + (b * ChannelCount + c) * MEL_BINS * FrameCount;
				Result.AppendTask(
					[this, SpecData, MelData, FrameCount]
					{
						//MelBasis[MelBins, FFTSize]
						//SpecData[FrameCount, FFTSize]
						//MelData[MelBins, FrameCount]
						//MelBasis * SpecData.T
						cblas_dgemm(
							CblasRowMajor,
							CblasNoTrans,
							CblasTrans,
							MEL_BINS,
							blasint(FrameCount),
							FFT_BINS,
							1.f,
							WEIGHTDBL.Data(),
							FFT_BINS,
							SpecData,
							blasint(FFT_BINS),
							0.f,
							MelData,
							blasint(FrameCount)
						);
						for (int i = 0; i < MEL_BINS * FrameCount; i++)
							MelData[i] = log(std::max(1e-5, MelData[i]));
					}
				);
			}
		}

		const auto BgnTime = clock();
		Result.Evaluate();
		if (_MyLogger)
			_MyLogger->Log((L"Mel Transform Use Time " + std::to_wstring(clock() - BgnTime) + L"ms"), Logger::LogLevel::Info);
		return std::move(Result.Evaluate());
	}

	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const
	{
		const auto SignalSize = Signal.Size(2);
		if (SignalSize < STFT_KERNEL.WINDOW_SIZE)
			_D_Dragonian_Lib_Throw_Exception("Signal is too short.");
		return operator()(Signal.Cast<Float32>());
	}

	Tensor<Float32, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const
	{
		auto [Mel, Spec] = WithSpec(Signal);
		return std::move(Mel.Evaluate());
	}

	Tensor<Float64, 4, Device::CPU> MFCCKernel::operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const
	{
		auto [Mel, Spec] = WithSpec(Signal);
		return std::move(Mel.Evaluate());
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