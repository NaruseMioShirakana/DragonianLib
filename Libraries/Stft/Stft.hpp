/**
 * @file Stft.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Implementation of Short-Time Fourier Transform (STFT) and Mel Extractor and other signal processing functions
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Tensor.h"
#include <numbers>

_D_Dragonian_Lib_Space_Begin

namespace FunctionTransform
{
	class MFCCKernel;

	void NormWindow(double* WindowNorm, const double* Window, int WindowSize, int HopSize, double NormScale = 1.);

	/**
	 * @brief Modified Bessel function of the first kind of order zero, I₀(x)
	 * @tparam T Type of input and output (float, double)
	 * @param x Input value
	 * @return I₀(x) value
	 */
	template <typename T>
	T BesselI0(T x)
	{
		T sum = T(1.0);
		T term = T(1.0);
		T factorial = T(1.0);
		T x_squared = x * x / T(4.0);

		if (x > T(15.0))
			return std::exp(x) / std::sqrt(T(2) * std::numbers::pi_v<T> *x);

		for (int i = 1; i <= 30; i++)
		{
			factorial *= static_cast<T>(i);
			term *= x_squared / (factorial * factorial);
			sum += term;

			if (term < sum * T(1e-12))
				break;
		}
		return sum;
	}

	/**
	 * @brief Creates a Hann window, W(i, N) = 0.5 * (1 - cos(2πi / [N - 1, N]))
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> HannWindow(
		size_t WindowSize,
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto Step = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = static_cast<T>(0.5) * (static_cast<T>(1) - cos(Step * static_cast<T>(i)));
		return Window;
	}

	/**
	 * @brief Creates a Hamming window, W(i, N) = α - (1 - α) * cos(2πi / [N - 1, N])
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Alpha Alpha (default: 0.54)
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> HammingWindow(
		size_t WindowSize,
		T Alpha = T(0.54),
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		Alpha = std::clamp(Alpha, T(0.), T(1.));
		const T Beta = T(1) - Alpha;
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto Step = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = Alpha - Beta * cos(Step * static_cast<T>(i));
		return Window;
	}

	/**
	 * @brief Creates a Blackman window, a2 = α / 2, a0 = 0.5 - a2, a1 = 0.5, W(i, N) = a0 - a1 * cos(2πi / [N - 1, N]) + a2 * cos(4πi / [N - 1, N])
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Alpha Alpha (default: 0.16)
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> BlackmanWindow(
		size_t WindowSize,
		T Alpha = T(0.16),
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto Step = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		const T a2 = Alpha / static_cast<T>(2);
		const T a0 = static_cast<T>(0.5) - a2;
		const T a1 = static_cast<T>(0.5);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = a0 - a1 * cos(Step * static_cast<T>(i)) + a2 * cos(T(2) * Step * static_cast<T>(i));
		return Window;
	}

	/**
	 * @brief Creates a Rectangular (uniform) window, W(i, N) = 1
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> RectangularWindow(
		size_t WindowSize
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = static_cast<T>(1);
		return Window;
	}

	/**
	 * @brief Creates a Bartlett (triangular) window, W(i, N) = 1 - 2 * |i - (N - 1) / 2| / (N - 1)
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> BartlettWindow(
		size_t WindowSize
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const T N1 = static_cast<T>(WindowSize - 1);
		const T HalfN1 = N1 / static_cast<T>(2);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = static_cast<T>(1) - static_cast<T>(2) * std::abs((static_cast<T>(i) - HalfN1) / N1);
		return Window;
	}

	/**
	 * @brief Creates a Blackman-Nuttall window, W(i, N) = 0.3635819 - 0.4891775 * cos(2πi / [N - 1, N]) + 0.1365995 * cos(4πi / [N - 1, N]) - 0.0106411 * cos(6πi / [N - 1, N])
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> BlackmanNuttallWindow(
		size_t WindowSize,
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto Step = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = static_cast<T>(0.3635819) - static_cast<T>(0.4891775) * cos(Step * static_cast<T>(i))
			+ static_cast<T>(0.1365995) * cos(T(2) * Step * static_cast<T>(i))
			- static_cast<T>(0.0106411) * cos(T(3) * Step * static_cast<T>(i));
		return Window;
	}

	/**
	 * @brief Creates a Blackman-Harris window, W(i, N) = 0.35875 - 0.48829 * cos(2πi / [N - 1, N]) + 0.14128 * cos(4πi / [N - 1, N]) - 0.01168 * cos(6πi / [N - 1, N])
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> BlackmanHarrisWindow(
		size_t WindowSize,
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto Step = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		for (size_t i = 0; i < WindowSize; i++)
			Window[i] = static_cast<T>(0.35875) - static_cast<T>(0.48829) * cos(Step * static_cast<T>(i))
			+ static_cast<T>(0.14128) * cos(2 * Step * static_cast<T>(i))
			- static_cast<T>(0.01168) * cos(3 * Step * static_cast<T>(i));
		return Window;
	}

	/**
	 * @brief Creates a Tukey (tapered cosine) window
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Alpha Width of the cosine-tapered portion (0 ≤ alpha ≤ 1)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> TukeyWindow(
		size_t WindowSize,
		T Alpha = T(0.5)
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t N = WindowSize;
		const T AlphaHalfN = Alpha * static_cast<T>(N) / static_cast<T>(2);

		for (size_t i = 0; i < WindowSize; i++)
		{
			if (i < AlphaHalfN)
				Window[i] = static_cast<T>(0.5) * (static_cast<T>(1) -
					cos(std::numbers::pi_v<T> *static_cast<T>(i) / AlphaHalfN));
			else if (i <= N - AlphaHalfN)
				Window[i] = static_cast<T>(1.0);
			else
				Window[i] = static_cast<T>(0.5) * (static_cast<T>(1) -
					cos(std::numbers::pi_v<T> *static_cast<T>(N - i) / AlphaHalfN));
		}
		return Window;
	}

	/**
	 * @brief Creates a custom cosine-sum window
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Coefficients Vector of cosine coefficients
	 * @param Div Divisor for the resulting window (default: 1.0)
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> CosineSumWindow(
		size_t WindowSize,
		const TemplateLibrary::Vector<T>& Coefficients,
		const T Div = T(1.),
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
		const auto AngularStep = T(2) * std::numbers::pi_v<T> / static_cast<T>(Denominator);
		for (size_t i = 0; i < WindowSize; i++)
		{
			Window[i] = Coefficients[0];
			for (size_t j = 1; j < Coefficients.Size(); j++)
			{
				if (j % 2 == 1)
					Window[i] -= Coefficients[j] * cos(j * AngularStep * static_cast<T>(i));
				else
					Window[i] += Coefficients[j] * cos(j * AngularStep * static_cast<T>(i));
			}
			Window[i] /= Div;
		}
		return Window;
	}

	template <typename T>
	TemplateLibrary::Vector<T> FlatTopWindow(
		size_t WindowSize,
		bool Periodic = false
	)
	{
		static TemplateLibrary::Vector<T> Coefficients{
			T(1.), T(1.93), T(1.29), T(0.388), T(0.0322)
		};
		return CosineSumWindow(
			WindowSize,
			Coefficients,
			T(1.),
			Periodic
		);
	}

	/**
	 * @brief Creates a Kaiser window
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Beta Shape parameter (higher values = wider main lobe and lower side lobes)
	 * @param Periodic Whether the window is periodic (true) or symmetric (false)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> KaiserWindow(
		size_t WindowSize,
		T Beta = T(8.6),
		bool Periodic = false
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);
		const T Denominator = static_cast<T>(Periodic ? WindowSize : WindowSize - 1);
		const T I0Beta = BesselI0(Beta);
		for (size_t i = 0; i < WindowSize; i++)
		{
			const T x = T(2.0) * static_cast<T>(i) / Denominator - T(1.0);

			const T arg = Beta * std::sqrt(T(1.0) - x * x);
			Window[i] = BesselI0(arg) / I0Beta;
		}
		return Window;
	}

	/**
	 * @brief Creates a Gaussian window, W(i, N) = exp(-0.5 * ((i - (N - 1) / 2) / (sigma * (N - 1) / 2))^2)
	 * @tparam T Type of window elements (e.g. float, double)
	 * @param WindowSize Size of the window
	 * @param Sigma Standard deviation as a fraction of half window size (default: 0.4)
	 * @return Vector containing the window coefficients
	 */
	template <typename T>
	TemplateLibrary::Vector<T> GaussianWindow(
		size_t WindowSize,
		T Sigma = T(0.4)
	)
	{
		TemplateLibrary::Vector<T> Window(WindowSize);

		const T N1 = static_cast<T>(WindowSize - 1);
		const T HalfN1 = N1 / static_cast<T>(2);
		const T ScaleFactor = Sigma * HalfN1;
		const T Denominator = static_cast<T>(2) * ScaleFactor * ScaleFactor;

		for (size_t i = 0; i < WindowSize; i++)
		{
			const T Diff = static_cast<T>(i) - HalfN1;
			Window[i] = std::exp(-(Diff * Diff) / Denominator);
		}

		return Window;
	}

	/**
	 * @brief Performs windowed resampling of a one-dimensional signal
	 * @tparam T Type of signal elements (e.g. float, double)
	 * @param Signal Input signal
	 * @param InputSampleRate Original sampling rate in Hz
	 * @param OutputSampleRate Target sampling rate in Hz
	 * @param KeepPower Whether to keep the power of the signal (default: true)
	 * @param InWindow Window function to use (default: "Kaiser")
	 */
	template <typename T>
	Tensor<T, 3, Device::CPU> WindowedResample(
		const Tensor<T, 3, Device::CPU>& Signal,
		size_t InputSampleRate,
		size_t OutputSampleRate,
		bool KeepPower = true,
		const TemplateLibrary::Vector<T>& InWindow = {}
	)
	{
		static auto DefaultWindow = KaiserWindow<T>(32);

		if (InputSampleRate == OutputSampleRate)
			return Signal;

		TemplateLibrary::Vector<T> Window;
		if (InWindow.Empty())
			Window = DefaultWindow;
		else
			Window = InWindow;

		const auto WindowSize = static_cast<SizeType>(Window.Size());

		const auto [BatchSize, Channel, SampleCount] = Signal.Size().RawArray();

		const auto ResampleRatio = double(OutputSampleRate) / double(InputSampleRate);
		const auto OutputSize = static_cast<Int64>(std::ceil(static_cast<double>(SampleCount) * ResampleRatio));

		auto Output = Tensor<T, 3, Device::CPU>::New(
			{ BatchSize, Channel, OutputSize }
		);

		if (!KeepPower)
		{
			T WindowSum = 0;
			for (SizeType i = 0; i < WindowSize; ++i)
				WindowSum += Window[i];
			for (SizeType i = 0; i < WindowSize; ++i)
				Window[i] /= WindowSum;
		}

		const auto FilterScale = std::min(1., ResampleRatio);
		const auto HalfWindow = WindowSize / 2;

		const auto OutputData = Output.Data();
		const auto InputData = Signal.Data();
		const auto StrideInp = Signal.Stride(2);

		for (SizeType B = 0; B < BatchSize; ++B)
		{
			for (SizeType C = 0; C < Channel; ++C)
			{
				const auto CurOutputData = OutputData + B * Output.Stride(0) + C * Output.Stride(1);
				const auto CurInputData = InputData + B * Signal.Stride(0) + C * Signal.Stride(1);
				Output.AppendTask(
					[=]
					{
						for (SizeType i = 0; i < OutputSize; ++i)
						{
							const auto InputPos = static_cast<double>(i) / ResampleRatio;
							const auto InputIdx = static_cast<SizeType>(InputPos);
							T Sum = 0;
							SizeType WinCount = 0;
							for (SizeType j = 0; j < WindowSize; ++j)
							{
								const SizeType WindowIdx = j;
								const SizeType InputSample = InputIdx - HalfWindow + j;
								if (InputSample >= 0 && InputSample < static_cast<int>(SampleCount))
								{
									const auto Delta = InputPos - static_cast<double>(InputSample);
									double SincVal;
									if (std::abs(Delta) < 1e-6)
										SincVal = 1.0;
									else
										SincVal = std::sin(std::numbers::pi_v<double> *FilterScale * Delta) /
										(std::numbers::pi_v<double> *FilterScale * Delta);
									Sum += CurInputData[InputSample * StrideInp] * static_cast<T>(SincVal) * Window[WindowIdx];
									WinCount++;
								}
							}
							if (WinCount > 0 && WinCount < WindowSize)
								Sum *= static_cast<T>(WindowSize) / static_cast<T>(WinCount);
							CurOutputData[i] = Sum;
						}
					}
				);
			}
		}
		return Output;
	}

	/**
	* @class StftKernel
	* @brief Implementation of Short-Time Fourier Transform (STFT)
	*/
	class StftKernel
	{
	public:
		StftKernel() = default; ///< Default constructor

		StftKernel(
			int NumFFT, int HopSize = -1, int WindowSize = -1, TemplateLibrary::Vector<Double> Window = {},
			bool Center = true, PaddingType Padding = PaddingType::Zero
		); ///< Parameterized constructor

		~StftKernel(); ///< Destructor

		friend class MFCCKernel; ///< Friend class
		inline static double PI = std::numbers::pi_v<double>; ///< Constant value of PI

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Int16, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Float32, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		Tensor<Float64, 4, Device::CPU> operator()(
			const Tensor<Float64, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		Tensor<Complex32, 4, Device::CPU> Execute(
			const Tensor<Float32, 3, Device::CPU>& Signal
		) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		Tensor<Complex64, 4, Device::CPU> Execute(
			const Tensor<Float64, 3, Device::CPU>& Signal
		) const;

		/**
		 * @brief Inverse Short-Time Fourier Transform
		 * @param Spectrogram Input spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 * @return Signal, Shape [Batch, Channel, SampleCount]
		 */
		Tensor<Float32, 3, Device::CPU> Inverse(
			const Tensor<Float32, 4, Device::CPU>& Spectrogram
		) const;

		/**
		 * @brief Inverse Short-Time Fourier Transform
		 * @param Spectrogram Input spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 * @return Signal, Shape [Batch, Channel, SampleCount]
		 */
		Tensor<Float32, 3, Device::CPU> Inverse(
			const Tensor<Complex32, 4, Device::CPU>& Spectrogram
		) const;

		/**
		 * @brief Inverse Short-Time Fourier Transform
		 * @param Spectrogram Input spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 * @return Signal, Shape [Batch, Channel, SampleCount]
		 */
		Tensor<Float64, 3, Device::CPU> Inverse(
			const Tensor<Complex64, 4, Device::CPU>& Spectrogram
		) const;

		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Float32, 3, Device::CPU>& Signal,
			Int64 HopSize
			) const;

		static Tensor<Float32, 3, Device::CPU> Inverse(
			const Tensor<Complex32, 4, Device::CPU>& Spectrogram,
			Int64 HopSize
		);

		StftKernel(const StftKernel&) = default; ///< Disable copy constructor
		StftKernel(StftKernel&&) = default; ///< Disable move constructor
		StftKernel& operator=(const StftKernel&) = default; ///< Disable copy assignment
		StftKernel& operator=(StftKernel&&) = default; ///< Disable move assignment

		auto GetStftSize() const
		{
			return NUM_FFT;
		}

		auto GetStftBins() const
		{
			return FFT_BINS;
		}

		auto GetHopSize() const
		{
			return HOP_SIZE;
		}

		auto GetWindowSize() const
		{
			return WINDOW_SIZE;
		}

		decltype(auto) GetWindow() const
		{
			return WINDOW;
		}

		auto GetFreqPerBin(Int64 SamplingRate) const
		{
			return double(SamplingRate) / 2. / double(FFT_BINS - 1);
		}

		void SetHopSize(int HopSize)
		{
			if (HopSize > 0)
				HOP_SIZE = HopSize;
		}
	private:
		int NUM_FFT = 2048; ///< FFT size
		int FFT_BINS = 1025; ///< FFT bins
		int HOP_SIZE = 512; ///< Hop size
		int WINDOW_SIZE = 2048; ///< Window size
		int PADDING = 0; ///< Padding size
		bool CENTER = true;
		int CENTER_PADDING_SIZE = 256;
		PaddingType PADDING_TYPE = PaddingType::Reflect;
		std::shared_ptr<TemplateLibrary::Vector<Double>> WINDOW;
		std::shared_ptr<TemplateLibrary::Vector<Double>> WINDOW_NORM;
	};

	/**
	 * @class MFCCKernel
	 * @brief Implementation of Mel Frequency Cepstral Coefficients (MFCC)
	 */
	class MFCCKernel
	{
	public:
		MFCCKernel() = delete; ///< Disable default constructor

		MFCCKernel(
			int SamplingRate, int NumFFT, int HopSize = -1, int WindowSize = -1, int MelBins = 0,
			double FreqMin = 20., double FreqMax = 11025., TemplateLibrary::Vector<Double> Window = {},
			bool Center = true, PaddingType Padding = PaddingType::Zero,
			DLogger _Logger = nullptr
		);

		~MFCCKernel() = default; ///< Default destructor

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Spectrogram Input spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Float32, 4, Device::CPU>& Spectrogram
			) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Spectrogram Input spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float64, 4, Device::CPU> operator()(
			const Tensor<Float64, 4, Device::CPU>& Spectrogram
			) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Int16, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(
			const Tensor<Float32, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float64, 4, Device::CPU> operator()(
			const Tensor<Float64, 3, Device::CPU>& Signal
			) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount], Spectrogram, Shape [Batch, Channel, FrameCount, StftBins]
		 */
		template <typename _Type>
		std::pair<Tensor<_Type, 4, Device::CPU>, Tensor<_Type, 4, Device::CPU>> WithSpec(const Tensor<_Type, 3, Device::CPU>& Signal) const
		{
			const auto SignalSize = Signal.Size(2);
			if (SignalSize < STFT_KERNEL.WINDOW_SIZE)
				_D_Dragonian_Lib_Throw_Exception("Signal is too short.");

			auto BgnTime = clock();
			auto Spec = STFT_KERNEL(Signal);
			if (_MyLogger)
				_MyLogger->Log((L"Stft Use Time " + std::to_wstring(clock() - BgnTime) + L"ms"), Logger::LogLevel::Info);

			return { operator()(Spec), std::move(Spec) };
		}

		const StftKernel& GetStftKernel() const
		{
			return STFT_KERNEL;
		}

		MFCCKernel(const MFCCKernel&) = default; ///< Disable copy constructor
		MFCCKernel(MFCCKernel&&) = default; ///< Disable move constructor
		MFCCKernel& operator=(const MFCCKernel&) = delete; ///< Disable copy assignment
		MFCCKernel& operator=(MFCCKernel&&) = default; ///< Disable move assignment

		auto GetMelBins() const
		{
			return MEL_BINS;
		}

		auto GetStftSize() const
		{
			return FFT_SIZE;
		}

		auto GetStftBins() const
		{
			return FFT_BINS;
		}

		auto GetSamplingRate() const
		{
			return SAMPLING_RATE;
		}

		decltype(auto) GetMelBasis() const
		{
			return WEIGHT;
		}

		decltype(auto) GetMelBasisDbl() const
		{
			return WEIGHTDBL;
		}

		auto GetFreqPerBin() const
		{
			return STFT_KERNEL.GetFreqPerBin(SAMPLING_RATE);
		}

		auto GetMaxFreq() const
		{
			return double(SAMPLING_RATE) / 2.;
		}

	private:
		StftKernel STFT_KERNEL; ///< STFT instance
		int MEL_BINS = 128; ///< Mel spectrum size
		int FFT_SIZE = 0; ///< FFT size
		int FFT_BINS = 0; ///< FFT bins
		int SAMPLING_RATE = 22050; ///< Sampling rate
		Tensor<Float32, 2, Device::CPU> WEIGHT; ///< Mel basis [MelBins, FFTSize]
		Tensor<Float64, 2, Device::CPU> WEIGHTDBL; ///< Mel basis [MelBins, FFTSize]
		DLogger _MyLogger = nullptr; ///< Logger
	};

	/**
	 * @brief Compute Constant-Q Transform (CQT)
	 * @param AudioData Input audio data
	 * @param SamplingRate Sampling rate
	 * @param HopSize Hop size
	 * @param FreqMin Minimum frequency
	 * @param CQTBins Number of CQT bins
	 * @param BinsPerOctave Number of bins per octave
	 * @param Tuning Tuning
	 * @param FilterScale Filter scale
	 * @param Norm Normalization
	 * @param Sparsity Sparsity
	 * @param Window Window type
	 * @param Scale Whether to scale
	 * @param PaddingMode Padding mode
	 * @param ResourceType Resource type
	 * @return Transformed data
	 */
	DragonianLibSTL::Vector<float> CQT(
		const DragonianLibSTL::Vector<float>& AudioData,
		int SamplingRate = 22050,
		int HopSize = 512,
		float FreqMin = 32.70f,
		int CQTBins = 84,
		int BinsPerOctave = 12,
		float Tuning = 0.f,
		float FilterScale = 1.f,
		float Norm = 1.f,
		float Sparsity = 0.01f,
		const char* Window = "Hann",
		bool Scale = true,
		const char* PaddingMode = "Constant",
		const char* ResourceType = "SOXR_HQ"
	);

	/**
	 * @brief Compute Variable-Q Transform (VQT)
	 * @param AudioData Input audio data
	 * @param SamplingRate Sampling rate
	 * @param HopSize Hop size
	 * @param FreqMin Minimum frequency
	 * @param CQTBins Number of CQT bins
	 * @param Intervals Interval type
	 * @param Gamma Gamma value
	 * @param BinsPerOctave Number of bins per octave
	 * @param Tuning Tuning
	 * @param FilterScale Filter scale
	 * @param Norm Normalization
	 * @param Sparsity Sparsity
	 * @param Window Window type
	 * @param Scale Whether to scale
	 * @param PaddingMode Padding mode
	 * @param ResourceType Resource type
	 * @return Transformed data
	 */
	DragonianLibSTL::Vector<float> VQT(
		const DragonianLibSTL::Vector<float>& AudioData,
		int SamplingRate = 22050,
		int HopSize = 512,
		float FreqMin = 32.70f,
		int CQTBins = 84,
		const char* Intervals = "Equal",
		float Gamma = 0.f,
		int BinsPerOctave = 12,
		float Tuning = 0.f,
		float FilterScale = 1.f,
		float Norm = 1.f,
		float Sparsity = 0.01f,
		const char* Window = "Hann",
		bool Scale = true,
		const char* PaddingMode = "Constant",
		const char* ResourceType = "SOXR_HQ"
	);

}

_D_Dragonian_Lib_Space_End
