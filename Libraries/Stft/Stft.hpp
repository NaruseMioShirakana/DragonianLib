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

_D_Dragonian_Lib_Space_Begin

namespace FunctionTransform
{
	class MFCCKernel;

	/**
	* @class StftKernel
	* @brief Implementation of Short-Time Fourier Transform (STFT)
	*/
	class StftKernel
	{
	public:
		StftKernel() = default; ///< Default constructor

		StftKernel(
			int NumFFT, int HopSize = -1, int WindowSize = -1, const double* Window = nullptr,
			bool Center = true, PaddingType Padding = PaddingType::Reflect
		); ///< Parameterized constructor

		~StftKernel(); ///< Destructor

		friend class MFCCKernel; ///< Friend class
		inline static double PI = 3.14159265358979323846; ///< Constant value of PI

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
		double WINDOW_POWER_SUM = 0.0; ///< Window power sum
		PaddingType PADDING_TYPE = PaddingType::Reflect;
		TemplateLibrary::Vector<Double> WINDOW;
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
			double FreqMin = 20., double FreqMax = 11025., const double* Window = nullptr,
			bool Center = true, PaddingType Padding = PaddingType::Reflect,
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
