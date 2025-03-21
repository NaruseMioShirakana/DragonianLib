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
		~StftKernel(); ///< Destructor
		StftKernel(int WindowSize, int HopSize, int FFTSize = 0); ///< Parameterized constructor
		friend class MFCCKernel; ///< Friend class
		inline static double PI = 3.14159265358979323846; ///< Constant value of PI

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, FFTSize]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, FFTSize]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const;

		/**
		 * @brief Short-Time Fourier Transform
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Spectrogram, Shape [Batch, Channel, FrameCount, FFTSize]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const;
	private:
		int WINDOW_SIZE = 2048; ///< Window size
		int HOP_SIZE = WINDOW_SIZE / 4; ///< Hop size
		int FFT_SIZE = WINDOW_SIZE / 2 + 1; ///< FFT size
	};

	/**
	 * @class MFCCKernel
	 * @brief Implementation of Mel Frequency Cepstral Coefficients (MFCC)
	 */
	class MFCCKernel
	{
	public:
		MFCCKernel() = delete; ///< Disable default constructor
		~MFCCKernel() = default; ///< Default destructor
		MFCCKernel(
			int WindowSize, int HopSize, int SamplingRate, int MelBins = 0,
			double FreqMin = 20., double FreqMax = 11025.,
			DLogger _Logger = nullptr
		);

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Float32, 3, Device::CPU>& Signal) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Float64, 3, Device::CPU>& Signal) const;

		/**
		 * @brief Mel Frequency Cepstral Coefficients
		 * @param Signal Input signal, Shape [Batch, Channel, SampleCount]
		 * @return Log mel spectrogram, Shape [Batch, Channel, MelBins, FrameCount]
		 */
		Tensor<Float32, 4, Device::CPU> operator()(const Tensor<Int16, 3, Device::CPU>& Signal) const;
	private:
		StftKernel _MyStftKernel; ///< STFT instance
		int _MyMelBins = 128; ///< Mel spectrum size
		int _MyFFTSize = 0; ///< FFT size
		int _MySamplingRate = 22050; ///< Sampling rate
		DragonianLibSTL::Vector<float> _MyMelBasis; ///< Mel basis [MelBins, FFTSize]
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
