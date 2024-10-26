/**
 * FileName: Stft.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "MyTemplateLibrary/Vector.h"

namespace DragonianLib
{
	namespace FunctionTransform
	{
		/**
		* @class STFT
		* @brief Implementation of Short-Time Fourier Transform (STFT)
		*/
		class STFT
		{
		public:
			STFT() = default; ///< Default constructor
			~STFT(); ///< Destructor
			STFT(int WindowSize, int HopSize, int FFTSize = 0); ///< Parameterized constructor
			inline static double PI = 3.14159265358979323846; ///< Constant value of PI
			/**
			 * @brief Perform STFT on audio data
			 * @param audioData Input audio data
			 * @return Transformed data and timestamp
			 */
			std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
		private:
			int WINDOW_SIZE = 2048; ///< Window size
			int HOP_SIZE = WINDOW_SIZE / 4; ///< Hop size
			int FFT_SIZE = WINDOW_SIZE / 2 + 1; ///< FFT size
		};

		/**
		 * @class Mel
		 * @brief Implementation of Mel Frequency Cepstral Coefficients (MFCC)
		 */
		class Mel
		{
		public:
			Mel() = delete; ///< Disable default constructor
			~Mel() = default; ///< Default destructor
			Mel(int WindowSize, int HopSize, int SamplingRate, int MelSize = 0, double FreqMin = 20., double FreqMax = 11025.); ///< Parameterized constructor
			/**
			 * @brief Get Mel spectrum
			 * @param audioData Input audio data
			 * @return Mel spectrum and timestamp
			 */
			std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<int16_t>& audioData) const;
			std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<double>& audioData) const;
			std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
		private:
			STFT stft; ///< STFT instance
			int MEL_SIZE = 128; ///< Mel spectrum size
			int FFT_SIZE = 0; ///< FFT size
			int sr = 22050; ///< Sampling rate
			DragonianLibSTL::Vector<float> MelBasis; ///< Mel basis
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
}
