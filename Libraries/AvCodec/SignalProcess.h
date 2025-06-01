/**
 * @file SignalProcess.h
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
 * @brief Implementation of signal processing functions
 * @changes
 *  > 2025/5/27 NaruseMioShirakana created <
 */

#pragma once

#include "TensorLib/Include/Base/Tensor/Functional.h"

_D_Dragonian_Lib_Space_Begin

namespace Signal
{
	/**
	 * @brief Sinc Interpolate
	 * @tparam T Value Type
	 */
	template <typename T>
	class ResampleKernel
	{
	public:
		ResampleKernel(
			TemplateLibrary::Vector<T> _Window,
			bool _Normalize = false
		) : _MyWindow(std::make_shared<TemplateLibrary::Vector<T>>(std::move(_Window)))
		{
			auto& Wn = *_MyWindow;
			const auto WindowSize = static_cast<Int64>(Wn.Size());
			if (_Normalize)
			{
				T WindowSum = 0;
				for (SizeType i = 0; i < WindowSize; ++i)
					WindowSum += Wn[i];
				for (SizeType i = 0; i < WindowSize; ++i)
					Wn[i] /= WindowSum;
			}
		}

		Tensor<T, 3, Device::CPU> operator()(
			const Tensor<T, 3, Device::CPU>& Signal,
			SizeType InputSampleRate,
			SizeType OutputSampleRate
		) const
		{
			if (InputSampleRate == OutputSampleRate)
				return Signal;

			const auto WindowSize = static_cast<SizeType>(_MyWindow->Size());

			const auto [BatchSize, Channel, SampleCount] = Signal.Size().RawArray();

			const auto ResampleRatio = double(OutputSampleRate) / double(InputSampleRate);
			const auto OutputSize = static_cast<Int64>(std::ceil(static_cast<double>(SampleCount) * ResampleRatio));

			auto Output = Tensor<T, 3, Device::CPU>::New(
				{ BatchSize, Channel, OutputSize }
			);

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
						[=](auto __Win, auto)  // NOLINT(performance-unnecessary-value-param)
						{
							const auto Window = __Win->Data();
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
						},
						_MyWindow,
						Signal.Buffer()
					);
				}
			}
			return Output;
		}

		Tensor<T, 2, Device::CPU> operator()(
			const Tensor<T, 2, Device::CPU>& Signal,
			SizeType InputSampleRate,
			SizeType OutputSampleRate
			) const
		{
			return operator()(Signal.UnSqueeze(0), InputSampleRate, OutputSampleRate).Squeeze(0);
		}

	private:
		std::shared_ptr<TemplateLibrary::Vector<T>> _MyWindow;
	};
}

_D_Dragonian_Lib_Space_End