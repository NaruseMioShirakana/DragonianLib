/**
 * @file Nsf-Hifigan.hpp
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
 * @brief Vocoder model for voice generation
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/Vocoder/Vocoder.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

class NsfHifigan : public VocoderBase
{
public:
	NsfHifigan() = delete;
	NsfHifigan(
		const std::wstring& _Path,
		const OnnxRuntimeEnvironment& _Environment,
		Int64 _SamplingRate = 16000,
		Int64 _MelBins = 128,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Vocoder_Space GetDefaultLogger()
	) : VocoderBase(_Path, _Environment, _SamplingRate, _MelBins, _Logger)
	{
	}
	~NsfHifigan() override = default;
	NsfHifigan(const NsfHifigan&) = default;
	NsfHifigan(NsfHifigan&&) noexcept = default;
	NsfHifigan& operator=(const NsfHifigan&) = default;
	NsfHifigan& operator=(NsfHifigan&&) noexcept = default;

	Tensor<Float32, 3, Device::CPU> Forward(
		const Tensor<Float32, 4, Device::CPU>& _Mel,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0 = std::nullopt
	) const override;
};

_D_Dragonian_Lib_Onnx_Vocoder_End

