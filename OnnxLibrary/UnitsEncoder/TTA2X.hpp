/**
 * @file TTA2X.hpp
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
 * @brief TTA2X model for units encoding
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 *  > 2025/3/21 NaruseMioShirakana Added TTA2X <
 */

#pragma once
#include "OnnxLibrary/UnitsEncoder/Hubert.hpp"

_D_Dragonian_Lib_Onnx_UnitsEncoder_Header

class TTA2X : public HubertBase
{
public:
	TTA2X() = delete;
	TTA2X(
		const std::wstring& _Path,
		const OnnxRuntimeEnvironment& _Environment,
		Int64 _SamplingRate = 16000,
		Int64 _UnitsDims = 768,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_UnitsEncoder_Space GetDefaultLogger()
	) : HubertBase(_Path, _Environment, _SamplingRate, _UnitsDims, _Logger)
	{
	}
	~TTA2X() override = default;

	/**
	 * @brief Inference with unit encoder
	 * @param _PCMData 16kHz PCM data (BatchSize, Channels, Samples)
	 * @param _SamplingRate Sampling rate of the input data
	 * @param _Mask Mask data (BatchSize, Channels, Samples)
	 * @return Unit features (BatchSize, Channels, Frames, UnitsDims)
	 */
	Tensor<Float32, 4, Device::CPU> Forward(
		const Tensor<Float32, 3, Device::CPU>& _PCMData,
		Int64 _SamplingRate = 16000,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask = std::nullopt
	) const override;

	TTA2X(const TTA2X&) = default;
	TTA2X(TTA2X&&) noexcept = default;
	TTA2X& operator=(const TTA2X&) = default;
	TTA2X& operator=(TTA2X&&) noexcept = default;
};

_D_Dragonian_Lib_Onnx_UnitsEncoder_End