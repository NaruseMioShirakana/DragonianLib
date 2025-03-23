/**
 * @file Hubert.hpp
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
 * @brief Hubert model for units encoding
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Created <
 *  > 2025/3/19 NaruseMioShirakana Added HubertBase <
 */

#pragma once
#include "OnnxLibrary/Base/OrtBase.hpp"

#define _D_Dragonian_Lib_Onnx_UnitsEncoder_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace UnitsEncoder {

#define _D_Dragonian_Lib_Onnx_UnitsEncoder_End } _D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Onnx_UnitsEncoder_Space \
	_D_Dragonian_Lib_Onnx_Runtime_Space UnitsEncoder::

_D_Dragonian_Lib_Onnx_UnitsEncoder_Header

DLogger& GetDefaultLogger() noexcept;

/**
 * @class HubertBase
 * @brief Hubert model for units encoding
 */
class HubertBase : public OnnxModelBase<HubertBase>
{
public:
	using _MyBase = OnnxModelBase<HubertBase>;
	friend _MyBase;

	HubertBase() = delete;
	HubertBase(
		const std::wstring& _Path,
		const OnnxRuntimeEnvironment& _Environment,
		Int64 _SamplingRate = 16000,
		Int64 _UnitsDims = 768,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_UnitsEncoder_Space GetDefaultLogger()
	);
	virtual ~HubertBase() = default;

	/**
	 * @brief Inference with unit encoder
	 * @param _PCMData 16kHz PCM data (BatchSize, Channels, Samples)
	 * @param _SamplingRate Sampling rate of the input data
	 * @param _Mask Mask data (BatchSize, Channels, Samples)
	 * @return Unit features (BatchSize, Channels, UnitsDims, Frames) or (BatchSize, Channels, Frames, UnitsDims)
	 */
	virtual Tensor<Float32, 4, Device::CPU> Forward(
		const Tensor<Float32, 3, Device::CPU>& _PCMData,
		Int64 _SamplingRate = 16000,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask = std::nullopt
	) const;

	/**
	 * @brief Get the sampling rate of the model
	 * @return Sampling rate
	 */
	Int64 GetSamplingRate() const noexcept
	{
		return _MySamplingRate;
	}

	/**
	 * @brief Get the units dims of the model
	 * @return Units dims
	 */
	Int64 GetUnitsDims() const noexcept
	{
		return _MyUnitsDims;
	}

	/**
	 * @brief Get the units axis of the model
	 * @return Units axis
	 */
	Int64 GetUnitsAxis() const noexcept
	{
		return _MyUnitsAxis;
	}

private:
	Int64 _MySamplingRate = 16000;
	Int64 _MyUnitsDims = 768;
	Int64 _MyUnitsAxis = -1;

public:
	HubertBase(const HubertBase&) = default;
	HubertBase(HubertBase&&) noexcept = default;
	HubertBase& operator=(const HubertBase&) = default;
	HubertBase& operator=(HubertBase&&) noexcept = default;

protected:
	Tensor<Float32, 4, Device::CPU> InferenceModel(
		const Tensor<Float32, 3, Device::CPU>& _PCMData,
		Int64 _SamplingRate = 16000,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask = std::nullopt
	) const;
};

using Hubert = HubertBase;
using ContentVec = HubertBase;
using UnitsEncoderBase = HubertBase;

_D_Dragonian_Lib_Onnx_UnitsEncoder_End