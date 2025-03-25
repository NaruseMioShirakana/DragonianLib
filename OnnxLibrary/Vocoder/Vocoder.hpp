/**
 * @file Vocoder.hpp
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
 * @brief Base class for Vocoder models
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/Base/OrtBase.hpp"

#define _D_Dragonian_Lib_Onnx_Vocoder_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace Vocoder {

#define _D_Dragonian_Lib_Onnx_Vocoder_End } _D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Onnx_Vocoder_Space \
	_D_Dragonian_Lib_Onnx_Runtime_Space Vocoder::

_D_Dragonian_Lib_Onnx_Vocoder_Header

DLogger& GetDefaultLogger() noexcept;

/**
 * @class VocoderBase
 * @brief Hubert model for units encoding
 */
class VocoderBase : public OnnxModelBase<VocoderBase>
{
public:
	using _MyBase = OnnxModelBase<VocoderBase>;
	friend _MyBase;
	VocoderBase() = delete;
	VocoderBase(
		const std::wstring& _Path,
		const OnnxRuntimeEnvironment& _Environment,
		Int64 _SamplingRate = 16000,
		Int64 _MelBins = 128,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Vocoder_Space GetDefaultLogger()
	);
	virtual ~VocoderBase() = default;

	Int64 GetSamplingRate() const noexcept
	{
		return _MySamplingRate;
	}

	Int64 GetMelBins() const noexcept
	{
		return _MyMelBins;
	}

	Int64 GetBinAxis() const noexcept
	{
		return _MyBinAxis;
	}

	/**
	 * @brief Inference with Mel spectrogram and F0
	 * @param _Mel Mel spectrogram, shape: [BatchSize, Channel, MelBins/Frames, Frames/MelBins]
	 * @param _F0 F0, shape: [BatchSize, Channel, Frames]
	 * @return Audio, shape: [BatchSize, Channel, AudioLength]
	 */
	virtual Tensor<Float32, 3, Device::CPU> Forward(
		const Tensor<Float32, 4, Device::CPU>& _Mel,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0 = std::nullopt
	) const;

private:
	Int64 _MySamplingRate = 16000;
	Int64 _MyMelBins = 128;
	Int64 _MyBinAxis = -1;

protected:
	Tensor<Float32, 3, Device::CPU> Inference(
		const Tensor<Float32, 4, Device::CPU>& _Mel,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0 = std::nullopt
	) const;

public:
	VocoderBase(const VocoderBase&) = default;
	VocoderBase(VocoderBase&&) noexcept = default;
	VocoderBase& operator=(const VocoderBase&) = default;
	VocoderBase& operator=(VocoderBase&&) noexcept = default;
};

_D_Dragonian_Lib_Onnx_Vocoder_End