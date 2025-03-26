/**
 * @file Diffusion-Svc.hpp
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
 * @brief Diffusion based Singing Voice Conversion
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "Ctrls.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Util/Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @class ProphesierDiffusion
 * @brief ProphesierDiffusion (DiffusionSvcV1)
 *
 * Following model path is required:
 * - "Ctrl" : Encoder mode path
 * - "Denoiser" : Denoiser path
 * - "NoisePredictor"[OPTIONAL] : Noise predictor path
 * - "AlphaCumprod"[OPTIONAL] : Alpha cumprod path
 *
 * Extended parameters:
 * - None
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - GTSpec[REQUIRED|AUTOGEN]: GTSpec, shape must be {BatchSize, Channels, MelBins, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * 
 * The output tensor is:
 * - MelSpec: MelSpec, shape is {BatchSize, Channels, MelBins, FrameCount}
 *
 */
class ProphesierDiffusion : public Unit2Ctrl
{
public:
	ProphesierDiffusion() = delete;
	ProphesierDiffusion(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~ProphesierDiffusion() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;

	ProphesierDiffusion(const ProphesierDiffusion&) = default;
	ProphesierDiffusion(ProphesierDiffusion&&) noexcept = default;
	ProphesierDiffusion& operator=(const ProphesierDiffusion&) = default;
	ProphesierDiffusion& operator=(ProphesierDiffusion&&) noexcept = default;

protected:
	OnnxRuntimeModel _MyDenoiser;
	OnnxRuntimeModel _MyNoisePredictor;
	OnnxRuntimeModel _MyAlphaCumprod;
};

/**
 * @class DiffusionSvc
 * @brief DiffusionSvc (DiffusionSvcV2 Later)
 *
 * Following model path is required:
 * - "Ctrl" : Encoder mode path
 * - "Denoiser" : Denoiser path
 * - "NoisePredictor"[OPTIONAL] : Noise predictor path
 * - "AlphaCumprod"[OPTIONAL] : Alpha cumprod path
 *
 * Extended parameters:
 * - None
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - GTSpec[REQUIRED|AUTOGEN]: GTSpec, shape must be {BatchSize, Channels, MelBins, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 *
 * The output tensor is:
 * - MelSpec: MelSpec, shape is {BatchSize, Channels, MelBins, FrameCount}
 *
 */
class DiffusionSvc : public Unit2Ctrl
{
public:
	DiffusionSvc() = delete;
	DiffusionSvc(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~DiffusionSvc() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;

	DiffusionSvc(const DiffusionSvc&) = default;
	DiffusionSvc(DiffusionSvc&&) noexcept = default;
	DiffusionSvc& operator=(const DiffusionSvc&) = default;
	DiffusionSvc& operator=(DiffusionSvc&&) noexcept = default;

protected:
	OnnxRuntimeModel _MyDenoiser;
	OnnxRuntimeModel _MyNoisePredictor;
	OnnxRuntimeModel _MyAlphaCumprod;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End