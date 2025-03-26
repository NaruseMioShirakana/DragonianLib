/**
 * @file Reflow-Svc.hpp
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
 * @brief Reflow based Singing Voice Conversion
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Model/Ctrls.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @class ReflowSvc
 * @brief Reflow based Singing Voice Conversion
 *
 * Following model path is required:
 * - "Ctrl" : Encoder mode or ddsp path
 * - "Velocity" : Velocity path
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
class ReflowSvc : public Unit2Ctrl
{
public:
	ReflowSvc() = delete;
	ReflowSvc(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~ReflowSvc() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;

	ReflowSvc(const ReflowSvc&) = default;
	ReflowSvc(ReflowSvc&&) noexcept = default;
	ReflowSvc& operator=(const ReflowSvc&) = default;
	ReflowSvc& operator=(ReflowSvc&&) noexcept = default;

protected:
	OnnxRuntimeModel _MyVelocity;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End