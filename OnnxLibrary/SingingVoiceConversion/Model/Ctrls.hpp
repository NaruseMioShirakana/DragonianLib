/**
 * @file Ctrls.hpp
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
 * @brief Base class of DDSP/DIFFUSION/REFLOW
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Util/Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @class Unit2Ctrl
 * @brief Unit2Ctrl (Encoder of diffusion model, Pre-model of [diffusion, reflow] or DDSP model.)
 *
 * Following model path is required:
 * - "Ctrl" : Ctrl mode path
 *
 * Extended parameters:
 * - None
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 *
 */
class Unit2Ctrl : public SingingVoiceConversionModule, public OnnxModelBase<Unit2Ctrl>
{
public:
	using _MyBase = OnnxModelBase<Unit2Ctrl>;

	Unit2Ctrl() = delete;
	Unit2Ctrl(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~Unit2Ctrl() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override = 0;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	OrtTuple Extract(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const;

protected:
	Int64 _MyMelBins = 128;

public:
	Unit2Ctrl(const Unit2Ctrl&) = default;
	Unit2Ctrl(Unit2Ctrl&&) noexcept = default;
	Unit2Ctrl& operator=(const Unit2Ctrl&) = default;
	Unit2Ctrl& operator=(Unit2Ctrl&&) noexcept = default;

protected:
	Unit2Ctrl(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger,
		bool
	);
	std::optional<Tensor<Float32, 4, Device::CPU>>& PreprocessSpec(
		std::optional<Tensor<Float32, 4, Device::CPU>>& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Int64 Seed,
		Float32 Scale,
		const DLogger& Logger = nullptr
	) const;
};

/**
 * @class Naive
 * @brief Naive (Pre-model Naive)
 *
 * Following model path is required:
 * - "Ctrl" : Ctrl mode path
 *
 * Extended parameters:
 * - None
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 *
 * The output tensor is:
 * - MelSpec: MelSpec, shape is {BatchSize, Channels, MelBins, FrameCount}
 *
 */
class Naive : public Unit2Ctrl
{
public:
	Naive() = delete;
	Naive(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~Naive() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;

	Naive(const Naive&) = default;
	Naive(Naive&&) noexcept = default;
	Naive& operator=(const Naive&) = default;
	Naive& operator=(Naive&&) noexcept = default;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End