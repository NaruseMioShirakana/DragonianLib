/**
 * @file Vits-Svc.hpp
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
 * @brief Vits based Singing Voice Conversion
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 *  > 2025/3/22 NaruseMioShirakana Add VitsSvc <
 *	> 2025/3/23 NaruseMioShirakana Add SoftVitsSvcV2 <
 *	> 2025/3/23 NaruseMioShirakana Add SoftVitsSvcV3 <
 *	> 2025/3/23 NaruseMioShirakana Add SoftVitsSvcV4 <
 *	> 2025/3/24 NaruseMioShirakana Add SoftVitsSvcV4Beta <
 *	> 2025/3/24 NaruseMioShirakana Add RetrievalBasedVitsSvc <
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Util/Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @class VitsSvc
 * @brief Vits based Singing Voice Conversion
 *
 * Following model path is required:
 * - "Model" : Vits based svc model path
 */
class VitsSvc : public SingingVoiceConversionModule, public OnnxModelBase<VitsSvc>
{
public:
	using _MyBase = OnnxModelBase<VitsSvc>;

	VitsSvc() = delete;
	VitsSvc(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~VitsSvc() override = default;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override = 0;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override = 0;

protected:
	Int64 _MyNoiseDims = 192;
	Int64 _MyWindowSize = 2048;

public:
	VitsSvc(const VitsSvc&) = default;
	VitsSvc(VitsSvc&&) noexcept = default;
	VitsSvc& operator=(const VitsSvc&) = default;
	VitsSvc& operator=(VitsSvc&&) noexcept = default;

protected:
	SliceDatas& PreprocessNoise(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Float32 Scale,
		Int64 Seed
	) const;
	SliceDatas& PreprocessStftNoise(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Float32 Scale
	) const;
};

/**
 * @class SoftVitsSvcV2
 * @brief Soft Vits Svc V2 (SoVitsSvc)
 *
 * Following model path is required:
 * - "Model" : Soft Vits Svc V2 model path
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - UnitsLength[REQUIRED|AUTOGEN]: UnitsLength, shape must be {BatchSize, Channels, 1}
 * - F0Embed[REQUIRED|AUTOGEN]: F0Embed, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "F0" is set
 * - F0[OPTIONAL]: F0, shape must be {BatchSize, Channels, FrameCount}, REQUIRED if "F0Embed" is not set
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 *
 * Hyper parameters example:
 * - SamplingRate/HopSize: In default [SamplingRate] equals to (50 * [HopSize]), as usual, the sampling rate is 32000 and the hop size is 640.
 */
class SoftVitsSvcV2 : public VitsSvc
{
public:
	SoftVitsSvcV2() = delete;
	SoftVitsSvcV2(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~SoftVitsSvcV2() override = default;

	SoftVitsSvcV2(const SoftVitsSvcV2&) = default;
	SoftVitsSvcV2(SoftVitsSvcV2&&) noexcept = default;
	SoftVitsSvcV2& operator=(const SoftVitsSvcV2&) = default;
	SoftVitsSvcV2& operator=(SoftVitsSvcV2&&) noexcept = default;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;
};

/**
 * @class SoftVitsSvcV3
 * @brief Soft Vits Svc V3 (SoVitsSvc3.0)
 *
 * Following model path is required:
 * - "Model" : Soft Vits Svc V3 model path
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - UnitsLength[REQUIRED|AUTOGEN]: UnitsLength, shape must be {BatchSize, Channels, 1}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 *
 * Hyper parameters example:
 * - SamplingRate/HopSize: In default [SamplingRate] equals to (100 * [HopSize]) or (150 * [HopSize]), as usual, the sampling rate is [32000/48000] and the hop size is [320].
 */
class SoftVitsSvcV3 : public SoftVitsSvcV2
{
public:
	SoftVitsSvcV3() = delete;
	SoftVitsSvcV3(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~SoftVitsSvcV3() override = default;

	SoftVitsSvcV3(const SoftVitsSvcV3&) = default;
	SoftVitsSvcV3(SoftVitsSvcV3&&) noexcept = default;
	SoftVitsSvcV3& operator=(const SoftVitsSvcV3&) = default;
	SoftVitsSvcV3& operator=(SoftVitsSvcV3&&) noexcept = default;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;
};

/**
 * @class SoftVitsSvcV4Beta
 * @brief Soft Vits Svc V4Beta (SoVitsSvc4.0-V2)
 *
 * Following model path is required:
 * - "Model" : Soft Vits Svc V4Beta model path
 *
 * Extended parameters:
 * - "NoiseDims" - The noise dims, default is 192
 * - "WindowSize" - The window size, default is 2048
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - StftNoise[REQUIRED|AUTOGEN]: StftNoise, shape must be {BatchSize, Channels, WindowSize, FrameCount}
 * - Noise[REQUIRED|AUTOGEN]: Noise, shape must be {BatchSize, Channels, NoiseDims, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 */
class SoftVitsSvcV4Beta : public VitsSvc
{
public:
	SoftVitsSvcV4Beta() = delete;
	SoftVitsSvcV4Beta(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~SoftVitsSvcV4Beta() override = default;

	SoftVitsSvcV4Beta(const SoftVitsSvcV4Beta&) = default;
	SoftVitsSvcV4Beta(SoftVitsSvcV4Beta&&) noexcept = default;
	SoftVitsSvcV4Beta& operator=(const SoftVitsSvcV4Beta&) = default;
	SoftVitsSvcV4Beta& operator=(SoftVitsSvcV4Beta&&) noexcept = default;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;
};

/**
 * @class SoftVitsSvcV4
 * @brief Soft Vits Svc V4 (SoVitsSvc4.0/SoVitsSvc4.1)
 *
 * Following model path is required:
 * - "Model" : Soft Vits Svc V4 model path
 *
 * Extended parameters:
 * - "NoiseDims" - The noise dims, default is 192
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Mel2Units[REQUIRED|AUTOGEN]: Mel2Units, used to gather units(like neaerest interpolation), shape must be {BatchSize, Channels, FrameCount}
 * - UnVoice[REQUIRED|AUTOGEN]: UnVoice, 0 if unvoice else 1, shape must be {BatchSize, Channels, WindowSize, FrameCount}
 * - Noise[REQUIRED|AUTOGEN]: Noise, shape must be {BatchSize, Channels, NoiseDims, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 */
class SoftVitsSvcV4 : public VitsSvc
{
public:
	SoftVitsSvcV4() = delete;
	SoftVitsSvcV4(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~SoftVitsSvcV4() override = default;

	SoftVitsSvcV4(const SoftVitsSvcV4&) = default;
	SoftVitsSvcV4(SoftVitsSvcV4&&) noexcept = default;
	SoftVitsSvcV4& operator=(const SoftVitsSvcV4&) = default;
	SoftVitsSvcV4& operator=(SoftVitsSvcV4&&) noexcept = default;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;
};

/**
 * @class RetrievalBasedVitsSvc
 * @brief Retrieval Based Voice Conversion (RVC)
 *
 * Following model path is required:
 * - "Model" : Soft Vits Svc V4 model path
 *
 * Extended parameters:
 * - "NoiseDims" - The noise dims, default is 192
 *
 * The input tensor is:
 * - Units[REQUIRED]: Units, shape must be {BatchSize, Channels, FrameCount, UnitsDim}
 * - UnitsLength[REQUIRED|AUTOGEN]: UnitsLength, shape must be {BatchSize, Channels, 1}
 * - F0Embed[REQUIRED|AUTOGEN]: F0Embed, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "F0" is set
 * - F0[REQUIRED]: F0, shape must be {BatchSize, Channels, FrameCount}
 * - Noise[REQUIRED|AUTOGEN]: Noise, shape must be {BatchSize, Channels, NoiseDims, FrameCount}
 * - SpeakerId[OPTIONAL|AUTOGEN]: SpeakerId, shape must be {BatchSize, Channels, 1}
 * - Speaker[OPTIONAL|AUTOGEN]: Speaker, shape must be {BatchSize, Channels, FrameCount, SpeakerCount}
 * - Volume[OPTIONAL|AUTOGEN]: Volume, shape must be {BatchSize, Channels, FrameCount}, AUTOGEN if "GTAudio" is set
 */
class RetrievalBasedVitsSvc : public VitsSvc
{
public:
	RetrievalBasedVitsSvc() = delete;
	RetrievalBasedVitsSvc(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~RetrievalBasedVitsSvc() override = default;

	RetrievalBasedVitsSvc(const RetrievalBasedVitsSvc&) = default;
	RetrievalBasedVitsSvc(RetrievalBasedVitsSvc&&) noexcept = default;
	RetrievalBasedVitsSvc& operator=(const RetrievalBasedVitsSvc&) = default;
	RetrievalBasedVitsSvc& operator=(RetrievalBasedVitsSvc&&) noexcept = default;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;

	Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End