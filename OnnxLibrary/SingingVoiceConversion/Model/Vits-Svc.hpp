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
#include "Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
* @class VitsSvc
* @brief Vits based Singing Voice Conversion, extended parameters is ["NoiseDims"], model path is ["Model"]
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
	) const override;

	SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override = 0;

protected:
	Int64 _MyNoiseDims = 192;

public:
	VitsSvc(const VitsSvc&) = default;
	VitsSvc(VitsSvc&&) noexcept = default;
	VitsSvc& operator=(const VitsSvc&) = default;
	VitsSvc& operator=(VitsSvc&&) noexcept = default;
};

/**
 * @class SoftVitsSvcV2
 * @brief Soft Vits Svc V2, extended parameters is ["NoiseDims"], model path is ["Model"], in common, the input count is 3 or 4, the output count is 1, the input tensor is [Hubert, AudioFrame, F0/F0Embed, [OPTIONAL:SpeakerId]], the output tensor is [Audio].
 *
 * In default [SamplingRate] equals to (50 * [HopSize]), as usual, the sampling rate is 32000 and the hop size is 640.
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
};

/**
 * @class SoftVitsSvcV2
 * @brief Soft Vits Svc V2, extended parameters is ["NoiseDims"], model path is ["Model"], in common, the input count is 3 or 4, the output count is 1, the input tensor is [Hubert, AudioFrame, F0, [OPTIONAL:SpeakerId]], the output tensor is [Audio].
 *
 * In default [SamplingRate] equals to (100 * [HopSize]) or (150 * [HopSize]), as usual, the sampling rate is 32000/48000 and the hop size is 320
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
};

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

private:
	Int64 _MyWindowSize = 2048;
};

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
};

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
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End