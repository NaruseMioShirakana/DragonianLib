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
		const OnnxRuntimeEnviroment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~VitsSvc() override = default;

	Tensor<Float32, 4, Device::CPU> Inference(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const override;

	SliceDatas PreProcess(
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

class SoftVitsSvcV2 : public VitsSvc
{
public:
	SoftVitsSvcV2() = delete;
	SoftVitsSvcV2(
		const OnnxRuntimeEnviroment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Onnx_Singing_Voice_Conversion_Space GetDefaultLogger()
	);
	~SoftVitsSvcV2() override = default;

	SoftVitsSvcV2(const SoftVitsSvcV2&) = default;
	SoftVitsSvcV2(SoftVitsSvcV2&&) noexcept = default;
	SoftVitsSvcV2& operator=(const SoftVitsSvcV2&) = default;
	SoftVitsSvcV2& operator=(SoftVitsSvcV2&&) noexcept = default;

	SliceDatas PreProcess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const override;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End