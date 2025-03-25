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

	Unit2Ctrl(const Unit2Ctrl&) = default;
	Unit2Ctrl(Unit2Ctrl&&) noexcept = default;
	Unit2Ctrl& operator=(const Unit2Ctrl&) = default;
	Unit2Ctrl& operator=(Unit2Ctrl&&) noexcept = default;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End