/**
 * @file FCPE.hpp
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
 * @brief F0Predictor 
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */

#pragma once
#include "OnnxLibrary/Base/OrtBase.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

class FCPE : public BaseF0Extractor, public OnnxRuntime::OnnxModelBase<FCPE>
{
public:
    FCPE(
		const void* _ModelHParams
    );
    ~FCPE() override = default;

	Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Float64, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
    ) const override;

	Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Float32, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
    ) const override;

	Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Int16, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
    ) const override;

    FCPE(const FCPE&) = default;
    FCPE(FCPE&&) = default;
    FCPE& operator=(const FCPE&) = default;
    FCPE& operator=(FCPE&&) = default;

private:
    Int64 _MySamplingRate = 16000;
};

_D_Dragonian_Lib_F0_Extractor_End