/**
 * FileName: VitsSvc.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "SvcBase.hpp"

_D_Dragonian_Lib_NCNN_Svc_Space_Header

class VitsSvc : public SvcBase
{
public:

    VitsSvc(
        const VitsSvcConfig& _Hps,
        const ProgressCallback& _ProgressCallback
    );
    ~VitsSvc() override;
    VitsSvc(const VitsSvc&) = delete;
    VitsSvc(VitsSvc&&) = delete;
    VitsSvc& operator=(const VitsSvc&) = delete;
    VitsSvc& operator=(VitsSvc&&) = delete;

    [[nodiscard]] DragonianLibSTL::Vector<float> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params
    ) override;

private:
    std::shared_ptr<void> VitsSvcModel;
    std::shared_ptr<void> HubertModel;

    std::wstring VitsSvcVersion;

    TensorXData SoVits4Preprocess(
        const DragonianLibSTL::Vector<float>& HiddenUnit,
        const DragonianLibSTL::Vector<float>& F0,
        const DragonianLibSTL::Vector<float>& Volume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
        const InferenceParams& Params,
        int32_t SourceSamplingRate,
        int64_t AudioSize
    ) const;

    TensorXData RVCTensorPreprocess(
        const DragonianLibSTL::Vector<float>& HiddenUnit,
        const DragonianLibSTL::Vector<float>& F0,
        const DragonianLibSTL::Vector<float>& Volume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
        const InferenceParams& Params,
        int32_t SourceSamplingRate,
        int64_t AudioSize
    ) const;

    TensorXData Preprocess(
        const DragonianLibSTL::Vector<float>& HiddenUnit,
        const DragonianLibSTL::Vector<float>& F0,
        const DragonianLibSTL::Vector<float>& Volume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
        const InferenceParams& Params,
        int32_t SourceSamplingRate,
        int64_t AudioSize
    ) const;
};

_D_Dragonian_Lib_NCNN_Svc_Space_End
