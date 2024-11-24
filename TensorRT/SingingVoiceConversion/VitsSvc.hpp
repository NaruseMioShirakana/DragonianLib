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

_D_Dragonian_Lib_TRT_Svc_Space_Header

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

    void EmptyCache() override;

private:
    std::unique_ptr<TrtModel> VitsSvcModel;
    std::shared_ptr<TrtModel> HubertModel;
    std::unordered_map<size_t, std::shared_ptr<InferenceSession>> VitsSvcSession;
    std::unordered_map<size_t, std::shared_ptr<InferenceSession>> HubertSession;

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

public:
    static inline const std::vector<DynaShapeSlice> VitsSvcDefaultsDynaSetting{
    {"source", nvinfer1::Dims3(1, 1, 320ll * 20), nvinfer1::Dims3(1, 1, 320ll * 200), nvinfer1::Dims3(1, 1, 320ll * 400)},
    {"c", nvinfer1::Dims3(1, 100, 256), nvinfer1::Dims3(1, 200, 256), nvinfer1::Dims3(1, 400, 256)},
    {"phone", nvinfer1::Dims3(1, 100, 256), nvinfer1::Dims3(1, 200, 256), nvinfer1::Dims3(1, 400, 256)},
    {"f0", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"pitch", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"pitchf", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"mel2ph", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"uv", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"noise", nvinfer1::Dims3(1, 192, 100), nvinfer1::Dims3(1, 192, 200), nvinfer1::Dims3(1, 192, 400)},
    {"sid", nvinfer1::Dims2(100, 1), nvinfer1::Dims2(200, 1), nvinfer1::Dims2(400, 1) },
    {"vol", nvinfer1::Dims2(1, 100), nvinfer1::Dims2(1, 200), nvinfer1::Dims2(1, 400) },
    {"rnd", nvinfer1::Dims3(1, 192, 100), nvinfer1::Dims3(1, 192, 200), nvinfer1::Dims3(1, 192, 400) }
    };
};

_D_Dragonian_Lib_TRT_Svc_Space_End
