/**
 * FileName: MoeSuperResolution.hpp
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
#include "TensorRT/TensorRTBase/TRTBase.hpp"
#include "Libraries/Image-Video/ImgVideo.hpp"

#define _D_Dragonian_Lib_TRT_Sr_Space_Header _D_Dragonian_TensorRT_Lib_Space_Header namespace SuperResolution {
#define _D_Dragonian_Lib_TRT_Sr_Space_End } _D_Dragonian_TensorRT_Lib_Space_End

_D_Dragonian_Lib_TRT_Sr_Space_Header

class MoeSR
{
public:
    MoeSR(
        const std::wstring& RGBModel,
        Int64 Scale,
        const TrtConfig& TrtSettings,
        ProgressCallback _Callback = nullptr
    );
    ~MoeSR();

	std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Bitmap
    );
private:
    MoeSR(const MoeSR&) = delete;
    MoeSR(MoeSR&&) = delete;
    MoeSR& operator=(const MoeSR&) = delete;
    MoeSR& operator=(MoeSR&&) = delete;

    ProgressCallback _MyCallback;
    std::unique_ptr<TrtModel> Model = nullptr;
    InferenceSession _MySession;
    Int64 _MyScaleH = 2;
    Int64 _MyScaleW = 2;
    static inline std::vector<DynaShapeSlice> DynaSetting{
        {
            "DynaArg0",
            nvinfer1::Dims4(1, 3, 64, 64),
            nvinfer1::Dims4(1, 3, 128, 128),
            nvinfer1::Dims4(1, 3, 192, 192)
        }
    };
};

_D_Dragonian_Lib_TRT_Sr_Space_End