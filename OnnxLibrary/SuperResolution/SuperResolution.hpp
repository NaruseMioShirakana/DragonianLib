/**
 * FileName: SuperResolution.hpp
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
#include "OnnxLibrary/Base/OrtBase.hpp"
#include "TensorLib/Include/Base/Tensor/Tensor.h"
#include "Libraries/Image-Video/ImgVideo.hpp"

#define _D_Dragonian_Lib_Lib_Super_Resolution_Header \
	_D_Dragonian_Lib_Onnx_Runtime_Header \
	namespace SuperResolution \
	{

#define _D_Dragonian_Lib_Lib_Super_Resolution_End \
	} \
	_D_Dragonian_Lib_Onnx_Runtime_End

#define _D_Dragonian_Lib_Onnx_Super_Resolution_Space _D_Dragonian_Lib_Onnx_Runtime_Space SuperResolution::

_D_Dragonian_Lib_Lib_Super_Resolution_Header

DLogger& GetDefaultLogger() noexcept;

struct HyperParameters
{
    std::wstring RGBModel;
    std::wstring AlphaModel;
    Int64 InputWidth = 64;
    Int64 InputHeight = 64;
    Int64 ScaleW = 2;
    Int64 ScaleH = 2;
    ProgressCallback Callback = nullptr;
};

class SuperResolution
{
public:
    SuperResolution(
        const HyperParameters& _Parameters
    ) : _MyInputWidth(_Parameters.InputWidth), _MyInputHeight(_Parameters.InputHeight),
        _MyScaleH(_Parameters.ScaleH), _MyScaleW(_Parameters.ScaleW), _MyCallback(_Parameters.Callback)
    {

    }
    virtual ~SuperResolution() = default;
    virtual std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const = 0;

    SuperResolution(const SuperResolution&) = default;
    SuperResolution(SuperResolution&&) noexcept = default;
    SuperResolution& operator=(const SuperResolution&) = default;
    SuperResolution& operator=(SuperResolution&&) noexcept = default;

protected:
    Int64 _MyInputWidth = 64;
    Int64 _MyInputHeight = 64;
    Int64 _MyScaleH = 2;
    Int64 _MyScaleW = 2;
    ProgressCallback _MyCallback = nullptr;
};

_D_Dragonian_Lib_Lib_Super_Resolution_End