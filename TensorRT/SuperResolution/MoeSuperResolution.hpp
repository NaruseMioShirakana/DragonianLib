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
#include "../TRTBase.hpp"
#include "Image-Video/ImgVideo.hpp"


namespace tlibsr
{
    using namespace TensorRTLib;

    class MoeSR
    {
    public:
        MoeSR(const std::wstring& RGBModel, long Scale, const TrtConfig& TrtSettings, ProgressCallback _Callback);
        ~MoeSR();

        DragonianLib::Image& Infer(DragonianLib::Image& _Image, const InferenceDeviceBuffer& _Buffer, int64_t _BatchSize) const;
    private:
        MoeSR(const MoeSR&) = delete;
        MoeSR(MoeSR&&) = delete;
        MoeSR& operator=(const MoeSR&) = delete;
        MoeSR& operator=(MoeSR&&) = delete;

        ProgressCallback Callback_;
        std::unique_ptr<TrtModel> Model = nullptr;
        long ScaleFactor = 2;
        DragonianLibSTL::Vector<DynaShapeSlice> DynaSetting{
	        {
	        	"DynaArg0",
	            nvinfer1::Dims4(1, 3, 64, 64),
	            nvinfer1::Dims4(1, 3, 64, 64),
	            nvinfer1::Dims4(1, 3, 64, 64)
	        }
        };
    };
}