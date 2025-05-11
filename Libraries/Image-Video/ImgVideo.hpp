/*
* file: ImgVideo.hpp
* info: Image data slicing class implementation
*
* Author: Maplespe(mapleshr@icloud.com)
*
* date: 2023-3-4 Create.
*/
#pragma once

#include "TensorLib/Include/Base/Tensor/Tensor.h"

#define _D_Dragonian_Lib_Image_Video_Header _D_Dragonian_Lib_Space_Begin namespace ImageVideo {
#define _D_Dragonian_Lib_Image_Video_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Image_Video_Header

using Image5D = Tensor<UInt8, 5, Device::CPU>;
using NormalizedImage5D = Tensor<Float32, 5, Device::CPU>;
using Image3D = Tensor<UInt8, 3, Device::CPU>;
using NormalizedImage3D = Tensor<Float32, 3, Device::CPU>;

std::tuple<Image5D, Int64, Int64> LoadAndSplitImage(
	const std::wstring& Path,
	Int64 WindowHeight = 0,
	Int64 WindowWidth = 0,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

std::tuple<NormalizedImage5D, Int64, Int64> LoadAndSplitImageNorm(
	const std::wstring& Path,
	Int64 WindowHeight = 0,
	Int64 WindowWidth = 0,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

std::tuple<Image5D, Int64, Int64> LoadAndSplitImage(
	const Image3D& ImageData,
	Int64 WindowHeight = 0,
	Int64 WindowWidth = 0,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

std::tuple<NormalizedImage5D, Int64, Int64> LoadAndSplitImageNorm(
	const NormalizedImage3D& ImageData,
	Int64 WindowHeight = 0,
	Int64 WindowWidth = 0,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

NormalizedImage3D CombineImage(
	const std::tuple<const NormalizedImage5D&, Int64, Int64>& BitMapSlice,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

NormalizedImage3D CombineImage(
	const std::tuple<NormalizedImage5D, Int64, Int64>& BitMapSlice,
	Int64 HopHeight = 0,
	Int64 HopWidth = 0
);

void SaveBitmap(
	const Image3D& ImageData,
	const std::wstring& Path,
	UInt Quality = 100
);

void SaveBitmap(
	const NormalizedImage3D& ImageData,
	const std::wstring& Path,
	UInt Quality = 100
);

_D_Dragonian_Lib_Image_Video_End
