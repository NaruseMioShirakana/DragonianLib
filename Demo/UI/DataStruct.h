#pragma once
#include "framework.h"

using FloatTensor2D = DragonianLib::Tensor<DragonianLib::Float32, 2, DragonianLib::Device::CPU>;
using Int16Tensor2D = DragonianLib::Tensor<DragonianLib::Int16, 2, DragonianLib::Device::CPU>;

struct MyAudioData
{
	DragonianLib::Int64 SamplingRate;
	FloatTensor2D Audio;
	FloatTensor2D F0;
	FloatTensor2D Spec;
	FloatTensor2D Mel;
};