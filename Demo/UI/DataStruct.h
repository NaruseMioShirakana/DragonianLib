#pragma once
#include "framework.h"

using FloatTensor1D = DragonianLib::Tensor<DragonianLib::Float32, 1, DragonianLib::Device::CPU>;
using FloatTensor2D = DragonianLib::Tensor<DragonianLib::Float32, 2, DragonianLib::Device::CPU>;
using Int16Tensor2D = DragonianLib::Tensor<DragonianLib::Int16, 2, DragonianLib::Device::CPU>;
using ImageTensor = DragonianLib::Tensor<DragonianLib::Int32, 3, DragonianLib::Device::CPU>;

struct MyAudioData
{
	MyAudioData(
		DragonianLib::Int64 _SamplingRate,
		FloatTensor2D _Audio,
		FloatTensor2D _F0,
		FloatTensor2D _Spec,
		FloatTensor2D _Mel,
		std::wstring _F0Path
	): 
		SamplingRate(_SamplingRate),
		Audio(std::move(_Audio)),
		F0(std::move(_F0)),
		Spec(std::move(_Spec)),
		Mel(std::move(_Mel)),
		F0Path(std::move(_F0Path))
	{
	}

	DragonianLib::Int64 SamplingRate;
	FloatTensor2D Audio;
	FloatTensor2D F0;
	FloatTensor2D Spec;
	FloatTensor2D Mel;
	std::wstring F0Path;
	int64_t ModifyCount = 0;

	MyAudioData(const MyAudioData&) = delete;
	MyAudioData(MyAudioData&&) = default;
	MyAudioData& operator=(const MyAudioData&) = delete;
	MyAudioData& operator=(MyAudioData&&) = default;

	~MyAudioData();
};