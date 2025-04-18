﻿#include "../../../header/InferTools/Sampler/BaseSampler.hpp"
#include "Libraries/Base.h"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_Header

BaseSampler::BaseSampler(Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory) :
	MelBins(Mel_Bins), Alpha(alpha), DenoiseFn(dfn), NoisePredictor(pred)
{
	_callback = _ProgressCallback;
	Memory = memory;
};

std::vector<Ort::Value> BaseSampler::Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

ReflowBaseSampler::ReflowBaseSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory) :
	MelBins_(MelBins), Velocity_(Velocity)
{
	Callback_ = _ProgressCallback;
	Memory_ = memory;
}

std::vector<Ort::Value> ReflowBaseSampler::Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_End