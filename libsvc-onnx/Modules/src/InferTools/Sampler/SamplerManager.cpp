#include "InferTools/Sampler/SamplerManager.hpp"
#include <map>
#include "Base.h"
#include "Util/Logger.h"
#include <ranges>

LibSvcHeader

std::map<std::wstring, GetSamplerFn> RegisteredSamplers;
std::map<std::wstring, GetReflowSamplerFn> RegisteredReflowSamplers;

SamplerWrp GetSampler(const std::wstring& _name,
	Ort::Session* alpha,
	Ort::Session* dfn,
	Ort::Session* pred,
	int64_t Mel_Bins,
	const BaseSampler::ProgressCallback& _ProgressCallback,
	Ort::MemoryInfo* memory)
{
	const auto f_Sampler = RegisteredSamplers.find(_name);
	if (f_Sampler != RegisteredSamplers.end())
		return f_Sampler->second(alpha, dfn, pred, Mel_Bins, _ProgressCallback, memory);
	throw std::runtime_error("Unable To Find An Available Sampler");
}

void RegisterSampler(const std::wstring& _name, const GetSamplerFn& _constructor_fn)
{
	if (RegisteredSamplers.contains(_name))
	{
		DragonianLibLogMessage(L"[Warn] SamplerNameConflict");
		return;
	}
	RegisteredSamplers[_name] = _constructor_fn;
}

std::vector<std::wstring> GetSamplerList()
{
	std::vector<std::wstring> SamplersVec;
	SamplersVec.reserve(RegisteredSamplers.size());
	for (const auto& i : RegisteredSamplers | std::ranges::views::keys)
		SamplersVec.emplace_back(i);
	return SamplersVec;
}

ReflowSamplerWrp GetReflowSampler(
	const std::wstring& _name, 
	Ort::Session* velocity, 
	int64_t Mel_Bins, 
	const BaseSampler::ProgressCallback& _ProgressCallback, 
	Ort::MemoryInfo* memory
)
{
	const auto f_Sampler = RegisteredReflowSamplers.find(_name);
	if (f_Sampler != RegisteredReflowSamplers.end())
		return f_Sampler->second(velocity, Mel_Bins, _ProgressCallback, memory);
	throw std::runtime_error("Unable To Find An Available Sampler");
}

void RegisterReflowSampler(const std::wstring& _name, const GetReflowSamplerFn& _constructor_fn)
{
	if (RegisteredReflowSamplers.contains(_name))
	{
		DragonianLibLogMessage(L"[Warn] SamplerNameConflict");
		return;
	}
	RegisteredReflowSamplers[_name] = _constructor_fn;
}

std::vector<std::wstring> GetReflowSamplerList()
{
	std::vector<std::wstring> SamplersVec;
	SamplersVec.reserve(RegisteredReflowSamplers.size());
	for (const auto& i : RegisteredReflowSamplers | std::ranges::views::keys)
		SamplersVec.emplace_back(i);
	return SamplersVec;
}

LibSvcEnd