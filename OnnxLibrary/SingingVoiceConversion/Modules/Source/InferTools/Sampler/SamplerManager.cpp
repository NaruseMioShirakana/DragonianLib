#include "../../../header/InferTools/Sampler/SamplerManager.hpp"
#include "../../../header/InferTools/Sampler/Samplers.hpp"
#include <map>
#include "Libraries/Base.h"
#include "Libraries/Util/Logger.h"
#include <ranges>

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_Header

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
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available Sampler");
}

void RegisterSampler(const std::wstring& _name, const GetSamplerFn& _constructor_fn)
{
	if (RegisteredSamplers.contains(_name))
	{
		LogWarn(L"Name Of Sampler Already Exists");
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
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available Sampler");
}

void RegisterReflowSampler(const std::wstring& _name, const GetReflowSamplerFn& _constructor_fn)
{
	if (RegisteredReflowSamplers.contains(_name))
	{
		LogWarn(L"Name Of Sampler Already Exists");
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

struct Init
{
	Init()
	{
		RegisterSampler(L"Pndm", [](Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred,
			int64_t Mel_Bins, const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> SamplerWrp
			{
				return std::make_shared<PndmSampler>(alpha, dfn, pred, Mel_Bins, _ProgressCallback, memory);
			});

		RegisterSampler(L"DDim", [](Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred,
			int64_t Mel_Bins, const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> SamplerWrp
			{
				return std::make_shared<DDimSampler>(alpha, dfn, pred, Mel_Bins, _ProgressCallback, memory);
			});

		RegisterReflowSampler(L"Eular", [](Ort::Session* velocity, int64_t Mel_Bins,
			const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> ReflowSamplerWrp
			{
				return std::make_shared<ReflowEularSampler>(velocity, Mel_Bins, _ProgressCallback, memory);
			});

		RegisterReflowSampler(L"Rk4", [](Ort::Session* velocity, int64_t Mel_Bins,
			const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> ReflowSamplerWrp
			{
				return std::make_shared<ReflowRk4Sampler>(velocity, Mel_Bins, _ProgressCallback, memory);
			});

		RegisterReflowSampler(L"Heun", [](Ort::Session* velocity, int64_t Mel_Bins,
			const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> ReflowSamplerWrp
			{
				return std::make_shared<ReflowHeunSampler>(velocity, Mel_Bins, _ProgressCallback, memory);
			});

		RegisterReflowSampler(L"Pecece", [](Ort::Session* velocity, int64_t Mel_Bins,
			const BaseSampler::ProgressCallback& _ProgressCallback,
			Ort::MemoryInfo* memory) -> ReflowSamplerWrp
			{
				return std::make_shared<ReflowPececeSampler>(velocity, Mel_Bins, _ProgressCallback, memory);
			});
	}
};
Init _Vardef_Init;

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_End