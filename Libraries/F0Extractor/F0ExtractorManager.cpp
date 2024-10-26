#include "F0Extractor/F0ExtractorManager.hpp"
#include <map>
#include <stdexcept>
#include <ranges>
#include "Base.h"
#include "Util/Logger.h"

_D_Dragonian_Lib_F0_Extractor_Header

std::map<std::wstring, GetF0ExtractorFn> RegisteredF0Extractors;

F0Extractor GetF0Extractor(
    const std::wstring& Name,
    const uint32_t SampleRate,
    const uint32_t HopSize,
    const uint32_t F0Bin,
    const double F0Max,
    const double F0Min)
{
    const auto F0ExtractorIt = RegisteredF0Extractors.find(Name);
    if (F0ExtractorIt != RegisteredF0Extractors.end())
        return F0ExtractorIt->second(SampleRate, HopSize, F0Bin, F0Max, F0Min);
    throw std::runtime_error("Unable To Find An Available F0Extractor");
}

void RegisterF0Extractor(const std::wstring& Name, const GetF0ExtractorFn& ConstructorFn)
{
    if (RegisteredF0Extractors.contains(Name))
    {
		LogWarn(L"Name Of F0Extractor Already Registered");
        return;
    }
    RegisteredF0Extractors[Name] = ConstructorFn;
}

std::vector<std::wstring> GetF0ExtractorList()
{
    std::vector<std::wstring> F0ExtractorsVec;
    F0ExtractorsVec.reserve(RegisteredF0Extractors.size());
    for (const auto& ExtractorName : RegisteredF0Extractors | std::ranges::views::keys)
        F0ExtractorsVec.emplace_back(ExtractorName);
    return F0ExtractorsVec;
}

_D_Dragonian_Lib_F0_Extractor_End
