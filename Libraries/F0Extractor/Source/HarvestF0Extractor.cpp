#include "../HarvestF0Extractor.hpp"
#include "world/harvest.h"
#include "world/stonemask.h"

_D_Dragonian_Lib_F0_Extractor_Header

Vector<float> HarvestF0Extractor::ExtractF0(
    const Vector<double>& PCMData,
    const F0ExtractorParams& Params
)
{
    auto RawF0 = Harvest(PCMData, Params);
    auto Output = Vector<float>(RawF0.Size());
    for (size_t i = 0; i < RawF0.Size(); ++i)
        Output[i] = (float)RawF0[i];
    return Output;
}

Vector<double> HarvestF0Extractor::Harvest(
    const Vector<double>& PCMData,
    const F0ExtractorParams& Params
)
{
    HarvestOption HarvestOption;
    InitializeHarvestOption(&HarvestOption);
    HarvestOption.f0_ceil = Params.F0Max;
    HarvestOption.f0_floor = Params.F0Min;
    HarvestOption.frame_period = 1000.0 * double(Params.HopSize) / double(Params.SamplingRate);
	const auto PCMLen = PCMData.Size();

    const size_t F0Length = GetSamplesForHarvest(int(Params.SamplingRate), (int)PCMLen, HarvestOption.frame_period);
    auto TemporalPositions = Vector<double>(F0Length);
    auto RawF0 = Vector<double>(F0Length);
    auto ResultF0 = Vector<double>(F0Length);
    ::Harvest(
        PCMData.Data(),
        (int)PCMLen,
        int(Params.SamplingRate),
        &HarvestOption,
        TemporalPositions.Data(),
        RawF0.Data()
    );
    StoneMask(
        PCMData.Data(),
        (int)PCMLen,
        int(Params.SamplingRate),
        TemporalPositions.Data(),
        RawF0.Data(),
        (int)F0Length,
        ResultF0.Data()
    );
	return ResultF0;
}

_D_Dragonian_Lib_F0_Extractor_End
