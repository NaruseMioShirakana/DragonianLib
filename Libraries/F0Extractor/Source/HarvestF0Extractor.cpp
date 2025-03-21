#include "../HarvestF0Extractor.hpp"
#include "world/harvest.h"
#include "world/stonemask.h"

_D_Dragonian_Lib_F0_Extractor_Header

Tensor<Float64, 2, Device::CPU> HarvestF0Extractor::Harvest(
    const Tensor<Float64, 2, Device::CPU>& PCMData,
    const F0ExtractorParams& Params
)
{
    HarvestOption HarvestOption;
    InitializeHarvestOption(&HarvestOption);
    HarvestOption.f0_ceil = Params.F0Max;
    HarvestOption.f0_floor = Params.F0Min;
    HarvestOption.frame_period = 1000.0 * double(Params.HopSize) / double(Params.SamplingRate);
    const auto PCMLen = PCMData.Size(1);

    const auto f0Length = GetSamplesForHarvest(
        int(Params.SamplingRate),
        (int)PCMLen,
        HarvestOption.frame_period
    );
    auto MySamplingRate = int(Params.SamplingRate);

    auto Shape = Dimensions<2>{ PCMData.Size(0), static_cast<Int64>(f0Length) };
    auto ResultF0 = Tensor<Float64, 2, Device::CPU>::New(Shape);

    for (SizeType i = 0; i < PCMData.Size(0); ++i)
    {
        ResultF0.AppendTask([=]
            {
                auto RawF0 = TemplateLibrary::Vector<double>(f0Length);
                auto TemporalPositions = TemplateLibrary::Vector<double>(f0Length);
                ::Harvest(
                    PCMData.Data() + i * PCMLen,
                    (int)PCMLen,
                    MySamplingRate,
                    &HarvestOption,
                    TemporalPositions.Data(),
                    RawF0.Data()
                );
                ::StoneMask(
                    PCMData.Data() + i * PCMLen,
                    (int)PCMLen,
                    MySamplingRate,
                    TemporalPositions.Data(),
                    RawF0.Data(),
                    f0Length,
                    ResultF0.Data() + i * f0Length
                );
            });
    }
    return ResultF0;
}

Tensor<Float32, 2, Device::CPU> HarvestF0Extractor::ExtractF0(
    const Tensor<Float64, 2, Device::CPU>& PCMData,
    const F0ExtractorParams& Params
)
{
    return Harvest(PCMData.Continuous().Evaluate(), Params).Cast<Float32>().Evaluate();
}

_D_Dragonian_Lib_F0_Extractor_End
