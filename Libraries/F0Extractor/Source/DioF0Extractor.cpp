#include "../DioF0Extractor.hpp"
#include "world/dio.h"
#include "world/stonemask.h"

_D_Dragonian_Lib_F0_Extractor_Header

Vector<float> DioF0Extractor::ExtractF0(
    const Vector<double>& PCMData,
    const F0ExtractorParams& Params
)
{
    auto RawF0 = Dio(PCMData, Params);
    auto Output = Vector<float>(RawF0.Size());
    for (size_t i = 0; i < RawF0.Size(); ++i)
        Output[i] = (float)RawF0[i];
    return Output;
}

Vector<double> DioF0Extractor::Dio(
    const Vector<double>& PCMData,
    const F0ExtractorParams& Params
)
{
    DioOption DioOption;
    InitializeDioOption(&DioOption);
    DioOption.f0_ceil = Params.F0Max;
    DioOption.f0_floor = Params.F0Min;
    DioOption.frame_period = 1000.0 * double(Params.HopSize) / double(Params.SamplingRate);
	const auto PCMLen = PCMData.Size();

    const auto f0Length = GetSamplesForDIO(
        int(Params.SamplingRate),
        (int)PCMLen,
        DioOption.frame_period
    );
    auto TemporalPositions = Vector<double>(f0Length);
    auto RawF0 = Vector<double>(f0Length);
    auto ResultF0 = Vector<double>(f0Length);
    ::Dio(
        PCMData.Data(),
        (int)PCMLen,
        int(Params.SamplingRate),
        &DioOption,
        TemporalPositions.Data(),
        RawF0.Data()
    );
    StoneMask(
        PCMData.Data(),
        (int)PCMLen,
        int(Params.SamplingRate),
        TemporalPositions.Data(),
        RawF0.Data(),
        (int)f0Length,
        ResultF0.Data()
    );
	return ResultF0;
}

_D_Dragonian_Lib_F0_Extractor_End
