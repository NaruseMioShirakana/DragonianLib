#include "../DioF0Extractor.hpp"
#include "world/dio.h"
#include "world/stonemask.h"

_D_Dragonian_Lib_F0_Extractor_Header

Tensor<Float64, 2, Device::CPU> DioF0Extractor::Dio(
	const Tensor<Float64, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
)
{
    DioOption DioOption;
    InitializeDioOption(&DioOption);
    DioOption.f0_ceil = Params.F0Max;
    DioOption.f0_floor = Params.F0Min;
    DioOption.frame_period = 1000.0 * double(Params.HopSize) / double(Params.SamplingRate);
    const auto PCMLen = PCMData.Size(1);

    const auto f0Length = GetSamplesForDIO(
        int(Params.SamplingRate),
        (int)PCMLen,
        DioOption.frame_period
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
				::Dio(
					PCMData.Data() + i * PCMLen,
					(int)PCMLen,
					MySamplingRate,
					&DioOption,
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

Tensor<Float32, 2, Device::CPU> DioF0Extractor::ExtractF0(
	const Tensor<Float64, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
) const
{
	return Dio(PCMData.Continuous().Evaluate(), Params).Cast<Float32>().Evaluate();
}

_D_Dragonian_Lib_F0_Extractor_End
