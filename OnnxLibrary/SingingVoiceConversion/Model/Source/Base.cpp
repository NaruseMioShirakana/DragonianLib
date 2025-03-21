#include "../Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::Inference(
	const Parameters& Params,
	const Tensor<Float32, 3, Device::CPU>& Audio,
	SizeType SamplingRate,
	const F0Extractor::F0Extractor& F0Extractor,
	const F0Extractor::Parameters& F0Params
) const
{
	
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::Inference(
	const Parameters& Params,
	const Tensor<Float32, 4, Device::CPU>& Units,
	const Tensor<Float32, 3, Device::CPU>& F0,
	SizeType SourceSampleCount,
	std::optional<Tensor<Float32, 3, Device::CPU>> Volume,
	std::optional<Tensor<Float32, 4, Device::CPU>> Speaker,
	std::optional<Tensor<Float32, 4, Device::CPU>> GTSpec
)
{
	
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::NormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec
) const
{
	return NormSpec(Spec, _MySpecMax, _MySpecMin);
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::DenormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec
) const
{
	return DenormSpec(Spec, _MySpecMax, _MySpecMin);
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::NormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec,
	float SpecMax,
	float SpecMin
)
{
	return (Spec - SpecMin) / (SpecMax - SpecMin) * 2 - 1;
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::DenormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec,
	float SpecMax,
	float SpecMin
)
{
	return (Spec + 1) / 2 * (SpecMax - SpecMin) + SpecMin;
}


_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End