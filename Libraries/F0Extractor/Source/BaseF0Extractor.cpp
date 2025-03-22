#include "../BaseF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::ExtractF0(
	const Tensor<Float64, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::ExtractF0(
	const Tensor<Float32, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
)
{
	return ExtractF0(PCMData.Cast<Float64>(), Params);
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::ExtractF0(
	const Tensor<Int16, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
)
{
	return ExtractF0(PCMData.Cast<Float64>() / 32768., Params);
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::operator()(const Tensor<Float32, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params)
{
	return ExtractF0(PCMData, Params);
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::operator()(const Tensor<Float64, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params)
{
	return ExtractF0(PCMData, Params);
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::operator()(const Tensor<Int16, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params)
{
	return ExtractF0(PCMData, Params);
}

_D_Dragonian_Lib_F0_Extractor_End