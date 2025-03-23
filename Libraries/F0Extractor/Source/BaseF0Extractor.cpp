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
#ifdef _DEBUG
	static auto _MyLogger = _D_Dragonian_Lib_Namespace GetDefaultLogger();
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
	auto F0 = ExtractF0(PCMData, Params);
	_MyLogger->LogInfo(
		L"Extract F0 from PCM data, Shape: [" +
		std::to_wstring(PCMData.Shape(0)) +
		L", " +
		std::to_wstring(PCMData.Shape(1)) +
		L"], Cost Time: " +
		std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TimeBegin).count()) +
		L"ms",
		L"F0Extractor"
	);
	return F0;
#else
	return ExtractF0(PCMData, Params);
#endif
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::operator()(const Tensor<Float64, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params)
{
#ifdef _DEBUG
	static auto _MyLogger = _D_Dragonian_Lib_Namespace GetDefaultLogger();
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
	auto F0 = ExtractF0(PCMData, Params);
	_MyLogger->LogInfo(
		L"Extract F0 from PCM data, Shape: [" +
		std::to_wstring(PCMData.Shape(0)) +
		L", " +
		std::to_wstring(PCMData.Shape(1)) +
		L"], Time: " +
		std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TimeBegin).count()) +
		L"ms",
		L"F0Extractor"
	);
	return F0;
#else
	return ExtractF0(PCMData, Params);
#endif
}

Tensor<Float32, 2, Device::CPU> BaseF0Extractor::operator()(const Tensor<Int16, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params)
{
#ifdef _DEBUG
	static auto _MyLogger = _D_Dragonian_Lib_Namespace GetDefaultLogger();
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
	auto F0 = ExtractF0(PCMData, Params);
	_MyLogger->LogInfo(
		L"Extract F0 from PCM data, Shape: [" +
		std::to_wstring(PCMData.Shape(0)) +
		L", " +
		std::to_wstring(PCMData.Shape(1)) +
		L"], Time: " +
		std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TimeBegin).count()) +
		L"ms",
		L"F0Extractor"
	);
	return F0;
#else
	return ExtractF0(PCMData, Params);
#endif
}

_D_Dragonian_Lib_F0_Extractor_End