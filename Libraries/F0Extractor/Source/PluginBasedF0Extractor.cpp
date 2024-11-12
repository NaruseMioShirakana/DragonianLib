#include "../PluginBasedF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

PluginF0Extractor::PluginF0Extractor(const Plugin::Plugin& Plugin, const void* UserParameter) :
	_MyInstance(Plugin->GetInstance(UserParameter)), _MyPlugin(Plugin),
	_MyGetF0Size((GetF0SizeFunctionType)Plugin->GetFunction("GetF0Size", true)) {}

PluginF0Extractor::~PluginF0Extractor()
{
	_MyPlugin->DestoryInstance(_MyInstance);
}

Vector<float> PluginF0Extractor::ExtractF0(const Vector<double>& PCMData, const F0ExtractorParams& Params)
{
	const auto F0Size = _MyGetF0Size(_MyInstance, Params.SamplingRate, Params.HopSize, Params.UserParameter);
	Vector<float> Output(F0Size);
	if (!_MyExtractPD)
		_MyExtractPD = (ExtractFunctionType)_MyPlugin->GetFunction("ExtractF0PD", true);
	_MyExtractPD(
		_MyInstance,
		PCMData.Data(), PCMData.Size(),
		Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
		Output.Data()
	);
	return Output;
}

Vector<float> PluginF0Extractor::ExtractF0(const Vector<float>& PCMData, const F0ExtractorParams& Params)
{
	const auto F0Size = _MyGetF0Size(_MyInstance, Params.SamplingRate, Params.HopSize, Params.UserParameter);
	Vector<float> Output(F0Size);
	if (!_MyExtractPS)
		_MyExtractPS = (ExtractFunctionType)_MyPlugin->GetFunction("ExtractF0PS", true);
	_MyExtractPS(
		_MyInstance,
		PCMData.Data(), PCMData.Size(),
		Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
		Output.Data()
	);
	return Output;
}

Vector<float> PluginF0Extractor::ExtractF0(const Vector<int16_t>& PCMData, const F0ExtractorParams& Params)
{
	const auto F0Size = _MyGetF0Size(_MyInstance, Params.SamplingRate, Params.HopSize, Params.UserParameter);
	Vector<float> Output(F0Size);
	if (!_MyExtractI16)
		_MyExtractI16 = (ExtractFunctionType)_MyPlugin->GetFunction("ExtractF0I16", true);
	_MyExtractI16(
		_MyInstance,
		PCMData.Data(), PCMData.Size(),
		Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
		Output.Data()
	);
	return Output;
}

_D_Dragonian_Lib_F0_Extractor_End