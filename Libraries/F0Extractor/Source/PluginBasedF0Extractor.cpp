#include "Libraries/F0Extractor/PluginBasedF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

PluginF0Extractor::PluginF0Extractor(const Plugin::Plugin& Plugin, const void* UserParameter) :
	_MyInstance(
		std::shared_ptr<void>(Plugin->GetInstance(UserParameter),
			[Plugin](void* Instance) { Plugin->DestoryInstance(Instance); })
	), _MyPlugin(Plugin),
	_MyGetF0Size((GetF0SizeFunctionType)Plugin->GetFunction("GetF0Size", true)),
	_MyExtractPD((ExtractFunctionTypeF64)_MyPlugin->GetFunction("ExtractF0PD")),
	_MyExtractPS((ExtractFunctionTypeF32)_MyPlugin->GetFunction("ExtractF0PS")),
	_MyExtractI16((ExtractFunctionTypeI16)_MyPlugin->GetFunction("ExtractF0I16"))
{

}

Tensor<Float32, 2, Device::CPU> PluginF0Extractor::ExtractF0(const Tensor<Float32, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params) const
{
	if (!_MyExtractPS)
		_D_Dragonian_Lib_Throw_Exception("ExtractF0PS is not available in the plugin");

	const auto Channel = PCMData.Size(0);
	const auto DataSize = PCMData.Size(1);

	const auto F0Size = _MyGetF0Size(_MyInstance.get(), Params.SamplingRate, Params.HopSize, Params.UserParameter);

	auto InputData = PCMData.Continuous().Evaluate();
	Dimensions<2> OutputSize{ Channel, F0Size };
	Tensor<Float32, 2, Device::CPU> Output = Tensor<Float32, 2, Device::CPU>::New(OutputSize);
	const auto InputBuffer = InputData.Data();
	const auto OutputBuffer = Output.Data();

	Output.AppendTask(
		[this, InputBuffer, Params, OutputBuffer, DataSize, Channel]
		(std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
		{
			_MyExtractPS(
				_MyInstance.get(),
				InputBuffer, DataSize, Channel,
				Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
				OutputBuffer
			);
		},
		InputData.Buffer()
	);

	return Output;
}

Tensor<Float32, 2, Device::CPU> PluginF0Extractor::ExtractF0(const Tensor<Float64, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params) const
{
	if (!_MyExtractPD)
		return ExtractF0(PCMData.Cast<Float32>(), Params);

	const auto Channel = PCMData.Size(0);
	const auto DataSize = PCMData.Size(1);

	const auto F0Size = _MyGetF0Size(_MyInstance.get(), Params.SamplingRate, Params.HopSize, Params.UserParameter);

	auto InputData = PCMData.Continuous().Evaluate();
	Dimensions<2> OutputSize{ Channel, F0Size };
	Tensor<Float32, 2, Device::CPU> Output = Tensor<Float32, 2, Device::CPU>::New(OutputSize);
	const auto InputBuffer = InputData.Data();
	const auto OutputBuffer = Output.Data();

	Output.AppendTask(
		[this, InputBuffer, Params, OutputBuffer, DataSize, Channel]
		(std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
		{
			_MyExtractPD(
				_MyInstance.get(),
				InputBuffer, DataSize, Channel,
				Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
				OutputBuffer
			);
		},
		InputData.Buffer()
	);

	return Output;
}

Tensor<Float32, 2, Device::CPU> PluginF0Extractor::ExtractF0(const Tensor<Int16, 2, Device::CPU>& PCMData, const F0ExtractorParams& Params) const
{
	if (!_MyExtractI16)
		return ExtractF0(PCMData.Cast<Float32>() / 32768.f, Params);

	const auto Channel = PCMData.Size(0);
	const auto DataSize = PCMData.Size(1);

	const auto F0Size = _MyGetF0Size(_MyInstance.get(), Params.SamplingRate, Params.HopSize, Params.UserParameter);

	auto InputData = PCMData.Continuous().Evaluate();
	Dimensions<2> OutputSize{ Channel, F0Size };
	Tensor<Float32, 2, Device::CPU> Output = Tensor<Float32, 2, Device::CPU>::New(OutputSize);
	const auto InputBuffer = InputData.Data();
	const auto OutputBuffer = Output.Data();

	Output.AppendTask(
		[this, InputBuffer, Params, OutputBuffer, DataSize, Channel]
		(std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
		{
			_MyExtractI16(
				_MyInstance.get(),
				InputBuffer, DataSize, Channel,
				Params.SamplingRate, Params.HopSize, Params.F0Bins, Params.F0Max, Params.F0Min, Params.UserParameter,
				OutputBuffer
			);
		},
		InputData.Buffer()
	);

	return Output;
}


_D_Dragonian_Lib_F0_Extractor_End