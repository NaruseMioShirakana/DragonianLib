#include "../Header/FishSpeech.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

constexpr std::array FFGanEncoderInputNames = { "audio" };
constexpr std::array FFGanDecoderInputNames = { "prompt" };

FireflyArchitecture::FireflyArchitecture(
	const std::wstring& _EncoderPath,
	const std::wstring& _DecoderPath,
	const ProgressCallback& _Progress,
	int64_t SampleRate,
	int64_t NumCodebooks,
	ExecutionProviders Provider,
	unsigned DeviceID,
	unsigned ThreadCount
) : LibTTSModule(Provider, DeviceID, ThreadCount), _MySampleRate(SampleRate), _MyNumCodebooks(NumCodebooks)
{
	ProgressCallbackFunction = _Progress;
	try
	{
		_MyEncoder = RefOrtCachedModel(_EncoderPath, *OrtApiEnv);
		_MyDecoder = RefOrtCachedModel(_DecoderPath, *OrtApiEnv);
	}
	catch (const std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

TemplateLibrary::Vector<Int64> FireflyArchitecture::Encode(
	TemplateLibrary::Vector<Float32>& _AudioFormat,
	int64_t _BatchSize
) const
{
	auto TotalSize = static_cast<Int64>(_AudioFormat.Size());
	Int64 InputShape[] = { _BatchSize, 1, TotalSize / _BatchSize };
	std::vector<Ort::Value> InputValues, OutputValues;
	InputValues.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			_AudioFormat.Data(),
			TotalSize,
			InputShape,
			3
		)
	);
	try
	{
		OutputValues = _MyEncoder->Run(
			Ort::RunOptions{ nullptr },
			FFGanEncoderInputNames.data(),
			InputValues.data(),
			FFGanEncoderInputNames.size(),
			FFGanDecoderInputNames.data(),
			FFGanDecoderInputNames.size()
		);
	}
	catch (const std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	auto OutputData = OutputValues[0].GetTensorMutableData<Int64>();
	auto OutputSize = OutputValues[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return { OutputData, OutputData + OutputSize };
}

TemplateLibrary::Vector<Float32> FireflyArchitecture::Decode(
	TemplateLibrary::Vector<Int64>& _PromptIds,
	int64_t _BatchSize
) const
{
	auto TotalSize = static_cast<Int64>(_PromptIds.Size());
	Int64 InputShape[] = { _BatchSize, _MyNumCodebooks, TotalSize / _BatchSize };
	std::vector<Ort::Value> InputValues, OutputValues;
	InputValues.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			_PromptIds.Data(),
			TotalSize,
			InputShape,
			3
		)
	);
	try {
		OutputValues = _MyDecoder->Run(
			Ort::RunOptions{ nullptr },
			FFGanDecoderInputNames.data(),
			InputValues.data(),
			FFGanDecoderInputNames.size(),
			FFGanEncoderInputNames.data(),
			FFGanEncoderInputNames.size()
		);
	}
	catch (const std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	auto OutputData = OutputValues[0].GetTensorMutableData<Float32>();
	auto OutputSize = OutputValues[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return { OutputData, OutputData + OutputSize };
}


_D_Dragonian_Lib_Lib_Text_To_Speech_End