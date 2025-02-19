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

TemplateLibrary::Vector<Int64> Llama::Inference(
	TemplateLibrary::Vector<Int64> TextPrompt,
	TemplateLibrary::Vector<Int64> PromptTokens,
	Int64 BatchSize,
	Int64 Seed,
	Int64 NumSamples,
	Int64 MaxNewTokens,
	Float32 TopP,
	Float32 RepetitionPenalty,
	Float32 Temperature,
	bool IterativePrompt,
	Int64 ChunkLength
)
{
	if (TextPrompt.Empty())
		_D_Dragonian_Lib_Throw_Exception("TextPrompt is empty.");
	if (BatchSize < 1)
		_D_Dragonian_Lib_Throw_Exception("BatchSize must be greater than 0.");
	if (NumSamples < 1)
		_D_Dragonian_Lib_Throw_Exception("NumSamples must be greater than 0.");
	if (MaxNewTokens < 0)
		_D_Dragonian_Lib_Throw_Exception("MaxNewTokens must be greater than or equal to 0.");
	if (TopP < 0.f)
		_D_Dragonian_Lib_Throw_Exception("TopP must be greater than 0.");
	if (TopP > 1.f)
		_D_Dragonian_Lib_Throw_Exception("TopP must be less than 1.");
	if (RepetitionPenalty < 0.f)
		_D_Dragonian_Lib_Throw_Exception("RepetitionPenalty must be greater than or equal to 0.");
	if (RepetitionPenalty > 2.f)
		_D_Dragonian_Lib_Throw_Exception("RepetitionPenalty must be less than 2.");
	if (Temperature < 0.f)
		_D_Dragonian_Lib_Throw_Exception("Temperature must be greater than or equal to 0.");
	if (Temperature > 2.f)
		_D_Dragonian_Lib_Throw_Exception("Temperature must be less than 2.");
	if (ChunkLength < 1)
		_D_Dragonian_Lib_Throw_Exception("ChunkLength must be greater than 0.");

	const auto TextSize = static_cast<Int64>(TextPrompt.Size());
	if (TextSize % BatchSize != 0)
		_D_Dragonian_Lib_Throw_Exception("TextPrompt size must be divisible by BatchSize.");
	//const auto TextShape = std::array{ BatchSize, 1, TextSize / BatchSize };

	const auto PromptSize = static_cast<Int64>(PromptTokens.Size());
	if (PromptSize % BatchSize != 0)
		_D_Dragonian_Lib_Throw_Exception("PromptTokens size must be divisible by BatchSize.");
	auto PromptCountPerBatch = PromptSize / BatchSize;
	if (PromptCountPerBatch % _MyNumCodebooks != 0)
		_D_Dragonian_Lib_Throw_Exception("PromptTokens size must be divisible by NumCodebooks.");
	auto PromptFrame = PromptCountPerBatch / _MyNumCodebooks;
	//const auto PromptShape = std::array{ BatchSize, _MyNumCodebooks, PromptFrame };

	TemplateLibrary::Vector<Int64> AllPrompt;
	AllPrompt.Reserve(TextSize + (PromptSize / _MyNumCodebooks) * 2);

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		/*Message(
            role="user",
            parts=[TextPart(text=string)],
            cal_loss=False,
        )*/
		{
			AllPrompt.EmplaceBack(_MyBegId);
			AllPrompt.EmplaceBack(_MyUserId);
			AllPrompt.Insert(
				AllPrompt.End(),
				TextPrompt.Begin() + Batch * TextSize / BatchSize,
				TextPrompt.Begin() + (Batch + 1) * TextSize / BatchSize
			);
			AllPrompt.EmplaceBack(_MyEndId);
		}

		/*Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vq_part],
                cal_loss=False,
            )
		 *Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )*/
		{
			AllPrompt.EmplaceBack(_MyBegId);
			AllPrompt.EmplaceBack(_MyAssistantId);
			AllPrompt.EmplaceBack(_MyVoiceId);
			if (PromptSize > 0)
			{
				const auto BatchOffset = Batch * PromptCountPerBatch;
				for (const auto Index : Ranges(PromptTokens.Data() + BatchOffset, PromptTokens.Data() + BatchOffset + PromptFrame))
					AllPrompt.EmplaceBack(Index); //TODO
				AllPrompt.EmplaceBack(_MyEndId);
			}
		}
	}
	auto AllPromptSize = static_cast<Int64>(AllPrompt.Size());
	auto AllPromptPerBatch = AllPromptSize / BatchSize;

	TemplateLibrary::Vector Prompts(AllPromptSize * (_MyNumCodebooks + 1), 0i64);
	auto PromptsShape = std::array{ BatchSize, _MyNumCodebooks + 1, AllPromptPerBatch };
	const auto PromptsPerBatch = AllPromptPerBatch * (_MyNumCodebooks + 1);

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		Ranges(Prompts.Data() + Batch * PromptsPerBatch, Prompts.Data() + Batch * PromptsPerBatch + AllPromptPerBatch) =
			Ranges(AllPrompt.Data() + Batch * AllPromptPerBatch, AllPrompt.Data() + (Batch + 1) * AllPromptPerBatch);
		if (PromptSize)
		{
			
		}
	}

	for (Int64 Sample = 0; Sample < NumSamples; ++Sample)
	{

	}
}


_D_Dragonian_Lib_Lib_Text_To_Speech_End