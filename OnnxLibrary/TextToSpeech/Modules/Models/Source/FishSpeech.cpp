#include "../Header/FishSpeech.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

constexpr TemplateLibrary::Array FFGanEncoderInputNames = { "audio" };
constexpr TemplateLibrary::Array FFGanDecoderInputNames = { "prompt" };

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
			FFGanEncoderInputNames.Data(),
			InputValues.data(),
			FFGanEncoderInputNames.Size(),
			FFGanDecoderInputNames.Data(),
			FFGanDecoderInputNames.Size()
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
	if (TotalSize % _BatchSize != 0)
		_D_Dragonian_Lib_Throw_Exception("PromptIds size must be divisible by BatchSize.");
	if (TotalSize / _BatchSize % _MyNumCodebooks != 0)
		_D_Dragonian_Lib_Throw_Exception("PromptIds size must be divisible by NumCodebooks.");
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
			FFGanDecoderInputNames.Data(),
			InputValues.data(),
			FFGanDecoderInputNames.Size(),
			FFGanEncoderInputNames.Data(),
			FFGanEncoderInputNames.Size()
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

Llama::PromptTensor Llama::Inference(
	const TemplateLibrary::Vector<Int64>& TextPrompt,
	std::optional<std::reference_wrapper<const TemplateLibrary::Vector<RefPrompt>>> VQPromptTokens,
	Int64 BatchSize,
	Int64 MaxLength,
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
	
	auto EncodedText = TemplateLibrary::Vector{ EncodeTextTokens(TextPrompt, BatchSize) };

	auto AllEncodedPrompt = TemplateLibrary::Vector{ EncodeSystemPrompts(BatchSize) };

	if (VQPromptTokens.has_value())
		EncodeVQTokens(VQPromptTokens->get(), AllEncodedPrompt, BatchSize);

	const auto TotalEncodedPromptLength = [&]()
		{
			Int64 Count = 0;
			for (const auto& i : AllEncodedPrompt)
				Count += i.Shape[2];
			return Count;
		}();

	for (Int64 Sample = 0; Sample < NumSamples; ++Sample)
	{
		TemplateLibrary::Vector<PromptTensor> GlobalEncoded;
		Int64 SegmentIndex = 0;

		while (SegmentIndex < static_cast<Int64>(EncodedText.Size()))
		{
			GlobalEncoded.EmplaceBack(std::move(EncodedText[SegmentIndex]));

			Int64 Count = 0;
			size_t I = 0;
			for (auto i = static_cast<int>(GlobalEncoded.Size()) - 1; i >= 0; --i)
			{
				Count += GlobalEncoded[i].Shape[2];
				if (Count + GlobalEncoded[i].Shape[2] > MaxLength - 1024 - TotalEncodedPromptLength)
					break;
				++I;
			}

			if (I != 0 && I % 2 == 0)
				--I;

			TemplateLibrary::Vector<std::reference_wrapper<const PromptTensor>> PartialEncoded;

			if (VQPromptTokens.has_value())
				for (const auto& i : AllEncodedPrompt)
					PartialEncoded.EmplaceBack(i);

			if (static_cast<long long>(I) < static_cast<Int64>(GlobalEncoded.Size()) - 2)
			{
				PartialEncoded.EmplaceBack(GlobalEncoded[0]);
				PartialEncoded.EmplaceBack(GlobalEncoded[1]);
				for (; I < GlobalEncoded.Size(); ++I)
					PartialEncoded.EmplaceBack(GlobalEncoded[I]);
			}
			else
				for (const auto& i : GlobalEncoded)
					PartialEncoded.EmplaceBack(i);

			PromptTensor CatEncoded;
			CatEncoded.Shape = { BatchSize, _MyNumCodebooks + 1, 0 };
			for (const auto& i : PartialEncoded)
				CatEncoded.Shape[2] += i.get().Shape[2];
			CatEncoded.Data.Reserve(CatEncoded.Shape[2] * CatEncoded.Shape[1] * CatEncoded.Shape[0]);

			for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
				for (Int64 CodeBook = 0; CodeBook < _MyNumCodebooks + 1; ++CodeBook)
					for (const auto& PartialEncodedSegment : PartialEncoded)
					{
						const auto EncodedPointer = PartialEncodedSegment.get().Data.Data() +
							Batch * PartialEncodedSegment.get().Shape[2] * PartialEncodedSegment.get().Shape[1];
						const auto EncodedBegin = EncodedPointer + CodeBook * PartialEncodedSegment.get().Shape[2];
						const auto EncodedEnd = EncodedBegin + PartialEncodedSegment.get().Shape[2];
						CatEncoded.Data.Insert(CatEncoded.Data.End(), EncodedBegin, EncodedEnd);
					}

			return CatEncoded;
		}
	}
}

Llama::PromptTensor Llama::EncodeTextTokens(
	const TemplateLibrary::Vector<Int64>& TextPrompt,
	Int64 BatchSize
)
{
	const auto TextPromptSize = static_cast<Int64>(TextPrompt.Size());
	const auto TextPromptShape = TemplateLibrary::Array{ BatchSize, 1ll, TextPromptSize / BatchSize };

	TemplateLibrary::Vector<Int64> AllPrompt;
	AllPrompt.Reserve(TextPromptSize + 114);

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		//Message(role="user",parts=[TextPart(text=string)],cal_loss=False)
		AllPrompt.EmplaceBack(_MyBegId);
		AllPrompt.EmplaceBack(_MyUserId);
		AllPrompt.Insert(
			AllPrompt.End(),
			TextPrompt.Begin() + Batch * TextPromptShape[2],
			TextPrompt.Begin() + (Batch + 1) * TextPromptShape[2]
		);
		AllPrompt.EmplaceBack(_MyEndId);

		//Message(role="assistant",parts=[TextPart(text="<|voice|>")],cal_loss=False,add_im_end=False,)
		AllPrompt.EmplaceBack(_MyBegId);
		AllPrompt.EmplaceBack(_MyAssistantId);
		AllPrompt.EmplaceBack(_MyVoiceId);
	}

	auto AllPromptSize = static_cast<Int64>(AllPrompt.Size());

	auto PromptsShape = TemplateLibrary::Array{ BatchSize, _MyNumCodebooks + 1, AllPromptSize / BatchSize };
	TemplateLibrary::Vector Prompts(AllPromptSize * PromptsShape[1], 0i64);
	const auto PromptsPerBatch = PromptsShape[2] * PromptsShape[1];

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		const auto PromptsPointer = Prompts.Data() + Batch * PromptsPerBatch;
		TemplateLibrary::Ranges(PromptsPointer, PromptsPointer + PromptsShape[2]) = TemplateLibrary::Ranges(AllPrompt.Data() + Batch * PromptsShape[2], AllPrompt.Data() + (Batch + 1) * PromptsShape[2]).C();
	}

	return { std::move(Prompts), PromptsShape };
}

void Llama::EncodeVQTokens(
	const TemplateLibrary::Vector<RefPrompt>& VQPrompts,
	TemplateLibrary::Vector<PromptTensor>& VQPromptsEncoded,
	Int64 BatchSize
)
{
	for (const auto& [VQPromptTokens, TextPrompt] : VQPrompts)
	{
		const auto TextSize = static_cast<Int64>(TextPrompt.Size());
		if (TextSize % BatchSize != 0)
			_D_Dragonian_Lib_Throw_Exception("TextPrompt size must be divisible by BatchSize.");
		const auto TextShape = TemplateLibrary::Array{ BatchSize, 1ll, TextSize / BatchSize };
		auto VQPromptShape = TemplateLibrary::Array{ BatchSize, _MyNumCodebooks, 0ll };
		const auto VQPromptSize = static_cast<Int64>(VQPromptTokens.Size());
		if (VQPromptSize % BatchSize != 0)
			_D_Dragonian_Lib_Throw_Exception("VQPromptTokens size must be divisible by BatchSize.");
		const auto VQPromptCountPerBatch = VQPromptSize / BatchSize;
		if (VQPromptCountPerBatch % _MyNumCodebooks != 0)
			_D_Dragonian_Lib_Throw_Exception("VQPromptTokens size must be divisible by NumCodebooks.");
		VQPromptShape[2] = VQPromptCountPerBatch / _MyNumCodebooks;

		TemplateLibrary::Vector<Int64> AllPrompt;
		AllPrompt.Reserve(TextSize + VQPromptSize + 114);

		for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
		{
			//Message(role="user",parts=[TextPart(text=string)],cal_loss=False)
			AllPrompt.EmplaceBack(_MyBegId);
			AllPrompt.EmplaceBack(_MyUserId);
			AllPrompt.Insert(
				AllPrompt.End(),
				TextPrompt.Begin() + Batch * TextShape[2],
				TextPrompt.Begin() + (Batch + 1) * TextShape[2]
			);
			AllPrompt.EmplaceBack(_MyEndId);

			//Message(role="assistant",parts=[TextPart(text="<|voice|>"), vq_part],cal_loss=False)
			{
				AllPrompt.EmplaceBack(_MyBegId);
				AllPrompt.EmplaceBack(_MyAssistantId);
				AllPrompt.EmplaceBack(_MyVoiceId);
				const auto VQPromptPointer = VQPromptTokens.Data() + Batch * VQPromptCountPerBatch;
				for (const auto Index : TemplateLibrary::Ranges(VQPromptPointer, VQPromptPointer + VQPromptShape[2]))
					AllPrompt.EmplaceBack(Index); //TODO
				AllPrompt.EmplaceBack(_MyEndId);
			}
		}
		auto AllPromptSize = static_cast<Int64>(AllPrompt.Size());

		auto PromptsShape = TemplateLibrary::Array{ BatchSize, _MyNumCodebooks + 1, AllPromptSize / BatchSize };
		TemplateLibrary::Vector Prompts(AllPromptSize * PromptsShape[1], 0i64);
		const auto PromptsPerBatch = PromptsShape[2] * PromptsShape[1];
		const auto OffsetVQ = TextShape[2] + 6;

		for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
		{
			const auto PromptsPointer = Prompts.Data() + Batch * PromptsPerBatch;
			TemplateLibrary::Ranges(PromptsPointer, PromptsPointer + PromptsShape[2]) = TemplateLibrary::Ranges(AllPrompt.Data() + Batch * PromptsShape[2], AllPrompt.Data() + (Batch + 1) * PromptsShape[2]).C();

			const auto VQPromptPointer = VQPromptTokens.Data() + Batch * VQPromptCountPerBatch;
			//values[0, encoded.vq_mask_tokens] = vq_parts[0] + tokenizer.semantic_begin_id
			{
				const auto VQBegin = PromptsPointer + OffsetVQ;
				const auto VQEnd = VQBegin + VQPromptShape[2];
				for (auto& i : TemplateLibrary::Ranges(VQBegin, VQEnd))
					i += _MySemanticBeginId;
			}

			const auto VoicePartBegin = PromptsPointer + PromptsShape[2];
			for (Int64 Codebook = 0; Codebook < _MyNumCodebooks; ++Codebook)
			{
				const auto CurrentVoicePartBegin = VoicePartBegin + Codebook * PromptsShape[2] + OffsetVQ;
				const auto CurrentVoicePartEnd = CurrentVoicePartBegin + VQPromptShape[2];
				const auto CurrentVQBegin = VQPromptPointer + Codebook * VQPromptShape[2];
				const auto CurrentVQEnd = CurrentVQBegin + VQPromptShape[2];
				TemplateLibrary::Ranges(CurrentVoicePartBegin, CurrentVoicePartEnd) = TemplateLibrary::Ranges(CurrentVQBegin, CurrentVQEnd);
			}
		}

		VQPromptsEncoded.EmplaceBack(std::move(Prompts), PromptsShape);
	}
}

Llama::PromptTensor Llama::EncodeSystemPrompts(Int64 BatchSize)
{
	TemplateLibrary::Vector<Int64> SystemPromptData;
	SystemPromptData.Reserve(_MySystemPrompt.Size() * BatchSize + 114);
	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		SystemPromptData.Insert(
			SystemPromptData.End(),
			_MySystemPrompt.Begin(),
			_MySystemPrompt.End()
		);
	}

	const auto SystemPromptSize = static_cast<Int64>(SystemPromptData.Size());
	const auto SystemPromptShape = TemplateLibrary::Array{ BatchSize, 1ll, SystemPromptSize / BatchSize };

	TemplateLibrary::Vector<Int64> AllPrompt;
	AllPrompt.Reserve(SystemPromptSize + 114);

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		AllPrompt.EmplaceBack(_MyBegId);
		AllPrompt.EmplaceBack(_MySystemId);
		AllPrompt.Insert(
			AllPrompt.End(),
			SystemPromptData.Begin() + Batch * SystemPromptShape[2],
			SystemPromptData.Begin() + (Batch + 1) * SystemPromptShape[2]
		);
		AllPrompt.EmplaceBack(_MyEndId);
	}

	auto AllPromptSize = static_cast<Int64>(AllPrompt.Size());

	auto PromptsShape = TemplateLibrary::Array{ BatchSize, _MyNumCodebooks + 1, AllPromptSize / BatchSize };
	TemplateLibrary::Vector Prompts(AllPromptSize * PromptsShape[1], 0i64);
	const auto PromptsPerBatch = PromptsShape[2] * PromptsShape[1];

	for (Int64 Batch = 0; Batch < BatchSize; ++Batch)
	{
		const auto PromptsPointer = Prompts.Data() + Batch * PromptsPerBatch;
		TemplateLibrary::Ranges(PromptsPointer, PromptsPointer + PromptsShape[2]) = TemplateLibrary::Ranges(AllPrompt.Data() + Batch * PromptsShape[2], AllPrompt.Data() + (Batch + 1) * PromptsShape[2]).C();
	}

	return { std::move(Prompts), PromptsShape };
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End