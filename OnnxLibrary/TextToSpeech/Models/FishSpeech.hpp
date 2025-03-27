﻿#pragma once

/*#include "ModelBase.hpp"
#include "Libraries/Dict/Dict.hpp"
#include "Libraries/MyTemplateLibrary/Array.h"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header
	class FireflyArchitecture : public LibTTSModule
{
public:
	FireflyArchitecture(
		const std::wstring& _EncoderPath,
		const std::wstring& _DecoderPath,
		const ProgressCallback& _Progress,
		int64_t SampleRate = 44100,
		int64_t NumCodebooks = 8,
		ExecutionProviders Provider = ExecutionProviders::CPU,
		unsigned DeviceID = 0,
		unsigned ThreadCount = 0
	);

	~FireflyArchitecture() override = default;
	FireflyArchitecture(const FireflyArchitecture&) = default;
	FireflyArchitecture& operator=(const FireflyArchitecture&) = default;
	FireflyArchitecture(FireflyArchitecture&&) noexcept = default;
	FireflyArchitecture& operator=(FireflyArchitecture&&) noexcept = default;

	TemplateLibrary::Vector<Int64> Encode(
		TemplateLibrary::Vector<Float32>& _AudioFormat,
		int64_t _BatchSize = 1
	) const;

	TemplateLibrary::Vector<Float32> Decode(
		TemplateLibrary::Vector<Int64>& _PromptIds,
		int64_t _BatchSize = 1
	) const;

	int64_t GetSampleRate() const
	{
		return _MySampleRate;
	}

	int64_t GetNumCodebooks() const
	{
		return _MyNumCodebooks;
	}

private:
	int64_t _MySampleRate = 44100;
	int64_t _MyNumCodebooks = 8;
	std::shared_ptr<Ort::Session> _MyEncoder = nullptr;
	std::shared_ptr<Ort::Session> _MyDecoder = nullptr;
};

class Llama
{
public:
	struct PromptTensor
	{
		TemplateLibrary::Vector<Int64> Data;
		TemplateLibrary::Array<Int64, 3> Shape{ 0, 0, 0 };
	};

	struct RefPrompt
	{
		TemplateLibrary::Vector<Int64> VQPrompt;
		TemplateLibrary::Vector<Int64> TextPrompt;
	};

	Llama() = delete;

	Llama(TemplateLibrary::Vector<Int64> SystemPrompt) : _MySystemPrompt(std::move(SystemPrompt)) {}

	PromptTensor Inference(
		const TemplateLibrary::Vector<Int64>& TextPrompt,
		std::optional<std::reference_wrapper<const TemplateLibrary::Vector<RefPrompt>>> VQPromptTokens = std::nullopt,
		Int64 BatchSize = 1,
		Int64 MaxLength = 2048,
		Int64 Seed = 114514,
		Int64 NumSamples = 1,
		Int64 MaxNewTokens = 0,
		Float32 TopP = 0.7f,
		Float32 RepetitionPenalty = 1.2f,
		Float32 Temperature = 0.7f,
		bool IterativePrompt = true,
		Int64 ChunkLength = 100
	);

protected:
	std::shared_ptr<Ort::Session> _MyLayer = nullptr;
	TemplateLibrary::Vector<Int64> _MySystemPrompt;
	Int64 _MyNumCodebooks = 8;

private:
	PromptTensor EncodeTextTokens(
		const TemplateLibrary::Vector<Int64>& TextPrompt,
		Int64 BatchSize = 1
	);

	void EncodeVQTokens(
		const TemplateLibrary::Vector<RefPrompt>& VQPrompts,
		TemplateLibrary::Vector<PromptTensor>& VQPromptsEncoded,
		Int64 BatchSize = 1
	);

	PromptTensor EncodeSystemPrompts(Int64 BatchSize = 1);

protected:
	Int64 _MyBegId = 0;
	Int64 _MyEndId = 1;
	Int64 _MyUserId = 2;
	Int64 _MyAssistantId = 3;
	Int64 _MySystemId = 4;
	Int64 _MyTextId = 5;
	Int64 _MyVoiceId = 6;
	Int64 _MyInterleaveId = 7;
	Int64 _MySemanticBeginId = 8;
};

class FishSpeech : public LibTTSModule
{
public:
	static void EncodeTokens(
		const DragonianLibSTL::Vector<Int64>& _PromptIds,
		const DragonianLibSTL::Vector<Int64>& _PromptTokens,
		long _NumCodebooks = 4,
		Dict::Tokenizer::TokenizerMethod _Method = Dict::Tokenizer::Maximum,
		bool _SkipNonLatin = true,
		size_t _MaximumTokenLength = 32,
		Int64 _PadID = 0
	)
	{
		DragonianLibSTL::Vector PromptIdsInt64(_NumCodebooks * _PromptIds.Size(), _PadID);

	}






};











_D_Dragonian_Lib_Lib_Text_To_Speech_End
*/