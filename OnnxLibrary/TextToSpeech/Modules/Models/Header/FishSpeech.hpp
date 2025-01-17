#pragma once
#include "ModelBase.hpp"
#include "Libraries/Dict/Dict.hpp"

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

class Llama : public LibTTSModule
{
public:









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
