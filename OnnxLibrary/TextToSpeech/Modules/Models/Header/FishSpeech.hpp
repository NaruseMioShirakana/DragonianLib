#pragma once
#include "ModelBase.hpp"
#include "Libraries/Dict/Dict.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

class FireflyArchitecture : public LibTTSModule
{
public:






private:
	int64_t _MySampleRate = 22050;
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
