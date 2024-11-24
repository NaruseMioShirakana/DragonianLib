#pragma once
#include "ModelBase.hpp"
#include "Dict/Dict.hpp"
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

	static void EncodeTokens(
		const Dict::Tokenizer& _MyTokenizer,
		const std::wstring _Text
		//PromptTokens = None,
		//long NumCodebooks = 4
	);








};













_D_Dragonian_Lib_Lib_Text_To_Speech_End
