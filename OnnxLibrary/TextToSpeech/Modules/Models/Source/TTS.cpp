#include "../header/TTS.hpp"
#include "Libraries/Util/Logger.h"
#include <ranges>

_D_Dragonian_Lib_Lib_Text_To_Speech_Header
	TextToSpeech::TextToSpeech(
	const ModelHParams& HParams,
	const ExecutionProviders& ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : LibTTSModule(ExecutionProvider_, DeviceID_, ThreadCount_),
SpeakerName2ID(HParams.SpeakerName2ID), Symbols(HParams.Symbols),
Language2ID(HParams.Language2ID), AddBlank(HParams.AddBlank),
SpeakerCount(static_cast<int64_t>(HParams.SpeakerName2ID.size()))
{
	if (SpeakerName2ID.empty())
		_D_Dragonian_Lib_Throw_Exception("Speaker Names Could Not Be Empty!");
	if (Symbols.empty())
		_D_Dragonian_Lib_Throw_Exception("Symbols Could Not Be Empty!");
	for (const auto& i : SpeakerName2ID)
		SpeakerID2Name[i.second] = i.first;

	{
		auto PhonemeUNK = Symbols.find(L"UNK");
		if (PhonemeUNK == Symbols.end())
			PhonemeUNK = Symbols.find(L"[UNK]");
		if (PhonemeUNK == Symbols.end())
		{
			Symbols[L"UNK"] = 0;
			Symbols[L"[UNK]"] = 0;
			LogWarn(L"Phoneme UNK/[UNK] Not Found, Use Default UNKID(0)! ");
			UNKID = 0;
		}
		else
		{
			UNKID = PhonemeUNK->second;
			Symbols[L"UNK"] = UNKID;
			Symbols[L"[UNK]"] = UNKID;
		}
	}

	{
		auto PhonemePAD = Symbols.find(L"PAD");
		if (PhonemePAD == Symbols.end())
			PhonemePAD = Symbols.find(L"[PAD]");
		if (PhonemePAD == Symbols.end())
			PhonemePAD = Symbols.find(HParams.PadSymbol);
		if (PhonemePAD == Symbols.end())
		{
			Symbols[L"PAD"] = 0;
			Symbols[L"[PAD]"] = 0;
			Symbols[HParams.PadSymbol] = 0;
			LogWarn(L"Phoneme PAD/[PAD] Not Found, Use Default PADID(0)! ");
			PADID = 0;
		}
		else
		{
			PADID = PhonemePAD->second;
			Symbols[L"PAD"] = PADID;
			Symbols[L"[PAD]"] = PADID;
			Symbols[HParams.PadSymbol] = PADID;
		}
	}
}

DragonianLibSTL::Vector<float> TextToSpeech::Inference(
	TTSInputData& InputData,
	TTSParams& Params,
	bool Inference
) const
{
	if (!Params.LanguageSymbol.empty())
	{
		const auto Res = Language2ID.find(Params.LanguageSymbol);
		if (Res != Language2ID.end())
			Params.LanguageID = Res->second;
	}

	if (!Params.SpeakerName.empty())
	{
		const auto Res = SpeakerName2ID.find(Params.SpeakerName);
		if (Res != SpeakerName2ID.end())
			Params.SpeakerID = Res->second;
	}

	Params.LanguageID %= static_cast<int64_t>(Language2ID.size());
	if (Params.LanguageID < 0)
		Params.LanguageID += static_cast<int64_t>(Language2ID.size());
	Params.SpeakerID %= static_cast<int64_t>(SpeakerName2ID.size());
	if (Params.SpeakerID < 0)
		Params.SpeakerID += static_cast<int64_t>(SpeakerName2ID.size());

	if (!InputData.GetPhonemes().Empty() && InputData._PhonemesIds.Empty())
		InputData._PhonemesIds = CleanedSeq2Indices(InputData.GetPhonemes());
	if (!InputData.GetRefPhonemes().Empty() && InputData._RefPhonemesIds.Empty())
		InputData._RefPhonemesIds = CleanedSeq2Indices(InputData.GetRefPhonemes());
	if (!InputData.GetLanguageSymbols().Empty() && InputData._LanguageIds.Empty())
		InputData._LanguageIds = LanguageSymbol2Indices(InputData.GetLanguageSymbols(), Params.LanguageID);
	if (InputData._SpeakerMixIds.empty())
		InputData._SpeakerMixIds = SpeakerMixSymbol2Indices(InputData.GetSpeakerMixSymbol(), Params.SpeakerID);

	if (Inference)
		return this->Inference(InputData, Params);

	return {};
}

DragonianLibSTL::Vector<float> TextToSpeech::Inference(
	TTSInputData& InputData,
	const TTSParams& Params
) const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

DragonianLibSTL::Vector<int64_t> TextToSpeech::CleanedSeq2Indices(
	const DragonianLibSTL::Vector<std::wstring>& Seq
) const
{
	if (Seq.Empty())
		_D_Dragonian_Lib_Fatal_Error;
	DragonianLibSTL::Vector<int64_t> Indices;
	Indices.Reserve(Seq.Size() * 3);
	for (const auto& i : Seq)
	{
		if (AddBlank)
			Indices.EmplaceBack(PADID);
		const auto Res = Symbols.find(i);
		if (Res != Symbols.end())
			Indices.EmplaceBack(Res->second);
		else
			Indices.EmplaceBack(UNKID);
	}
	if (AddBlank)
		Indices.EmplaceBack(PADID);
	return Indices;
}

DragonianLibSTL::Vector<int64_t> TextToSpeech::LanguageSymbol2Indices(
	const DragonianLibSTL::Vector<std::wstring>& Seq,
	int64_t LanguageID
) const
{
	if (Seq.Empty())
		_D_Dragonian_Lib_Fatal_Error;
	DragonianLibSTL::Vector<int64_t> Indices;
	Indices.Reserve(Seq.Size() * 3);
	for (const auto& i : Seq)
	{
		const auto Res = Language2ID.find(i);
		if (Res != Language2ID.end())
		{
			if (AddBlank)
				Indices.EmplaceBack(Res->second);
			Indices.EmplaceBack(Res->second);
		}
		else
		{
			if (AddBlank)
				Indices.EmplaceBack(LanguageID);
			Indices.EmplaceBack(LanguageID);
		}
	}
	if (AddBlank)
		Indices.EmplaceBack(Indices.Back());
	return Indices;
}

std::map<int64_t, float> TextToSpeech::SpeakerMixSymbol2Indices(
	const DragonianLibSTL::Vector<std::pair<std::wstring, float>>& Seq,
	int64_t SpeakerID
) const
{
	if (Seq.Empty())
		return { { SpeakerID, 1.0f } };

	std::map<int64_t, float> Indices;
	for (const auto& i : Seq)
	{
		const auto Res = SpeakerName2ID.find(i.first);
		if (Res != SpeakerName2ID.end())
			Indices[Res->second] = i.second;
		else
			Indices[SpeakerID] = i.second;
	}

	auto Sum = 0.0f;
	for (const auto& i : Indices | std::ranges::views::values)
		Sum += i;

	if (Sum > 0.0001f)
		for (auto& i : Indices | std::ranges::views::values)
			i /= Sum;
	else
		Indices[SpeakerID] = 1.0f;

	return Indices;
}

DragonianLibSTL::Vector<DragonianLibSTL::Vector<bool>> TextToSpeech::generatePath(
	float* duration,
	size_t durationSize,
	size_t maskSize
)
{
	for (size_t i = 1; i < maskSize; ++i)
		duration[i] = duration[i - 1] + duration[i];
	DragonianLibSTL::Vector path(durationSize, DragonianLibSTL::Vector(maskSize, false));
	/*
	const auto path = new float[maskSize * durationSize];
	for (size_t i = 0; i < maskSize; ++i)
		for (size_t j = 0; j < durationSize; ++j)
			path[i][j] = (j < (size_t)duration[i] ? 1.0f : 0.0f);
	for (size_t i = maskSize - 1; i > 0ull; --i)
		for (size_t j = 0; j < durationSize; ++j)
			path[i][j] -= path[i-1][j];
	 */
	auto dur = (size_t)duration[0];
	for (size_t j = 0; j < dur; ++j)
		path[j][0] = true;
	/*
	for (size_t i = maskSize - 1; i > 0ull; --i)
		for (size_t j = 0; j < durationSize; ++j)
			path[i][j] = (j < (size_t)duration[i] && j >= (size_t)duration[i - 1]);
	Vector<Vector<float>> tpath(durationSize, Vector<float>(maskSize));
	for (size_t i = 0; i < maskSize; ++i)
		for (size_t j = 0; j < durationSize; ++j)
			tpath[j][i] = path[i][j];
	 */
	for (size_t j = maskSize - 1; j > 0ull; --j)
	{
		dur = (size_t)duration[j];
		for (auto i = (size_t)duration[j - 1]; i < dur && i < durationSize; ++i)
			path[i][j] = true;
	}
	return path;
}

DragonianLibSTL::Vector<int64_t> TextToSpeech::GetAligments(size_t DstLen, size_t SrcLen)
{
	DragonianLibSTL::Vector bert2ph(DstLen + 1, 0ll);

	size_t startFrame = 0;
	const double ph_durs = static_cast<double>(DstLen) / static_cast<double>(SrcLen);
	for (size_t iph = 0; iph < SrcLen; ++iph)
	{
		const auto endFrame = static_cast<size_t>(round(static_cast<double>(iph) * ph_durs + ph_durs));
		for (auto j = startFrame; j < endFrame + 1; ++j)
			bert2ph[j] = static_cast<long long>(iph) + 1;
		startFrame = endFrame + 1;
	}
	return bert2ph;
}

ContextModel::ContextModel(
	const std::wstring& ModelPath,
	const ExecutionProviders& ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : LibTTSModule(ExecutionProvider_, DeviceID_, ThreadCount_),
Tokenizer(ModelPath + L"/Tokenizer.json")
{
	Session = std::make_shared<Ort::Session>(*OnnxEnv, (ModelPath + L"/model.onnx").c_str(), *SessionOptions);
}

std::pair<DragonianLibSTL::Vector<float>, int64_t> ContextModel::Inference(
	const std::wstring& InputData,
	Dict::Tokenizer::TokenizerMethod _Method,
	bool _SkipNonLatin,
	size_t _MaximumMatching
) const
{
	DragonianLibSTL::Vector<std::wstring> Tokens;
	Tokenizer.Tokenize(
		InputData,
		Tokens,
		_Method,
		_SkipNonLatin,
		_MaximumMatching
	);
	auto input_ids = Tokenizer(Tokens);
	return Inference(input_ids);
}

std::pair<DragonianLibSTL::Vector<float>, int64_t> ContextModel::Inference(DragonianLibSTL::Vector<int64_t>& TokenIds) const
{
	std::vector<int64_t> attention_mask(TokenIds.Size(), 1), token_type_ids(TokenIds.Size(), 0);
	const int64_t AttentionShape[2] = { 1, (int64_t)TokenIds.Size() };
	std::vector<Ort::Value> AttentionInput, AttentionOutput;
	AttentionInput.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo, TokenIds.Data(), TokenIds.Size(), AttentionShape, 2));
	if (Session->GetInputCount() == 3)
		AttentionInput.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, attention_mask.data(), attention_mask.size(), AttentionShape, 2));
	AttentionInput.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo, token_type_ids.data(), token_type_ids.size(), AttentionShape, 2));
	try
	{
		AttentionOutput = Session->Run(Ort::RunOptions{ nullptr },
			InputNames.data(),
			AttentionInput.data(),
			Session->GetInputCount(),
			OutputNames.data(),
			1);
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Bert\n") + e.what()));
	}
	const auto AttentionOutputTensor = AttentionOutput[0].GetTensorData<float>();
	const auto AttentionOutputSize = AttentionOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();

	return {
		{ AttentionOutputTensor, AttentionOutputTensor + AttentionOutputSize },
		std::max(
			AttentionOutput[0].GetTensorTypeAndShapeInfo().GetShape().back(),
			AttentionOutput[0].GetTensorTypeAndShapeInfo().GetShape().front()
		)
	};
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End