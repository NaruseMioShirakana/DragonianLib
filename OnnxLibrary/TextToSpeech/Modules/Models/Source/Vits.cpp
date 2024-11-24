#include "../Header/Vits.hpp"
#include <random>
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

Vits::Vits(
	const ModelHParams& _Config,
	const ProgressCallback& _ProgressCallback,
	const DurationCallback& _DurationCallback,
	ExecutionProviders ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : TextToSpeech(_Config, ExecutionProvider_, DeviceID_, ThreadCount_),
VitsType(_Config.VitsConfig.VitsType), DefBertSize(_Config.DefBertSize), VQCodeBookSize(_Config.VQCodeBookSize),
BertCount(_Config.BertCount), UseTone(_Config.UseTone), UseBert(_Config.UseBert), UseLength(_Config.UseLength),
UseLanguage(_Config.UseLanguageIds), EncoderG(_Config.EncoderG), ReferenceBert(_Config.ReferenceBert),
UseVQ(_Config.UseVQ), UseClap(_Config.UseClap), Emotion2Id(_Config.Emotion2Id)
{
	if (UseLanguage && Language2ID.empty())
		_D_Dragonian_Lib_Throw_Exception("Language Symbol Could Not Be Empty!");

	ProgressCallbackFunction = _ProgressCallback;
	CustomDurationCallback = _DurationCallback;
	ModelSamplingRate = _Config.SamplingRate;

	//InputNames
	if (UseLength)
		EncoderInputNames.emplace_back("x_lengths");
	if (UseTone)
		EncoderInputNames.emplace_back("t");
	if (Emotion)
	{
		EmotionVector = EmotionLoader(_Config.EmotionFilePath);
		EncoderInputNames.emplace_back("emo");
	}
	if (UseLanguage)
		EncoderInputNames.emplace_back("language");

	if (UseBert)
	{
		if (BertCount > 0 && BertCount < static_cast<int64_t>(VistBertInputNames.size()))
		{
			for (int64_t i = 0; i < BertCount; ++i)
			{
				EncoderInputNames.emplace_back(VistBertInputNames[i]);
			}
		}
	}

	if (UseClap)
		EncoderInputNames.emplace_back("emo");

	sessionDec = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.DecoderPath.c_str(), *SessionOptions);
	sessionEnc_p = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.EncoderPath.c_str(), *SessionOptions);
	sessionFlow = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.FlowPath.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.SpeakerEmbedding.c_str(), 0) != -1)
		sessionEmb = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.SpeakerEmbedding.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.DurationPredictorPath.c_str(), 0) != -1)
		sessionDp = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.DurationPredictorPath.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.SDurationPredictorPath.c_str(), 0) != -1)
		sessionSdp = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.SDurationPredictorPath.c_str(), *SessionOptions);

	if (!sessionDp && !sessionSdp)
		_D_Dragonian_Lib_Throw_Exception("You must have a duration predictor");

	if (sessionEmb)
	{
		if (EncoderG) EncoderInputNames.emplace_back("g");
		SdpInputNames.emplace_back("g");
		DpInputNames.emplace_back("g");
		FlowInputNames.emplace_back("g");
		DecInputNames.emplace_back("g");
	}

	if (VitsType == "BertVits" && sessionEnc_p->GetInputCount() == EncoderInputNames.size() + 2)
	{
		EncoderInputNames.emplace_back("vqidx");
		EncoderInputNames.emplace_back("sid");
		UseVQ = true;
	}
}

DragonianLibSTL::Vector<float> Vits::GetEmotionVector(
	const DragonianLibSTL::Vector<std::wstring>& EmotionSymbol
) const
{
	if (EmotionSymbol.Empty())
		return { 1024, 0.f, GetMemoryProvider(Device::CPU) };
	DragonianLibSTL::Vector Result(1024, 0.f);
	uint64_t mul = 0;
	for (const auto& EmotionSymb : EmotionSymbol)
	{
		size_t emoId;
		auto EmoID = Emotion2Id.find(EmotionSymb);
		if (EmoID != Emotion2Id.end())
			emoId = EmoID->second;
		else
			emoId = _wtoi64(EmotionSymb.c_str());

		auto emoVec = EmotionVector[emoId];
		for (size_t i = 0; i < 1024; ++i)
			Result[i] += (emoVec[i] - Result[i]) / (float)(mul + 1ull);
		++mul;
	}
	return Result;
}

DragonianLibSTL::Vector<float> Vits::Inference(
	TTSInputData& InputData,
	const TTSParams& Params
) const
{
	std::mt19937_64 Generate(Params.Seed);
	std::normal_distribution FloatRandFn(0.f, 1.f);

	const auto PhonemeSize = InputData._PhonemesIds.Size();

	if (InputData._SpeakerMixIds.empty())
		_D_Dragonian_Lib_Throw_Exception("SpeakerMixIds Could Not Be Empty!");

	if (InputData._PhonemesIds.Empty())
		_D_Dragonian_Lib_Throw_Exception("PhonemesIds Could Not Be Empty!");
	if (AddBlank && PhonemeSize < 3)
		_D_Dragonian_Lib_Throw_Exception("PhonemesIds Size Must Be Greater Than 3!");

	if (UseTone && InputData._Tones.Empty())
		_D_Dragonian_Lib_Throw_Exception("Tones Could Not Be Empty!");
	if (InputData._Tones.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._Tones.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> Tones;
			Tones.Reserve(PhonemeSize);
			for (auto Tone : InputData._Tones)
			{
				Tones.EmplaceBack(Tone);
				Tones.EmplaceBack(Tone);
			}
			Tones.EmplaceBack(InputData._Tones.Back());
			InputData._Tones = std::move(Tones);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Tones Size Not Matched!");
	}

	if (UseLanguage && InputData._LanguageIds.Empty())
		InputData._LanguageIds = { PhonemeSize, Params.LanguageID, GetMemoryProvider(Device::CPU) };
	if (InputData._LanguageIds.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._LanguageIds.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> LanguageIds;
			LanguageIds.Reserve(PhonemeSize);
			for (auto LanguageId : InputData._LanguageIds)
			{
				LanguageIds.EmplaceBack(LanguageId);
				LanguageIds.EmplaceBack(LanguageId);
			}
			LanguageIds.EmplaceBack(InputData._LanguageIds.Back());
			InputData._LanguageIds = std::move(LanguageIds);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("LanguageIds Size Not Matched!");
	}

	if (!InputData._Durations.Empty() && InputData._Durations.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._Durations.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> Durations;
			Durations.Reserve(PhonemeSize);
			for (auto Duration : InputData._Durations)
			{
				Durations.EmplaceBack(Duration);
				Durations.EmplaceBack(Duration);
			}
			Durations.EmplaceBack(InputData._Durations.Back());
			InputData._Durations = std::move(Durations);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Durations Size Not Matched!");
	}

	if (Emotion)
	{
		if (InputData._Emotion.Empty())
			InputData._Emotion = GetEmotionVector(Params.EmotionPrompt);
		if (InputData._Emotion.Size() != 1024)
			_D_Dragonian_Lib_Throw_Exception("Emotion Size Not Matched!");
	}

	if (UseBert)
	{
		if (InputData._BertVec.Empty())
			_D_Dragonian_Lib_Throw_Exception("BertVec Could Not Be Empty!");
		if (InputData._BertVec.Size() != static_cast<size_t>(BertCount))
			_D_Dragonian_Lib_Throw_Exception("BertVec Size Not Matched!");

		size_t BertBufferSize = 0;
		for (const auto& BertVec : InputData._BertVec)
		{
			if (!BertBufferSize)
				BertBufferSize = BertVec.Size();
			else if (BertBufferSize != BertVec.Size())
				_D_Dragonian_Lib_Throw_Exception("BertVec Size Not Matched!");
		}
		if (BertBufferSize == 0)
			_D_Dragonian_Lib_Throw_Exception("BertVec Could Not Be Empty!");

		for (auto& BertVec : InputData._BertVec)
			if (BertVec.Empty())
				BertVec = { BertBufferSize, 0.f, GetMemoryProvider(Device::CPU) };

		auto BertSize = BertBufferSize / InputData._BertDims;
		if (BertSize != PhonemeSize)
		{
			if (PhonemeSize != InputData._Token2Phoneme.Size())
				InputData._Token2Phoneme = GetAligments(PhonemeSize, BertSize);
			const auto TargetBertSize = PhonemeSize * InputData._BertDims;
			for (auto& BertVec : InputData._BertVec)
			{
				DragonianLibSTL::Vector<float> BertData(TargetBertSize);
				if (BertVec.Size() == BertBufferSize)
				{
					for (size_t IndexOfSrcVector = 0; IndexOfSrcVector < PhonemeSize; ++IndexOfSrcVector)
						memcpy(
							BertData.Data() + IndexOfSrcVector * InputData._BertDims,
							BertVec.Data() + (InputData._Token2Phoneme[IndexOfSrcVector]) * InputData._BertDims,
							InputData._BertDims * sizeof(float)
						);
				}
				else
					memset(BertData.Data(), 0, TargetBertSize * sizeof(float));
				BertVec = std::move(BertData);
			}
		}
	}

	std::vector<Ort::Value> EncoderInputs, EncoderOutputs;

	const int64_t TextSeqShape[2] = { 1, (int64_t)PhonemeSize };
	EncoderInputs.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._PhonemesIds.Data(),
			PhonemeSize,
			TextSeqShape,
			2
		)
	);

	auto Length = (int64_t)PhonemeSize;
	constexpr int64_t LengthShape[1] = { 1 };
	if (UseLength)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				&Length,
				1,
				LengthShape,
				1
			)
		);

	constexpr int64_t EmotionShape[1] = { 1024 };
	if (Emotion)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._Emotion.Data(),
				1024,
				EmotionShape,
				1
			)
		);

	if (UseTone)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._Tones.Data(),
				PhonemeSize,
				TextSeqShape,
				2
			)
		);

	if (UseLanguage)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._LanguageIds.Data(),
				PhonemeSize,
				TextSeqShape,
				2
			)
		);

	int64_t BertShape[2] = { (int64_t)PhonemeSize, InputData._BertDims };
	if (UseBert)
		for (auto BertVec : InputData._BertVec)
			EncoderInputs.emplace_back(
				Ort::Value::CreateTensor(
					*MemoryInfo,
					BertVec.Data(),
					BertVec.Size(),
					BertShape,
					2
				)
			);

	constexpr int64_t ClapShape[2] = { 512, 1 };
	if (UseClap)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._ClapVec.Data(),
				512,
				ClapShape,
				2
			)
		);


	DragonianLibSTL::Vector<float> SpeakerVector;
	int64_t SpeakerShape[3] = { 1, 0, 1 };
	if (sessionEmb)
	{
		SpeakerVector = GetSpeakerEmbedding(InputData._SpeakerMixIds);
		SpeakerShape[1] = (int64_t)SpeakerVector.Size();
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				SpeakerVector.Data(),
				SpeakerVector.Size(),
				SpeakerShape,
				3
			)
		);
	}

	int64_t VQIndices[] = { Params.VQIndex };
	int64_t SidIndices[] = { Params.SpeakerID };
	if (UseVQ)
	{
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				VQIndices,
				1,
				LengthShape,
				1
			)
		);
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				SidIndices,
				1,
				LengthShape,
				1
			)
		);
	}

	try
	{
		EncoderOutputs = sessionEnc_p->Run(Ort::RunOptions{ nullptr },
			EncoderInputNames.data(),
			EncoderInputs.data(),
			EncoderInputs.size(),
			EncoderOutputNames.data(),
			EncoderOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: enc_p\n") + e.what()));
	}

	std::vector<float>
		m_p(EncoderOutputs[1].GetTensorData<float>(), EncoderOutputs[1].GetTensorData<float>() + EncoderOutputs[1].GetTensorTypeAndShapeInfo().GetElementCount()),
		logs_p(EncoderOutputs[2].GetTensorData<float>(), EncoderOutputs[2].GetTensorData<float>() + EncoderOutputs[2].GetTensorTypeAndShapeInfo().GetElementCount()),
		x_mask(EncoderOutputs[3].GetTensorData<float>(), EncoderOutputs[3].GetTensorData<float>() + EncoderOutputs[3].GetTensorTypeAndShapeInfo().GetElementCount());

	const auto xshape = EncoderOutputs[0].GetTensorTypeAndShapeInfo().GetShape();

	std::vector w_ceil(PhonemeSize, 1.f);
	if (InputData._Durations.Empty())
	{
		const int64_t zinputShape[3] = { xshape[0],2,xshape[2] };
		const int64_t zinputCount = xshape[0] * xshape[2] * 2;
		std::vector<float> zinput(zinputCount);
		for (auto& it : zinput)
			it = FloatRandFn(Generate) * Params.DurationPredictorNoiseScale;
		std::vector<Ort::Value> DurationPredictorInput;
		DurationPredictorInput.emplace_back(std::move(EncoderOutputs[0]));
		DurationPredictorInput.emplace_back(std::move(EncoderOutputs[3]));
		DurationPredictorInput.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				zinput.data(),
				zinputCount,
				zinputShape,
				3
			)
		);
		if (sessionEmb)
			DurationPredictorInput.emplace_back(
				Ort::Value::CreateTensor(
					*MemoryInfo,
					SpeakerVector.Data(),
					SpeakerVector.Size(),
					SpeakerShape,
					3
				)
			);
		if (sessionSdp)
		{
			std::vector<Ort::Value> StochasticDurationPredictorOutput;
			try
			{
				StochasticDurationPredictorOutput = sessionSdp->Run(Ort::RunOptions{ nullptr },
					SdpInputNames.data(),
					DurationPredictorInput.data(),
					DurationPredictorInput.size(),
					SdpOutputNames.data(),
					SdpOutputNames.size());
			}
			catch (Ort::Exception& e)
			{
				_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dp\n") + e.what()));
			}
			const auto w_data = StochasticDurationPredictorOutput[0].GetTensorData<float>();
			const auto w_data_length = StochasticDurationPredictorOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
			if (w_data_length != w_ceil.size())
				w_ceil.resize(w_data_length, 0.f);
			float SdpFactor = 1.f - Params.FactorDpSdp;
			if (sessionDp)
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i] * SdpFactor) * x_mask[i] * Params.LengthScale);
			else
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i]) * x_mask[i] * Params.LengthScale);
		}
		if (sessionDp)
		{
			std::vector<Ort::Value> DurationPredictorOutput;
			DurationPredictorInput.erase(DurationPredictorInput.begin() + 2);
			try
			{
				DurationPredictorOutput = sessionDp->Run(Ort::RunOptions{ nullptr },
					DpInputNames.data(),
					DurationPredictorInput.data(),
					DurationPredictorInput.size(),
					DpOutputNames.data(),
					DpOutputNames.size());
			}
			catch (Ort::Exception& e)
			{
				_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dp\n") + e.what()));
			}
			const auto w_data = DurationPredictorOutput[0].GetTensorData<float>();
			const auto w_data_length = DurationPredictorOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
			if (w_data_length != w_ceil.size())
				w_ceil.resize(w_data_length, 0.f);
			if (sessionSdp)
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] += ceil(exp(w_data[i] * Params.FactorDpSdp) * x_mask[i] * Params.LengthScale);
			else
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i]) * x_mask[i] * Params.LengthScale);
			CustomDurationCallback(w_ceil.data(), w_ceil.data() + w_ceil.size());
		}
	}
	else
		for (size_t i = 0; i < w_ceil.size(); ++i)
			w_ceil[i] = float(InputData._Durations[i]);

	const auto maskSize = x_mask.size();
	float y_length_f = 0.0;
	int64_t y_length;
	for (size_t i = 0; i < w_ceil.size(); ++i)
		y_length_f += w_ceil[i];
	if (y_length_f < 1.0f)
		y_length = 1;
	else
		y_length = (int64_t)y_length_f;

	auto attn = generatePath(w_ceil.data(), y_length, maskSize);
	std::vector logVec(192, std::vector(y_length, 0.0f));
	std::vector mpVec(192, std::vector(y_length, 0.0f));
	std::vector<float> nlogs_pData(192 * y_length);
	for (size_t i = 0; i < static_cast<size_t>(y_length); ++i)
	{
		for (size_t j = 0; j < 192; ++j)
		{
			for (size_t k = 0; k < maskSize; k++)
			{
				if (attn[i][k])
				{
					mpVec[j][i] += m_p[j * maskSize + k];
					logVec[j][i] += logs_p[j * maskSize + k];
				}
			}
			nlogs_pData[j * y_length + i] = mpVec[j][i] + FloatRandFn(Generate) * exp(logVec[j][i]) * Params.NoiseScale;
		}
	}
	std::vector y_mask(y_length, 1.0f);
	const int64_t zshape[3] = { 1,192,y_length };
	const int64_t yshape[3] = { 1,1,y_length };

	std::vector<Ort::Value> FlowDecInputs, FlowDecOutputs;
	FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
		*MemoryInfo, nlogs_pData.data(), 192 * y_length, zshape, 3));
	FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
		*MemoryInfo, y_mask.data(), y_length, yshape, 3));
	if (sessionEmb)
		FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
			*MemoryInfo, SpeakerVector.Data(), SpeakerVector.Size(), SpeakerShape, 3));

	try
	{
		FlowDecOutputs = sessionFlow->Run(Ort::RunOptions{ nullptr },
			FlowInputNames.data(),
			FlowDecInputs.data(),
			FlowDecInputs.size(),
			FlowOutputNames.data(),
			FlowOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dec & flow\n") + e.what()));
	}
	FlowDecInputs[0] = std::move(FlowDecOutputs[0]);
	if (sessionEmb)
		FlowDecInputs[1] = std::move(FlowDecInputs[2]);
	FlowDecInputs.pop_back();
	try
	{

		FlowDecOutputs = sessionDec->Run(Ort::RunOptions{ nullptr },
			DecInputNames.data(),
			FlowDecInputs.data(),
			FlowDecInputs.size(),
			DecOutputNames.data(),
			DecOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dec & flow\n") + e.what()));
	}
	const auto shapeOut = FlowDecOutputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto outData = FlowDecOutputs[0].GetTensorData<float>();
	return { outData, outData + shapeOut };
}

DragonianLibSTL::Vector<float> Vits::GetSpeakerEmbedding(const std::map<int64_t, float>& SpeakerMixIds) const
{
	DragonianLibSTL::Vector<float> SpeakerVector;

	constexpr int64_t LengthShape[1] = { 1 };
	for (const auto& SpeakerMixId : SpeakerMixIds)
	{
		std::vector<Ort::Value> EmbiddingInput;
		std::vector<Ort::Value> EmbiddingOutput;
		if (SpeakerMixId.first >= SpeakerCount || SpeakerMixId.second < 0.0001f || SpeakerMixId.first < 0)
			continue;
		int64_t Character[1] = { SpeakerMixId.first };
		EmbiddingInput.push_back(Ort::Value::CreateTensor(
			*MemoryInfo, Character, 1, LengthShape, 1));
		try
		{
			EmbiddingOutput = sessionEmb->Run(Ort::RunOptions{ nullptr },
				EmbiddingInputNames.data(),
				EmbiddingInput.data(),
				EmbiddingInput.size(),
				EmbiddingOutputNames.data(),
				EmbiddingOutputNames.size());
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: emb\n") + e.what()));
		}
		const auto GOutData = EmbiddingOutput[0].GetTensorData<float>();
		const auto GOutCount = EmbiddingOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
		if (SpeakerVector.Empty())
		{
			SpeakerVector = { GOutData, GOutData + GOutCount };
			for (auto& idx : SpeakerVector)
				idx *= SpeakerMixId.second;
		}
		else
			for (size_t i = 0; i < GOutCount; ++i)
				SpeakerVector[i] += GOutData[i] * SpeakerMixId.second;
	}

	return SpeakerVector;
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End