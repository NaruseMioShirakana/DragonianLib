#include "../header/GPT-SoVits.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

GptSoVits::GptSoVits(
	const ModelHParams& _Config,
	const ProgressCallback& _ProgressCallback,
	const DurationCallback& _DurationCallback,
	ExecutionProviders ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : TextToSpeech(_Config, ExecutionProvider_, DeviceID_, ThreadCount_),
NumLayers(_Config.NumLayers), EmbeddingDim(_Config.EmbeddingDim), EOSId(_Config.EOSId)
{
	ProgressCallbackFunction = _ProgressCallback;
	CustomDurationCallback = _DurationCallback;
	ModelSamplingRate = _Config.SamplingRate;

	try
	{
		sessionVits = std::make_shared<Ort::Session>(*OnnxEnv, _Config.GptSoVitsConfig.VitsPath.c_str(), *SessionOptions);
		sessionSSL = std::make_shared<Ort::Session>(*OnnxEnv, _Config.GptSoVitsConfig.SSLPath.c_str(), *SessionOptions);
		sessionEncoder = std::make_shared<Ort::Session>(*OnnxEnv, _Config.GptSoVitsConfig.EncoderPath.c_str(), *SessionOptions);
		sessionDecoder = std::make_shared<Ort::Session>(*OnnxEnv, _Config.GptSoVitsConfig.DecoderPath.c_str(), *SessionOptions);
		sessionFDecoder = std::make_shared<Ort::Session>(*OnnxEnv, _Config.GptSoVitsConfig.FDecoderPath.c_str(), *SessionOptions);
	}
	catch (Ort::Exception& _exception)
	{
		_D_Dragonian_Lib_Throw_Exception(_exception.what());
	}
}

DragonianLibSTL::Vector<float> GptSoVits::Inference(TTSInputData& InputData, const TTSParams& Params) const
{
	if (InputData._BertVec.Size() != 2)
		_D_Dragonian_Lib_Throw_Exception("Missing Reference or Target Bert Data");

	if (InputData._ReferenceAudio16KSr.Empty() || InputData._ReferenceAudioSrc.Empty())
		_D_Dragonian_Lib_Throw_Exception("Reference Audio Could Not Be Empty!");

	if (InputData._PhonemesIds.Empty() || InputData._RefPhonemesIds.Empty())
		_D_Dragonian_Lib_Throw_Exception("PhonemesIds Could Not Be Empty!");

	const int64_t SSLShape[] = { 1, (int64_t)InputData._ReferenceAudio16KSr.Size() };
	std::vector<Ort::Value> SSLInput, SSLOutPut;
	SSLInput.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._ReferenceAudio16KSr.Data(),
			InputData._ReferenceAudio16KSr.Size(),
			SSLShape,
			2
		)
	);
	try
	{
		SSLOutPut = sessionSSL->Run(Ort::RunOptions{ nullptr },
			SSLInputNames.data(),
			SSLInput.data(),
			SSLInputNames.size(),
			SSLOutputNames.data(),
			SSLOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: SSL\n") + e.what()));
	}

	const auto PhonemeSize = InputData._PhonemesIds.Size();
	const auto RefPhonemeSize = InputData._RefPhonemesIds.Size();
	// Reference
	{
		auto& BertVec = InputData._BertVec[0];
		const auto RefBertSize = InputData._BertDims * RefPhonemeSize;

		if (BertVec.Size() != RefBertSize)
		{
			DragonianLibSTL::Vector<float> BertData(RefBertSize);
			if (const auto SrcBertCount = BertVec.Size() / InputData._BertDims; SrcBertCount)
			{
				auto Token2Ph = GetAligments(RefPhonemeSize, SrcBertCount);
				for (size_t IndexOfSrcVector = 0; IndexOfSrcVector < RefPhonemeSize; ++IndexOfSrcVector)
					memcpy(
						BertData.Data() + IndexOfSrcVector * InputData._BertDims,
						BertVec.Data() + Token2Ph[IndexOfSrcVector] * InputData._BertDims,
						InputData._BertDims * sizeof(float)
					);
			}
			else
				memset(BertData.Data(), 0, RefBertSize * sizeof(float));
			BertVec = std::move(BertData);
		}
	}
	// Target
	{
		auto& BertVec = InputData._BertVec[1];
		const auto TargetBertSize = InputData._BertDims * PhonemeSize;

		if (BertVec.Size() != TargetBertSize)
		{
			DragonianLibSTL::Vector<float> BertData(TargetBertSize);
			if (const auto SrcBertCount = BertVec.Size() / InputData._BertDims; SrcBertCount)
			{
				auto Token2Ph = GetAligments(PhonemeSize, SrcBertCount);
				for (size_t IndexOfSrcVector = 0; IndexOfSrcVector < PhonemeSize; ++IndexOfSrcVector)
					memcpy(
						BertData.Data() + IndexOfSrcVector * InputData._BertDims,
						BertVec.Data() + Token2Ph[IndexOfSrcVector] * InputData._BertDims,
						InputData._BertDims * sizeof(float)
					);
			}
			else
				memset(BertData.Data(), 0, TargetBertSize * sizeof(float));
			BertVec = std::move(BertData);
		}
	}

	int64_t RefSeqShape[] = { 1, (int64_t)RefPhonemeSize };
	int64_t RefBertShape[] = { RefSeqShape[1], 1024 };
	int64_t TargetSeqShape[] = { 1, (int64_t)PhonemeSize };
	int64_t TargetBertShape[] = { TargetSeqShape[1], 1024 };

	std::vector<Ort::Value> EncoderInpTensor;
	EncoderInpTensor.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._RefPhonemesIds.Data(),
			InputData._RefPhonemesIds.Size(),
			RefSeqShape,
			2
		)
	);
	EncoderInpTensor.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._PhonemesIds.Data(),
			InputData._PhonemesIds.Size(),
			TargetSeqShape,
			2
		)
	);
	EncoderInpTensor.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._BertVec[0].Data(),
			InputData._BertVec[0].Size(),
			RefBertShape,
			2
		)
	);
	EncoderInpTensor.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._BertVec[1].Data(),
			InputData._BertVec[1].Size(),
			TargetBertShape,
			2
		)
	);
	EncoderInpTensor.emplace_back(std::move(SSLOutPut[0]));

	std::vector<Ort::Value> EncoderOutput, FDecoderOutput, DecoderOutput;

	try
	{
		EncoderOutput = sessionEncoder->Run(Ort::RunOptions{ nullptr },
			EncoderInputNames.data(),
			EncoderInpTensor.data(),
			EncoderInputNames.size(),
			EncoderOutputNames.data(),
			EncoderOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Encoder\n") + e.what()));
	}

	try
	{
		FDecoderOutput = sessionFDecoder->Run(Ort::RunOptions{ nullptr },
			FDecoderInputNames.data(),
			EncoderOutput.data(),
			FDecoderInputNames.size(),
			FDecoderOutputNames.data(),
			FDecoderOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: FDecoder\n") + e.what()));
	}

	int64_t idx = 1;
	for (; idx < Params.MaxDecodeStep; ++idx)
	{
		try
		{
			DecoderOutput = sessionDecoder->Run(Ort::RunOptions{ nullptr },
				DecoderInputNames.data(),
				FDecoderOutput.data(),
				DecoderInputNames.size(),
				DecoderOutputNames.data(),
				DecoderOutputNames.size());
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Decoder\n") + e.what()));
		}

		const auto Logit = DecoderOutput[4].GetTensorData<float>();
		int64_t MaxIdx = 0;
		for (int64_t midx = 0; midx < EOSId + 1; ++midx)
			if (Logit[midx] > Logit[MaxIdx])
				MaxIdx = midx;
		if (MaxIdx == EOSId)
			break;
		if (*DecoderOutput[5].GetTensorData<int32_t>() == EOSId)
			break;

		FDecoderOutput[0] = std::move(DecoderOutput[0]);
		FDecoderOutput[1] = std::move(DecoderOutput[1]);
		FDecoderOutput[2] = std::move(DecoderOutput[2]);
		FDecoderOutput[3] = std::move(DecoderOutput[3]);
	}

	auto PredSemanticPtr = DecoderOutput[0].GetTensorData<int64_t>();
	auto PredSemanticShape = DecoderOutput[0].GetTensorTypeAndShapeInfo().GetShape();
	PredSemanticShape.insert(PredSemanticShape.begin(), 1);
	std::vector PredSemantic(PredSemanticPtr + std::max((PredSemanticShape[2] - idx - 1), 0ll), PredSemanticPtr + PredSemanticShape[2]);
	if (PredSemantic[PredSemantic.size() - 1] == EOSId)
		PredSemantic[PredSemantic.size() - 1] = 0;
	PredSemanticShape[2] = int64_t(PredSemantic.size());

	std::vector<Ort::Value> VitsTensors;
	VitsTensors.emplace_back(std::move(EncoderInpTensor[1]));
	VitsTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, PredSemantic.data(), PredSemantic.size(), PredSemanticShape.data(), PredSemanticShape.size()));
	int64_t AudioShape[] = { 1, (int64_t)InputData._ReferenceAudioSrc.Size() };
	VitsTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, InputData._ReferenceAudioSrc.Data(), InputData._ReferenceAudioSrc.Size(), AudioShape, 2));

	try
	{
		VitsTensors = sessionVits->Run(Ort::RunOptions{ nullptr },
			VitsInputNames.data(),
			VitsTensors.data(),
			VitsInputNames.size(),
			VitsOutputNames.data(),
			VitsOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: SoVits\n") + e.what()));
	}

	const auto shapeOut = VitsTensors[0].GetTensorTypeAndShapeInfo().GetShape();
	const auto outData = VitsTensors[0].GetTensorData<float>();
	int64_t Size = 0;
	for (const auto& i : shapeOut)
		Size += i;
	return { outData, outData + Size };
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End