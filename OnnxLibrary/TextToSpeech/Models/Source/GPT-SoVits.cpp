#include "../GPT-SoVits.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

namespace GptSoVits
{
	T2SAR::T2SAR(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& _Config,
		const std::shared_ptr<Logger>& _Logger
	) : _MyPromptModel(_Environment, _Config.ModelPaths.at(L"Prompt"), _Logger),
		_MyDecodeModel(_Environment, _Config.ModelPaths.at(L"Decode"), _Logger)
	{
		if (_Config.Parameters.contains(L"EOSId"))
			_MyEOSId = std::stoll(_Config.Parameters.at(L"EOSId"));
	}

	Tensor<Int64, 2, Device::CPU> T2SAR::Forward(
		const Tensor<Int64, 2, Device::CPU>& _PhonemeIds,
		const Tensor<Int64, 2, Device::CPU>& _RefPhonemeIds,
		const Tensor<Float32, 3, Device::CPU>& _BertFeature,
		Float32 _TopP,
		Float32 _Temperature,
		Float32 _RepetitionPenalty
	)
	{
		if (_PhonemeIds.Null())
			_D_Dragonian_Lib_Throw_Exception("PhonemeIds Could Not Be Null!");
		if (_RefPhonemeIds.Null())
			_D_Dragonian_Lib_Throw_Exception("RefPhonemeIds Could Not Be Null!");
		if (_BertFeature.Null())
			_D_Dragonian_Lib_Throw_Exception("BertFeature Could Not Be Null!");

		constexpr Int64 One = 1;
		Int64 PhonemeSize = _PhonemeIds.Shape(1);
		Int64 RefPhonemeSize = _RefPhonemeIds.Shape(1);

		InputTensorsType PromptInputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyPromptModel.GetMemoryInfo(),
					_PhonemeIds,
					_MyPromptModel.GetInputTypes()[0],
					_MyPromptModel.GetInputDims()[0],
					{ L"BatchSize", L"SequnceLength" },
					"Phoneme",
					_MyPromptModel.GetLoggerPtr()
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				Ort::Value::CreateTensor(
					*_MyPromptModel.GetMemoryInfo(),
					&PhonemeSize,
					1,
					&One,
					1
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyPromptModel.GetMemoryInfo(),
					_RefPhonemeIds,
					_MyPromptModel.GetInputTypes()[2],
					_MyPromptModel.GetInputDims()[2],
					{ L"BatchSize", L"SequnceLength" },
					"RefPhoneme",
					_MyPromptModel.GetLoggerPtr()
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				Ort::Value::CreateTensor(
					*_MyPromptModel.GetMemoryInfo(),
					&RefPhonemeSize,
					1,
					&One,
					1
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyPromptModel.GetMemoryInfo(),
					_BertFeature,
					_MyPromptModel.GetInputTypes()[4],
					_MyPromptModel.GetInputDims()[4],
					{ L"BatchSize", L"SequnceLength", L"FeatureSize" },
					"BertFeature",
					_MyPromptModel.GetLoggerPtr()
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				Ort::Value::CreateTensor(
					*_MyPromptModel.GetMemoryInfo(),
					&_TopP,
					1,
					&One,
					1
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				Ort::Value::CreateTensor(
					*_MyPromptModel.GetMemoryInfo(),
					&_RepetitionPenalty,
					1,
					&One,
					1
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				Ort::Value::CreateTensor(
					*_MyPromptModel.GetMemoryInfo(),
					&_Temperature,
					1,
					&One,
					1
				)
			);
		);

		OrtTuple Outputs;

		_D_Dragonian_Lib_Rethrow_Block(Outputs = _MyPromptModel.RunModel(PromptInputTensors););

		auto Sample = *Outputs.back().GetTensorData<Int64>();
		Outputs.pop_back();
		Outputs.emplace_back(std::move(PromptInputTensors[5]));
		Outputs.emplace_back(std::move(PromptInputTensors[6]));
		Outputs.emplace_back(std::move(PromptInputTensors[7]));

		TemplateLibrary::Vector<Int64> Samples;
		Samples.Reserve(2000);

		auto Inputs = std::move(Outputs);

		Int64 Idx = 1;
		for (; Idx < 1500; ++Idx)
		{
			if (Sample == _MyEOSId)
				break;
			Samples.EmplaceBack(Sample);

			_D_Dragonian_Lib_Rethrow_Block(Outputs = _MyDecodeModel.RunModel(Inputs););

			Inputs[0] = std::move(Outputs[0]);
			Inputs[1] = std::move(Outputs[1]);
			Inputs[2] = std::move(Outputs[2]);
			Inputs[3] = std::move(Outputs[3]);
			Inputs[4] = std::move(Outputs[4]);
			Sample = *Outputs[5].GetTensorData<Int64>();
		}

		return Functional::FromVector(std::move(Samples)).View(1, -1);
	}

	SoVits::SoVits(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& _Config,
		const std::shared_ptr<Logger>& _Logger
	) : OnnxModelBase(_Environment, _Config.ModelPaths.at(L"Vits"), _Logger),
		_MyExtract(_Environment, _Config.ModelPaths.at(L"Extract"), _Logger),
		_MySamplingRate(_Config.SamplingRate)
	{

	}

	Tensor<Float32, 3, Device::CPU> SoVits::Forward(
		const Tensor<Int64, 2, Device::CPU>& _Phonemes,
		const Tensor<Int64, 2, Device::CPU>& _PredSemantic,
		const Tensor<Float32, 2, Device::CPU>& _RefAudio
	)
	{
		if (_Phonemes.Null())
			_D_Dragonian_Lib_Throw_Exception("Phonemes Could Not Be Null!");
		if (_PredSemantic.Null())
			_D_Dragonian_Lib_Throw_Exception("PredSemantic Could Not Be Null!");
		if (_RefAudio.Null())
			_D_Dragonian_Lib_Throw_Exception("RefAudio Could Not Be Null!");

		InputTensorsType PromptInputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					_Phonemes,
					GetInputTypes()[0],
					GetInputDims()[0],
					{ L"BatchSize", L"SequnceLength" },
					"Phonemes",
					GetLoggerPtr()
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					_PredSemantic,
					GetInputTypes()[1],
					GetInputDims()[1],
					{ L"BatchSize", L"SequnceLength" },
					"PredSemantic",
					GetLoggerPtr()
				)
			);
		);
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					_RefAudio,
					GetInputTypes()[2],
					GetInputDims()[2],
					{ L"BatchSize", L"SequnceLength" },
					"RefAudio",
					GetLoggerPtr()
				)
			);
		);

		OrtTuple Outputs;

		_D_Dragonian_Lib_Rethrow_Block(Outputs = RunModel(PromptInputTensors););
		Dimensions<3> Shape;
		auto OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
		if (OutputShape.size() == 3)
			Shape = { OutputShape[0], OutputShape[1], OutputShape[2] };
		else if (OutputShape.size() == 2)
			Shape = { 1, OutputShape[0], OutputShape[1] };
		else if (OutputShape.size() == 1)
			Shape = { 1, 1, OutputShape[0] };
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid Output Shape");

		_D_Dragonian_Lib_Rethrow_Block(
			return CreateTensorViewFromOrtValue<Float>(std::move(Outputs[0]), Shape);
		);
	}

	Tensor<Int64, 3, Device::CPU> SoVits::ExtractLatent(
		const Tensor<Float32, 3, Device::CPU>& _RefSSlContext
	)
	{
		if (_RefSSlContext.Null())
			_D_Dragonian_Lib_Throw_Exception("RefAudio Could Not Be Null!");

		InputTensorsType PromptInputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			PromptInputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyExtract.GetMemoryInfo(),
					_RefSSlContext,
					_MyExtract.GetInputTypes()[0],
					_MyExtract.GetInputDims()[0],
					{ L"BatchSize", L"SequnceLength", L"UnitsDim" },
					"SSL",
					_MyExtract.GetLoggerPtr()
				)
			);
		);

		OrtTuple Outputs;

		_D_Dragonian_Lib_Rethrow_Block(Outputs = _MyExtract.RunModel(PromptInputTensors););
		Dimensions<3> Shape;
		auto OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
		if (OutputShape.size() == 3)
			Shape = { OutputShape[0], OutputShape[1], OutputShape[2] };
		else if (OutputShape.size() == 2)
			Shape = { 1, OutputShape[0], OutputShape[1] };
		else if (OutputShape.size() == 1)
			Shape = { 1, 1, OutputShape[0] };
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid Output Shape");

		_D_Dragonian_Lib_Rethrow_Block(
			return CreateTensorViewFromOrtValue<Int64>(std::move(Outputs[0]), Shape);
		);
	}


}

_D_Dragonian_Lib_Lib_Text_To_Speech_End



/*#include "../GPT-SoVits.hpp"

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

_D_Dragonian_Lib_Lib_Text_To_Speech_End*/