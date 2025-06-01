#include <iostream>

#ifndef DRAGONIANLIB_USE_SHARED_LIBS
#include "Libraries/AvCodec/AvCodec.h"
#include "OnnxLibrary/SingingVoiceConversion/Model/Reflow-Svc.hpp"
#include "OnnxLibrary/UnitsEncoder/Register.hpp"
#include "OnnxLibrary/Vocoder/Register.hpp"
#endif
#include "OnnxLibrary/SingingVoiceConversion/Api/NativeApi.h"

#ifdef DRAGONIANLIB_USE_SHARED_LIBS
const auto CBOBJ = DragonianLib::DefaultProgressCallback(std::cout);
static void CBFN(bool c, INT64 p) { CBOBJ(c, p); }
#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	DragonianLib::OnnxRuntime::OnnxEnvironmentOptions Options{
		DragonianLib::Device::CUDA,
		0,
		8,
		4,
		ORT_LOGGING_LEVEL_WARNING
	};

	auto Env = DragonianLib::OnnxRuntime::CreateEnvironment(
		Options
	);

	DragonianLib::OnnxRuntime::SingingVoiceConversion::HParams hParams;
	hParams.ProgressCallback = DragonianLib::DefaultProgressCallback(std::cout);
	hParams.F0Max = 800;
	hParams.F0Min = 65;
	hParams.HopSize = 512;
	hParams.MelBins = 128;
	hParams.UnitsDim = 768;
	hParams.OutputSamplingRate = 44100;
	hParams.SpeakerCount = 92;
	hParams.HasSpeakerEmbedding = true;
	hParams.HasVolumeEmbedding = true;
	hParams.HasSpeakerMixLayer = true;
	hParams.ModelPaths[L"Ctrl"] = __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/encoder.onnx)";
	hParams.ModelPaths[L"Velocity"] = __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/velocity.onnx)";

	auto Model = DragonianLib::OnnxRuntime::SingingVoiceConversion::ReflowSvc(
		Env,
		hParams
	);

	auto Vocoder = DragonianLib::OnnxRuntime::Vocoder::New(
		L"Nsf-HiFi-GAN",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/nsf-hifigan-n.onnx)",
		Env
	);

	DragonianLib::OnnxRuntime::SingingVoiceConversion::Parameters Params;
	Params.SpeakerId = 0;
	Params.PitchOffset = 0.f;
	Params.StftNoiseScale = 1.f;
	Params.NoiseScale = 1.f;
	Params.Reflow.Begin = 0.f;
	Params.Reflow.End = 1.f;
	Params.Reflow.Stride = 0.05f;
	Params.F0HasUnVoice = false;
	Params.Reflow.Sampler = L"Eular";

	//vec-768-layer-12-f16
	const auto UnitsEncoder = DragonianLib::OnnxRuntime::UnitsEncoder::New(
		L"ContentVec-768-l12",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/vec-768-layer-12-f16.onnx)",
		Env,
		16000,
		768
	);

	const auto F0Extractor = DragonianLib::F0Extractor::New(
		L"Dio",
		nullptr
	);

	constexpr auto F0Params = DragonianLib::F0Extractor::Parameters{
		44100,
		320,
		256,
		2048,
		800,
		65,
		0.03f,
		nullptr
	};

	DragonianLib::OnnxRuntime::SingingVoiceConversion::SliceDatas MyData;
	auto AudioStream = DragonianLib::AvCodec::OpenInputStream(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal.wav)"
	);
	auto Audio = AudioStream.DecodeAll(44100, 2);
	auto Input = Audio[{"882000:1323000", "0"}].Transpose(-1, -2).Contiguous();

	auto Mel = Model.Inference(
		Params,
		Input.UnSqueeze(0),
		44100,
		UnitsEncoder,
		F0Extractor,
		F0Params,
		std::nullopt,
		std::nullopt,
		&MyData
	);

	Mel = Model.DenormSpec(Mel);
	auto Output = Vocoder->Forward(Mel, MyData.F0);

	auto AudioOutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal-SVC.wav)"
	);
	AudioOutStream.EncodeAll(
		Output.GetCRng(), 44100
	);
#else
	DragonianVoiceSvcEnviromentSetting EnvSetting;
	DragonianVoiceSvcHyperParameters HyperParameters;
	DragonianVoiceSvcF0ExtractorParameters F0ExtractorParameters;
	DragonianVoiceSvcInferenceParameters InferenceParameters;

	DragonianVoiceSvcEnviroment Enviroment = 0;
	DragonianVoiceSvcModel SvcModel = 0;
	DragonianVoiceSvcUnitsEncoder UnitsEncoder = 0;
	DragonianVoiceSvcF0Extractor F0Extractor = 0;
	DragonianVoiceSvcVocoder Vocoder = 0;

	DragonianVoiceSvcFloatTensor Audio = 0;
	DragonianVoiceSvcFloatTensor RawF0 = 0;
	DragonianVoiceSvcFloatTensor Units = 0;
	DragonianVoiceSvcFloatTensor F0 = 0;
	DragonianVoiceSvcFloatTensor NetOutput = 0;
	DragonianVoiceSvcFloatTensor OutputAudio = 0;

	float* OutputAudioBuffer = 0;
	float* SpecBuffer = 0;
	INT64 AudioLength = 0;
	INT64 SpecLength = 0;
	INT64 i = 0;


	struct { const wchar_t* Key; const wchar_t* Value; } ModelPaths[]
	{
		{ L"Ctrl", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/encoder.onnx)" },
		{ L"Velocity", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/velocity.onnx)" },
		{ 0, 0 },
		{ L"Nsf-HiFi-GAN", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/nsf-hifigan-n.onnx)" },
		{ L"ContentVec-768-l12", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/vec-768-layer-12-f16.onnx)" },
		{ L"Dio", 0 }
	};

	DragonianVoiceSvcInitEnviromentSetting(&EnvSetting);
	EnvSetting.Provider = 1;
	Enviroment = DragonianVoiceSvcCreateEnviroment(&EnvSetting);

	DragonianVoiceSvcInitHyperParameters(&HyperParameters);
	HyperParameters.ModelPaths = (DragonianVoiceSvcArgDict)ModelPaths;
	HyperParameters.ModelType = DragonianVoiceSvcReflowSvc;
	HyperParameters.ProgressCallback = CBFN;
	HyperParameters.F0Max = 800;
	HyperParameters.F0Min = 65;
	HyperParameters.HopSize = 512;
	HyperParameters.MelBins = 128;
	HyperParameters.UnitsDim = 768;
	HyperParameters.OutputSamplingRate = 44100;
	HyperParameters.SpeakerCount = 92;
	HyperParameters.HasSpeakerEmbedding = true;
	HyperParameters.HasVolumeEmbedding = true;
	HyperParameters.HasSpeakerMixLayer = true;
	SvcModel = DragonianVoiceSvcLoadModel(&HyperParameters, Enviroment);


	UnitsEncoder = DragonianVoiceSvcLoadUnitsEncoder(
		ModelPaths[4].Key,
		ModelPaths[4].Value,
		16000,
		768,
		Enviroment
	);


	F0Extractor = DragonianVoiceSvcCreateF0Extractor(
		ModelPaths[5].Key,
		ModelPaths[5].Value,					//Should be set when using fcpe or rmvpe
		0,										//Should be set when using fcpe or rmvpe
		HyperParameters.OutputSamplingRate		//Should be set when using fcpe or rmvpe
	);


	Vocoder = DragonianVoiceSvcLoadVocoder(
		ModelPaths[3].Key,
		ModelPaths[3].Value,
		44100,
		128,
		Enviroment
	);


	//</LoadYourAudioHere>
	auto InputAudio = DragonianLib::AvCodec::OpenInputStream(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal.wav)"
	).DecodeAll(44100, 2)[{"882000:1323000", "0"}].Transpose(-1, -2).Contiguous().Evaluate();
	//<LoadYourAudioHere/>
	Audio = DragonianVoiceSvcCreateFloatTensor(
		InputAudio.Data(),
		1,
		1,
		1,
		InputAudio.Size(1)
	);


	DragonianVoiceSvcInitF0ExtractorParameters(&F0ExtractorParameters);
	F0ExtractorParameters.HopSize = (INT32)HyperParameters.HopSize;
	F0ExtractorParameters.SamplingRate = (INT32)HyperParameters.OutputSamplingRate;
	F0ExtractorParameters.F0Max = HyperParameters.F0Max;
	F0ExtractorParameters.F0Min = HyperParameters.F0Min;
	F0ExtractorParameters.F0Bins = (INT32)HyperParameters.F0Bin;
	RawF0 = DragonianVoiceSvcExtractF0(
		Audio,
		&F0ExtractorParameters,
		F0Extractor
	);


	Units = DragonianVoiceSvcEncodeUnits(
		Audio,
		44100,
		UnitsEncoder
	);


	DragonianVoiceSvcInitInferenceParameters(&InferenceParameters);
	InferenceParameters.SpeakerId = 0;
	InferenceParameters.PitchOffset = 0.f;
	InferenceParameters.StftNoiseScale = 1.f;
	InferenceParameters.NoiseScale = 1.f;
	InferenceParameters.Reflow.Begin = 0.f;
	InferenceParameters.Reflow.End = 1.f;
	InferenceParameters.Reflow.Stride = 0.05f;
	InferenceParameters.F0HasUnVoice = false;
	InferenceParameters.Reflow.Sampler = L"Eular";
	NetOutput = DragonianVoiceSvcInference(
		Audio,
		44100,
		Units,
		RawF0,
		0,
		0,
		&InferenceParameters,
		SvcModel,
		&F0
	);


	SpecBuffer = DragonianVoiceSvcGetTensorData(NetOutput);
	SpecLength = DragonianVoiceSvcGetTensorShape(NetOutput)[2] * DragonianVoiceSvcGetTensorShape(NetOutput)[3];


	//If mel is normalized, denorm mel
	for (i = 0; i < SpecLength; ++i)
		SpecBuffer[i] = (SpecBuffer[i] + 1) / 2 * (HyperParameters.SpecMax - HyperParameters.SpecMin) + HyperParameters.SpecMin;


	OutputAudio = DragonianVoiceSvcInferVocoder(
		NetOutput,
		F0,
		Vocoder
	);


	OutputAudioBuffer = DragonianVoiceSvcGetTensorData(OutputAudio);
	AudioLength = DragonianVoiceSvcGetTensorShape(OutputAudio)[3];


	//</OutputAudioHere>
	DragonianLib::AvCodec::OpenOutputStream(
		44100,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal-SVC.wav)"
	).EncodeAll(
		DragonianLib::TemplateLibrary::Ranges(OutputAudioBuffer, AudioLength + OutputAudioBuffer).RawConst(),
		44100
	);
	//<OutputAudioHere/>


	//Calling "DragonianVoiceSvcUnrefGlobalCache" function means you don't need this model anymore, if you call this function, the model will be unloaded when all instances of this model are unrefed. If Enviroment is destoryed, all models will be unloaded.

	DragonianVoiceSvcDestoryFloatTensor(OutputAudio);
	DragonianVoiceSvcDestoryFloatTensor(F0);
	DragonianVoiceSvcDestoryFloatTensor(NetOutput);
	DragonianVoiceSvcDestoryFloatTensor(Units);
	DragonianVoiceSvcDestoryFloatTensor(RawF0);
	DragonianVoiceSvcDestoryFloatTensor(Audio);

	DragonianVoiceSvcUnrefVocoder(Vocoder);
	DragonianVoiceSvcUnrefGlobalCache(ModelPaths[3].Value, Enviroment);

	DragonianVoiceSvcUnrefF0Extractor(F0Extractor);
	//DragonianVoiceSvcUnrefGlobalCache(ModelPaths[5].Value, Enviroment);     //When using fcpe or rmvpe

	DragonianVoiceSvcUnrefUnitsEncoder(UnitsEncoder);
	DragonianVoiceSvcUnrefGlobalCache(ModelPaths[4].Value, Enviroment);

	DragonianVoiceSvcUnrefModel(SvcModel);
	DragonianVoiceSvcUnrefGlobalCache(ModelPaths[0].Value, Enviroment);
	DragonianVoiceSvcUnrefGlobalCache(ModelPaths[1].Value, Enviroment);


	DragonianVoiceSvcDestoryEnviroment(Enviroment);
#endif

}