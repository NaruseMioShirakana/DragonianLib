#include <chrono>
#include <iostream>

#include "Libraries/AvCodec/AvCodec.h"

#include "TensorLib/Include/Base/Tensor/Einops.h"

static auto MyLastTime = std::chrono::high_resolution_clock::now();
static int64_t TotalStep = 0;

template <typename Fn>
[[maybe_unused]] static void WithTimer(const Fn& fn)
{
	auto start = std::chrono::high_resolution_clock::now();
	fn();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Task completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << "\n\n";
}

[[maybe_unused]] static void ProgressCb(bool a, int64_t b)
{
	if (a)
		TotalStep = b;
	else
		ShowProgressBar(b);
}

[[maybe_unused]] static auto TestG2PW(const std::wstring& Text)
{
	using namespace DragonianLib;
	const auto Env = OnnxRuntime::CreateEnvironment({ Device::CPU, 0 });
	G2P::CppPinYinConfigs Configs{
		LR"(C:\DataSpace\libsvc\PythonScript\pypinyin_dict.json)",
		LR"(C:\DataSpace\libsvc\PythonScript\pypinyin_pinyin_dict.json)",
		LR"(C:\DataSpace\libsvc\PythonScript\bopomofo_to_pinyin_wo_tune_dict.json)",
		LR"(C:\DataSpace\libsvc\PythonScript\char_bopomofo_dict.json)"
	};
	G2P::G2PWModelHParams gParams{
		&Configs,
		LR"(C:\DataSpace\libsvc\PythonScript\POLYPHONIC_CHARS.txt)",
		LR"(C:\DataSpace\libsvc\PythonScript\tokens.txt)",
		LR"(C:\DataSpace\libsvc\PythonScript\G2PW.onnx)",
		&Env,
		&_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger(),
		512
	};
	G2P::G2PWModel G2PW(
		&gParams
	);

	G2P::CppPinYinParameters Parameters;
	Parameters.Style = G2P::CppPinYinParameters::TONE3;
	Parameters.NumberStyle = G2P::CppPinYinParameters::SPLITCHINESE;
	Parameters.Heteronym = false;
	Parameters.ReplaceASV = true;

	auto [PinYin, Tones] = G2PW.Convert(
		Text,
		"",
		&Parameters
	);

	auto [PinYinNew, Phoneme2Word] = decltype(G2PW)::SplitYunmu(PinYin);

	return PinYinNew;
}

[[maybe_unused]] static void TestGptSoVits()
{
	using namespace DragonianLib;
	G2P::RegisterG2PModules(LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");
	auto Jap = G2P::New(L"BasicCleaner", LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");

	const auto Env = OnnxRuntime::CreateEnvironment({ Device::CPU, 0 });
	Env->SetExecutionMode(ORT_PARALLEL);
	Env->EnableMemPattern(false);

	/*OnnxRuntime::ContextModel::ContextModel Bert(
		Env,
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Bert.onnx)"
	);
	Dict::Tokenizer Tokenizer(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Vocab.json)",
		L"[CLS]",
		L"[SEP]",
		L"[EOS]",
		L"[UNK]"
	);*/
	OnnxRuntime::UnitsEncoder::HubertBase CnHubert(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\cnhubert.onnx)",
		Env,
		16000,
		768
	);

	Dict::IdsDict Text2Seq(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\symbols1.json)"
	);

	OnnxRuntime::Vocoder::VocoderBase BigVGan = OnnxRuntime::Vocoder::VocoderBase(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\BigVGAN.onnx)",
		Env,
		24000,
		100
	);

	OnnxRuntime::Text2Speech::GptSoVits::T2SAR ARModel(
		Env,
		{
			{
				{
					L"Prompt",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\PromptProcessor.onnx)"
				},
			   {
					L"Decode",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\DecodeNextToken.onnx)"
			   }
			},
			0,
			{}
		}
	);

	OnnxRuntime::Text2Speech::GptSoVits::VQModel GSV(
		Env,
		{
			{
				{
					L"Vits",
				   LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\GptSoVits.onnx)"
				},
				{
					L"Extract",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Extractor.onnx)"
				},
				{
					L"Cfm",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\GptSoVits_cfm.onnx)"
				}
			},
			32000,
			{}
		}
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\ttt.wav)"
	);
	auto SrcAudio = AudioInStream.DecodeAll(
		44100
	).View(1, -1);

	std::wstring RefText = LR"(テレジアはもう戻らぬ旅路に就いたわ. 残された私は, "文明の存続"の中に存在するプログラムでしかない. アーミヤの思いがあったからこそ, 私はこの姿で現れたの. テレジアとは別人よ. たとえ私にかつての記憶が全てあったとしてもよ. )";
	std::wstring InputText = L"この会議室のことは覚えているわ. この席は, テレジアのために空けられているのかしら? いいえ……私は要らないわ. ";
	bool UV = false;

	auto [RefPhonemes, _] = Jap->Convert(
		RefText, "Japanese", &UV
	);
	auto [InputPhonemes, __] = Jap->Convert(
		InputText, "Japanese", &UV
	);

	auto RefPhonemeIds = Functional::FromVector(
		Text2Seq(RefPhonemes)
	).UnSqueeze(0);
	auto InputPhonemeIds = Functional::FromVector(
		Text2Seq(InputPhonemes)
	).UnSqueeze(0);

	auto SSL = CnHubert.Forward(
		SrcAudio.UnSqueeze(0), 44100
	);

	auto PhonemeIds = Functional::Cat(
		RefPhonemeIds, InputPhonemeIds, 1
	);
	auto BertFeature = Functional::Zeros(
		IDim(1, PhonemeIds.Size(1), 1024)
	);
	auto Prompt = GSV.ExtractLatent(
		SSL.Squeeze(0)
	).Squeeze(0);

	auto Ret = ARModel.Forward(
		PhonemeIds,
		Prompt,
		BertFeature,
		0.6f,
		0.6f,
		1.35f
	);

	auto Res = GSV.Forward(
		InputPhonemeIds,
		Ret,
		SrcAudio,
		44100,
		RefPhonemeIds,
		Prompt
	);

	auto AudioOutStream = AvCodec::OpenOutputStream(
		32000,
		LR"(C:\DataSpace\MediaProj\PlayList\Test.wav)"
	);
	AudioOutStream.EncodeAll(Res.GetCRng(), 32000);
}

[[maybe_unused]] static void TestStft()
{
	using namespace DragonianLib;

	FunctionTransform::StftKernel Stft(
		2048, 512, 2048
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism.wav)"
	);
	auto [SrcAudio, SamplingRate] = AudioInStream.DecodeAudio(
		2, true
	);

	auto Spec = Stft.Execute(SrcAudio.UnSqueeze(0));

	auto Signal = Functional::MinMaxNormalize(Stft.Inverse(Spec), -1).Evaluate();

	auto AudioOutStream = AvCodec::OpenOutputStream(
		SamplingRate,
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism-ISTFT.wav)"
	);
	AudioOutStream.EncodeAll(
		Signal.GetCRng(), SamplingRate, 2, true
	);
}

[[maybe_unused]] static void TestSvc()
{
	auto OutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals-n.mp3)"
	);

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
	hParams.ProgressCallback = ProgressCb;
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
	hParams.ModelPaths[L"Ctrl"] = LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\encoder.onnx)";
	hParams.ModelPaths[L"Velocity"] = LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\velocity.onnx)";
	//hParams.ModelPaths[L"Ctrl"] = LR"(D:\VSGIT\MoeVoiceStudio\Diffusion-SVC-2.0_dev\checkpoints\d-hifigan\d-hifigan_encoder.onnx)";
	//hParams.ModelPaths[L"Velocity"] = LR"(D:\VSGIT\MoeVoiceStudio\Diffusion-SVC-2.0_dev\checkpoints\d-hifigan\d-hifigan_velocity.onnx)";

	auto Model = DragonianLib::OnnxRuntime::SingingVoiceConversion::ReflowSvc(
		Env,
		hParams
	);

	auto Vocoder = DragonianLib::OnnxRuntime::Vocoder::New(
		L"Nsf-HiFi-GAN",
		LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)",
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
		LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\vec-768-layer-12-f16.onnx)",
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
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)"
	);
	auto Audio = AudioStream.DecodeAll(44100, 2);
	auto Input = Audio[{"900000:944100", "0"}].Transpose(-1, -2).Contiguous();

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

	WithTimer(
		[&]
		{
			Mel = Model.Forward(
				Params,
				MyData
			);
		}
	);
	WithTimer(
		[&]
		{
			Mel = Model.Forward(
				Params,
				MyData
			);
		}
	);
	WithTimer(
		[&]
		{
			Mel = Model.Forward(
				Params,
				MyData
			);
		}
	);
	WithTimer(
		[&]
		{
			Mel = Model.Forward(
				Params,
				MyData
			);
		}
	);
	WithTimer(
		[&]
		{
			Mel = Model.Forward(
				Params,
				MyData
			);
		}
	);

	Mel = Model.DenormSpec(Mel);
	auto Output = Vocoder->Forward(Mel, MyData.F0);

	auto AudioOutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		LR"(C:\DataSpace\MediaProj\PlayList\Test-IStft.wav)"
	);
	AudioOutStream.EncodeAll(
		Output.GetCRng(), 44100
	);
}

[[maybe_unused]] static void TestVocoder()
{
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
	auto Vocoder = DragonianLib::OnnxRuntime::Vocoder::New(
		L"Nsf-HiFi-GAN",
		LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)",
		Env
	);
	auto MFCCKernel = DragonianLib::FunctionTransform::MFCCKernel(
		44100, 2048, 512, 2048, 128
	);
	auto F0Extractor = DragonianLib::F0Extractor::New(
		L"Dio",
		nullptr
	);

	auto AudioStream = DragonianLib::AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)"
	);
	auto Audio = AudioStream.DecodeAll(44100, 2);
	auto Input = Audio[{"900000:1341000", "0"}].Transpose(-1, -2);

	auto F0 = F0Extractor->ExtractF0(Input, {44100, 512, 256, 2048, 800, 65, 0.03f}).UnSqueeze(0);
	auto Mel = MFCCKernel(Input.UnSqueeze(0));
	DragonianLib::Functional::NumpySave(L"C:/DataSpace/MediaProj/PlayList/Echoism_mel.npy", Mel);
	DragonianLib::Functional::NumpySave(L"C:/DataSpace/MediaProj/PlayList/Echoism.npy", Input.Contiguous().Evaluate());
	Mel = DragonianLib::Functional::NumpyLoad<float, 4>(L"C:/DataSpace/MediaProj/PlayList/Echoism_mel2.npy");
	F0 = F0[{":", ":", ":-1"}];
	Audio = Vocoder->Forward(
		Mel,
		F0
	).Squeeze(0);

	auto OutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals-test.mp3)"
	);
	OutStream.EncodeAll(
		Audio.GetCRng(), 44100, 1, true
	);
}

[[maybe_unused]] static void TestApi()
{
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
		{ L"Ctrl", LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\encoder.onnx)" },
		{ L"Velocity", LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\velocity.onnx)" },
		{ 0, 0 },
		{ L"Nsf-HiFi-GAN", LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)" },
		{L"ContentVec-768-l12", LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\vec-768-layer-12-f16.onnx)" },
		{ L"Dio", 0 }
	};

	DragonianVoiceSvcInitEnviromentSetting(&EnvSetting);
	EnvSetting.Provider = 1;
	Enviroment = DragonianVoiceSvcCreateEnviroment(&EnvSetting);

	DragonianVoiceSvcInitHyperParameters(&HyperParameters);
	HyperParameters.ModelPaths = (DragonianVoiceSvcArgDict)ModelPaths;
	HyperParameters.ModelType = DragonianVoiceSvcReflowSvc;
	HyperParameters.ProgressCallback = ProgressCb;
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
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)"
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
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals-n.mp3)"
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
}

[[maybe_unused]] static void TestSR()
{
	using namespace DragonianLib;
	auto Image = ImageVideo::LoadAndSplitImageNorm(
		LR"(C:\DataSpace\MediaProj\Wallpaper\T\1747414157401.png)",
		256,
		256,
		248,
		248
	);

	std::get<0>(Image) + std::get<0>(Image);

	OnnxRuntime::SuperResolution::HyperParameters Parameters;
	Parameters.RGBModel = LR"(D:\VSGIT\白叶的AI工具箱\Models\real-hatgan\x2\x2_universal-fix1.onnx)";
	Parameters.Callback = ProgressCb;

	auto Env = OnnxRuntime::CreateOnnxRuntimeEnvironment({
			Device::CUDA,
		});
	OnnxRuntime::SuperResolution::SuperResolutionBCRGBHW Model(
		Env,
		Parameters
	);

	//Image = Model.Infer(Image, 1);

	////TestApi();
	//ImageVideo::SaveBitmap(
	//	ImageVideo::CombineImage(Image, 248ll * 2, 248ll * 2),
	//	LR"(C:\DataSpace\MediaProj\Wallpaper\T\Theresa.png)"
	//);
}

int main()
{
	using namespace DragonianLib;
	std::wcout.imbue(std::locale("zh_CN"));
	SetWorkerCount(16);
	SetMaxTaskCountPerOperator(4);
	SetTaskPoolSize(4);

	//TestSR();

	//TestStft();
	return 0;
}
