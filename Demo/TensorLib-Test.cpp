#include <iostream>
#include "Libraries/AvCodec/AvCodec.h"
#include "OnnxLibrary/SingingVoiceConversion/Model/Vits-Svc.hpp"

auto MyLastTime = std::chrono::high_resolution_clock::now();
size_t TotalStep = 0;
void ShowProgressBar(size_t progress) {
	int barWidth = 70;
	float progressRatio = static_cast<float>(progress) / float(TotalStep);
	int pos = static_cast<int>(float(barWidth) * progressRatio);

	std::cout << "\r";
	std::cout.flush();
	auto TimeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - MyLastTime).count();
	MyLastTime = std::chrono::high_resolution_clock::now();
	std::cout << "[Speed: " << 6000.0f / static_cast<float>(TimeUsed) << " it/s] ";
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
	MyLastTime = std::chrono::high_resolution_clock::now();
}

void ProgressCb(size_t a, size_t b)
{
	if (a == 0)
		TotalStep = b;
	ShowProgressBar(a);
}

template <typename Fn>
void WithTimer(const Fn& fn)
{
	auto start = std::chrono::high_resolution_clock::now();
	fn();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Task completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << "\n\n";
}
	
int main()
{
	auto AudioStream = DragonianLib::AvCodec::OpenInputStream(LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)");
	auto Tensor = AudioStream.DecodeAll(44100, 2);
	auto OutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals-n.mp3)"
	);

	auto Tensor1 = Tensor[{"900000:1341000"}].Transpose(-1, -2).Continuous().Evaluate();

	std::cout << static_cast<float>(DragonianLib::Float16(44100.f));

	DragonianLib::OnnxRuntime::SingingVoiceConversion::HParams hParams;
	hParams.UnitsDim = 768;
	hParams.OutputSamplingRate = 44100;
	hParams.HopSize = 512;
	hParams.ModelPaths = {
		{ L"Model", LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\Model.onnx)" }
	};
	hParams.SpeakerCount = 1;
	hParams.HasSpeakerMixLayer = false;
	hParams.HasSpeakerEmbedding = true;
	hParams.HasVolumeEmbedding = false;

	DragonianLib::OnnxRuntime::OnnxEnvironmentOptions Options{
		DragonianLib::Device::DIRECTX,
		1,
		8,
		4,
		ORT_LOGGING_LEVEL_WARNING
	};

	Options.SetCUDAOptions("gpu_mem_limit", std::to_string(1024ll * 1024 * 1024 * 6));

	auto Env = DragonianLib::OnnxRuntime::CreateEnvironment(
		Options
	);

	auto Model = DragonianLib::OnnxRuntime::SingingVoiceConversion::SoftVitsSvcV4(
			Env,
			hParams
		);
	
	DragonianLib::OnnxRuntime::SingingVoiceConversion::Parameters Params;
	Params.PitchOffset = 0.f;
	Params.StftNoiseScale = 0.8f;
	Params.NoiseScale = 0.5f;

	const auto UnitsEncoder = DragonianLib::OnnxRuntime::UnitsEncoder::New(
		L"HubertBase",
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
		1100.0,
		50.0,
		nullptr
	};

	WithTimer(
		[&]
		{
			auto Audio = Model.Inference(
				Params,
				Tensor1[0].View(1, 1, -1),
				44100,
				UnitsEncoder,
				F0Extractor,
				F0Params
			);
		}
	);
	WithTimer(
		[&]
		{
			auto Audio = Model.Inference(
				Params,
				Tensor1[0].View(1, 1, -1),
				44100,
				UnitsEncoder,
				F0Extractor,
				F0Params
			);
		}
	);
	WithTimer(
		[&]
		{
			auto Audio = Model.Inference(
				Params,
				Tensor1[0].View(1, 1, -1),
				44100,
				UnitsEncoder,
				F0Extractor,
				F0Params
			);
		}
	);
	WithTimer(
		[&]
		{
			auto Audio = Model.Inference(
				Params,
				Tensor1[0].View(1, 1, -1),
				44100,
				UnitsEncoder,
				F0Extractor,
				F0Params
			);
		}
	);
	WithTimer(
		[&]
		{
			auto Audio = Model.Inference(
				Params,
				Tensor1[0].View(1, 1, -1),
				44100,
				UnitsEncoder,
				F0Extractor,
				F0Params
			);
			OutStream.EncodeAll(
				DragonianLib::TemplateLibrary::CRanges(Audio.Data(), Audio.Data() + Audio.ElementCount()),
				static_cast<DragonianLib::UInt32>(hParams.OutputSamplingRate),
				1
			);
		}
	);

	return 0;

	/*DragonianLib::SingingVoiceConversion::ReflowSvc Model(
		{
			L"DDSPSvc",
			LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)",
			{},
			{},
			{
				LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\encoder.onnx)",
				LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\velocity.onnx)",
				LR"(D:\VSGIT\VC & TTS Python\DDSP-SVC-6.2\model\after.onnx)",
			},
			{},
			44100,
			512,
			768,
			92,
			true,
			true,
			false,
			128,
			0,
			1000,
			-12,
			2,
			65.f,
			850.f,
			1000.f,
		},
		ProgressCb,
		DragonianLib::SingingVoiceConversion::LibSvcModule::ExecutionProviders::CUDA,
		0,
		8
	);

	DragonianLib::SingingVoiceConversion::InferenceParams Params;
	DragonianLib::SingingVoiceConversion::CrossFadeParams Crossfade;
	Params.VocoderModel = DragonianLib::DragonianLibOrtEnv::RefOrtCachedModel(
		LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)",
		Model.GetDlEnv()
	);
	Params.Step = 10;
	Params.TBegin = 1.0;
	Params.MelFactor = 1.4f;
	try
	{
		Audio = Model.InferenceWithCrossFade(
		   CRanges(Audio),
		   44100,
		   Params,
		   Crossfade,
		   {
			   44100, 512, 256, 2048, 850., 65., nullptr
		   },
		   L"Dio",
		   {},
		   -60.f
	   );
	}
	catch (const std::exception& e)
	{
		std::wcout << e.what() << '\n';
		return 0;
	}*/

	/*
	auto Codec = DragonianLib::AvCodec::AvCodec();

	auto Audio = Codec.DecodeFloat(
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)",
		44100
	);

	auto Env = DragonianVoiceSvcCreateEnv(8, 1, DragonianVoiceSvcDMLEP);
	std::wstring VocoderPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)";
	std::wstring ModelPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\NaruseMioShirakana\NaruseMioShirakana_RVC.onnx)";
	std::wstring F0ModelPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\F0Predictor\RMVPE.onnx)";
	std::wstring VecModelPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)";
	std::wstring ModelType = L"RVC";
	DragonianVoiceSvcHparams Hparams{
			ModelType.data(),
			VecModelPath.data(),
			{},
			{ModelPath.data()},
			{},
			{},
			40000,
			320,
			768,
			1,
			false,
			false,
			false,
			128,
			0,
			1000,
			-12,
			2,
			65.f,
			1100.f,
			1000.f
	};
	auto Model = DragonianVoiceSvcLoadModel(
		&Hparams,
		Env,
		ProgressCb
	);

	DragonianVoiceSvcF0ExtractorSetting F0Setting{
		44100,
		320,
		256,
		1100.0,
		50.0,
		nullptr,
		Env,
		F0ModelPath.data()
	};

	float* OutputAudio = nullptr;
	size_t OutputAudioSize = 0;

	DragonianVoiceSvcParams Params;
	DragonianVoiceSvcInitInferenceParams(&Params);
	Params.VocoderModel = DragonianVoiceSvcLoadVocoder(
		VocoderPath.data(),
		Env
	);
	Params.Keys = -8;

	MyLastTime = std::chrono::high_resolution_clock::now();
	DragonianVoiceSvcInferAudio(
		Model,
		&Params,
		Audio.Data(),
		Audio.Size(),
		44100,
		L"Rmvpe",
		&F0Setting,
		5,
		1,
		-60.,
		&OutputAudio,
		&OutputAudioSize
	);

	DragonianLib::Byte* OutputAudioBytes = reinterpret_cast<DragonianLib::Byte*>(OutputAudio);
	DragonianLib::Byte* OutputAudioEnd = OutputAudioBytes + OutputAudioSize * sizeof(float);
	Codec.Encode(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-aaaa.mp3)",
		{ OutputAudioBytes, OutputAudioEnd },
		44100
	);

	DragonianVoiceSvcFreeData(OutputAudio);
	DragonianVoiceSvcDestoryEnv(Env);
	 */

	/*TextToSpeech::Llama LLAMAModel({ 666, 777, 888, 999 });

	auto VQ = TemplateLibrary::Vector<TextToSpeech::Llama::RefPrompt>{ {TemplateLibrary::Arange(8 * 4ll, 8 * 8ll), { 123,456,789 }} };

	auto Tokens = LLAMAModel.Inference({ 114, 514, 1919, 810 }, VQ);

	auto Tensor = Functional::FromBuffer(Tokens.Data.Data(), Tokens.Data.Size()).EvalMove();
	auto TTensor = Tensor.View(Tokens.Shape);
	std::cout << TTensor;
	return 0;

	SingingVoiceConversion::VitsSvc Model{
		{
			L"RVC",
			LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)",
			{},
			{LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\NaruseMioShirakana\NaruseMioShirakana_RVC.onnx)"},
			{},
			{},
			40000,
			320,
			768,
			1,
			false,
			false,
			false,
			128,
			0,
			1000,
			-12,
			2,
			65.f
		},
		ProgressCb,
		SingingVoiceConversion::LibSvcModule::ExecutionProviders::CPU,
		0,
		8
	};

	auto Audio = DragonianLib::AvCodec::AvCodec().DecodeFloat(
		R"(C:\DataSpace\MediaProj\PlayList\Echoism_vocals.wav)",
		44100
	);

	AvCodec::SlicerSettings SlicerConfig{
		44100,
		-60.,
		2.,
		4410*2,
		4410
	};

	std::wstring VocoderPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)";
	_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space InferenceParams Params;
	Params.VocoderSamplingRate = 44100;
	Params.VocoderHopSize = 512;
	Params.VocoderMelBins = 128;
	Params.VocoderModel = DragonianLib::RefOrtCachedModel(
		VocoderPath,
		Model.GetDlEnv()
	);
	Params.Keys = -8;
	//const auto SliPos = TemplateLibrary::Arange(0ull, Audio.Size(), 441000ull);
	const auto SliPos = AvCodec::SliceAudio(Audio, SlicerConfig);
	auto Slices = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::GetAudioSlice(Audio, SliPos, 44100, -60.);
	_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::PreProcessAudio(Slices, { 44100, 512 }, L"Rmvpe", {LR"(D:\VSGIT\MoeVS-SVC\Build\Release\F0Predictor\RMVPE.onnx)", &Model.GetDlEnvPtr()});
	size_t Proc = 0;
	Params.Keys = -12.f;
	DragonianLibSTL::Vector<float> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	TotalStep = Slices.Slices.Size() * Params.Step;
 	TemplateLibrary::Vector<float> Output;
	for (auto& Single : Slices.Slices)
	{
		Single.F0 = SingingVoiceConversion::TensorExtractor::LibSvcTensorExtractor::GetInterpedF0(Single.F0);
		try
		{
			MyLastTime = std::chrono::high_resolution_clock::now();
			auto Now = std::chrono::high_resolution_clock::now();
			auto Out = Model.SliceInference(Single, Params, Proc);
			std::cout << "Rtf: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - Now).count()) / 10000.0 << '\n';
			DragonianLib::AvCodec::WritePCMData(
				(LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-)" + std::to_wstring(Proc) + L".wav").c_str(),
				Out,
				40000
			);
			Proc++;
			Output.Insert(Output.End(), Out.Begin(), Out.End());
		}
		catch (const std::exception& e)
		{
			std::wcout << e.what() << '\n';
		}
	}
	DragonianLib::AvCodec::WritePCMData(
		(LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-aaaa.wav)"),
		Output,
		40000
	);*/
}
