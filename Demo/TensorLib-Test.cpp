﻿#include <chrono>
#include <iostream>

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

#include "Libraries/AvCodec/AvCodec.h"
#include "OnnxLibrary/TextToSpeech/Models/Vits.hpp"
#include "OnnxLibrary/Vocoder/Register.hpp"
	
int main()
{
	using namespace DragonianLib;
	OnnxRuntime::Text2Speech::HParams Hparams;
	Hparams.ModelPaths = {
		{ L"Encoder", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_enc_p.onnx)" },
		{ L"Embedding", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_emb.onnx)" },
		{ L"DP", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_dp.onnx)" },
		{ L"SDP", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_sdp.onnx)" },
		{ L"Flow", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_flow.onnx)" },
		{ L"Decoder", LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits2.4PT\BertVits2.4PT_emb.onnx)" }
	};

	Hparams.Parameters = {
		{ L"HasLength", L"false" },
		{ L"HasEmotion", L"false" },
		{ L"HasTone", L"true" },
		{ L"HasLanguage", L"true" },
		{ L"HasBert", L"true" },
		{ L"HasClap", L"true" },
		{ L"HasSpeaker", L"true" },
		{ L"EncoderSpeaker", L"true" },
		{ L"HasVQ", L"false" },
		{ L"SpeakerCount", L"1" },
		{ L"GinChannel", L"256" },
		{ L"VQCodebookSize", L"10" },
		{ L"EmotionDims", L"1024"},
		{ L"BertDims", L"2048"},
		{ L"ClapDims", L"512"},
		{ L"BertCount", L"1"},
		{ L"ZinDims", L"2" }
	};

	auto Env = OnnxRuntime::CreateEnvironment(
		{}
	);

	OnnxRuntime::Text2Speech::Vits::SpeakerEmbedding SpeakerEmbedding(
		Env,
		Hparams
	);

	auto Emb = SpeakerEmbedding.Forward(
		Functional::Ones(IDim(9, 1))
	);

	OnnxRuntime::Text2Speech::Vits::Encoder Encoder(
		Env,
		Hparams
	);

	auto Enc = Encoder.Forward(
		Functional::Ones<Int64>(IDim(1, 20)),
		Functional::Ones<Float32>(IDim(1, 256)),
		Functional::Ones<Float32>(IDim(1, 1024)),
		Functional::Ones<Int64>(IDim(1, 20)),
		Functional::Ones<Int64>(IDim(1, 20)),
		Functional::Ones<Float32>(IDim(1, 1, 20, 2048)),
		Functional::Ones<Float32>(IDim(1, 512))
	);

	OnnxRuntime::Text2Speech::Vits::DurationPredictor DurationPredictor(
		Env,
		Hparams
	);

	auto Dur = DurationPredictor.Forward(
		Enc.X,
		Enc.X_mask,
		Functional::Ones<Float32>(IDim(1, 256)),
		0.8f,
		1.0f,
		114514
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
