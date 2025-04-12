#include <chrono>
#include <iostream>

#include "OnnxLibrary/BertClap/Context.hpp"
#include "OnnxLibrary/TextToSpeech/Models/GPT-SoVits.hpp"
#include "OnnxLibrary/UnitsEncoder/Register.hpp"
#include "OnnxLibrary/Vocoder/Register.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/DDSP-Svc.hpp"
#include "Libraries/G2P/G2PModule.hpp"

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
#include "Libraries/F0Extractor/DioF0Extractor.hpp"
#include "OnnxLibrary/Vocoder/Nsf-Hifigan.hpp"
#include "OnnxLibrary/G2P/G2PW.hpp"
#include "OnnxLibrary/UVR/UVR.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/Reflow-Svc.hpp"

void TestUVR()
{
	using namespace DragonianLib;
	const auto Env = OnnxRuntime::CreateEnvironment({Device::CUDA});
	OnnxRuntime::UltimateVocalRemover::CascadedNet Net(
		LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\UVR\HP5_only_main_vocal.onnx)",
		Env,
		OnnxRuntime::UltimateVocalRemover::CascadedNet::GetPreDefinedHParams(L"4band_v2")
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\Echoism.wav)"
	);
	auto AudioData = AudioInStream.DecodeAll(
		44100, 2, true
	);
	//AudioData = AudioData[{":", "441000:882000"}];
	//auto AudioDatas = AudioData.Split(441000, -1);

	auto [Vocal, Instrument] =
		Net.Forward(AudioData, 85, 0.5f, 44100);

	auto OutputStream = AvCodec::OpenOutputStream(
		44100, LR"(C:\DataSpace\MediaProj\PlayList\Echoism-Vocal.wav)"
	);
	OutputStream.EncodeAll(Vocal.GetCRng(), 44100, 2, true);
	OutputStream = AvCodec::OpenOutputStream(
		44100, LR"(C:\DataSpace\MediaProj\PlayList\Echoism-Instrument.wav)"
	);
	OutputStream.EncodeAll(Instrument.GetCRng(), 44100, 2, true);
}

auto TestG2PW(const std::wstring& Text)
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

void TestGptSoVits()
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

void TestStft()
{
	using namespace DragonianLib;

	FunctionTransform::StftKernel Stft(
		2048, 512, 2048, true
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\ttt.wav)"
	);
	auto SrcAudio = AudioInStream.DecodeAll(
		32000
	).View(1, 1, -1);
	SrcAudio = SrcAudio.Interpolate<Operators::InterpolateMode::Linear>(
		IDim(-1),
		IScale(44100.0f / 32000.0f)
	);

	auto Spec = Stft.Execute(SrcAudio);

	auto Signal = Stft.Inverse(Spec);

	auto AudioOutStream = AvCodec::OpenOutputStream(
		32000,
		LR"(C:\DataSpace\MediaProj\PlayList\Test-IStft.wav)"
	);
	AudioOutStream.EncodeAll(
		Signal.GetCRng(), 44100
	);
}

void TestSvc()
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
	Params.Reflow.End = 0.99f;
	Params.Reflow.Stride = 0.1f;
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
	auto Input = Audio[{"900000:1341000", "0"}].Transpose(-1, -2).Contiguous();

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
		LR"(C:\DataSpace\MediaProj\PlayList\Test-IStft.wav)"
	);
	AudioOutStream.EncodeAll(
		Output.GetCRng(), 44100
	);
}

void TestVocoder()
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

int main()
{
	using namespace DragonianLib;
	std::wcout.imbue(std::locale("zh_CN"));
	SetWorkerCount(8);
	SetMaxTaskCountPerOperator(4);
	SetTaskPoolSize(4);

	TestSvc();

}
