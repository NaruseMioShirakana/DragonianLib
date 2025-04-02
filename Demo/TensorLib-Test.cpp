#include <chrono>
#include <iostream>

#include "OnnxLibrary/BertClap/Context.hpp"
#include "OnnxLibrary/TextToSpeech/Models/GPT-SoVits.hpp"
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
	const auto Env = OnnxRuntime::CreateEnvironment({});
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
	AudioData = AudioData[{":", "441000:882000"}];

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

void TestGptSoVits()
{
	using namespace DragonianLib;
	SetWorkerCount(8);
	SetMaxTaskCountPerOperator(4);
	G2P::RegisterG2PModules(LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");
	auto Jap = G2P::New(L"BasicCleaner", LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");

	const auto Env = OnnxRuntime::CreateEnvironment({Device::CPU, 0});
	OnnxRuntime::ContextModel::ContextModel Bert(
		Env,
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Bert.onnx)"
	);
	Dict::Tokenizer Tokenizer(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Vocab.json)",
		L"[CLS]",
		L"[SEP]",
		L"[EOS]",
		L"[UNK]"
	);
	OnnxRuntime::UnitsEncoder::HubertBase CnHubert(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\cnhubert.onnx)",
		Env,
		16000,
		768
	);
	Dict::IdsDict IdsDict(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\symbols1.json)"
	);
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
	OnnxRuntime::Text2Speech::GptSoVits::T2SAR Model(
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
			}
		}
	);
	OnnxRuntime::Text2Speech::GptSoVits::SoVits GSV(
		Env,
		{
			{
				{
					L"Vits",
				   LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\GptSoVitsV1V2.onnx)"
				},
				{
					L"Extract",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Extractor.onnx)"
				}
			},
			32000
		}
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\ttt.wav)"
	);
	auto AudioData = AudioInStream.DecodeAll(
		44100
	).View(1, -1).Interpolate<Operators::InterpolateMode::Linear>(
		IDim(-1),
		IScale(16000. / 44100.)
	).Evaluate();

	G2P::CppPinYinParameters Parameters;
	Parameters.Style = G2P::CppPinYinParameters::TONE3;
	Parameters.NumberStyle = G2P::CppPinYinParameters::SPLITCHINESE;
	Parameters.Heteronym = false;
	Parameters.ReplaceASV = true;


	std::wstring RefText = LR"(テレジアはもう戻らぬ旅路に就いたわ. 残された私は, "文明の存続"の中に存在するプログラムでしかない. アーミヤの思いがあったからこそ, 私はこの姿で現れたの. テレジアとは別人よ. たとえ私にかつての記憶が全てあったとしてもよ. )";
	std::wstring InputText = L"この会議室のことは覚えているわ. この席は, テレジアのために空けられているのかしら? いいえ……私は要らないわ. ";
	bool UV = false;
	auto [RefPhonemes, _] = Jap->Convert(RefText, "Japanese", &UV);
	auto [InputPhonemes, __] = Jap->Convert(InputText, "Japanese", &UV);
	auto RefPhonemeIds = Functional::FromVector(IdsDict(RefPhonemes)).UnSqueeze(0);
	auto InputPhonemeIds = Functional::FromVector(IdsDict(InputPhonemes)).UnSqueeze(0);
	auto PhonemeIds = Functional::Cat(RefPhonemeIds, InputPhonemeIds, 1);

	auto BertFeature = Functional::Zeros(IDim(1, PhonemeIds.Size(1), 1024));
	auto SSL = CnHubert.Forward(AudioData.UnSqueeze(0));
	auto Prompt = GSV.ExtractLatent(SSL.Squeeze(0)).Squeeze(0);

	auto Ret = Model.Forward(PhonemeIds, Prompt, BertFeature, 0.6f, 0.6f, 1.35f);

	auto Res = GSV.Forward(InputPhonemeIds, Ret, AudioData);

	auto AudioOutStream = AvCodec::OpenOutputStream(
		32000,
		LR"(C:\DataSpace\MediaProj\PlayList\Test.wav)"
	);
	AudioOutStream.EncodeAll(Res.GetCRng(), 32000);
}
	
int main()
{
	std::wcout.imbue(std::locale("zh_CN"));
	
	TestGptSoVits();
	
	//SetTaskPoolSize(4);

	

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
