#include "TensorLib/Include/Base/Tensor/Functional.h"
#include "TensorLib/Include/Base/Module/Convolution.h"
#include "TensorLib/Include/Base/Module/Embedding.h"
#include "TensorLib/Include/Base/Module/Linear.h"
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
	std::cout << "[Speed: " << 1000.0f / static_cast<float>(TimeUsed) << " it/s] ";
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
}

void ProgressCb(size_t a, size_t b)
{
	if (a == 0)
		TotalStep = b;
	ShowProgressBar(a);
}

class MyModule : public DragonianLib::Graph::Module
{
public:
	MyModule() : Module(nullptr, L"MyModule"),
		DragonianLibRegisterLayer(_List),
		DragonianLibRegisterLayer(_Seq)
	{
		using emb = DragonianLib::Graph::Embedding<float, DragonianLib::Device::CPU>;
		using linear = DragonianLib::Graph::Linear<float, DragonianLib::Device::CPU>;
		using conv1d = DragonianLib::Graph::Conv1D<float, DragonianLib::Device::CPU>;
		_List.Append(
			DragonianLibLayerItem(
				emb,
				DragonianLib::Graph::EmbeddingParam{ 1919, 810 }
			)
		);
		_List.Append(
			DragonianLibLayerItem(
				linear,
				DragonianLib::Graph::LinearParam{ 514, 114 }
			)
		);
		_List.Append(
			DragonianLibLayerItem(
				conv1d,
				DragonianLib::Graph::Conv1DParam{ 114, 514, 9 }
			)
		);
	}
private:
	DragonianLib::Graph::ModuleList _List;
	DragonianLib::Graph::Sequential _Seq;
};

struct Integer
{
	int i = 0;
	operator std::string() const { return std::to_string(i); }
};

template <typename T>
std::enable_if_t<std::is_same_v<T, Integer>, std::string> DragonianLibCvtToString(const T& t)
{
	return std::to_string(t.i);
}

struct my_struct
{
	int ccc = 0;
};

template <typename Fn>
void WithTimer(const Fn& fn)
{
	auto start = std::chrono::high_resolution_clock::now();
	fn();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

#include "OnnxLibrary/SingingVoiceConversion/Modules/Header/Models/VitsSvc.hpp"
#include "OnnxLibrary/TextToSpeech/Modules/Models/Header/FishSpeech.hpp"
#include "Libraries/AvCodec/AvCodec.h"

struct aaa
{
	int a;
	
};

int main()
{
	using namespace DragonianLib;
	auto Tmp = Functional::Arange(0, 100, 1);
	auto tensor = Tmp.View(5, 20);
	for (auto [i, iter] : Enumrate(tensor))
		for (auto [j, value] : Enumrate(iter))
			std::cout << i << ' ' << j << ' ' << value << '\n';

	return 0;

	TextToSpeech::Llama LLAMAModel({ 666, 777, 888, 999 });

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
	);
}
