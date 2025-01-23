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
	int	a = 0;
	int	b = 0;
	int c = 0;
};

template <typename Fn>
void WithTimer(const Fn& fn)
{
	auto start = std::chrono::high_resolution_clock::now();
	fn();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

#include "OnnxLibrary/SingingVoiceConversion/Modules/Header/Models/ReflowSvc.hpp"
#include "Libraries/AvCodec/AvCodec.h"

int main()
{
	using namespace DragonianLib;

	SingingVoiceConversion::ReflowSvc Model{
		{
			L"ReflowSvc",
			LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)",
			{},
			{},
			{
				LR"(D:\VSGIT\MoeVoiceStudio\Diffusion-SVC-2.0_dev\checkpoints\d-hifigan\d-hifigan_encoder.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio\Diffusion-SVC-2.0_dev\checkpoints\d-hifigan\d-hifigan_velocity.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio\Diffusion-SVC-2.0_dev\checkpoints\d-hifigan\d-hifigan_after.onnx)",
			},
			{},
			44100,
			512,
			768,
			282,
			true,
			true,
			false,
			128,
			0,
			1000,
			-12,
			2,
			65.f
		},
		ProgressCb,
		SingingVoiceConversion::LibSvcModule::ExecutionProviders::DML,
		1,
		8
	};

	auto Audio = DragonianLib::AvCodec::AvCodec().DecodeFloat(
		R"(D:\VSGIT\MoeSS - Release\Testdata\testaudio114.wav)",
		44100
	);

	AvCodec::SlicerSettings SlicerConfig{
		44100,
		-30.,
		3.,
		1024,
		256
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
	const auto SliPos = TemplateLibrary::Arange(0ull, Audio.Size(), 441000ull);
	//const auto PosSlice = AvCodec::SliceAudio(Audio, SlicerConfig);
	auto Slices = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::GetAudioSlice(Audio, SliPos, 44100, -30.);
	_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::PreProcessAudio(Slices, { 44100, 512 }, L"Rmvpe", {LR"(D:\VSGIT\MoeVS-SVC\Build\Release\F0Predictor\RMVPE.onnx)", &Model.GetDlEnvPtr()});
	size_t Proc = 0;
	Params.Step = 10;
	Params.TBegin = 1.f;
	DragonianLibSTL::Vector<float> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	TotalStep = Slices.Slices.Size() * Params.Step;
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
				(LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-)" + std::to_wstring(Proc / Params.Step) + L".wav").c_str(),
				Out,
				44100
			);
		}
		catch (const std::exception& e)
		{
			std::wcout << e.what() << '\n';
		}
	}
}
