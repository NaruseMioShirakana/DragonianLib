#include <iostream>
#include "AvCodec.h"
#include "Modules.hpp"
#include "NativeApi.h"

int main()
{
#ifdef _WIN32
	if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
		SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
	libsvc::SetupKernel();
	libsvc::Hparams Config;
	Config.TensorExtractor = L"DiffusionSvc";
	Config.SamplingRate = 44100;
	Config.HopSize = 512;
	Config.HubertPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)";
	Config.SpeakerCount = 8;
	Config.HiddenUnitKDims = 768;
	Config.EnableCharaMix = true;
	Config.EnableVolume = true;
	Config.MaxStep = 100;
	Config.MelBins = 128;
	Config.DiffusionSvc.Encoder = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\ComboSummerPockets\ComboSummerPockets_encoder.onnx)";
	Config.DiffusionSvc.Alpha = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\ComboSummerPockets\ComboSummerPockets_alpha.onnx)";
	Config.DiffusionSvc.Denoise = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\ComboSummerPockets\ComboSummerPockets_denoise.onnx)";
	Config.DiffusionSvc.Pred = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\ComboSummerPockets\ComboSummerPockets_pred.onnx)";
	Config.DiffusionSvc.After = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\ComboSummerPockets\ComboSummerPockets_after.onnx)";
	long SrcSr = Config.SamplingRate;

	size_t TotalStep = 0;
	auto ProgressCb = [&](size_t a, size_t)
		{
			printf("%lf%c\n", double(a) / double(TotalStep) * 100., '%');
		};
	auto Model = libsvc::UnionSvcModel(Config, ProgressCb, 2, 8, 0);
	auto Audio = DragonianLib::AvCodec().DecodeSigned16(
		R"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)",
		SrcSr
	);
	libsvc::SlicerSettings SlicerConfig{
		SrcSr,
		40.,
		5.,
		2048,
		512
	};
	const auto SliPos = SliceAudio(Audio, SlicerConfig);
	auto Slices = libsvc::SingingVoiceConversion::GetAudioSlice(Audio, SliPos, SlicerConfig);
	libsvc::SingingVoiceConversion::PreProcessAudio(Slices, SrcSr, 512, L"Dio");
	libsvc::InferenceParams Params;
	LibSvcSetGlobalEnv(8, 0, 2);

	std::wstring VocoderPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf_hifigan.onnx)";
	Params.SrcSamplingRate = SrcSr;
	Params.VocoderSamplingRate = Config.SamplingRate;
	Params.VocoderHopSize = Config.HopSize;
	Params.VocoderMelBins = static_cast<int>(Config.MelBins);
	Params.VocoderModel = LibSvcLoadVocoder(VocoderPath.data());
	if(!Params.VocoderModel)
	{
		auto Str = LibSvcGetError(0);
		std::cout << DragonianLib::WideStringToUTF8(Str);
		LibSvcFreeString(Str);
		return 0;
	}

	size_t Proc = 0;
	DragonianLibSTL::Vector<int16_t> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	Params.Step = 100;
	Params.Pndm = 10;
	TotalStep = Slices.Slices.Size() * 10;
	//std::vector<>;
	for (const auto& Single : Slices.Slices)
	{
		DragonianLib::WritePCMData(
			(LR"(D:/VSGIT/MoeSS - Release/Testdata/Input-PCM-SignedInt-16-)" + std::to_wstring(Proc) + L".wav").c_str(),
			Single.Audio,
			Config.SamplingRate
		);
		const auto SliceResampleLen = Single.OrgLen * 16000ll / SrcSr;
		const auto WavPaddedSize = ((SliceResampleLen / DRAGONIANLIB_PADDING_COUNT) + 1) * DRAGONIANLIB_PADDING_COUNT;
		const auto SliceTime = double(WavPaddedSize) / 16000.;
		auto BeginTime = clock();
		auto Out = Model.SliceInference(Single, Params, Proc);
		auto InferenceTime = double(clock() - BeginTime) / 1000.;
		printf("Time Per Sec: %lf, Rtf: %lf\n", SliceTime / InferenceTime, InferenceTime / SliceTime);
		DragonianLib::WritePCMData(
			(LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-SignedInt-16-)" + std::to_wstring(Proc) + L".wav").c_str(),
			Out,
			Config.SamplingRate
		);
		OutAudio.Insert(OutAudio.end(), Out.begin(), Out.end());
	}

	DragonianLib::WritePCMData(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/Output-PCM-SignedInt-16.wav)",
		OutAudio,
		Config.SamplingRate
	);
	return 0;
}