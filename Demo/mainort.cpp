#include <iostream>
#include "AvCodec.h"
#include "Modules.hpp"
#ifdef _WIN32
#include <Windows.h>
#endif

int main()
{
#ifdef _WIN32
	if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
		SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
	libsvc::SetupKernel();
	libsvc::Hparams Config;
	Config.TensorExtractor = L"RVC";
	Config.SamplingRate = 40000;
	Config.HopSize = 320;
	Config.HubertPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)";
	Config.SpeakerCount = 1;
	Config.HiddenUnitKDims = 768;
	Config.VitsSvc.VitsSvc = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\Models\NaruseMioShirakana\NaruseMioShirakana_RVC.onnx)";
	auto ProgressCb = [](size_t a, size_t b)
		{
			printf("%lf%c\n", double(a) / double(b) * 100., '%');
		};
	auto Model = libsvc::VitsSvc(Config, ProgressCb, libsvc::LibSvcModule::ExecutionProviders::DML, 0, 8);
	auto Audio = DragonianLib::AvCodec().DecodeSigned16(
		R"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)",
		40000
	);
	libsvc::SlicerSettings SlicerConfig{
		40000,
		40.,
		5.,
		2048,
		512
	};
	const auto SliPos = SliceAudio(Audio, SlicerConfig);
	auto Slices = libsvc::SingingVoiceConversion::GetAudioSlice(Audio, SliPos, SlicerConfig);
	libsvc::SingingVoiceConversion::PreProcessAudio(Slices, 40000, 512, L"Dio");
	libsvc::InferenceParams Params;
	Params.SrcSamplingRate = 40000;
	size_t Proc = 0;
	DragonianLibSTL::Vector<int16_t> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	for (const auto& Single : Slices.Slices)
	{
		const auto SliceResampleLen = Single.OrgLen * 16000ll / 40000ll;
		const auto WavPaddedSize = ((SliceResampleLen / DRAGONIANLIB_PADDING_COUNT) + 1) * DRAGONIANLIB_PADDING_COUNT;
		const auto SliceTime = double(WavPaddedSize) / 16000.;
		ProgressCb(Proc++, Slices.Slices.Size());
		auto BeginTime = clock();
		auto Out = Model.SliceInference(Single, Params, Proc);
		auto InferenceTime = double(clock() - BeginTime) / 1000.;
		printf("Time Per Sec: %lf, Rtf: %lf\n", SliceTime / InferenceTime, InferenceTime / SliceTime);
		OutAudio.Insert(OutAudio.end(), Out.begin(), Out.end());
	}

	DragonianLib::WritePCMData(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/testAudioFloat.wav)",
		OutAudio,
		40000
	);
	return 0;
}