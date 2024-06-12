#include <iostream>
#include "AvCodec.h"
#include "Modules.hpp"

int main()
{
	libsvc::SetupKernel();
	libsvc::Hparams Config;
	Config.TensorExtractor = L"RVC";
	Config.SamplingRate = 40000;
	Config.HopSize = 320;
	Config.HubertPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-768-layer-12.onnx)";
	Config.SpeakerCount = 1;
	Config.HiddenUnitKDims = 768;
	Config.VitsSvc.VitsSvc = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\Models\NaruseMioShirakana\NaruseMioShirakana_RVC.onnx)";
	auto ProgressCb = [](size_t a, size_t b) { std::cout << double(a) / double(b) * 100. << "%\n"; };
	auto Model = libsvc::VitsSvc(Config, ProgressCb, libsvc::LibSvcModule::ExecutionProviders::CPU, 0, 8);
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
	for(const auto& sli : Slices.Slices)
	{
		ProgressCb(Proc++, Slices.Slices.Size());
		auto Out = Model.SliceInference(sli, Params, Proc);
		OutAudio.Insert(OutAudio.end(), Out.begin(), Out.end());
		if(Proc == 10)
			break;
	}
	DragonianLib::WritePCMData(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/testAudioFloat.wav)",
		OutAudio,
		40000
	);
	return 0;
}