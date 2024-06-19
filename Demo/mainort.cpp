#include <iostream>
#include <tchar.h>
#include "AvCodec.h"
#include "Modules.hpp"
#include "NativeApi.h"
#include "MusicTranscription/PianoTranscription.hpp"
#include "SuperResolution/SuperResolution.hpp"

size_t TotalStep = 0;
void ProgressCb(size_t a, size_t)
{
	printf("%lf%c\n", double(a) / double(TotalStep) * 100., '%');
}

void ProgressCbS(size_t a, size_t b)
{
	printf("%lf%c\n", double(a) / double(b) * 100., '%');
}

std::string WideStringToUTF8(const std::wstring& input)
{
#ifdef _WIN32
	std::vector<char> ByteString(input.length() * 6);
	WideCharToMultiByte(
		CP_UTF8,
		0,
		input.c_str(),
		int(input.length()),
		ByteString.data(),
		int(ByteString.size()),
		nullptr,
		nullptr
	);
	return ByteString.data();
#else
	//TODO
#endif
}

void LibSvcTest();

void LibSrTest()
{
	libsr::RealESRGan Model(
		{
			LR"(D:\VSGIT\白叶的AI工具箱\Models\RealESRGAN_x4plus\model.onnx)",
			LR"(D:\VSGIT\白叶的AI工具箱\Models\RealESRGAN_x4plus\model_alpha.onnx)",
			64,
			64,
			4
		},
		ProgressCbS,
		8,
		0,
		2
	);

	DragonianLib::GdiInit();

	DragonianLib::ImageSlicer Image(
		LR"(D:\VSGIT\CG000002.BMP)",
		64,
		64, 
		16, 
		0.f, 
		false
	);

	Model.Infer(Image, 50);

	Image.MergeWrite(LR"(D:\VSGIT\CG000002-N.png)", 4, 100);
}

void LibMtsTest()
{
	libmts::PianoTranScription Model(
		{
			LR"(D:\VSGIT\libsvc\model.onnx)"
		},
		ProgressCbS,
		8,
		0,
		0
	);

	libmts::PianoTranScription::Hparams _Config;
	auto Audio = DragonianLib::AvCodec().DecodeFloat(
		R"(C:\DataSpace\MediaProj\Fl Proj\Childish White.mp3)",
		16000
	);

	auto Midi = Model.Infer(Audio, _Config, 1);
	WriteMidiFile(LR"(C:\DataSpace\MediaProj\Fl Proj\Childish White Mts.mid)", Midi, 0, 384 * 2);
}

int main()
{
#ifdef _WIN32
	if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
		SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
	//LibMtsTest();
	LibSrTest();
	system("pause");
	return 0;
}

void LibSvcTest()
{
#ifdef DRAGONIANLIB_IMPORT
	LibSvcInit();
#else
	libsvc::SetupKernel();
#endif
	constexpr auto EProvider = 1;
	constexpr auto NumThread = 8;
	constexpr auto DeviceId = 0;

	if (LibSvcSetGlobalEnv(NumThread, DeviceId, EProvider))
	{
		auto ErrorMessage = LibSvcGetError(0);
		std::cout << WideStringToUTF8(ErrorMessage);
		LibSvcFreeString(ErrorMessage);
		return;
	}
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
#include <corecrt.h>
#ifdef DRAGONIANLIB_IMPORT
	LibSvcHparams DynConfig{
		Config.TensorExtractor.data(),
		Config.HubertPath.data(),
		{
			Config.DiffusionSvc.Encoder.data(),
			Config.DiffusionSvc.Denoise.data(),
			Config.DiffusionSvc.Pred.data(),
			Config.DiffusionSvc.After.data(),
			Config.DiffusionSvc.Alpha.data(),
			Config.DiffusionSvc.Naive.data(),
			Config.DiffusionSvc.DiffSvc.data()
		},
		{
			Config.VitsSvc.VitsSvc.data()
		},
		{
			Config.ReflowSvc.Encoder.data(),
			Config.ReflowSvc.VelocityFn.data(),
			Config.ReflowSvc.After.data()
		},
		{
			Config.Cluster.ClusterCenterSize,
			Config.Cluster.Path.data(),
			Config.Cluster.Type.data()
		},
		Config.SamplingRate,
		Config.HopSize,
		Config.HiddenUnitKDims,
		Config.SpeakerCount,
		Config.EnableCharaMix,
		Config.EnableVolume,
		Config.VaeMode,
		Config.MelBins,
		Config.Pndms,
		Config.MaxStep,
		Config.SpecMin,
		Config.SpecMax,
		Config.Scale
	};
#endif

#ifdef DRAGONIANLIB_IMPORT
	auto Model = LibSvcLoadModel(
		1,
		&DynConfig,
		ProgressCb,
		EProvider,
		DeviceId,
		NumThread
	);
	if (!Model)
	{
		auto ErrorMessage = LibSvcGetError(0);
		std::cout << WideStringToUTF8(ErrorMessage);
		LibSvcFreeString(ErrorMessage);
		return 0;
	}

	wchar_t AudioInputPath[] = LR"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)";
	auto Audio = LibSvcAllocateAudio();
	auto Error = LibSvcReadAudio(
		AudioInputPath,
		SrcSr,
		Audio
	);
	if (Error)
	{
		auto ErrorMessage = LibSvcGetError(0);
		std::cout << WideStringToUTF8(ErrorMessage);
		LibSvcFreeString(ErrorMessage);
		return 0;
	}
#else
	auto Model = libsvc::UnionSvcModel(Config, ProgressCb, EProvider, NumThread, DeviceId);
	auto Audio = DragonianLib::AvCodec().DecodeSigned16(
		R"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)",
		SrcSr
	);
#endif

	libsvc::SlicerSettings SlicerConfig{
		SrcSr,
		40.,
		5.,
		2048,
		512
	};

#ifdef DRAGONIANLIB_IMPORT
	LibSvcSlicerSettings SlicerConf{
		SlicerConfig.SamplingRate,
		SlicerConfig.Threshold,
		SlicerConfig.MinLength,
		SlicerConfig.WindowLength,
		SlicerConfig.HopSize
	};
#endif

	std::wstring VocoderPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf_hifigan.onnx)";
	libsvc::InferenceParams Params;
	Params.SrcSamplingRate = SrcSr;
	Params.VocoderSamplingRate = Config.SamplingRate;
	Params.VocoderHopSize = Config.HopSize;
	Params.VocoderMelBins = static_cast<int>(Config.MelBins);
	Params.VocoderModel = LibSvcLoadVocoder(VocoderPath.data());
	if (!Params.VocoderModel)
	{
		auto Str = LibSvcGetError(0);
		std::cout << WideStringToUTF8(Str);
		LibSvcFreeString(Str);
		return;
	}

#ifdef DRAGONIANLIB_IMPORT
	const auto SliPos = LibSvcAllocateOffset();
	Error = LibSvcSliceAudio(
		Audio,
		&SlicerConf,
		SliPos
	);
	if (Error)
	{
		auto ErrorMessage = LibSvcGetError(0);
		std::cout << WideStringToUTF8(ErrorMessage);
		LibSvcFreeString(ErrorMessage);
		return 0;
	}

	wchar_t F0MethodS[] = L"Dio";
	auto Slices = LibSvcAllocateSliceData();
	Error = LibSvcPreprocess(
		Audio,
		SliPos,
		SrcSr,
		512,
		SlicerConfig.Threshold,
		F0MethodS,
		Slices
	);
	if (Error)
	{
		auto ErrorMessage = LibSvcGetError(0);
		std::cout << WideStringToUTF8(ErrorMessage);
		LibSvcFreeString(ErrorMessage);
		return 0;
	}
#else
	const auto SliPos = SliceAudio(Audio, SlicerConfig);
	auto Slices = libsvc::SingingVoiceConversion::GetAudioSlice(Audio, SliPos, SlicerConfig);
	libsvc::SingingVoiceConversion::PreProcessAudio(Slices, SrcSr, 512, L"Dio");
#endif

	size_t Proc = 0;
	Params.Step = 100;
	Params.Pndm = 10;
#ifdef DRAGONIANLIB_IMPORT
	auto OutAudio = LibSvcAllocateAudio();
	TotalStep = LibSvcGetSliceCount(Slices) * 10;
#else
	DragonianLibSTL::Vector<int16_t> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	TotalStep = Slices.Slices.Size() * 10;
#endif

#ifdef DRAGONIANLIB_IMPORT
	LibSvcParams DynParams{
		Params.NoiseScale,
		Params.Seed,
		Params.SpeakerId,
		Params.SrcSamplingRate,
		Params.SpkCount,
		Params.IndexRate,
		Params.ClusterRate,
		Params.DDSPNoiseScale,
		Params.Keys,
		Params.MeanWindowLength,
		Params.Pndm,
		Params.Step,
		Params.TBegin,
		Params.TEnd,
		Params.Sampler.data(),
		Params.ReflowSampler.data(),
		Params.F0Method.data(),
		Params.UseShallowDiffusion,
		Params.VocoderModel,
		Params.ShallowDiffusionModel,
		Params.ShallowDiffusionUseSrcAudio,
		Params.VocoderHopSize,
		Params.VocoderMelBins,
		Params.VocoderSamplingRate,
		Params.ShallowDiffuisonSpeaker
	};
	auto TestAudio = LibSvcAllocateAudio();
	auto __BeginTime = clock();
	LibSvcInferAudio(Model, 1, Slices, &DynParams, LibSvcGetAudioSize(Audio) * 2, &Proc, TestAudio);
	auto __InferenceTime = double(clock() - __BeginTime) / 1000.;
	std::cout << "RTF: " << __InferenceTime / ((double)LibSvcGetAudioSize(Audio) / (double)SrcSr) << '\n';
#endif

#ifdef DRAGONIANLIB_IMPORT
	for (size_t i = 0; i < LibSvcGetSliceCount(Slices); ++i)
	{
		const auto Single = LibSvcGetSlice(Slices, i);
#else
	for (const auto& Single : Slices.Slices)
	{
#endif
#ifndef DRAGONIANLIB_IMPORT
		DragonianLib::WritePCMData(
			(LR"(D:/VSGIT/MoeSS - Release/Testdata/Input-PCM-SignedInt-16-)" + std::to_wstring(Proc) + L".wav").c_str(),
			Single.Audio,
			Config.SamplingRate
		);
#endif
#ifdef DRAGONIANLIB_IMPORT
		const auto SliceResampleLen = LibSvcGetSrcLength(Single) * 16000ll / SrcSr;
#else
		const auto SliceResampleLen = Single.OrgLen * 16000ll / SrcSr;
#endif
		const auto WavPaddedSize = ((SliceResampleLen / DRAGONIANLIB_PADDING_COUNT) + 1) * DRAGONIANLIB_PADDING_COUNT;
		const auto SliceTime = double(WavPaddedSize) / 16000.;
		auto BeginTime = clock();
#ifdef DRAGONIANLIB_IMPORT
		auto OutputObj = LibSvcAllocateAudio();
		Error = LibSvcInferSlice(Model, 1, Single, &DynParams, &Proc, OutputObj);
		if (Error)
		{
			auto ErrorMessage = LibSvcGetError(0);
			std::cout << WideStringToUTF8(ErrorMessage);
			LibSvcFreeString(ErrorMessage);
			return 0;
		}
#else
		auto Out = Model.SliceInference(Single, Params, Proc);
#endif
		auto InferenceTime = double(clock() - BeginTime) / 1000.;
		printf("Time Per Sec: %lf, Rtf: %lf\n", SliceTime / InferenceTime, InferenceTime / SliceTime);
#ifndef DRAGONIANLIB_IMPORT
		DragonianLib::WritePCMData(
			(LR"(D:/VSGIT/MoeSS - Release/Testdata/OutPut-PCM-SignedInt-16-)" + std::to_wstring(Proc) + L".wav").c_str(),
			Out,
			Config.SamplingRate
		);
#endif
#ifdef DRAGONIANLIB_IMPORT
		LibSvcInsertAudio(OutAudio, OutputObj);
		LibSvcReleaseAudio(OutputObj);
#else
		OutAudio.Insert(OutAudio.end(), Out.begin(), Out.end());
#endif
#ifdef DRAGONIANLIB_IMPORT
	}
#else
	}
#endif

#ifdef DRAGONIANLIB_IMPORT
	WCHAR OutPutPath[] = LR"(D:/VSGIT/MoeSS - Release/Testdata/Output-PCM-SignedInt-16.wav)";
	LibSvcWriteAudioFile(
		OutAudio,
		OutPutPath,
		Config.SamplingRate
	);
	LibSvcReleaseOffset(SliPos);
	LibSvcReleaseSliceData(Slices);
	LibSvcReleaseAudio(OutAudio);
	LibSvcReleaseAudio(Audio);
#else
	DragonianLib::WritePCMData(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/Output-PCM-SignedInt-16.wav)",
		OutAudio,
		Config.SamplingRate
	);
#endif
}