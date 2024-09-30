#include <iostream>
#include <tchar.h>
#ifndef DRAGONIANLIB_IMPORT
#include "AvCodec/AvCodec.h"
#include "MusicTranscription/ByteDancePianoTranScription.hpp"
#include "SingingVoiceConversion/Modules/header/Modules.hpp"
#include "MusicTranscription/MoePianoTranScription.hpp"
#include "SuperResolution/MoeSuperResolution.hpp"
#include "Tensor/Tensor.h"
#include "tlibsvc/VitsSvc.hpp"
#ifdef _WIN32
#include <mmeapi.h>
#pragma comment(lib, "winmm.lib") 
#endif
#else
#include <functional>
#include "SingingVoiceConversion/Modules/header/Models/ModelBase.hpp"
#include "SingingVoiceConversion/Modules/header/Models/Params.hpp"
#endif
#include "SingingVoiceConversion/Api/header/NativeApi.h"

class WithTimer
{
public:
	WithTimer(const std::function<void()>& _Fn)
	{
		LARGE_INTEGER Time1, Time2, Freq;
		QueryPerformanceFrequency(&Freq);
		QueryPerformanceCounter(&Time1);
		_Fn();
		QueryPerformanceCounter(&Time2);
		std::cout << " CostTime:" << double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart << "ms\n";
	}
};

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

#ifndef DRAGONIANLIB_IMPORT
template<typename _T = float>
void PrintTensor(DragonianLib::Tensor& _Tensor)
{
	for (DragonianLib::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (DragonianLib::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << _Tensor.Item<_T>({ i,j }) << ", ";
		std::cout << "],\n";
	}
	std::cout << "\n";
}

template<>
void PrintTensor<bool>(DragonianLib::Tensor& _Tensor)
{
	for (DragonianLib::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (DragonianLib::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << ((_Tensor.Item<bool>({ i,j })) ? "true " : "false") << ", ";
		std::cout << "]\n";
	}
	std::cout << "\n";
}
#endif

void LibSvcTest();

#ifndef DRAGONIANLIB_IMPORT
void LibSrTest();
void LibMtsTest();
void TensorLibDemo();
void OperatorTest();
#endif

int main()
{
#ifdef _WIN32
	if (GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS)
		SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#endif
	//LibMtsTest();
	//LibSrTest();
	LibSvcTest();
	//OperatorTest();
	system("pause");
	return 0;
}

void LibSvcTest()
{
#ifdef DRAGONIANLIB_IMPORT
	LibSvcInit();
#else
	LibSvcSpace SetupKernel();
#endif
	constexpr auto EProvider = 1;
	constexpr auto NumThread = 16;
	constexpr auto DeviceId = 0;

	auto GlobalEnv = LibSvcCreateEnv(NumThread, DeviceId, EProvider);
	if (!GlobalEnv)
		return;
	LibSvcSpace Hparams Config;
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
		return;

	wchar_t AudioInputPath[] = LR"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)";
	auto Audio = LibSvcAllocateFloatVector();
	auto Error = LibSvcReadAudio(
		AudioInputPath,
		SrcSr,
		Audio
	);
	if (Error)
		return;
#else
	auto Model = LibSvcSpace DiffusionSvc(
		Config,
		ProgressCb,
		DragonianLib::SingingVoiceConversion::LibSvcModule::ExecutionProviders(EProvider),
		DeviceId,
		NumThread
	);
	auto Audio = DragonianLib::AvCodec().DecodeFloat(
		R"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)",
		SrcSr
	);
#endif

	LibSvcSpace SlicerSettings SlicerConfig{
		SrcSr,
		40. / 32768.,
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

	std::wstring VocoderPath = LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hifigan\nsf-hifigan-n.onnx)";
	LibSvcSpace InferenceParams Params;
	Params.VocoderSamplingRate = Config.SamplingRate;
	Params.VocoderHopSize = Config.HopSize;
	Params.VocoderMelBins = static_cast<int>(Config.MelBins);
#ifndef DRAGONIANLIB_IMPORT
	Params.VocoderModel = LibSvcSpace SingingVoiceConversion::RefOrtCachedModel(
		VocoderPath,
		*(DragonianLib::DragonianLibOrtEnv*)GlobalEnv
	);
#endif

#ifdef DRAGONIANLIB_IMPORT
	const auto SliPos = LibSvcAllocateUInt64Vector();
	Error = LibSvcSliceAudio(
		Audio,
		&SlicerConf,
		SliPos
	);
	if (Error)
		return;

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
		return;
#else
	const auto SliPos = SliceAudio(Audio, SlicerConfig);
	auto Slices = LibSvcSpace SingingVoiceConversion::GetAudioSlice(Audio, SliPos, SlicerConfig);
	LibSvcSpace SingingVoiceConversion::PreProcessAudio(Slices, SrcSr, 512, L"Dio");
#endif

	size_t Proc = 0;
	Params.Step = 100;
	Params.Pndm = 5;
#ifdef DRAGONIANLIB_IMPORT
	auto OutAudio = LibSvcAllocateFloatVector();
	TotalStep = LibSvcGetSliceCount(Slices) * Params.Step / Params.Pndm;
#else
	DragonianLibSTL::Vector<float> OutAudio;
	OutAudio.Reserve(Audio.Size() * 2);
	TotalStep = Slices.Slices.Size() * Params.Step / Params.Pndm;
#endif

#ifdef DRAGONIANLIB_IMPORT
	auto VocoderModel = LibSvcLoadVocoder(
		VocoderPath.c_str(),
		GlobalEnv
	);

	LibSvcParams DynParams{
		Params.NoiseScale,
		Params.Seed,
		Params.SpeakerId,
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
		VocoderModel,
		Params.VocoderHopSize,
		Params.VocoderMelBins,
		Params.VocoderSamplingRate,
		0
	};
	auto TestAudio = LibSvcAllocateFloatVector();
	auto __BeginTime = clock();
	LibSvcInferAudio(Model, Slices, &DynParams, LibSvcGetFloatVectorSize(Audio) * 2, &Proc, TestAudio);
	LibSvcReleaseFloatVector(TestAudio);
	auto __InferenceTime = double(clock() - __BeginTime) / 1000.;
	std::cout << "RTF: " << __InferenceTime / ((double)LibSvcGetFloatVectorSize(Audio) / (double)SrcSr) << '\n';
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
		auto OutputObj = LibSvcAllocateFloatVector();
		Error = LibSvcInferSlice(Model, Single, &DynParams, &Proc, OutputObj);
		if (Error)
			return;
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
		LibSvcInsertFloatVector(OutAudio, OutputObj);
		LibSvcReleaseFloatVector(OutputObj);
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
	LibSvcReleaseUInt64Vector(SliPos);
	LibSvcReleaseSliceData(Slices);
	LibSvcReleaseFloatVector(OutAudio);
	LibSvcReleaseFloatVector(Audio);
	LibSvcUnloadVocoder(VocoderPath.c_str(), GlobalEnv);
	LibSvcDestoryEnv(GlobalEnv);
#else
	DragonianLib::WritePCMData(
		LR"(D:/VSGIT/MoeSS - Release/Testdata/Output-PCM-SignedInt-16.wav)",
		OutAudio,
		Config.SamplingRate
	);
#endif
}

#ifndef DRAGONIANLIB_IMPORT
//#include "tlibsr/MoeSuperResolution.hpp"
void OperatorTest()
{
	using namespace DragonianLib;
	Tensor Test{ {114,514,1919}, TensorType::Float32, Device::CPU };
	std::cout << Test.Item<float>();
	/*tlibsvc::VitsSvcConfig Config{
		LR"(D:\VSGIT\MoeVS-SVC\Build\Release\Models\SoVits4\SoVits4_SoVits.onnx)",
		LR"(D:\VSGIT\MoeVS-SVC\Build\Release\hubert\vec-768-layer-12.onnx)",
		L"SoVits4.0",
		nullptr,
		{},
		{},
		40000,
		320,
		768,
	};
	tlibsvc::VitsSvc Model(
		Config,
		[](size_t, size_t) {}
	);
	auto Audio = DragonianLib::AvCodec().DecodeSigned16(
		R"(D:/VSGIT/MoeVoiceStudioSvc - Core - Cmd/libdlvoicecodec/input.wav)",
		44100
	);
	LibSvcSpace SlicerSettings SlicerConfig{
		44100,
		40.,
		5.,
		2048,
		512
	};
	const auto SliPos = SliceAudio(Audio, SlicerConfig);
	auto Slices = tlibsvc:: GetAudioSlice(Audio, SliPos, SlicerConfig.Threshold);
	tlibsvc::PreProcessAudio(Slices, 44100, 512, L"Dio");*/
	/*
	
	auto ddddd = adddd(1, 2.f, 3, 4.f, 5);
	std::cout << ddddd << '\n';
	DragonianLib::Tensor::SetThreadCount(8);
	DragonianLib::Tensor::EnableTimeLogger(false);
	DragonianLib::ThreadPool Thp;
	Thp.EnableTimeLogger(false);
	Thp.Init(8);
	DragonianLib::Tensor Ten1919810({ 1,768,10000 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
	for (int64_t i = 0; i < 20; ++i)
	{
		Ten1919810.RandFix(&Thp);
		WithTimer(
			[&]()
			{
				auto Res = Ten1919810 + Ten1919810 * 2.;
			}
		);
	}
	*/
}

void TensorLibDemo()
{
	DragonianLib::Tensor aaaaaaaaaaaaa{ {114,514,810}, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU };
	aaaaaaaaaaaaa.Assign(1.f);
	aaaaaaaaaaaaa.Permute({ 2,0,1 }).Assign(1.f);

	DragonianLib::Tensor::SetThreadCount(8);
	DragonianLib::Tensor::EnableTimeLogger(false);
	DragonianLib::ThreadPool Thp;
	Thp.EnableTimeLogger(false);
	Thp.Init(8);
	DragonianLib::Tensor::Arange(1., 5., 0.3).UnSqueeze(0).Invoke(1, PrintTensor);
	constexpr float Temp[10]{ 114,514,1919,810,1453,721,996,7,1919,810 };
	DragonianLib::Tensor Ten({ 3,5 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
	Ten.RandnFix();
	auto a = Ten;
	std::cout << "\nTen:\n";
	Ten.Invoke(1, PrintTensor);
	std::cout << "\nGather Op Test\n";
	auto Indices = DragonianLib::Tensor::ConstantOf({ 2,2 }, 0ll, DragonianLib::TensorType::Int64);
	Indices[0][0] = 0ll; Indices[0][1] = 1ll; Indices[1][0] = 2ll; Indices[1][1] = 1ll;
	Indices.Invoke(1, PrintTensor<DragonianLib::SizeType>);
	Ten.Gather(Indices, 1).Invoke(1, PrintTensor);
	std::cout << "\nCumSum Op Test\n";
	DragonianLib::Tensor::CumSum(Ten, 0, nullptr).UnSqueeze(0).Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten = DragonianLib::Tensor::Stack({ Ten ,Ten, Ten }, 0);
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten += Ten;
	Ten.Sin_();
	std::cout << "Op Test: \n";
	Ten.Invoke(1, PrintTensor);
	std::cout << "Sin Op Test: \n";
	Ten.Sin().Invoke(1, PrintTensor);
	std::cout << "Cos Op Test: \n";
	Ten.Cos().Invoke(1, PrintTensor);
	std::cout << "Tan Op Test: \n";
	Ten.Tan().Invoke(1, PrintTensor);
	std::cout << "Abs Op Test: \n";
	Ten.Abs().Invoke(1, PrintTensor);
	std::cout << "Ceil Op Test: \n";
	Ten.Ceil().Invoke(1, PrintTensor);
	std::cout << "Floor Op Test: \n";
	Ten.Floor().Invoke(1, PrintTensor);
	std::cout << "Compare Op Test: \n";
	(Ten.Abs() == Ten).Invoke(1, PrintTensor<bool>);
	(Ten.Abs() != Ten).Invoke(1, PrintTensor<bool>);
	((Ten.Abs() != Ten) + (Ten.Abs() == Ten)).Invoke(1, PrintTensor<bool>);
	std::cout << "Op Test End.\n\n\n";
	auto Tens = DragonianLib::Tensor::Cat({ Ten ,Ten }, 2);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens = DragonianLib::Tensor::Cat({ Ten ,Ten }, -1);
	Tens.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tens.Cast(DragonianLib::TensorType::Int64).Invoke(1, PrintTensor<int64_t>);
	std::cout << '\n';
	DragonianLib::Tensor::Pad(Ten, { {1, 2}, {1, 2} }, DragonianLib::PaddingType::Reflect).Invoke(1, PrintTensor);
	std::cout << '\n';
	const DragonianLib::Tensor Ten114514({ 1,514,1,1919 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
	LARGE_INTEGER Time1, Time2, Freq;
	QueryPerformanceFrequency(&Freq);
	Indices = DragonianLib::Tensor::ConstantOf({ 1000 }, 0ll, DragonianLib::TensorType::Int64);
	Indices[1].Assign(1ll);
	const DragonianLib::Tensor Embedding({ 1000,768 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);

	for (int64_t i = 0; i < 20; ++i)
	{
		DragonianLib::Tensor Ten1919810({ 1,768,100000 }, DragonianLib::TensorType::Float32, DragonianLib::Device::CPU);
		Ten1919810.RandFix(&Thp);
		Embedding[0].Assign(i);
		Embedding[1].Assign(i + 1);
		//Ten1919810.Assign(i);
		auto Emb = Embedding[0];
		QueryPerformanceCounter(&Time1);
		//auto Out = Embedding.Gather(Indices, 0, Thp);
		//Embedding.Assign(Embedding);
		//Ten114514.Permute({ 3,1,2,0 }).Clone();
		//DragonianLib::Tensor::Pad(Ten114514, {DragonianLib::None,19 },DragonianLib::PaddingType::Zero, DragonianLib::TensorType::Float32,nullptr, &Thp);
		/*DragonianLib::Tensor::Pad(
			Ten1919810,
			{DragonianLib::None,1 },
			DragonianLib::PaddingType::Replicate,
			DragonianLib::TensorType::Float32,
			nullptr, &Thp
		);*/
		//auto a = DragonianLib::Tensor::Diff(Ten1919810, 1, &Thp);
		//DragonianLib::Tensor::Stack({Ten1919810.Squeeze()}, 0, &Thp);
		//auto a = Ten1919810.Permute({ 0,2,1 });
		//DragonianLib::Tensor::Repeat(Ten1919810, { {0, 2} }, &Thp);
		//a.Continuous(&Thp);
		auto Res = ((Ten1919810 + Ten1919810) == Ten1919810 * 2.);
		std::cout << (bool)*(Res.Buffer()) << '\n';
		std::cout << (bool)*(Res.Buffer() + 1) << '\n';
		std::cout << (bool)*(Res.Buffer() + 2) << '\n';
		std::cout << (bool)*(Res.Buffer() + 3) << '\n';
		std::cout << (bool)*(Res.Buffer() + 4) << '\n';

		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{0,192} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{192,384} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{384,572} }).Continuous(); });
		//Thp.Commit([&]() { a.Slice({ DragonianLib::None,{572,768} }).Continuous(); });
		//Thp.Join();
		QueryPerformanceCounter(&Time2);
		std::cout << i << " CostTime:" << double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart << "ms\n";
		//Out.Invoke(1, PrintTensor);
	}
	std::cout << "\n\n\n";
	Ten.FixOnes();
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Fix(114514.);
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Assign(Temp, sizeof(Temp));
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto TenA = Ten[0];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(1.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(2.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(3.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(4.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(5.f);
	TenA = Ten[1];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(6.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(7.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(8.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(9.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(10.f);
	TenA = TenA[2];
	TenA = TenA[4];
	TenA = TenA[0];
	TenA = TenA[0];
	auto TenB = std::move(Ten);
	TenA = std::move(TenB);
	auto TenC = TenA.Slice({ {-1, -1,-3},{0,-1 },{2,2,-1} });
	TenA.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Assign(Temp, sizeof(Temp));
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto Tennnnn = TenA.Clone();
	TenA.Permute({ 2,0,1 }).Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Clone().Invoke(1, PrintTensor);
}

void LibSrTest()
{
	DragonianLib::LibSuperResolution::MoeSR Model(
		{
			LR"(D:\VSGIT\白叶的AI工具箱\Models\real-hatgan\x2\x2_universal-fix1.onnx)",
			LR"(None)",
			0,
			0,
			2
		},
		ProgressCbS,
		8,
		0,
		1
	);

	DragonianLib::GdiInit();

	DragonianLib::Image Image(
		LR"(C:\DataSpace\MediaProj\PlayList\a.jpg)",
		192,
		192,
		16,
		0.f,
		false/*,
		LR"(D:\VSGIT\CG000002-DEB.png)"*/
	);
	/*Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TN.png)", 1, 100))
		std::cout << "1-Complete!\n";
	Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TNN.png)", 1, 100))
		std::cout << "2-Complete!\n";*/

	Model.Infer(Image, 50);

	if (Image.MergeWrite(LR"(C:\DataSpace\MediaProj\PlayList\CG.png)", 2, 100))
		std::cout << "Complete!\n";

	DragonianLib::GdiClose();
}

void LibMtsTest()
{
	DragonianLib::LibMusicTranscription::ByteDancePianoTranScription Model(
		{
			LR"(D:\VSGIT\白叶的AI工具箱\Models\PianoTranscription\modeln.onnx)",
			L"null",
		},
		ProgressCbS,
		8,
		0,
		0
	);

	DragonianLib::LibMusicTranscription::Hparams _Config;
	auto Audio = DragonianLib::AvCodec().DecodeFloat(
		R"(C:\DataSpace\MediaProj\PlayList\list.ogg)",
		16000
	);
	_Config.FrameThreshold = 0.5;
	_Config.OnsetThreshold = 0.8;
	_Config.OffsetThreshold = 0.2;
	_Config.OnsetAligSize = 3;
	_Config.FliterSize = 8;
	_Config.FliterCount = 5;

	auto Midi = Model.Inference(Audio, _Config, 5);
	WriteMidiFile(LR"(C:\DataSpace\MediaProj\PlayList\list.mid)", Midi, 0, 384 / 2);
}

#endif