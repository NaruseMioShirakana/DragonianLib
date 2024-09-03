#include "../header/NativeApi.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <deque>
#include <mutex>
#include <string>

#include "AvCodec/AvCodec.h"
#include "Base.h"
#include "../../Modules/header/Modules.hpp"


#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif



const wchar_t* LibSvcNullString = L"";

#define LibSvcNullStrCheck(Str) ((Str)?(Str):(LibSvcNullString))

#ifndef _WIN32
BSTR SysAllocString(const wchar_t* _String)
{
	wchar_t* ret = new wchar_t[wcslen(_String)];
	wcscpy(ret, _String);
	return ret;
}

void SysFreeString(BSTR _String)
{
	delete[] _String;
}
#endif

std::deque<std::wstring> ErrorQueue;
size_t MaxErrorCount = 20;

using Config = LibSvcSpace Hparams;
using VitsSvc = LibSvcSpace VitsSvc;
using UnionSvc = LibSvcSpace UnionSvcModel;
using ReflowSvc = LibSvcSpace ReflowSvc;
using ClusterBase = DragonianLib::BaseCluster;
using TensorExtractorBase = LibSvcSpace LibSvcTensorExtractor;
using ProgressCallback = LibSvcSpace LibSvcModule::ProgressCallback;
using ExecutionProvider = LibSvcSpace LibSvcModule::ExecutionProviders;
using Slices = LibSvcSpace SingleAudio;
using SingleSlice = LibSvcSpace SingleSlice;
using Params = LibSvcSpace InferenceParams;

using AudioContainer = DragonianLibSTL::Vector<int16_t>;
using OffsetContainer = DragonianLibSTL::Vector<size_t>;
using MelContainer = std::pair<DragonianLibSTL::Vector<float>, int64_t>;
using DataContainer = Slices;

std::unordered_map<std::wstring, DlCodecStft::Mel*> MelOperators;

void InitLibSvcHparams(LibSvcHparams* _Input)
{
	_Input->TensorExtractor = nullptr;
	_Input->HubertPath = nullptr;
	_Input->DiffusionSvc = {
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr
	};
	_Input->VitsSvc = {
		nullptr
	};
	_Input->ReflowSvc = {
		nullptr,
		nullptr,
		nullptr
	};
	_Input->Cluster = {
		10000,
		nullptr,
		nullptr
	};

	_Input->SamplingRate = 22050;

	_Input->HopSize = 320;
	_Input->HiddenUnitKDims = 256;
	_Input->SpeakerCount = 1;
	_Input->EnableCharaMix = false;
	_Input->EnableVolume = false;
	_Input->VaeMode = true;

	_Input->MelBins = 128;
	_Input->Pndms = 100;
	_Input->MaxStep = 1000;
	_Input->SpecMin = -12;
	_Input->SpecMax = 2;
	_Input->Scale = 1000.f;
}

void InitLibSvcParams(LibSvcParams* _Input)
{
	//通用
	_Input->NoiseScale = 0.3f;							//噪声修正因子          0-10
	_Input->Seed = 52468;									//种子
	_Input->SpeakerId = 0;								//角色ID
	_Input->SrcSamplingRate = 48000;						//源采样率
	_Input->SpkCount = 2;									//模型角色数

	//SVC
	_Input->IndexRate = 0.f;								//索引比               0-1
	_Input->ClusterRate = 0.f;							//聚类比               0-1
	_Input->DDSPNoiseScale = 0.8f;						//DDSP噪声修正因子      0-10
	_Input->Keys = 0.f;									//升降调               -64-64
	_Input->MeanWindowLength = 2;						//均值滤波器窗口大小     1-20
	_Input->Pndm = 100;									//Diffusion加速倍数    1-200
	_Input->Step = 1000;									//Diffusion总步数      1-1000
	_Input->TBegin = 0.f;
	_Input->TEnd = 1.f;
	_Input->Sampler = nullptr;							//Diffusion采样器
	_Input->ReflowSampler = nullptr;						//Reflow采样器
	_Input->F0Method = nullptr;							//F0提取算法
	_Input->UseShallowDiffusionOrEnhancer = false;                  //使用浅扩散
	_Input->_VocoderModel = nullptr;
	_Input->_ShallowDiffusionModel = nullptr;
	_Input->ShallowDiffusionUseSrcAudio = 1;
	_Input->VocoderHopSize = 512;
	_Input->VocoderMelBins = 128;
	_Input->VocoderSamplingRate = 44100;
	_Input->ShallowDiffuisonSpeaker = 0;
	_Input->__DEBUG__MODE__ = 0;
}

void InitLibSvcSlicerSettings(LibSvcSlicerSettings* _Input)
{
	_Input->SamplingRate = 48000;
	_Input->Threshold = 30.;
	_Input->MinLength = 3.;
	_Input->WindowLength = 2048;
	_Input->HopSize = 512;
}

float* LibSvcGetFloatVectorData(void* _Obj)
{
	auto& Obj = *(DragonianLibSTL::Vector<float>*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetFloatVectorSize(void* _Obj)
{
	auto& Obj = *(DragonianLibSTL::Vector<float>*)_Obj;
	return Obj.Size();
}

void* LibSvcGetDFloatVectorData(void* _Obj, size_t _Index)
{
	auto& Obj = *(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>*)_Obj;
	return Obj.Data() + _Index;
}

size_t LibSvcGetDFloatVectorSize(void* _Obj)
{
	auto& Obj = *(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>*)_Obj;
	return Obj.Size();
}

void* LibSvcAllocateAudio()
{
	return new AudioContainer;
}

void* LibSvcAllocateMel()
{
	return new MelContainer;
}

void* LibSvcAllocateOffset()
{
	return new OffsetContainer;
}

void* LibSvcAllocateSliceData()
{
	return new DataContainer;
}

void LibSvcReleaseAudio(void* _Obj)
{
	delete (AudioContainer*)_Obj;
}

void LibSvcReleaseMel(void* _Obj)
{
	delete (MelContainer*)_Obj;
}

void LibSvcReleaseOffset(void* _Obj)
{
	delete (OffsetContainer*)_Obj;
}

void LibSvcSetOffsetLength(void* _Obj, size_t _Size)
{
	auto& Obj = *(OffsetContainer*)_Obj;
	Obj.Resize(_Size);
}

void LibSvcReleaseSliceData(void* _Obj)
{
	delete (DataContainer*)_Obj;
}

size_t* LibSvcGetOffsetData(void* _Obj)
{
	auto& Obj = *(OffsetContainer*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetOffsetSize(void* _Obj)
{
	auto& Obj = *(OffsetContainer*)_Obj;
	return Obj.Size();
}

void LibSvcSetAudioLength(void* _Obj, size_t _Size)
{
	auto& Obj = *(AudioContainer*)_Obj;
	Obj.Resize(_Size);
}

void LibSvcInsertAudio(void* _ObjA, void* _ObjB)
{
	auto& ObjA = *(AudioContainer*)_ObjA;
	auto& ObjB = *(AudioContainer*)_ObjB;
	ObjA.Insert(ObjA.end(), ObjB.begin(), ObjB.end());
}

int16_t* LibSvcGetAudioData(void* _Obj)
{
	auto& Obj = *(AudioContainer*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetAudioSize(void* _Obj)
{
	auto& Obj = *(AudioContainer*)_Obj;
	return Obj.Size();
}

void* LibSvcGetMelData(void* _Obj)
{
	auto& Obj = *(MelContainer*)_Obj;
	return &Obj.first;
}

int64_t LibSvcGetMelSize(void* _Obj)
{
	auto& Obj = *(MelContainer*)_Obj;
	return Obj.second;
}

void LibSvcSetMaxErrorCount(size_t Count)
{
	MaxErrorCount = Count;
}

BSTR LibSvcGetAudioPath(void* _Obj)
{
	auto& Obj = *(DataContainer*)_Obj;
	return SysAllocString(Obj.Path.c_str());
}

void* LibSvcGetSlice(void* _Obj, size_t _Index)
{
	auto& Obj = *(DataContainer*)_Obj;
	return Obj.Slices.Data() + _Index;
}

size_t LibSvcGetSliceCount(void* _Obj)
{
	auto& Obj = *(DataContainer*)_Obj;
	return Obj.Slices.Size();
}

void* LibSvcGetAudio(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return &Obj.Audio;
}

void* LibSvcGetF0(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return &Obj.F0;
}

void* LibSvcGetVolume(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return &Obj.Volume;
}

void* LibSvcGetSpeaker(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return &Obj.Speaker;
}

INT32 LibSvcGetSrcLength(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return Obj.OrgLen;
}

INT32 LibSvcGetIsNotMute(void* _Obj)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return Obj.IsNotMute;
}

void LibSvcSetSpeakerMixDataSize(void* _Obj, size_t _NSpeaker)
{
	auto& Obj = *(SingleSlice*)_Obj;
	Obj.Speaker.Resize(_NSpeaker, DragonianLibSTL::Vector(Obj.F0.Size(), 0.f));
}

void LibSvcSetGlobalEnvDir(
	LPWSTR _Dir
)
{
	DragonianLib::SetGlobalEnvDir(_Dir);
}

void LibSvcInit()
{
	LibSvcSpace SetupKernel();
}

std::mutex ErrorMx;

void LibSvcFreeString(BSTR _String)
{
	SysFreeString(_String);
}

BSTR LibSvcGetError(size_t Index)
{
	const auto& Ref = ErrorQueue.at(Index);
	auto Ret = SysAllocString(Ref.c_str());
	ErrorQueue.erase(ErrorQueue.begin() + ptrdiff_t(Index));
	ErrorMx.unlock();
	return Ret;
}

void RaiseError(const std::wstring& _Msg)
{
	ErrorMx.lock();
	ErrorQueue.emplace_front(_Msg);
	if (ErrorQueue.size() > MaxErrorCount)
		ErrorQueue.pop_back();
}

DragonianLib::DragonianLibOrtEnv* GlobalEnv = nullptr;
size_t GlobalEnvRefCount = 0;

INT32 LibSvcSetGlobalEnv(UINT32 ThreadCount, UINT32 DeviceID, UINT32 Provider)
{
	if (GlobalEnvRefCount)
	{
		RaiseError(L"You Must Unload All Vocoder First!");
		return 1;
	}
	try
	{
		delete GlobalEnv;
		GlobalEnv = new DragonianLib::DragonianLibOrtEnv(ThreadCount,DeviceID,Provider);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

int32_t LibSvcSliceAudio(
	const void* _Audio, //DragonianLibSTL::Vector<int16_t> By "LibSvcAllocateAudio()"
	const void* _Setting, //LibSvcSlicerSettings
	void* _Output //DragonianLibSTL::Vector<size_t> By "LibSvcAllocateOffset()"
)
{
	if (!_Audio)
	{
		RaiseError(L"Audio Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	const LibSvcSlicerSettings* _SettingInp = (const LibSvcSlicerSettings*)_Setting;
	auto& Ret = *(DragonianLibSTL::Vector<size_t>*)(_Output);
	LibSvcSpace SlicerSettings SliSetting{
		_SettingInp->SamplingRate,
		_SettingInp->Threshold,
		_SettingInp->MinLength,
		_SettingInp->WindowLength,
		_SettingInp->HopSize
	};
	
	try
	{
		Ret = LibSvcSpace SliceAudio(*(const AudioContainer*)(_Audio), SliSetting);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

int32_t LibSvcPreprocess(
	const void* _Audio, //DragonianLibSTL::Vector<int16_t> By "LibSvcAllocateAudio()"
	const void* _SlicePos, //DragonianLibSTL::Vector<size_t> By "LibSvcAllocateOffset()"
	int32_t _SamplingRate,
	int32_t _HopSize,
	double _Threshold,
	const wchar_t* _F0Method,
	void* _Output // Slices By "LibSvcAllocateSliceData()"
)
{
	LibSvcSpace SlicerSettings _Setting{
		.Threshold = _Threshold
	};

	if (!_Audio)
	{
		RaiseError(L"Audio Could Not Be Null!");
		return 1;
	}

	if (!_SlicePos)
	{
		RaiseError(L"Slice Pos Could Not Be Null!");
		return 1;
	}

	if(!_Output) 
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	Slices& Ret = *static_cast<Slices*>(_Output);
	try
	{
		Ret = LibSvcSpace SingingVoiceConversion::GetAudioSlice(
			*(const AudioContainer*)(_Audio),
			*(const OffsetContainer*)(_SlicePos),
			_Setting
		);
		LibSvcSpace SingingVoiceConversion::PreProcessAudio(Ret, _SamplingRate, _HopSize, _F0Method);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 LibSvcStft(
	const void* _Audio,
	INT32 _SamplingRate,
	INT32 _Hopsize,
	INT32 _MelBins,
	void* _Output
)
{
	if (!_Audio)
	{
		RaiseError(L"Audio Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	if (MelOperators.size() > 5)
	{
		delete MelOperators.begin()->second;
		MelOperators.erase(MelOperators.begin());
	}

	try
	{
		const std::wstring _Name = L"S" +
			std::to_wstring(_SamplingRate) +
			L"H" + std::to_wstring(_Hopsize) +
			L"M" + std::to_wstring(_MelBins);
		if (!MelOperators.contains(_Name))
			MelOperators[_Name] = new DlCodecStft::Mel(_Hopsize * 4, _Hopsize, _SamplingRate, _MelBins);
		auto _NormalizedAudio = InterpResample(
			*(const AudioContainer*)_Audio,
			_SamplingRate,
			_SamplingRate,
			32768.
		);
		*(MelContainer*)(_Output) = MelOperators.at(_Name)->operator()(_NormalizedAudio);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcInferSlice(
	void* _Model,
	UINT32 _T,
	const void* _Slice,
	const void* _InferParams,
	size_t* _Process,
	void* _Output
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Slice)
	{
		RaiseError(L"_Slice Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		RaiseError(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Process)
	{
		RaiseError(L"_Process Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	const auto& InpParam = *(const LibSvcParams*)(_InferParams);

	if (!InpParam._VocoderModel && _T == 1)
	{
		RaiseError(L"_VocoderModel Could Not Be Null!");
		return 1;
	}

	const Params Param
	{
		InpParam.NoiseScale,
		InpParam.Seed,
		InpParam.SpeakerId,
		InpParam.SrcSamplingRate,
		InpParam.SpkCount,
		InpParam.IndexRate,
		InpParam.ClusterRate,
		InpParam.DDSPNoiseScale,
		InpParam.Keys,
		InpParam.MeanWindowLength,
		InpParam.Pndm,
		InpParam.Step,
		InpParam.TBegin,
		InpParam.TEnd,
		LibSvcNullStrCheck(InpParam.Sampler),
		LibSvcNullStrCheck(InpParam.ReflowSampler),
		LibSvcNullStrCheck(InpParam.F0Method),
		(bool)InpParam.UseShallowDiffusionOrEnhancer,
		InpParam._VocoderModel,
		InpParam._ShallowDiffusionModel,
		(bool)InpParam.ShallowDiffusionUseSrcAudio,
		InpParam.VocoderHopSize,
		InpParam.VocoderMelBins,
		InpParam.VocoderSamplingRate,
		InpParam.ShallowDiffuisonSpeaker
	};

	try
	{
		if (_T == 0)
			*(AudioContainer*)(_Output) = ((VitsSvc*)(_Model))->SliceInference(*(const SingleSlice*)(_Slice), Param, *_Process);
		else if (_T == 1)
			*(AudioContainer*)(_Output) = ((UnionSvc*)(_Model))->SliceInference(*(const SingleSlice*)(_Slice), Param, *_Process);
		else
		{
			RaiseError(L"UnSupported Model Type!");
			return 1;
		}
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcInferAudio(
	SvcModel _Model,
	UINT32 _T,
	SlicesType _Audio,
	const void* _InferParams,
	UINT64 _SrcLength,
	size_t* _Process,
	Int16Vector _Output
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Audio)
	{
		RaiseError(L"_Audio Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		RaiseError(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Process)
	{
		RaiseError(L"_Process Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	const auto& InpParam = *(const LibSvcParams*)(_InferParams);

	if (!InpParam._VocoderModel && _T == 1)
	{
		RaiseError(L"_VocoderModel Could Not Be Null!");
		return 1;
	}

	const Params Param
	{
		InpParam.NoiseScale,
		InpParam.Seed,
		InpParam.SpeakerId,
		InpParam.SrcSamplingRate,
		InpParam.SpkCount,
		InpParam.IndexRate,
		InpParam.ClusterRate,
		InpParam.DDSPNoiseScale,
		InpParam.Keys,
		InpParam.MeanWindowLength,
		InpParam.Pndm,
		InpParam.Step,
		InpParam.TBegin,
		InpParam.TEnd,
		LibSvcNullStrCheck(InpParam.Sampler),
		LibSvcNullStrCheck(InpParam.ReflowSampler),
		LibSvcNullStrCheck(InpParam.F0Method),
		(bool)InpParam.UseShallowDiffusionOrEnhancer,
		InpParam._VocoderModel,
		InpParam._ShallowDiffusionModel,
		(bool)InpParam.ShallowDiffusionUseSrcAudio,
		InpParam.VocoderHopSize,
		InpParam.VocoderMelBins,
		InpParam.VocoderSamplingRate,
		InpParam.ShallowDiffuisonSpeaker
	};
	auto __Slices = *(const Slices*)(_Audio);

	auto& OutPutAudio = *(AudioContainer*)(_Output);
	OutPutAudio.Reserve(_SrcLength);
	try
	{
		if (_T == 0)
		{
			for (const auto& Single : __Slices.Slices)
			{
				auto Out = ((VitsSvc*)(_Model))->SliceInference(Single, Param, *_Process);
				OutPutAudio.Insert(OutPutAudio.end(), Out.begin(), Out.end());
			}
		}
		else if (_T == 1)
		{
			for (const auto& Single : __Slices.Slices)
			{
				auto Out = ((UnionSvc*)(_Model))->SliceInference(Single, Param, *_Process);
				OutPutAudio.Insert(OutPutAudio.end(), Out.begin(), Out.end());
			}
		}
		else
		{
			RaiseError(L"UnSupported Model Type!");
			return 1;
		}
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcInferPCMData(
	SvcModel _Model,							//SingingVoiceConversion Model
	UINT32 _T,
	CInt16Vector _PCMData,
	const void* _InferParams,					//Ptr Of LibSvcParams
	Int16Vector _Output							//DragonianLibSTL::Vector<int16_t> By "LibSvcAllocateAudio()"
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_PCMData)
	{
		RaiseError(L"_PCMData Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		RaiseError(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	const auto& InpParam = *(const LibSvcParams*)(_InferParams);

	if (!InpParam._VocoderModel && _T == 1)
	{
		RaiseError(L"_VocoderModel Could Not Be Null!");
		return 1;
	}

	const Params Param
	{
		InpParam.NoiseScale,
		InpParam.Seed,
		InpParam.SpeakerId,
		InpParam.SrcSamplingRate,
		InpParam.SpkCount,
		InpParam.IndexRate,
		InpParam.ClusterRate,
		InpParam.DDSPNoiseScale,
		InpParam.Keys,
		InpParam.MeanWindowLength,
		InpParam.Pndm,
		InpParam.Step,
		InpParam.TBegin,
		InpParam.TEnd,
		LibSvcNullStrCheck(InpParam.Sampler),
		LibSvcNullStrCheck(InpParam.ReflowSampler),
		LibSvcNullStrCheck(InpParam.F0Method),
		(bool)InpParam.UseShallowDiffusionOrEnhancer,
		InpParam._VocoderModel,
		InpParam._ShallowDiffusionModel,
		(bool)InpParam.ShallowDiffusionUseSrcAudio,
		InpParam.VocoderHopSize,
		InpParam.VocoderMelBins,
		InpParam.VocoderSamplingRate,
		InpParam.ShallowDiffuisonSpeaker
	};

	auto& InputData = *(const AudioContainer*)(_PCMData);

	try
	{
		if (_T == 0)
			*(AudioContainer*)(_Output) = ((VitsSvc*)(_Model))->InferPCMData(InputData, (long)InputData.Size(), Param);
		else if (_T == 1)
			*(AudioContainer*)(_Output) = ((UnionSvc*)(_Model))->InferPCMData(InputData, (long)InputData.Size(), Param);
		else
		{
			RaiseError(L"UnSupported Model Type!");
			return 1;
		}
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcShallowDiffusionInference(
	void* _Model,
	const void* _16KAudioHubert,
	void* _Mel,
	const void* _SrcF0,
	const void* _SrcVolume,
	const void* _SrcSpeakerMap,
	INT64 _SrcSize,
	const void* _InferParams,
	size_t* _Process,
	void* _Output
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_16KAudioHubert)
	{
		RaiseError(L"_16KAudioHubert Could Not Be Null!");
		return 1;
	}

	if (!_Mel)
	{
		RaiseError(L"_Mel Could Not Be Null!");
		return 1;
	}

	if (!_SrcF0)
	{
		RaiseError(L"_SrcF0 Could Not Be Null!");
		return 1;
	}

	if (!_SrcVolume)
	{
		RaiseError(L"_SrcVolume Could Not Be Null!");
		return 1;
	}

	if (!_SrcSpeakerMap)
	{
		RaiseError(L"_SrcSpeakerMap Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		RaiseError(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Process)
	{
		RaiseError(L"_Process Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	const auto& InpParam = *(const LibSvcParams*)(_InferParams);

	if(!InpParam._VocoderModel)
	{
		RaiseError(L"_VocoderModel Could Not Be Null!");
		return 1;
	}

	const Params Param
	{
		InpParam.NoiseScale,
		InpParam.Seed,
		InpParam.SpeakerId,
		InpParam.SrcSamplingRate,
		InpParam.SpkCount,
		InpParam.IndexRate,
		InpParam.ClusterRate,
		InpParam.DDSPNoiseScale,
		InpParam.Keys,
		InpParam.MeanWindowLength,
		InpParam.Pndm,
		InpParam.Step,
		InpParam.TBegin,
		InpParam.TEnd,
		LibSvcNullStrCheck(InpParam.Sampler),
		LibSvcNullStrCheck(InpParam.ReflowSampler),
		LibSvcNullStrCheck(InpParam.F0Method),
		(bool)InpParam.UseShallowDiffusionOrEnhancer,
		InpParam._VocoderModel,
		InpParam._ShallowDiffusionModel,
		(bool)InpParam.ShallowDiffusionUseSrcAudio,
		InpParam.VocoderHopSize,
		InpParam.VocoderMelBins,
		InpParam.VocoderSamplingRate,
		InpParam.ShallowDiffuisonSpeaker
	};

	auto _NormalizedAudio = InterpResample(
		*(const AudioContainer*)_16KAudioHubert,
		16000,
		16000,
		32768.f
	);

	try
	{
		*(AudioContainer*)(_Output) = ((UnionSvc*)(_Model))->ShallowDiffusionInference(
			_NormalizedAudio,
			Param,
			*(MelContainer*)(_Mel),
			*(const DragonianLibSTL::Vector<float>*)(_SrcF0),
			*(const DragonianLibSTL::Vector<float>*)(_SrcVolume),
			*(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>*)(_SrcSpeakerMap),
			*_Process,
			_SrcSize
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcVocoderEnhance(
	void* _Model,
	void* _Mel,
	const void* _F0,
	INT32 _VocoderMelBins,
	void* _Output
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_F0)
	{
		RaiseError(L"_16KAudioHubert Could Not Be Null!");
		return 1;
	}

	if (!_Mel)
	{
		RaiseError(L"_Mel Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	auto Rf0 = *(const DragonianLibSTL::Vector<float>*)(_F0);
	auto& MelTemp = *(MelContainer*)(_Mel);
	if (Rf0.Size() != (size_t)MelTemp.second)
		Rf0 = InterpFunc(Rf0, (long)Rf0.Size(), (long)MelTemp.second);
	try
	{
		*(AudioContainer*)(_Output) = LibSvcSpace VocoderInfer(
		   MelTemp.first,
		   Rf0,
		   _VocoderMelBins,
		   MelTemp.second,
		   GlobalEnv->GetMemoryInfo(),
		   _Model
	   );
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	
	return 0;
}

void* LibSvcLoadModel(
	UINT32 _T,
	const void* _Config,
	ProgCallback _ProgressCallback,
	UINT32 _ExecutionProvider,
	UINT32 _DeviceID,
	UINT32 _ThreadCount
)
{
	if (!_Config)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return nullptr;
	}

	auto& Config = *(const LibSvcHparams*)(_Config);

	//printf("%lld", (long long)(Config.DiffusionSvc.Encoder));

	LibSvcSpace Hparams ModelConfig{
		LibSvcNullStrCheck(Config.TensorExtractor),
		LibSvcNullStrCheck(Config.HubertPath),
		{
			LibSvcNullStrCheck(Config.DiffusionSvc.Encoder),
			LibSvcNullStrCheck(Config.DiffusionSvc.Denoise),
			LibSvcNullStrCheck(Config.DiffusionSvc.Pred),
			LibSvcNullStrCheck(Config.DiffusionSvc.After),
			LibSvcNullStrCheck(Config.DiffusionSvc.Alpha),
			LibSvcNullStrCheck(Config.DiffusionSvc.Naive),
			LibSvcNullStrCheck(Config.DiffusionSvc.DiffSvc)
		},
		{
			LibSvcNullStrCheck(Config.VitsSvc.VitsSvc)
		},
		{
			LibSvcNullStrCheck(Config.ReflowSvc.Encoder),
			LibSvcNullStrCheck(Config.ReflowSvc.VelocityFn),
			LibSvcNullStrCheck(Config.ReflowSvc.After)
		},
		{
			Config.Cluster.ClusterCenterSize,
			LibSvcNullStrCheck(Config.Cluster.Path),
			LibSvcNullStrCheck(Config.Cluster.Type)
		},
		Config.SamplingRate,
		Config.HopSize,
		Config.HiddenUnitKDims,
		Config.SpeakerCount,
		(bool)Config.EnableCharaMix,
		(bool)Config.EnableVolume,
		(bool)Config.VaeMode,
		Config.MelBins,
		Config.Pndms,
		Config.MaxStep,
		Config.SpecMin,
		Config.SpecMax,
		Config.Scale
	};
	
	try
	{
		if(_T == 0)
		{
			return new VitsSvc(ModelConfig, _ProgressCallback, static_cast<LibSvcSpace LibSvcModule::ExecutionProviders>(_ExecutionProvider), _DeviceID, _ThreadCount);
		}
		return new UnionSvc(ModelConfig, _ProgressCallback, int(_ExecutionProvider), int(_DeviceID), int(_ThreadCount));
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

INT32 LibSvcUnloadModel(
	UINT32 _T,
	void* _Model
)
{
	try
	{
		if (_T == 0)
			delete (VitsSvc*)_Model;
		else
			delete (UnionSvc*)_Model;
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

void* LibSvcLoadVocoder(LPWSTR VocoderPath)
{
	if (!VocoderPath)
	{
		RaiseError(L"VocoderPath Could Not Be Null");
		return nullptr;
	}

	try
	{
		auto VocoderL = new Ort::Session(*GlobalEnv->GetEnv(), VocoderPath, *GlobalEnv->GetSessionOptions());
		if (VocoderL)
			++GlobalEnvRefCount;
		return VocoderL;
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

INT32 LibSvcUnloadVocoder(void* _Model)
{
	try
	{
		delete (Ort::Session*)_Model;
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	if (_Model)
		--GlobalEnvRefCount;
	return 0;
}

INT32 LibSvcReadAudio(LPWSTR _AudioPath, INT32 _SamplingRate, void* _Output)
{
	try
	{
		*(DragonianLibSTL::Vector<int16_t>*)(_Output) = DragonianLib::AvCodec().DecodeSigned16(
			DragonianLib::WideStringToUTF8(_AudioPath).c_str(),
			_SamplingRate
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

void LibSvcWriteAudioFile(void* _PCMData, LPWSTR _OutputPath, INT32 _SamplingRate)
{
	DragonianLib::WritePCMData(
		_OutputPath,
		*(DragonianLibSTL::Vector<int16_t>*)(_PCMData),
		_SamplingRate
	);
}