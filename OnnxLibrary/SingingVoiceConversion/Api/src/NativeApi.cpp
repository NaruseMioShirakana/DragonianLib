#include "../header/NativeApi.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <mutex>
#include <string>
#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/Base.h"
#include "../../Modules/header/Modules.hpp"
#include "Libraries/F0Extractor/NetF0Predictors.hpp"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"

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

using Config = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space Hparams;
using VitsSvc = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space VitsSvc;
using DiffusionSvc = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space DiffusionSvc;
using ReflowSvc = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space ReflowSvc;
using ClusterBase = DragonianLib::Cluster::BaseCluster;
using TensorExtractorBase = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space TensorExtractor::LibSvcTensorExtractor;
using ProgressCallback = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space LibSvcModule::ProgressCallback;
using ExecutionProvider = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space LibSvcModule::ExecutionProviders;
using Slices = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingleAudio;
using SingleSlice = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingleSlice;
using Params = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space InferenceParams;

using DInt16Vector = DragonianLibSTL::Vector<int16_t>;
using DFloat32Vector = DragonianLibSTL::Vector<float>;
using DDFloat32Vector = DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>;
using DUInt64Vector = DragonianLibSTL::Vector<size_t>;
using MelContainer = std::pair<DragonianLibSTL::Vector<float>, int64_t>;
using DataContainer = Slices;
using OrtModelType = std::shared_ptr<Ort::Session>*;
using SvcModelType = DragonianLib::SingingVoiceConversion::SingingVoiceConversion*;

std::unordered_map<std::wstring, DragonianLib::FunctionTransform::Mel*> MelOperators;

void InitLibSvcHparams(
	LibSvcHparams* _Input
)
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

void InitLibSvcParams(
	LibSvcParams* _Input
)
{
	_Input->NoiseScale = 0.3f;					
	_Input->Seed = 52468;							
	_Input->SpeakerId = 0;					
	_Input->SpkCount = 2;							
	_Input->IndexRate = 0.f;				
	_Input->ClusterRate = 0.f;					
	_Input->DDSPNoiseScale = 0.8f;				
	_Input->Keys = 0.f;									
	_Input->MeanWindowLength = 2;					
	_Input->Pndm = 100;								
	_Input->Step = 1000;							
	_Input->TBegin = 0.f;
	_Input->TEnd = 1.f;
	_Input->Sampler = nullptr;						
	_Input->ReflowSampler = nullptr;					
	_Input->F0Method = nullptr;						
	_Input->VocoderModel = nullptr;
	_Input->VocoderHopSize = 512;
	_Input->VocoderMelBins = 128;
	_Input->VocoderSamplingRate = 44100;
	_Input->F0Bins = 256;	///< F0 bins		
	_Input->F0Max = 1100.0;  ///< F0 max
	_Input->F0Min = 50.0;   ///< F0 min
	_Input->F0ExtractorUserParameter = nullptr;   ///< F0 extractor user parameter
	_Input->__DEBUG__MODE__ = 0;
}

void InitLibSvcF0ExtractorSetting(
	LibSvcF0ExtractorSetting* _Input
)
{
	_Input->HopSize = 480;
	_Input->SamplingRate = 48000;
	_Input->F0Bins = 256;
	_Input->F0Max = 1100.0;
	_Input->F0Min = 50.0;
	_Input->UserParameter = nullptr;
	_Input->ModelPath = LibSvcNullString;
	_Input->Env = nullptr;
}

void InitLibSvcSlicerSettings(
	LibSvcSlicerSettings* _Input
)
{
	_Input->SamplingRate = 48000;
	_Input->Threshold = 30. / 32768.;
	_Input->MinLength = 3.;
	_Input->WindowLength = 2048;
	_Input->HopSize = 512;
}

//FloatVector

float* LibSvcGetFloatVectorData(
	LibSvcFloatVector _Obj
)
{
	auto& Obj = *(DFloat32Vector*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetFloatVectorSize(
	LibSvcFloatVector _Obj
)
{
	auto& Obj = *(DFloat32Vector*)_Obj;
	return Obj.Size();
}

LibSvcFloatVector LibSvcAllocateFloatVector()
{
	return LibSvcFloatVector(new DFloat32Vector);
}

void LibSvcReleaseFloatVector(
	LibSvcFloatVector _Obj
)
{
	delete (DFloat32Vector*)_Obj;
}

void LibSvcInsertFloatVector(LibSvcFloatVector _ObjA, LibSvcFloatVector _ObjB)
{
	auto& ObjA = *(DFloat32Vector*)_ObjA;
	auto& ObjB = *(DFloat32Vector*)_ObjB;
	ObjA.Insert(ObjA.end(), ObjB.begin(), ObjB.end());
}

//DFloatVector

LibSvcFloatVector LibSvcGetDFloatVectorData(
	LibSvcDoubleDimsFloatVector _Obj,
	size_t _Index
)
{
	auto& Obj = *(DDFloat32Vector*)_Obj;
	return LibSvcFloatVector(Obj.Data() + _Index);
}

size_t LibSvcGetDFloatVectorSize(
	LibSvcDoubleDimsFloatVector _Obj
)
{
	auto& Obj = *(DDFloat32Vector*)_Obj;
	return Obj.Size();
}

//Int16Vector

LibSvcInt16Vector LibSvcAllocateInt16Vector()
{
	return LibSvcInt16Vector(new DInt16Vector);
}

void LibSvcReleaseInt16Vector(
	LibSvcInt16Vector _Obj
)
{
	delete (DInt16Vector*)_Obj;
}

void LibSvcSetInt16VectorLength(
	LibSvcInt16Vector _Obj,
	size_t _Size
)
{
	auto& Obj = *(DInt16Vector*)_Obj;
	Obj.Resize(_Size);
}

void LibSvcInsertInt16Vector(
	LibSvcInt16Vector _ObjA,
	LibSvcInt16Vector _ObjB
)
{
	auto& ObjA = *(DInt16Vector*)_ObjA;
	auto& ObjB = *(DInt16Vector*)_ObjB;
	ObjA.Insert(ObjA.end(), ObjB.begin(), ObjB.end());
}

int16_t* LibSvcGetInt16VectorData(
	LibSvcInt16Vector _Obj
)
{
	auto& Obj = *(DInt16Vector*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetInt16VectorSize(
	LibSvcInt16Vector _Obj
)
{
	auto& Obj = *(DInt16Vector*)_Obj;
	return Obj.Size();
}

//UInt64Vector

LibSvcUInt64Vector LibSvcAllocateUInt64Vector()
{
	return LibSvcUInt64Vector(new DUInt64Vector);
}

void LibSvcReleaseUInt64Vector(
	LibSvcUInt64Vector _Obj
)
{
	delete (DUInt64Vector*)_Obj;
}

void LibSvcSetUInt64VectorLength(
	LibSvcUInt64Vector _Obj,
	size_t _Size
)
{
	auto& Obj = *(DUInt64Vector*)_Obj;
	Obj.Resize(_Size);
}

size_t* LibSvcGetUInt64VectorData(
	LibSvcUInt64Vector _Obj
)
{
	auto& Obj = *(DUInt64Vector*)_Obj;
	return Obj.Data();
}

size_t LibSvcGetUInt64VectorSize(
	LibSvcUInt64Vector _Obj
)
{
	auto& Obj = *(DUInt64Vector*)_Obj;
	return Obj.Size();
}

//Mel

LibSvcMelType LibSvcAllocateMel()
{
	return LibSvcMelType(new MelContainer);
}

void LibSvcReleaseMel(
	LibSvcMelType _Obj
)
{
	delete (MelContainer*)_Obj;
}

LibSvcFloatVector LibSvcGetMelData(
	LibSvcMelType _Obj
)
{
	auto& Obj = *(MelContainer*)_Obj;
	return LibSvcFloatVector(&Obj.first);
}

int64_t LibSvcGetMelSize(
	LibSvcMelType _Obj
)
{
	auto& Obj = *(MelContainer*)_Obj;
	return Obj.second;
}

//Slice

LibSvcFloatVector LibSvcGetAudio(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return LibSvcFloatVector(&Obj.Audio);
}

LibSvcFloatVector LibSvcGetF0(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return LibSvcFloatVector(&Obj.F0);
}

LibSvcFloatVector LibSvcGetVolume(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return LibSvcFloatVector(&Obj.Volume);
}

LibSvcDoubleDimsFloatVector LibSvcGetSpeaker(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return LibSvcDoubleDimsFloatVector(&Obj.Speaker);
}

UINT64 LibSvcGetSrcLength(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return Obj.OrgLen;
}

INT32 LibSvcGetIsNotMute(
	LibSvcSliceType _Obj
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	return Obj.IsNotMute;
}

void LibSvcSetSpeakerMixDataSize(
	LibSvcSliceType _Obj,
	size_t _NSpeaker
)
{
	auto& Obj = *(SingleSlice*)_Obj;
	Obj.Speaker.Resize(_NSpeaker, DragonianLibSTL::Vector(Obj.F0.Size(), 0.f));
}

//Array Of Slice - MoeVoiceStudioSvcSlice

LibSvcSlicesType LibSvcAllocateSliceData()
{
	return LibSvcSlicesType(new DataContainer);
}

void LibSvcReleaseSliceData(
	LibSvcSlicesType _Obj
)
{
	delete (DataContainer*)_Obj;
}

BSTR LibSvcGetAudioPath(
	LibSvcSlicesType _Obj
)
{
	auto& Obj = *(DataContainer*)_Obj;
	return SysAllocString(Obj.Path.c_str());
}

LibSvcSliceType LibSvcGetSlice(
	LibSvcSlicesType _Obj,
	size_t _Index
)
{
	auto& Obj = *(DataContainer*)_Obj;
	return LibSvcSliceType(Obj.Slices.Data() + _Index);
}

size_t LibSvcGetSliceCount(
	LibSvcSlicesType _Obj
)
{
	auto& Obj = *(DataContainer*)_Obj;
	return Obj.Slices.Size();
}

/***************************************Fun*******************************************/

void RaiseError(const std::wstring& _Msg)
{
	DragonianLib::LogError(_Msg.c_str());
}

void LibSvcSetGlobalEnvDir(
	LPCWSTR _Dir
)
{
	DragonianLib::SetGlobalEnvDir(_Dir);
}

void LibSvcSetLoggerId(
	LPCWSTR _Id
)
{
	DragonianLib::SetLoggerId(_Id);
}

void LibSvcSetLoggerLevel(
	INT32 _Level
)
{
	SetLoggerLevel(static_cast<DragonianLib::LogLevel>(_Level));
}

void LibSvcSetLoggerFunction(
	LibSvcLoggerFunction _Logger
)
{
	DragonianLib::SetLoggerFunction(_Logger);
}

void LibSvcInit()
{
	
}

void LibSvcFreeString(BSTR _String)
{
	SysFreeString(_String);
}

LibSvcEnv LibSvcCreateEnv(
	UINT32 ThreadCount,
	UINT32 DeviceID,
	UINT32 Provider
)
{
	try
	{
		return LibSvcEnv(DragonianLib::DragonianLibOrtEnv::CreateEnv(ThreadCount, DeviceID, Provider).get());
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void LibSvcDestoryEnv(
	LibSvcEnv Env
)
{
	auto& _MyEnv = *(DragonianLib::DragonianLibOrtEnv*)Env;
	DragonianLib::DragonianLibOrtEnv::DestroyEnv(_MyEnv.GetCurThreadCount(), _MyEnv.GetCurDeviceID(), _MyEnv.GetCurProvider());
}

INT32 LibSvcSliceAudioI16(
	LibSvcCInt16Vector _Audio,
	const LibSvcSlicerSettings* _Setting,
	LibSvcUInt64Vector _Output
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

	auto& Ret = *(DUInt64Vector*)(_Output);
	_D_Dragonian_Lib_Lib_Av_Codec_Space SlicerSettings SliSetting{
		_Setting->SamplingRate,
		_Setting->Threshold,
		_Setting->MinLength,
		_Setting->WindowLength,
		_Setting->HopSize
	};

	try
	{
		Ret = _D_Dragonian_Lib_Lib_Av_Codec_Space SliceAudio(
			InterpResample(
				*(const DInt16Vector*)(_Audio),
				1,
				1,
				32768.f
			),
			SliSetting
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 LibSvcSliceAudio(
	LibSvcCFloatVector _Audio,
	const LibSvcSlicerSettings* _Setting,
	LibSvcUInt64Vector _Output
)
{
	if (!_Audio)
	{
		RaiseError(L"Audio Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		RaiseError(L"Output Could Not Be Null!");
		return 1;
	}

	auto& Ret = *(DUInt64Vector*)(_Output);

	_D_Dragonian_Lib_Lib_Av_Codec_Space SlicerSettings SliSetting{
		_Setting->SamplingRate,
		_Setting->Threshold,
		_Setting->MinLength,
		_Setting->WindowLength,
		_Setting->HopSize
	};

	try
	{
		Ret = _D_Dragonian_Lib_Lib_Av_Codec_Space SliceAudio(
			*(const DFloat32Vector*)(_Audio),
			SliSetting
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 LibSvcPreprocessI16(
	LibSvcCInt16Vector _Audio,
	LibSvcCUInt64Vector _SlicePos,
	const LibSvcF0ExtractorSetting* _Settings,
	double _Threshold,
	const wchar_t* _F0Method,
	LibSvcSlicesType _Output
)
{
	_D_Dragonian_Lib_Lib_Av_Codec_Space SlicerSettings _Setting{
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

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	Slices& Ret = *(Slices*)_Output;
	try
	{
		Ret = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::GetAudioSlice(
			InterpResample(
				*(const DInt16Vector*)(_Audio),
				1,
				1,
				32768.f
			),
			*(const DUInt64Vector*)(_SlicePos),
			_Setting.SamplingRate,
			_Setting.Threshold
		);

		_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::PreProcessAudio(
			Ret,
			{
				_Settings->SamplingRate,
				_Settings->HopSize,
				_Settings->F0Bins,
				_Settings->F0Max,
				_Settings->F0Min,
				_Settings->UserParameter,
			},
			_F0Method,
			{
				_Settings->ModelPath,
				_Settings->Env
			}
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 LibSvcPreprocess(
	LibSvcCFloatVector _Audio,
	LibSvcCUInt64Vector _SlicePos,
	const LibSvcF0ExtractorSetting* _Settings,
	double _Threshold,
	const wchar_t* _F0Method,
	LibSvcSlicesType _Output
)
{
	_D_Dragonian_Lib_Lib_Av_Codec_Space SlicerSettings _Setting{
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

	if (!_Output)
	{
		RaiseError(L"_Output Could Not Be Null!");
		return 1;
	}

	Slices& Ret = *(Slices*)_Output;
	try
	{
		Ret = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::GetAudioSlice(
			*(const DFloat32Vector*)(_Audio),
			*(const DUInt64Vector*)(_SlicePos),
			_Setting.SamplingRate,
			_Setting.Threshold
		);

		_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::PreProcessAudio(
			Ret,
			{
				_Settings->SamplingRate,
				_Settings->HopSize,
				_Settings->F0Bins,
				_Settings->F0Max,
				_Settings->F0Min,
				_Settings->UserParameter,
			},
			_F0Method,
			{
				_Settings->ModelPath,
				_Settings->Env
			}
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 LibSvcStftI16(
	LibSvcCInt16Vector _Audio,
	INT32 _SamplingRate,
	INT32 _Hopsize,
	INT32 _MelBins,
	LibSvcMelType _Output
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
			MelOperators[_Name] = new DragonianLib::FunctionTransform::Mel(_Hopsize * 4, _Hopsize, _SamplingRate, _MelBins);
		auto _NormalizedAudio = InterpResample(
			*(const DInt16Vector*)_Audio,
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

INT32 LibSvcStft(
	LibSvcCFloatVector _Audio,
	INT32 _SamplingRate,
	INT32 _Hopsize,
	INT32 _MelBins,
	LibSvcMelType _Output
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
			MelOperators[_Name] = new DragonianLib::FunctionTransform::Mel(_Hopsize * 4, _Hopsize, _SamplingRate, _MelBins);
		auto _NormalizedAudio = InterpResample<double>(
			*(const DFloat32Vector*)_Audio,
			_SamplingRate,
			_SamplingRate
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
	LibSvcModel _Model,
	LibSvcCSliceType _Slice,
	const LibSvcParams* _InferParams,
	size_t* _Process,
	LibSvcFloatVector _Output
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

	const Params Param
	{
		_InferParams->NoiseScale,
		_InferParams->Seed,
		_InferParams->SpeakerId,
		_InferParams->SpkCount,
		_InferParams->IndexRate,
		_InferParams->ClusterRate,
		_InferParams->DDSPNoiseScale,
		_InferParams->Keys,
		_InferParams->MeanWindowLength,
		_InferParams->Pndm,
		_InferParams->Step,
		_InferParams->TBegin,
		_InferParams->TEnd,
		LibSvcNullStrCheck(_InferParams->Sampler),
		LibSvcNullStrCheck(_InferParams->ReflowSampler),
		LibSvcNullStrCheck(_InferParams->F0Method),
		*OrtModelType(_InferParams->VocoderModel),
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter,
		{},
		{},
		{},
		{}
	};

	try
	{
		*(DFloat32Vector*)(_Output) = (SvcModelType(_Model))->SliceInference(
			*(const SingleSlice*)(_Slice),
			Param,
			*_Process
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcInferAudio(
	LibSvcModel _Model,
	LibSvcSlicesType _Audio,
	const LibSvcParams* _InferParams,
	UINT64 _SrcLength,
	size_t* _Process,
	LibSvcFloatVector _Output
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

	const Params Param
	{
		_InferParams->NoiseScale,
		_InferParams->Seed,
		_InferParams->SpeakerId,
		_InferParams->SpkCount,
		_InferParams->IndexRate,
		_InferParams->ClusterRate,
		_InferParams->DDSPNoiseScale,
		_InferParams->Keys,
		_InferParams->MeanWindowLength,
		_InferParams->Pndm,
		_InferParams->Step,
		_InferParams->TBegin,
		_InferParams->TEnd,
		LibSvcNullStrCheck(_InferParams->Sampler),
		LibSvcNullStrCheck(_InferParams->ReflowSampler),
		LibSvcNullStrCheck(_InferParams->F0Method),
		*OrtModelType(_InferParams->VocoderModel),
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter,
		{},
		{},
		{},
		{}
	};

	auto __Slices = *(const Slices*)(_Audio);

	auto& OutPutAudio = *(DFloat32Vector*)(_Output);
	OutPutAudio.Reserve(_SrcLength);
	try
	{
		for (const auto& Single : __Slices.Slices)
		{
			auto Out = SvcModelType(_Model)->SliceInference(Single, Param, *_Process);
			OutPutAudio.Insert(OutPutAudio.end(), Out.begin(), Out.end());
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
	LibSvcModel _Model,
	LibSvcCFloatVector _PCMData,
	const LibSvcParams* _InferParams,
	INT32 SamplingRate,
	LibSvcFloatVector _Output
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

	const Params Param
	{
		_InferParams->NoiseScale,
		_InferParams->Seed,
		_InferParams->SpeakerId,
		_InferParams->SpkCount,
		_InferParams->IndexRate,
		_InferParams->ClusterRate,
		_InferParams->DDSPNoiseScale,
		_InferParams->Keys,
		_InferParams->MeanWindowLength,
		_InferParams->Pndm,
		_InferParams->Step,
		_InferParams->TBegin,
		_InferParams->TEnd,
		LibSvcNullStrCheck(_InferParams->Sampler),
		LibSvcNullStrCheck(_InferParams->ReflowSampler),
		LibSvcNullStrCheck(_InferParams->F0Method),
		*OrtModelType(_InferParams->VocoderModel),
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter,
		{},
		{},
		{},
		{}
	};

	auto& InputData = *(const DFloat32Vector*)(_PCMData);

	try
	{
		*(DFloat32Vector*)(_Output) = SvcModelType(_Model)->InferPCMData(
			InputData,
			SamplingRate,
			Param
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 LibSvcShallowDiffusionInference(
	LibSvcModel _Model,
	LibSvcFloatVector _16KAudioHubert,
	LibSvcMelType _Mel,
	LibSvcCFloatVector _SrcF0,
	LibSvcCFloatVector _SrcVolume,
	LibSvcCDoubleDimsFloatVector _SrcSpeakerMap,
	INT64 _SrcSize,
	const LibSvcParams* _InferParams,
	size_t* _Process,
	LibSvcFloatVector _Output
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

	const Params Param
	{
		_InferParams->NoiseScale,
		_InferParams->Seed,
		_InferParams->SpeakerId,
		_InferParams->SpkCount,
		_InferParams->IndexRate,
		_InferParams->ClusterRate,
		_InferParams->DDSPNoiseScale,
		_InferParams->Keys,
		_InferParams->MeanWindowLength,
		_InferParams->Pndm,
		_InferParams->Step,
		_InferParams->TBegin,
		_InferParams->TEnd,
		LibSvcNullStrCheck(_InferParams->Sampler),
		LibSvcNullStrCheck(_InferParams->ReflowSampler),
		LibSvcNullStrCheck(_InferParams->F0Method),
		*OrtModelType(_InferParams->VocoderModel),
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter,
		{},
		{},
		{},
		{}
	};

	try
	{
		*(DFloat32Vector*)(_Output) = SvcModelType(_Model)->ShallowDiffusionInference(
			*(DFloat32Vector*)_16KAudioHubert,
			Param,
			*(MelContainer*)(_Mel),
			*(const DFloat32Vector*)(_SrcF0),
			*(const DFloat32Vector*)(_SrcVolume),
			*(const DDFloat32Vector*)(_SrcSpeakerMap),
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
	LibSvcVocoderModel _Model,
	LibSvcEnv _Env,
	LibSvcMelType _Mel,
	LibSvcCFloatVector _F0,
	INT32 _VocoderMelBins,
	LibSvcFloatVector _Output
)
{
	if (!_Model)
	{
		RaiseError(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Env)
	{
		RaiseError(L"_Env Could Not Be Null!");
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

	auto Rf0 = *(const DFloat32Vector*)(_F0);
	auto& MelTemp = *(MelContainer*)(_Mel);
	if (Rf0.Size() != (size_t)MelTemp.second)
		Rf0 = InterpFunc(Rf0, (long)Rf0.Size(), (long)MelTemp.second);
	try
	{
		*(DFloat32Vector*)(_Output) = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space VocoderInfer(
			MelTemp.first,
			Rf0,
			_VocoderMelBins,
			MelTemp.second,
			((DragonianLib::DragonianLibOrtEnv*)_Env)->GetMemoryInfo(),
			*(OrtModelType)_Model
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

LibSvcModel LibSvcLoadModel(
	UINT32 _T,
	const LibSvcHparams* _Config,
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

	//printf("%lld", (long long)(Config.DiffusionSvc.Encoder));

	_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space Hparams ModelConfig{
		LibSvcNullStrCheck(_Config->TensorExtractor),
		LibSvcNullStrCheck(_Config->HubertPath),
		{
			LibSvcNullStrCheck(_Config->DiffusionSvc.Encoder),
			LibSvcNullStrCheck(_Config->DiffusionSvc.Denoise),
			LibSvcNullStrCheck(_Config->DiffusionSvc.Pred),
			LibSvcNullStrCheck(_Config->DiffusionSvc.After),
			LibSvcNullStrCheck(_Config->DiffusionSvc.Alpha),
			LibSvcNullStrCheck(_Config->DiffusionSvc.Naive),
			LibSvcNullStrCheck(_Config->DiffusionSvc.DiffSvc)
		},
		{
			LibSvcNullStrCheck(_Config->VitsSvc.VitsSvc)
		},
		{
			LibSvcNullStrCheck(_Config->ReflowSvc.Encoder),
			LibSvcNullStrCheck(_Config->ReflowSvc.VelocityFn),
			LibSvcNullStrCheck(_Config->ReflowSvc.After)
		},
		{
			_Config->Cluster.ClusterCenterSize,
			LibSvcNullStrCheck(_Config->Cluster.Path),
			LibSvcNullStrCheck(_Config->Cluster.Type)
		},
		_Config->SamplingRate,
		_Config->HopSize,
		_Config->HiddenUnitKDims,
		_Config->SpeakerCount,
		(bool)_Config->EnableCharaMix,
		(bool)_Config->EnableVolume,
		(bool)_Config->VaeMode,
		_Config->MelBins,
		_Config->Pndms,
		_Config->MaxStep,
		_Config->SpecMin,
		_Config->SpecMax,
		_Config->Scale
	};

	try
	{
		if (_T == 0)
			return LibSvcModel(new VitsSvc(ModelConfig, _ProgressCallback, static_cast<_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space LibSvcModule::ExecutionProviders>(_ExecutionProvider), _DeviceID, _ThreadCount));
		if (_T == 1)
			return LibSvcModel(new DiffusionSvc(ModelConfig, _ProgressCallback, static_cast<_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space LibSvcModule::ExecutionProviders>(_ExecutionProvider), _DeviceID, _ThreadCount));
		if (_T == 2)
			return LibSvcModel(new ReflowSvc(ModelConfig, _ProgressCallback, static_cast<_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space LibSvcModule::ExecutionProviders>(_ExecutionProvider), _DeviceID, _ThreadCount));
		return nullptr;
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

INT32 LibSvcUnloadModel(
	LibSvcModel _Model
)
{
	try
	{
		delete (SvcModelType)_Model;
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

LibSvcVocoderModel LibSvcLoadVocoder(
	LPCWSTR VocoderPath,
	LibSvcEnv _Env
)
{
	if (!VocoderPath)
	{
		RaiseError(L"VocoderPath Could Not Be Null");
		return nullptr;
	}

	if (!_Env)
	{
		RaiseError(L"_Env Could Not Be Null");
		return nullptr;
	}

	try
	{
		return LibSvcVocoderModel(
			&DragonianLib::DragonianLibOrtEnv::RefOrtCachedModel(
				VocoderPath,
				*(DragonianLib::DragonianLibOrtEnv*)_Env
			)
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

INT32 LibSvcUnloadCachedModel(
	LPCWSTR ModelPath,
	LibSvcEnv _Env
)
{
	if (!ModelPath)
	{
		RaiseError(L"ModelPath Could Not Be Null");
		return 1;
	}

	if (!_Env)
	{
		RaiseError(L"_Env Could Not Be Null");
		return 1;
	}

	try
	{
		DragonianLib::DragonianLibOrtEnv::UnRefOrtCachedModel(
			ModelPath,
			*(DragonianLib::DragonianLibOrtEnv*)_Env
		);
	}
	catch (std::exception& e)
	{
		RaiseError(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

void LibSvcClearCachedModel()
{
	DragonianLib::DragonianLibOrtEnv::ClearModelCache();
}

INT32 LibSvcReadAudio(
	LPCWSTR _AudioPath,
	INT32 _SamplingRate,
	LibSvcFloatVector _Output
)
{
	try
	{
		*(DFloat32Vector*)(_Output) = _D_Dragonian_Lib_Lib_Av_Codec_Space AvCodec().DecodeFloat(
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

void LibSvcWriteAudioFile(
	LibSvcFloatVector _PCMData,
	LPCWSTR _OutputPath,
	INT32 _SamplingRate
)
{
	_D_Dragonian_Lib_Lib_Av_Codec_Space WritePCMData(
		_OutputPath,
		*(DFloat32Vector*)(_PCMData),
		_SamplingRate
	);
}
