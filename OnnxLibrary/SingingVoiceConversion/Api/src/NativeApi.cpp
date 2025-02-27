#include <mutex>
#include <string>

#include "Libraries/Base.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/Util/StringPreprocess.h"

#include "../header/NativeApi.h"
#include "../../Modules/header/Modules.hpp"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

const wchar_t* _GDragonianLibSvcNullString = L"";
std::wstring _GDragonianLibSvcLastError = L"";

#define _GDragonianLibSvcNullStrCheck(Str) ((Str)?(Str):(_GDragonianLibSvcNullString))

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

void _Dragonian_Lib_Svc_Add_Prefix(InitHparams)(
	_Dragonian_Lib_Svc_Add_Prefix(Hparams)* _Input
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
	_Input->F0Min = 50.f;
	_Input->F0Max = 1100.f;
	_Input->Scale = 1000.f;
}

void _Dragonian_Lib_Svc_Add_Prefix(InitInferenceParams)(
	_Dragonian_Lib_Svc_Add_Prefix(Params)* _Input
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

void _Dragonian_Lib_Svc_Add_Prefix(InitF0ExtractorSetting)(
	_Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Input
	)
{
	_Input->HopSize = 480;
	_Input->SamplingRate = 48000;
	_Input->F0Bins = 256;
	_Input->F0Max = 1100.0;
	_Input->F0Min = 50.0;
	_Input->UserParameter = nullptr;
	_Input->ModelPath = _GDragonianLibSvcNullString;
	_Input->Env = nullptr;
}

void _Dragonian_Lib_Svc_Add_Prefix(InitSlicerSettings)(
	_Dragonian_Lib_Svc_Add_Prefix(SlicerSettings)* _Input
	)
{
	_Input->SamplingRate = 48000;
	_Input->Threshold = -60.;
	_Input->MinLength = 5.;
	_Input->WindowLength = 8192;
	_Input->HopSize = 1024;
}

/***************************************Fun*******************************************/

void _Dragonian_Lib_Svc_Add_Prefix(RaiseError)(const std::wstring& _Msg)
{
	DragonianLib::LogError(_Msg.c_str());
	_GDragonianLibSvcLastError = _Msg;
}

BSTR _Dragonian_Lib_Svc_Add_Prefix(GetLastError)()
{
	return SysAllocString(_GDragonianLibSvcLastError.c_str());
}

void _Dragonian_Lib_Svc_Add_Prefix(SetGlobalEnvDir)(
	LPCWSTR _Dir
	)
{
	DragonianLib::SetGlobalEnvDir(_Dir);
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerId)(
	LPCWSTR _Id
	)
{
	DragonianLib::SetLoggerId(_Id);
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerLevel)(
	INT32 _Level
	)
{
	SetLoggerLevel(static_cast<DragonianLib::LogLevel>(_Level));
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerFunction)(
	_Dragonian_Lib_Svc_Add_Prefix(LoggerFunction) _Logger
	)
{
	DragonianLib::SetLoggerFunction(_Logger);
}

void _Dragonian_Lib_Svc_Add_Prefix(Init)()
{

}

void _Dragonian_Lib_Svc_Add_Prefix(FreeString)(
	BSTR _String
	)
{
	SysFreeString(_String);
}

void _Dragonian_Lib_Svc_Add_Prefix(FreeData)(
	void* _Ptr
	)
{
	DragonianLib::TemplateLibrary::CPUAllocator::deallocate(_Ptr);
}

_Dragonian_Lib_Svc_Add_Prefix(Env) _Dragonian_Lib_Svc_Add_Prefix(CreateEnv)(
	UINT32 _ThreadCount,
	UINT32 _DeviceID,
	_Dragonian_Lib_Svc_Add_Prefix(ExecutionProvider) _Provider
)
{
	try
	{
		return _Dragonian_Lib_Svc_Add_Prefix(Env)(&DragonianLib::DragonianLibOrtEnv::CreateEnv(_ThreadCount, _DeviceID, _Provider));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(DestoryEnv)(
	_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
	)
{
	auto& _MyEnv = *(std::shared_ptr<DragonianLib::DragonianLibOrtEnv>*)_Env;
	DragonianLib::DragonianLibOrtEnv::DestroyEnv(_MyEnv->GetCurThreadCount(), _MyEnv->GetCurDeviceID(), _MyEnv->GetCurProvider());
}

_Dragonian_Lib_Svc_Add_Prefix(Model) _Dragonian_Lib_Svc_Add_Prefix(LoadModel)(
	const _Dragonian_Lib_Svc_Add_Prefix(Hparams)* _Config,
	_Dragonian_Lib_Svc_Add_Prefix(Env) _Env,
	_Dragonian_Lib_Svc_Add_Prefix(ProgressCallback) _ProgressCallback
	)
{
	auto _MyEnv = *(std::shared_ptr<DragonianLib::DragonianLibOrtEnv>*)_Env;

	if (!_Config)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return nullptr;
	}

	_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space Hparams ModelConfig{
		_GDragonianLibSvcNullStrCheck(_Config->TensorExtractor),
		_GDragonianLibSvcNullStrCheck(_Config->HubertPath),
		{
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.Encoder),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.Denoise),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.Pred),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.After),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.Alpha),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.Naive),
			_GDragonianLibSvcNullStrCheck(_Config->DiffusionSvc.DiffSvc)
		},
		{
			_GDragonianLibSvcNullStrCheck(_Config->VitsSvc.VitsSvc)
		},
		{
			_GDragonianLibSvcNullStrCheck(_Config->ReflowSvc.Encoder),
			_GDragonianLibSvcNullStrCheck(_Config->ReflowSvc.VelocityFn),
			_GDragonianLibSvcNullStrCheck(_Config->ReflowSvc.After)
		},
		{
			_Config->Cluster.ClusterCenterSize,
			_GDragonianLibSvcNullStrCheck(_Config->Cluster.Path),
			_GDragonianLibSvcNullStrCheck(_Config->Cluster.Type)
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
		_Config->F0Min,
		_Config->F0Max,
		_Config->Scale
	};

	try
	{
		if (!ModelConfig.VitsSvc.VitsSvc.empty())
			return _Dragonian_Lib_Svc_Add_Prefix(Model)(new DragonianLib::SingingVoiceConversion::VitsSvc(ModelConfig, _ProgressCallback, _MyEnv));
		if (!ModelConfig.DiffusionSvc.Encoder.empty() || !ModelConfig.DiffusionSvc.DiffSvc.empty())
			return _Dragonian_Lib_Svc_Add_Prefix(Model)(new DragonianLib::SingingVoiceConversion::DiffusionSvc(ModelConfig, _ProgressCallback, _MyEnv));
		if (!ModelConfig.ReflowSvc.Encoder.empty())
			return _Dragonian_Lib_Svc_Add_Prefix(Model)(new DragonianLib::SingingVoiceConversion::ReflowSvc(ModelConfig, _ProgressCallback, _MyEnv));
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Model not recognized!");
		return nullptr;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

_Dragonian_Lib_Svc_Add_Prefix(VocoderModel) _Dragonian_Lib_Svc_Add_Prefix(LoadVocoder)(
	LPCWSTR VocoderPath,
	_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
	)
{
	if (!VocoderPath)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"VocoderPath Could Not Be Null");
		return nullptr;
	}

	if (!_Env)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Env Could Not Be Null");
		return nullptr;
	}

	try
	{
		return _Dragonian_Lib_Svc_Add_Prefix(VocoderModel)(
			&DragonianLib::DragonianLibOrtEnv::RefOrtCachedModel(
				VocoderPath,
				**(std::shared_ptr<DragonianLib::DragonianLibOrtEnv>*)_Env
			)
			);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(UnloadModel)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model
	)
{
	try
	{
		delete (DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(UnloadCachedModel)(
	LPCWSTR ModelPath,
	_Dragonian_Lib_Svc_Add_Prefix(Env) _Env
	)
{
	if (!ModelPath)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"ModelPath Could Not Be Null");
		return 1;
	}

	if (!_Env)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Env Could Not Be Null");
		return 1;
	}

	try
	{
		DragonianLib::DragonianLibOrtEnv::UnRefOrtCachedModel(
			ModelPath,
			**(std::shared_ptr<DragonianLib::DragonianLibOrtEnv>*)_Env
		);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

void _Dragonian_Lib_Svc_Add_Prefix(ClearCachedModel)()
{
	DragonianLib::DragonianLibOrtEnv::ClearModelCache();
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(SliceAudio)(
	const _Dragonian_Lib_Svc_Add_Prefix(SlicerSettings)* _Setting,
	const float* _Audio,
	size_t _AudioSize,
	size_t** _SlicePos,
	size_t* _SlicePosSize
	)
{
	if (!_Setting)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Setting Could Not Be Null!");
		return 1;
	}

	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Null!");
		return 1;
	}

	if (_AudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Empty!");
		return 1;
	}

	if (!_SlicePos)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"SlicePos Could Not Be Null!");
		return 1;
	}

	if (!_SlicePosSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"SlicePosSize Could Not Be Null!");
		return 1;
	}

	_D_Dragonian_Lib_Lib_Av_Codec_Space SlicerSettings SliSetting{
		_Setting->SamplingRate,
		_Setting->Threshold,
		_Setting->MinLength,
		_Setting->WindowLength,
		_Setting->HopSize
	};

	try
	{
		auto [_MyPos, _MySize] = _D_Dragonian_Lib_Lib_Av_Codec_Space SliceAudio(
			DragonianLib::TemplateLibrary::Ranges(_Audio, _Audio + _AudioSize),
			SliSetting
		).Release();
		*_SlicePos = _MyPos;
		*_SlicePosSize = _MySize;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(PreprocessInferenceData)(
	const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Settings,
	const float* _Audio,
	size_t _AudioSize,
	const size_t* _SlicePos,
	size_t _SlicePosSize,
	double _DbThreshold,
	const wchar_t* _F0Method,
	_Dragonian_Lib_Svc_Add_Prefix(InferenceData)* _OutputSlices,
	size_t* _OutputSlicesSize
	)
{
	if (!_Settings)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Settings Could Not Be Null!");
		return 1;
	}

	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Null!");
		return 1;
	}

	if (_AudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Empty!");
		return 1;
	}

	if (!_SlicePos)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Slice Pos Could Not Be Null!");
		return 1;
	}

	if (_SlicePosSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Slice Pos Could Not Be Empty!");
		return 1;
	}

	if (!_F0Method)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"F0 Method Could Not Be Null!");
		return 1;
	}

	if (!_OutputSlices)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Output Slices Could Not Be Null!");
		return 1;
	}

	if (!_OutputSlicesSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Output Slices Size Could Not Be Null!");
		return 1;
	}

	*_OutputSlices = (_Dragonian_Lib_Svc_Add_Prefix(InferenceData))(new DragonianLib::SingingVoiceConversion::SingleAudio());
	auto& Ret = *(DragonianLib::SingingVoiceConversion::SingleAudio*)(*_OutputSlices);
	try
	{
		Ret = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space SingingVoiceConversion::GetAudioSlice(
			DragonianLib::TemplateLibrary::Ranges(_Audio, _Audio + _AudioSize),
			DragonianLib::TemplateLibrary::Ranges(_SlicePos, _SlicePos + _SlicePosSize),
			_Settings->SamplingRate,
			_DbThreshold
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
			_GDragonianLibSvcNullStrCheck(_F0Method),
			{
				_GDragonianLibSvcNullStrCheck(_Settings->ModelPath),
				_Settings->Env
			}
		);

		*_OutputSlicesSize = Ret.Slices.Size();
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(ReleaseInferenceData)(
	_Dragonian_Lib_Svc_Add_Prefix(InferenceData) _Data
	)
{
	try
	{
		delete (DragonianLib::SingingVoiceConversion::SingleAudio*)_Data;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

_Dragonian_Lib_Svc_Add_Prefix(Slice) _Dragonian_Lib_Svc_Add_Prefix(GetSlice)(
	_Dragonian_Lib_Svc_Add_Prefix(InferenceData) _Data,
	size_t _Index,
	size_t* _NumFrames
	)
{
	auto& _MyData = *(DragonianLib::SingingVoiceConversion::SingleAudio*)_Data;
	if (_Index >= _MyData.Slices.Size())
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Index Out Of Range!");
		return nullptr;
	}
	*_NumFrames = _MyData.Slices[_Index].F0.Size();
	return (_Dragonian_Lib_Svc_Add_Prefix(Slice))(&_MyData.Slices[_Index]);
}

float* _Dragonian_Lib_Svc_Add_Prefix(GetAudio)(
	_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice,
	size_t* _AudioSize
	)
{
	auto& _MySlice = *(DragonianLib::SingingVoiceConversion::SingleSlice*)_Slice;
	*_AudioSize = _MySlice.Audio.Size();
	return _MySlice.Audio.Data();
}

float* _Dragonian_Lib_Svc_Add_Prefix(GetF0)(
	_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
	)
{
	auto& _MySlice = *(DragonianLib::SingingVoiceConversion::SingleSlice*)_Slice;
	return _MySlice.F0.Data();
}

float* _Dragonian_Lib_Svc_Add_Prefix(GetVolume)(
	_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
	)
{
	auto& _MySlice = *(DragonianLib::SingingVoiceConversion::SingleSlice*)_Slice;
	return _MySlice.Volume.Data();
}

_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Dragonian_Lib_Svc_Add_Prefix(GetSpeaker)(
	_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice
	)
{
	return (_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData))(&((DragonianLib::SingingVoiceConversion::SingleSlice*)_Slice)->Speaker);
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(ReshapeSpeakerMixData)(
	_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Speaker,
	size_t _SpeakerCount,
	size_t _NumFrame
	)
{
	auto& _MySpeaker = *(decltype(DragonianLib::SingingVoiceConversion::SingleSlice::Speaker)*)_Speaker;
	try
	{
		_MySpeaker.Resize(_SpeakerCount);
		for (auto& _MyData : _MySpeaker)
			_MyData.Resize(_NumFrame);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

float* _Dragonian_Lib_Svc_Add_Prefix(GetSpeakerData)(
	_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _Speaker,
	size_t _Index
	)
{
	auto& _MySpeaker = *(decltype(DragonianLib::SingingVoiceConversion::SingleSlice::Speaker)*)_Speaker;
	return _MySpeaker[_Index].Data();
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(Stft)(
	const float* _Audio,
	size_t _AudioSize,
	INT32 _SamplingRate,
	INT32 _Hopsize,
	INT32 _MelBins,
	float** _OutputMel,
	size_t* _OutputMelSize
	)
{
	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Null!");
		return 1;
	}

	if (_AudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"Audio Could Not Be Empty!");
		return 1;
	}

	if (!_OutputMel)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"OutputMel Could Not Be Null!");
		return 1;
	}

	if (!_OutputMelSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"OutputMelSize Could Not Be Null!");
		return 1;
	}

	try
	{
		DragonianLib::TemplateLibrary::Vector<double> _InputAudio;
		_InputAudio.Resize(_AudioSize);
		for (size_t i = 0; i < _AudioSize; ++i)
			_InputAudio[i] = _Audio[i];
		auto [_MyMel, _MyMelSize] = DragonianLib::SingingVoiceConversion::GetMelOperator(_SamplingRate, _Hopsize, _MelBins)(_InputAudio);
		*_OutputMelSize = _MyMel.Size();
		*_OutputMel = _MyMel.Release().first;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(InferSlice)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
	const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
	_Dragonian_Lib_Svc_Add_Prefix(Slice) _Slice,
	size_t* _Process,
	float** _Output,
	size_t* _OutputSize
	)
{
	
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Slice)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Slice Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Process)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Process Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Output Could Not Be Null!");
		return 1;
	}

	if (!_OutputSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_OutputSize Could Not Be Null!");
		return 1;
	}

	const DragonianLib::SingingVoiceConversion::InferenceParams Param
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
		_GDragonianLibSvcNullStrCheck(_InferParams->Sampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->ReflowSampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->F0Method),
		*(std::shared_ptr<Ort::Session>*)_InferParams->VocoderModel,
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter
	};

	try
	{
		auto [_MyData, _MySize] = DragonianLib::TemplateLibrary::InterpResample<float>(
			((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->SliceInference(
				*(const DragonianLib::SingingVoiceConversion::SingleSlice*)(_Slice),
				Param,
				*_Process
			),
			((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->GetSamplingRate(),
			static_cast<long>(((const DragonianLib::SingingVoiceConversion::SingleSlice*)_Slice)->SamplingRate)
		).Release();
		*_Output = _MyData;
		*_OutputSize = _MySize;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(InferAudio)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
	const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
	const float* _Audio,
	size_t _AudioSize,
	long _AudioSamplingRate,
	const wchar_t* _F0Method,
	const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorSetting)* _Settings,
	float _SliceTime,
	float _CrossFadeTime,
	double _DbThreshold,
	float** _Output,
	size_t* _OutputSize
	)
{
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null!");
		return 1;
	}

	if (_AudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Empty!");
		return 1;
	}

	if (!_InferParams)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_F0Method)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_F0Method Could Not Be Null!");
		return 1;
	}

	if (!_Settings)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Settings Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Output Could Not Be Null!");
		return 1;
	}

	if (!_OutputSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_OutputSize Could Not Be Null!");
		return 1;
	}

	const DragonianLib::SingingVoiceConversion::InferenceParams Param
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
		_GDragonianLibSvcNullStrCheck(_InferParams->Sampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->ReflowSampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->F0Method),
		*(std::shared_ptr<Ort::Session>*)_InferParams->VocoderModel,
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter
	};

	auto [OutPutData, OutPutSize] = ((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->InferenceWithCrossFade(
		DragonianLib::TemplateLibrary::Ranges(_Audio, _Audio + _AudioSize),
		_AudioSamplingRate,
		Param,
		{ _CrossFadeTime, _SliceTime },
		{ _AudioSamplingRate, _Settings->HopSize, _Settings->F0Bins, _Settings->F0Max, _Settings->F0Min, _Settings->UserParameter },
		_GDragonianLibSvcNullStrCheck(_F0Method),
		{ _GDragonianLibSvcNullStrCheck(_Settings->ModelPath), _Settings->Env },
		_DbThreshold
	).Release();
	*_Output = OutPutData;
	*_OutputSize = OutPutSize;
	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(InferPCMData)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
	const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
	const float* _Audio,
	size_t _AudioSize,
	INT32 _InputSamplingRate,
	float** _Output,
	size_t* _OutputSize
	)
{
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null!");
		return 1;
	}

	if (_AudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Empty!");
		return 1;
	}

	if (!_InferParams)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_InferParams Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Output Could Not Be Null!");
		return 1;
	}

	if (!_OutputSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_OutputSize Could Not Be Null!");
		return 1;
	}

	const DragonianLib::SingingVoiceConversion::InferenceParams Param
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
		_GDragonianLibSvcNullStrCheck(_InferParams->Sampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->ReflowSampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->F0Method),
		*(std::shared_ptr<Ort::Session>*)_InferParams->VocoderModel,
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter
	};

	try
	{
		auto InputData = DragonianLib::TemplateLibrary::Vector<float>(_Audio, _Audio + _AudioSize);
		auto [_MyData, _MySize] = DragonianLib::TemplateLibrary::InterpResample<float>(
			((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->InferPCMData(
				InputData,
				_InputSamplingRate,
				Param
			),
			((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->GetSamplingRate(),
			_InputSamplingRate
		).Release();
		*_Output = _MyData;
		*_OutputSize = _MySize;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(ShallowDiffusionInference)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
	const _Dragonian_Lib_Svc_Add_Prefix(Params)* _InferParams,
	const float* _16KAudioHubert,
	size_t _16KAudioSize,
	const float* _Mel,
	size_t _SrcMelSize,
	size_t _MelSize,
	const float* _SrcF0,
	size_t _SrcF0Size,
	const float* _SrcVolume,
	size_t _SrcVolumeSize,
	_Dragonian_Lib_Svc_Add_Prefix(SpeakerMixData) _SrcSpeakerMap,
	size_t* _Process,
	float** _Output,
	size_t* _OutputSize
	)
{
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_16KAudioHubert)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_16KAudioHubert Could Not Be Null!");
		return 1;
	}

	if (_16KAudioSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_16KAudioHubert Could Not Be Empty!");
		return 1;
	}

	if (!_Mel)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Mel Could Not Be Null!");
		return 1;
	}

	if (_SrcMelSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Mel Could Not Be Empty!");
		return 1;
	}

	if (!_SrcF0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcF0 Could Not Be Null!");
		return 1;
	}

	if (_SrcF0Size == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcF0 Could Not Be Empty!");
		return 1;
	}

	if (!_SrcVolume)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcVolume Could Not Be Null!");
		return 1;
	}

	if (_SrcVolumeSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcVolume Could Not Be Empty!");
		return 1;
	}

	if (!_SrcSpeakerMap)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcSpeakerMap Could Not Be Null!");
		return 1;
	}

	if (!_Process)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Process Could Not Be Null!");
		return 1;
	}

	if (!_Output)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Output Could Not Be Null!");
		return 1;
	}

	if (!_OutputSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_OutputSize Could Not Be Null!");
		return 1;
	}

	if (!_InferParams)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_InferParams Could Not Be Null!");
		return 1;
	}

	const DragonianLib::SingingVoiceConversion::InferenceParams Param
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
		_GDragonianLibSvcNullStrCheck(_InferParams->Sampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->ReflowSampler),
		_GDragonianLibSvcNullStrCheck(_InferParams->F0Method),
		*(std::shared_ptr<Ort::Session>*)_InferParams->VocoderModel,
		_InferParams->VocoderHopSize,
		_InferParams->VocoderMelBins,
		_InferParams->VocoderSamplingRate,
		_InferParams->F0Bins,
		_InferParams->F0Max,
		_InferParams->F0Min,
		_InferParams->F0ExtractorUserParameter
	};

	try
	{
		auto _16kAudio = DragonianLib::TemplateLibrary::Vector(_16KAudioHubert, _16KAudioHubert + _16KAudioSize);
		std::pair<DragonianLib::TemplateLibrary::Vector<float>, int64_t> _MelData{ {_Mel, _Mel + _SrcMelSize}, _MelSize };
		auto [_MyData, _MySize] = ((DragonianLib::SingingVoiceConversion::SingingVoiceConversion*)_Model)->ShallowDiffusionInference(
			_16kAudio,
			Param,
			_MelData,
			{ _SrcF0, _SrcF0 + _SrcF0Size },
			{ _SrcVolume, _SrcVolume + _SrcVolumeSize },
			*(decltype(DragonianLib::SingingVoiceConversion::SingleSlice::Speaker)*)_SrcSpeakerMap,
			*_Process,
			0
		).Release();
		*_Output = _MyData;
		*_OutputSize = _MySize;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(VocoderEnhance)(
	_Dragonian_Lib_Svc_Add_Prefix(VocoderModel) _Model,
	_Dragonian_Lib_Svc_Add_Prefix(Env) _Env,
	const float* _Mel,
	size_t _SrcMelSize,
	const float* _SrcF0,
	size_t _SrcF0Size,
	INT32 _VocoderMelBins,
	float** _Output,
	size_t* _OutputSize
	)
{
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null!");
		return 1;
	}

	if (!_Mel)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Mel Could Not Be Null!");
		return 1;
	}

	if (_SrcMelSize == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Mel Could Not Be Empty!");
		return 1;
	}

	if (!_SrcF0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcF0 Could Not Be Null!");
		return 1;
	}

	if (_SrcF0Size == 0)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_SrcF0 Could Not Be Empty!");
		return 1;
	}

	if (!_Output)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Output Could Not Be Null!");
		return 1;
	}

	if (!_OutputSize)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_OutputSize Could Not Be Null!");
		return 1;
	}

	auto Rf0 = DragonianLib::TemplateLibrary::Vector(_SrcF0, _SrcF0 + _SrcF0Size);
	std::pair<DragonianLib::TemplateLibrary::Vector<float>, int64_t> MelTemp{ {_Mel, _Mel + _SrcMelSize}, _SrcMelSize / _VocoderMelBins };
	if (Rf0.Size() != (size_t)MelTemp.second)
		Rf0 = InterpFunc(Rf0, (long)Rf0.Size(), (long)MelTemp.second);

	try
	{
		auto [_MyData, _MySize] = _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space VocoderInfer(
			MelTemp.first,
			Rf0,
			_VocoderMelBins,
			MelTemp.second,
			(*(std::shared_ptr<DragonianLib::DragonianLibOrtEnv>*)_Env)->GetMemoryInfo(),
			*(std::shared_ptr<Ort::Session>*)_Model
		).Release();
		*_Output = _MyData;
		*_OutputSize = _MySize;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}

	return 0;
}
