#include <mutex>
#include <string>

#include "Libraries/Base.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/Util/StringPreprocess.h"
#include "Libraries/Cluster/ClusterManager.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"

#include "OnnxLibrary/Vocoder/Register.hpp"
#include "OnnxLibrary/UnitsEncoder/Register.hpp"

#include "OnnxLibrary/SingingVoiceConversion/Api/NativeApi.h"
#include "OnnxLibrary/SingingVoiceConversion/Model/DDSP-Svc.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/Diffusion-Svc.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/Vits-Svc.hpp"

#include "TensorLib/Include/Base/Tensor/Einops.h"

#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif

std::wstring _GDragonianLibSvcLastError;  // NOLINT(misc-use-internal-linkage)

static DragonianLib::DLogger& GetDragonianVoiceApiLogger()
{
	static DragonianLib::DLogger Logger = std::make_shared<DragonianLib::Logger>(
		*DragonianLib::GetDefaultLogger(),
		L"SvcApi"
	);
	return Logger;
}

static const wchar_t* _GDragonianLibSvcNullStrCheck(LPCWSTR Str)
{
	static const wchar_t* DragonianLibSvcNullString = L"";
	return Str ? Str : DragonianLibSvcNullString;
}

struct _Dragonian_Lib_Svc_Class_Name(Enviroment)
{
	_Dragonian_Lib_Svc_Class_Name(Enviroment)(
		const _Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting) * _Setting
		)
	{
		DragonianLib::OnnxRuntime::OnnxEnvironmentOptions Options{
			static_cast<DragonianLib::Device>(_Setting->Provider),
			_Setting->DeviceID,
			_Setting->IntraOpNumThreads,
			_Setting->InterOpNumThreads,
			static_cast<OrtLoggingLevel>(_Setting->LoggingLevel),
			_Setting->LoggerId ? DragonianLib::WideStringToUTF8(_Setting->LoggerId) : "DragonianLib"
		};

		if (auto Iter = _Setting->CUDAConfig)
			while ((*Iter)[0] != nullptr && (*Iter)[1] != nullptr)
			{
				const auto Key = DragonianLib::WideStringToUTF8(_GDragonianLibSvcNullStrCheck((*Iter)[0]));
				const auto Value = DragonianLib::WideStringToUTF8(_GDragonianLibSvcNullStrCheck((*Iter)[1]));
				Options.SetCUDAOptions(
					Key, Value
				);
				++Iter;
			}

		Object = DragonianLib::OnnxRuntime::CreateOnnxRuntimeEnvironment(Options);
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::OnnxRuntime::OnnxRuntimeEnvironment& ()
	{
		return Object;
	}
private:
	DragonianLib::OnnxRuntime::OnnxRuntimeEnvironment Object;
};

struct _Dragonian_Lib_Svc_Class_Name(Model)
{
	_Dragonian_Lib_Svc_Class_Name(Model)(
		const _Dragonian_Lib_Svc_Add_Prefix(HyperParameters)* _HyperParameters,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		)
	{
		DragonianLib::OnnxRuntime::SingingVoiceConversion::HParams _Params;
		_Params.OutputSamplingRate = _HyperParameters->OutputSamplingRate;
		_Params.UnitsDim = _HyperParameters->UnitsDim;
		_Params.HopSize = _HyperParameters->HopSize;
		_Params.SpeakerCount = _HyperParameters->SpeakerCount;
		_Params.HasVolumeEmbedding = _HyperParameters->HasVolumeEmbedding;
		_Params.HasSpeakerEmbedding = _HyperParameters->HasSpeakerEmbedding;
		_Params.HasSpeakerMixLayer = _HyperParameters->HasSpeakerMixLayer;
		_Params.SpecMax = _HyperParameters->SpecMax;
		_Params.SpecMin = _HyperParameters->SpecMin;
		_Params.F0Max = _HyperParameters->F0Max;
		_Params.F0Min = _HyperParameters->F0Min;
		_Params.MelBins = _HyperParameters->MelBins;
		if (_HyperParameters->ProgressCallback)
			_Params.ProgressCallback = _HyperParameters->ProgressCallback;
		if (auto Iter = _HyperParameters->ExtendedParameters)
			while ((*Iter)[0] != nullptr && (*Iter)[1] != nullptr)
			{
				const auto Key = _GDragonianLibSvcNullStrCheck((*Iter)[0]);
				const auto Value = _GDragonianLibSvcNullStrCheck((*Iter)[1]);
				_Params.ExtendedParameters[Key] = Value;
				++Iter;
			}
		if (auto Iter = _HyperParameters->ModelPaths)
			while ((*Iter)[0] != nullptr && (*Iter)[1] != nullptr)
			{
				const auto Key = _GDragonianLibSvcNullStrCheck((*Iter)[0]);
				const auto Value = _GDragonianLibSvcNullStrCheck((*Iter)[1]);
				_Params.ModelPaths[Key] = Value;
				++Iter;
			}

		switch (_HyperParameters->ModelType)
		{
		case _Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV2):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::SoftVitsSvcV2>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV3):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::SoftVitsSvcV3>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV4):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::SoftVitsSvcV4>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(SoVitsSvcV4b):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::SoftVitsSvcV4Beta>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(RVC):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::RetrievalBasedVitsSvc>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(DiffusionSvc):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::DiffusionSvc>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(ReflowSvc):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::ReflowSvc>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		case _Dragonian_Lib_Svc_Add_Prefix(DDSPSvc):
			Object = std::make_shared<DragonianLib::OnnxRuntime::SingingVoiceConversion::DDSPSvc>(
				*_Enviroment,
				_Params,
				GetDragonianVoiceApiLogger()
			);
			break;
		}
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::OnnxRuntime::SingingVoiceConversion::SvcModel& ()
	{
		return Object;
	}
private:
	DragonianLib::OnnxRuntime::SingingVoiceConversion::SvcModel Object;
};

struct _Dragonian_Lib_Svc_Class_Name(UnitsEncoder)
{
	_Dragonian_Lib_Svc_Class_Name(UnitsEncoder)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _SamplingRate,
		INT64 _UnitDims,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		)
	{
		Object = DragonianLib::OnnxRuntime::UnitsEncoder::New(
			_GDragonianLibSvcNullStrCheck(_Name),
			_GDragonianLibSvcNullStrCheck(_Path),
			*_Enviroment,
			_SamplingRate,
			_UnitDims,
			GetDragonianVoiceApiLogger()
		);
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::OnnxRuntime::UnitsEncoder::UnitsEncoder& ()
	{
		return Object;
	}
private:
	DragonianLib::OnnxRuntime::UnitsEncoder::UnitsEncoder Object;
};

struct _Dragonian_Lib_Svc_Class_Name(Vocoder)
{
	_Dragonian_Lib_Svc_Class_Name(Vocoder)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _SamplingRate,
		INT64 _MelBins,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
		)
	{
		Object = DragonianLib::OnnxRuntime::Vocoder::New(
			_GDragonianLibSvcNullStrCheck(_Name),
			_GDragonianLibSvcNullStrCheck(_Path),
			*_Enviroment,
			_SamplingRate,
			_MelBins,
			GetDragonianVoiceApiLogger()
		);
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::OnnxRuntime::Vocoder::Vocoder& ()
	{
		return Object;
	}
private:
	DragonianLib::OnnxRuntime::Vocoder::Vocoder Object;
};

struct _Dragonian_Lib_Svc_Class_Name(Cluster)
{
	_Dragonian_Lib_Svc_Class_Name(Cluster)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		INT64 _ClusterDimension,
		INT64 _ClusterSize
		)
	{
		Object = DragonianLib::Cluster::New(
			_GDragonianLibSvcNullStrCheck(_Name),
			_GDragonianLibSvcNullStrCheck(_Path),
			_ClusterDimension,
			_ClusterSize
		);
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::Cluster::Cluster& ()
	{
		return Object;
	}
private:
	DragonianLib::Cluster::Cluster Object;
};

struct _Dragonian_Lib_Svc_Class_Name(F0Extractor)
{
	_Dragonian_Lib_Svc_Class_Name(F0Extractor)(
		LPCWSTR _Name,
		LPCWSTR _Path,
		_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment,
		INT64 _SamplingRate
		)
	{
		DragonianLib::F0Extractor::PEModelHParams Params{
			_GDragonianLibSvcNullStrCheck(_Path),
			_Enviroment,
			&GetDragonianVoiceApiLogger(),
			_SamplingRate
		};
		Object = DragonianLib::F0Extractor::New(
			_GDragonianLibSvcNullStrCheck(_Name),
			&Params
		);
	}

	auto operator->() const
	{
		return Object.get();
	}

	operator DragonianLib::F0Extractor::F0Extractor& ()
	{
		return Object;
	}
private:
	DragonianLib::F0Extractor::F0Extractor Object;
};

struct _Dragonian_Lib_Svc_Class_Name(FloatTensor)
{
	using MyValueType = DragonianLib::Tensor<DragonianLib::Float32, 4, DragonianLib::Device::CPU>;

	_Dragonian_Lib_Svc_Class_Name(FloatTensor)() = default;

	_Dragonian_Lib_Svc_Class_Name(FloatTensor)(MyValueType&& _Value) : Object(std::move(_Value)) {}

	auto operator->()
	{
		return &Object;
	}

	operator DragonianLib::Tensor<DragonianLib::Float32, 4, DragonianLib::Device::CPU>&()
	{
		return Object;
	}

	MyValueType& Set()
	{
		return Object;
	}

	const MyValueType& Get() const
	{
		return Object;
	}

private:
	DragonianLib::Tensor<DragonianLib::Float32, 4, DragonianLib::Device::CPU> Object;
};

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

void _Dragonian_Lib_Svc_Add_Prefix(InitEnviromentSetting)(
	_Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting)* _Input
	)
{
	*_Input = {
		static_cast<INT32>(DragonianLib::Device::CPU),
		0,
		4,
		2,
		ORT_LOGGING_LEVEL_WARNING,
		nullptr,
		nullptr
	};
}

void _Dragonian_Lib_Svc_Add_Prefix(InitHyperParameters)(
	_Dragonian_Lib_Svc_Add_Prefix(HyperParameters)* _Input
	)
{
	*_Input = {
		_Dragonian_Lib_Svc_Add_Prefix(RVC),
		nullptr,
		44100,
		256,
		512,
		1,
		0,
		0,
		0,
		2.f,
		-12.f,
		256,
		1100.f,
		50.f,
		128,
		nullptr,
		nullptr
	};
}

void _Dragonian_Lib_Svc_Add_Prefix(InitInferenceParameters)(
	_Dragonian_Lib_Svc_Add_Prefix(InferenceParameters)* _Input
	)
{
	static wchar_t PNDM[] = L"Pndm";
	static wchar_t EULAR[] = L"Pndm";
	*_Input = {
		0.3f,
		0,
		0.f,
		52468,
		0.5f,
		0,
		{1, 0, 100, PNDM, 1.f, nullptr },
		{0.1f, 0.f, 1.f, 1000.f, EULAR, 1.f, nullptr },
		0.8f,
		nullptr
	};
}

void _Dragonian_Lib_Svc_Add_Prefix(InitF0ExtractorParameters)(
	_Dragonian_Lib_Svc_Add_Prefix(F0ExtractorParameters)* _Input
	)
{
	_Input->HopSize = 512;
	_Input->SamplingRate = 44100;
	_Input->F0Bins = 256;
	_Input->WindowSize = 2048;
	_Input->F0Max = 1100.0;
	_Input->F0Min = 50.0;
	_Input->Threshold = 0.03f;
	_Input->UserParameter = nullptr;
}

/***************************************Fun*******************************************/

void _Dragonian_Lib_Svc_Add_Prefix(RaiseError)(const std::wstring& _Msg)
{
	GetDragonianVoiceApiLogger()->LogError(_Msg);
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
	DragonianLib::SetGlobalEnvDir(_GDragonianLibSvcNullStrCheck(_Dir));
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerId)(
	LPCWSTR _Id
	)
{
	GetDragonianVoiceApiLogger()->SetLoggerId((_Id));
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLoggerLevel)(
	INT32 _Level
	)
{
	GetDragonianVoiceApiLogger()->SetLoggerLevel(static_cast<DragonianLib::Logger::LogLevel>(_Level));
}

void _Dragonian_Lib_Svc_Add_Prefix(SetLogFunction)(
	_Dragonian_Lib_Svc_Add_Prefix(LogFunction) _Logger
	)
{
	GetDragonianVoiceApiLogger()->SetLoggerFunction(_Logger);
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

_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Dragonian_Lib_Svc_Add_Prefix(CreateEnviroment)(
	const _Dragonian_Lib_Svc_Add_Prefix(EnviromentSetting)* _Setting
	)
{
	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(Enviroment)(_Setting);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(DestoryEnviroment)(
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)
{
	try
	{
		delete _Enviroment;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(Model) _Dragonian_Lib_Svc_Add_Prefix(LoadModel)(
	const _Dragonian_Lib_Svc_Add_Prefix(HyperParameters)* _HyperParameters,
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)
{
	if (!_Enviroment)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Enviroment Could Not Be Null");
		return nullptr;
	}

	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(Model)(_HyperParameters, _Enviroment);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(UnrefModel)(
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model
	)
{
	try
	{
		delete _Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(Vocoder) _Dragonian_Lib_Svc_Add_Prefix(LoadVocoder)(
	LPCWSTR _Name,
	LPCWSTR _Path,
	INT64 _SamplingRate,
	INT64 _MelBins,
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)
{
	if (!_Enviroment)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Enviroment Could Not Be Null");
		return nullptr;
	}

	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(Vocoder)(_Name, _Path, _SamplingRate, _MelBins, _Enviroment);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(UnrefVocoder)(
	_Dragonian_Lib_Svc_Add_Prefix(Vocoder) _Model
	)
{
	try
	{
		delete _Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Dragonian_Lib_Svc_Add_Prefix(LoadUnitsEncoder)(
	LPCWSTR _Name,
	LPCWSTR _Path,
	INT64 _SamplingRate,
	INT64 _UnitDims,
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)
{
	if (!_Enviroment)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Enviroment Could Not Be Null");
		return nullptr;
	}

	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(UnitsEncoder)(_Name, _Path, _SamplingRate, _UnitDims, _Enviroment);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(UnrefUnitsEncoder)(
	_Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Model
	)
{
	try
	{
		delete _Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(UnrefGlobalCache)(
	LPCWSTR _ModelPath,
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)

{
	if (!_ModelPath)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_ModelPath Could Not Be Null");
		return 1;
	}

	if (!_Enviroment)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Enviroment Could Not Be Null");
		return 1;
	}

	try
	{
		(*_Enviroment)->UnRefOnnxRuntimeModel(_ModelPath);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

INT32 _Dragonian_Lib_Svc_Add_Prefix(ClearGlobalCache)(
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment
	)
{
	if (!_Enviroment)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Enviroment Could Not Be Null");
		return 1;
	}

	try
	{
		(*_Enviroment)->ClearOnnxRuntimeModel();
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return 1;
	}
	return 0;
}

_Dragonian_Lib_Svc_Add_Prefix(Cluster) _Dragonian_Lib_Svc_Add_Prefix(CreateCluster)(
	LPCWSTR _Name,
	LPCWSTR _Path,
	INT64 _ClusterDimension,
	INT64 _ClusterSize
	)
{
	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(Cluster)(_Name, _Path, _ClusterDimension, _ClusterSize);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(DestoryCluster)(
	_Dragonian_Lib_Svc_Add_Prefix(Cluster) _Model
	)
{
	try
	{
		delete _Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Dragonian_Lib_Svc_Add_Prefix(CreateF0Extractor)(
	LPCWSTR _Name,
	LPCWSTR _Path,
	_Dragonian_Lib_Svc_Add_Prefix(Enviroment) _Enviroment,
	INT64 _SamplingRate
	)
{
	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(F0Extractor)(_Name, _Path, _Enviroment, _SamplingRate);
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

void _Dragonian_Lib_Svc_Add_Prefix(UnrefF0Extractor)(
	_Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Model
	)
{
	try
	{
		delete _Model;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(CreateFloatTensor)(
	float* _Buffer,
	INT64 _0,
	INT64 _1,
	INT64 _2,
	INT64 _3
	)
{
	try
	{
		auto Tensor = new _Dragonian_Lib_Svc_Class_Name(FloatTensor)();
		Tensor->Set() = DragonianLib::Tensor<DragonianLib::Float32, 4, DragonianLib::Device::CPU>::FromBuffer(
			DragonianLib::Dimensions{ _0, _1, _2, _3 },
			_Buffer,
			_0 * _1 * _2 * _3
		);
		return Tensor;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

float* _Dragonian_Lib_Svc_Add_Prefix(GetTensorData)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
	)
{
	if (!_Tensor)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Tensor Could Not Be Null");
		return nullptr;
	}

	return (*_Tensor)->Data();
}

const INT64* _Dragonian_Lib_Svc_Add_Prefix(GetTensorShape)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
	)
{
	if (!_Tensor)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Tensor Could Not Be Null");
		return nullptr;
	}

	return (*_Tensor)->Size().Data();
}

void _Dragonian_Lib_Svc_Add_Prefix(DestoryFloatTensor)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Tensor
	)
{
	try
	{
		delete _Tensor;
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
	}
}

_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(EncodeUnits)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
	INT64 _SourceSamplingRate,
	_Dragonian_Lib_Svc_Add_Prefix(UnitsEncoder) _Model
	)
{
	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null");
		return nullptr;
	}
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null");
		return nullptr;
	}

	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(
			(*_Model)->Forward(
				_Audio->Get().Squeeze(0),
				_SourceSamplingRate
			));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(ClusterSearch)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Units,
	INT64 _CodeBookId,
	_Dragonian_Lib_Svc_Add_Prefix(Cluster) _Model
	)
{
	if (!_Units)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null");
		return nullptr;
	}
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null");
		return nullptr;
	}

	try
	{
		return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(
			(*_Model)->Search(
				_Units->Get().AutoView(-2, -1),
				static_cast<DragonianLib::Long>(_CodeBookId)
			).AutoView(1, 1, -2, -1));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(ExtractF0)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
	const _Dragonian_Lib_Svc_Add_Prefix(F0ExtractorParameters)* _Parameters,
	_Dragonian_Lib_Svc_Add_Prefix(F0Extractor) _Model
	)
{
	if (!_Audio)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null");
		return nullptr;
	}
	if (!_Parameters)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Parameters Could Not Be Null");
		return nullptr;
	}
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null");
		return nullptr;
	}

	try
	{
		auto Ret = (*_Model)->ExtractF0(
			_Audio->Get().AutoView(-2, -1),
			{
				_Parameters->SamplingRate,
				_Parameters->HopSize,
				_Parameters->F0Bins,
				_Parameters->WindowSize,
				_Parameters->F0Max,
				_Parameters->F0Min,
				_Parameters->Threshold,
				_Parameters->UserParameter
			}
		);
		return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(Ret.View(1, 1, Ret.Size(0), Ret.Size(1)));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(Inference)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Audio,
	INT64 SourceSamplingRate,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Units,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _F0,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _SpeakerMix,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Spec,
	const _Dragonian_Lib_Svc_Add_Prefix(InferenceParameters)* _Parameters,
	_Dragonian_Lib_Svc_Add_Prefix(Model) _Model,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor)* _OutF0
	)
{
	if (!_Audio)
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Audio Could Not Be Null");

	if (!_Units)
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Units Could Not Be Null");

	if (!_F0)
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_F0 Could Not Be Null");

	if (!_Parameters)
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Parameters Could Not Be Null");

	if (!_Model)
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null");

	DragonianLib::OnnxRuntime::SingingVoiceConversion::Parameters Params{
		_Parameters->NoiseScale,
		_Parameters->SpeakerId,
		_Parameters->PitchOffset,
		_Parameters->Seed,
		_Parameters->ClusterRate,
		static_cast<bool>(_Parameters->F0HasUnVoice),
		{
			_Parameters->Diffusion.Stride,
			_Parameters->Diffusion.Begin,
			_Parameters->Diffusion.End,
			_GDragonianLibSvcNullStrCheck(_Parameters->Diffusion.Sampler),
			_Parameters->Diffusion.MelFactor,
			_Parameters->Diffusion.UserParameters
		},
		{
			_Parameters->Reflow.Stride,
			_Parameters->Reflow.Begin,
			_Parameters->Reflow.End,
			_Parameters->Reflow.Scale,
			_GDragonianLibSvcNullStrCheck(_Parameters->Reflow.Sampler),
			_Parameters->Reflow.MelFactor,
			_Parameters->Reflow.UserParameters
		},
		_Parameters->StftNoiseScale,
		nullptr,
		_Parameters->UserParameters
	};

	DragonianLib::OnnxRuntime::SingingVoiceConversion::SliceDatas Slice;
	Slice.SourceSampleCount = _Audio->Get().ElementCount();
	Slice.SourceSampleRate = SourceSamplingRate;
	Slice.GTAudio = _Audio->Get().Squeeze(0).View();
	Slice.GTSampleRate = SourceSamplingRate;
	if (_Spec)
		Slice.GTSpec = _Spec->Get().View();
	if (_SpeakerMix)
		Slice.Speaker = _SpeakerMix->Get().View();
	Slice.Units = _Units->Get().View();
	Slice.F0 = _F0->Get().Squeeze(0).View();

	try
	{
		auto Data = (*_Model)->VPreprocess(Params, std::move(Slice));
		if (_OutF0)
			*_OutF0 = new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(Data.F0.UnSqueeze(0));
		return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)((*_Model)->Forward(Params, Data));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}

_Dragonian_Lib_Svc_Api _Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Dragonian_Lib_Svc_Add_Prefix(InferVocoder)(
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _Mel,
	_Dragonian_Lib_Svc_Add_Prefix(FloatTensor) _F0,
	_Dragonian_Lib_Svc_Add_Prefix(Vocoder) _Model
	)
{
	if (!_Mel)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Mel Could Not Be Null");
		return nullptr;
	}
	if (!_Model)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(L"_Model Could Not Be Null");
		return nullptr;
	}

	try
	{
		if (_F0)
		{
			auto F0 = _F0->Get().Squeeze(0);
			return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(
			   (*_Model)->Forward(
				   _Mel->Get(),
				   F0
			   ).UnSqueeze(0));
		}
		return new _Dragonian_Lib_Svc_Class_Name(FloatTensor)(
			(*_Model)->Forward(
				_Mel->Get()
			).UnSqueeze(0));
	}
	catch (std::exception& e)
	{
		_Dragonian_Lib_Svc_Add_Prefix(RaiseError)(DragonianLib::UTF8ToWideString(e.what()));
		return nullptr;
	}
}