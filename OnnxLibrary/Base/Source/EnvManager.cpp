#ifdef DRAGONIANLIB_ONNXRT_LIB
#include <thread>
#include <ranges>
#include <providers/dml/dml_provider_factory.h>

#include "Libraries/Util/Logger.h"
#include "OnnxLibrary/Base/EnvManager.hpp"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Onnx_Runtime_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		*_D_Dragonian_Lib_Namespace GetDefaultLogger(),
		L"OnnxRuntime"
	);
	return _MyLogger;
}

std::unordered_map<std::wstring, std::shared_ptr<OnnxRuntimeEnviromentBase>> GlobalOrtEnvCache;

const char* logger_id = "DragonianLib-OnnxRuntime";

void DragonianLibOrtLoggingFn(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
	const char* message)
{
	std::string ort_message = severity == ORT_LOGGING_LEVEL_ERROR ? "[Error" : severity == ORT_LOGGING_LEVEL_WARNING ? "[Warning" : severity == ORT_LOGGING_LEVEL_INFO ? "[Info" : severity == ORT_LOGGING_LEVEL_VERBOSE ? "[Verbose" : severity == ORT_LOGGING_LEVEL_FATAL ? "[Fatal" : "[Unknown";
	ort_message += "; @OnnxRuntime::";
	ort_message += code_location;
	ort_message += "]: ";
	ort_message += message;
	_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->LogMessage(UTF8ToWideString(ort_message));
}

inline const std::vector GlobalOrtCUDAOptionKeys{
	"device_id",
	"gpu_mem_limit",
	"arena_extend_strategy",
	"cudnn_conv_algo_search",
	"do_copy_in_default_stream",
	"cudnn_conv_use_max_workspace",
	"cudnn_conv1d_pad_to_nc1d",
	"enable_cuda_graph",
	"enable_skip_layer_norm_strict_mode"
};
inline std::vector GlobalOrtCUDAOptionValues{
	"0",
	"2147483648",
	"kNextPowerOfTwo",
	"EXHAUSTIVE",
	"1",
	"1",
	"1",
	"0",
	"0"
};
inline std::vector<std::string> GlobalOrtCUDAOptionValueStrings{
	"0",
	"2147483648",
	"kNextPowerOfTwo",
	"EXHAUSTIVE",
	"1",
	"1",
	"1",
	"0",
	"0"
};

OnnxRuntimeEnviromentBase::~OnnxRuntimeEnviromentBase()
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << this;
		wss >> HexPtr;
	}
	_MyStaticLogger->LogMessage(L"Destroying Envireoment: Instance[PTR:" + HexPtr + L", Provider:" + std::to_wstring(GetCurProvider()) + L", DeviceID:" + std::to_wstring(GetCurDeviceID()) + L", ThreadCount:" + std::to_wstring(GetCurThreadCount()) + L']');
	GlobalOrtModelCache.clear();
	_MyStaticLogger->LogMessage(L"Envireoment Destroyed: Instance[PTR:" + HexPtr + L']');
}

OnnxRuntimeEnviromentBase::OnnxRuntimeEnviromentBase(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
{
	try
	{
		Load(ThreadCount, DeviceID, Provider);
	}
	catch (std::exception& e)
	{
		throw std::exception(e.what());
	}
}

void OnnxRuntimeEnviromentBase::Load(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
{
	try
	{
		Create(ThreadCount, DeviceID, Provider);
	}
	catch (std::exception& e)
	{
		throw std::exception(e.what());
	}
}

void OnnxRuntimeEnviromentBase::Create(unsigned ThreadCount_, unsigned DeviceID_, unsigned ExecutionProvider_)
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	_MyStaticLogger->LogInfo(
		L"Creating Envireoment With Provider:[" +
		std::to_wstring(ExecutionProvider_) +
		L"], DeviceID:[" +
		std::to_wstring(DeviceID_)
		+ L"], ThreadCount:[" +
		std::to_wstring(ThreadCount_) +
		L"]"
	);

	static const OrtApi& GlobalOrtApi = Ort::GetApi();
	_MyOrtSessionOptions = std::make_shared<Ort::SessionOptions>();
	if (ExecutionProvider_ == 0)
	{
		if (ThreadCount_ == 0)
			ThreadCount_ = std::thread::hardware_concurrency();
		_MyOrtEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, logger_id, DragonianLibOrtLoggingFn, nullptr);
		_MyOrtSessionOptions->SetIntraOpNumThreads(static_cast<int>(ThreadCount_));
		_MyOrtSessionOptions->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	else if (ExecutionProvider_ == 1)
	{
		const auto AvailableProviders = Ort::GetAvailableProviders();
		bool ret = true;
		for (const auto& it : AvailableProviders)
			if (it.find("CUDA") != std::string::npos)
				ret = false;
		if (ret)
			throw std::exception("CUDA Provider Not Found");

		{
			OrtCUDAProviderOptionsV2* TmpCudaProviderOptionsV2 = nullptr;
			GlobalOrtApi.CreateCUDAProviderOptions(&TmpCudaProviderOptionsV2);
			_MyCudaOptionsV2 = std::shared_ptr<OrtCUDAProviderOptionsV2>(
				TmpCudaProviderOptionsV2,
				GlobalOrtApi.ReleaseCUDAProviderOptions
			);
			GlobalOrtCUDAOptionValues[0] = std::to_string(DeviceID_).c_str();
			GlobalOrtApi.UpdateCUDAProviderOptions(
				_MyCudaOptionsV2.get(),
				GlobalOrtCUDAOptionKeys.data(),
				GlobalOrtCUDAOptionValues.data(),
				GlobalOrtCUDAOptionKeys.size()
			);
		}

		_MyOrtEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, logger_id, DragonianLibOrtLoggingFn, nullptr);
		_MyOrtSessionOptions->AppendExecutionProvider_CUDA_V2(*_MyCudaOptionsV2);
		_MyOrtSessionOptions->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
		_MyOrtSessionOptions->SetIntraOpNumThreads((int)std::thread::hardware_concurrency());
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	else if (ExecutionProvider_ == 2)
	{
		const auto AvailableProviders = Ort::GetAvailableProviders();
		std::string ret;
		for (const auto& it : AvailableProviders)
			if (it.find("Dml") != std::string::npos)
				ret = it;
		if (ret.empty())
			throw std::exception("DML Provider Not Found");

		const OrtDmlApi* ortDmlApi = nullptr;
		GlobalOrtApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
		Ort::ThreadingOptions threading_options;
		threading_options.SetGlobalInterOpNumThreads(static_cast<int>(ThreadCount_));
		_MyOrtEnv = std::make_shared<Ort::Env>(threading_options, DragonianLibOrtLoggingFn, nullptr, ORT_LOGGING_LEVEL_WARNING, logger_id);
		_MyOrtEnv->DisableTelemetryEvents();
		ortDmlApi->SessionOptionsAppendExecutionProvider_DML(*_MyOrtSessionOptions, int(DeviceID_));
		_MyOrtSessionOptions->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
		_MyOrtSessionOptions->DisablePerSessionThreads();
		_MyOrtSessionOptions->SetExecutionMode(ORT_SEQUENTIAL);
		_MyOrtSessionOptions->DisableMemPattern();
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
	}
	else
	{
		_D_Dragonian_Lib_Throw_Exception("Invalid Execution Provider");
	}

	_MyStaticLogger->LogInfo(L"Envireoment Created With Provider:[" + std::to_wstring(ExecutionProvider_) + L"], DeviceID:[" + std::to_wstring(DeviceID_) + L"], ThreadCount:[" + std::to_wstring(ThreadCount_) + L"]");
}

OnnxRuntimeModel& OnnxRuntimeEnviromentBase::RefOrtCachedModel(
	const std::wstring& Path_,
	const OnnxRuntimeEnviroment& Env_
)
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();

	std::wstring EnvPtr;
	{
		std::wstringstream wss;
		wss << std::hex << Env_.get();
		wss >> EnvPtr;
	}
	const auto RawID = EnvPtr +
		L" EP:" + std::to_wstring(Env_->GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_->GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_->GetCurThreadCount());
	const auto ID = RawID + L" PATH:" + Path_;

	auto Iter = Env_->GlobalOrtModelCache.find(ID);
	if (Iter != Env_->GlobalOrtModelCache.end())
	{
		std::wstring HexPtr;
		{
			std::wstringstream wss;
			wss << std::hex << Iter->second.get();
			wss >> HexPtr;
		}
		_MyStaticLogger->LogInfo(L"Referencing Model: Instance[PTR:" + HexPtr + L", PATH:\"" + Path_ + L"\"], Current Referece Count: " + std::to_wstring(Iter->second.use_count()));
		return Iter->second;
	}
	try
	{
		_MyStaticLogger->LogInfo(L"Loading Model: \"" + Path_ + L"\" With OnnxEnvironment: Instance[PTR:" + RawID + L"], Current Referece Count: 1");
		auto _DeleterLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
		return Env_->GlobalOrtModelCache[ID] = std::shared_ptr<Ort::Session>(
			new Ort::Session(*Env_->GetEnv(), Path_.c_str(), *Env_->GetSessionOptions()),
			[_DeleterLogger, Path_](const Ort::Session* Ptr)
			{
				std::wstring HexPtr;
				{
					std::wstringstream wss;
					wss << std::hex << Ptr;
					wss >> HexPtr;
				}
				delete Ptr;
				_DeleterLogger->LogInfo(L"Model Unloaded: Instance[PTR:" + HexPtr + L", PATH:\"" + Path_ + L"\"], Current Referece Count: 0");
			}
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

void OnnxRuntimeEnviromentBase::UnRefOrtCachedModel(
	const std::wstring& Path_,
	const OnnxRuntimeEnviroment& Env_
)
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring EnvPtr;
	{
		std::wstringstream wss;
		wss << std::hex << Env_.get();
		wss >> EnvPtr;
	}
	const auto RawID = EnvPtr +
		L" EP:" + std::to_wstring(Env_->GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_->GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_->GetCurThreadCount());
	const auto ID = RawID + L" PATH:" + Path_;
	auto Iter = Env_->GlobalOrtModelCache.find(ID);
	if (Iter != Env_->GlobalOrtModelCache.end())
	{
		std::wstring HexPtr;
		{
			std::wstringstream wss;
			wss << std::hex << Iter->second.get();
			wss >> HexPtr;
		}
		_MyStaticLogger->LogInfo(L"UnReference Model: Instance[PTR:" + HexPtr + L", PATH:\"" + Path_ + L"\"], Current Referece Count: " + std::to_wstring(Iter->second.use_count()));
		Env_->GlobalOrtModelCache.erase(Iter);
	}
}

void OnnxRuntimeEnviromentBase::ClearModelCache(
	const OnnxRuntimeEnviroment& Env_
)
{
	Env_->GlobalOrtModelCache.clear();
}

void OnnxRuntimeEnviromentBase::SetCUDAOption(
	const std::string& Key,
	const std::string& Value
)
{
	auto Iter = std::ranges::find(GlobalOrtCUDAOptionKeys, Key);
	if (Iter == GlobalOrtCUDAOptionKeys.end())
		_D_Dragonian_Lib_Throw_Exception("Invalid CUDA Option Key");
	const auto Index = Iter - GlobalOrtCUDAOptionKeys.begin();
	GlobalOrtCUDAOptionValueStrings[Index] = Value;
	GlobalOrtCUDAOptionValues[Index] = GlobalOrtCUDAOptionValueStrings[Index].c_str();
}

OnnxRuntimeEnviroment& OnnxRuntimeEnviromentBase::CreateEnv(
	unsigned ThreadCount, unsigned DeviceID, unsigned Provider
)
{
	const auto ID = L"EP:" + std::to_wstring(Provider) +
		L" DEVICE:" + std::to_wstring(DeviceID) +
		L" THREAD:" + std::to_wstring(ThreadCount);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		return Iter->second;
	return GlobalOrtEnvCache[ID] = OnnxRuntimeEnviroment(
		new OnnxRuntimeEnviromentBase(ThreadCount, DeviceID, Provider)
	);
}

void OnnxRuntimeEnviromentBase::DestroyEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
{
	const auto ID = L"EP:" + std::to_wstring(Provider) +
		L" DEVICE:" + std::to_wstring(DeviceID) +
		L" THREAD:" + std::to_wstring(ThreadCount);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		GlobalOrtEnvCache.erase(Iter);
}

_D_Dragonian_Lib_Onnx_Runtime_End

#endif
