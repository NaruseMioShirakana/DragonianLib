#ifdef DRAGONIANLIB_ONNXRT_LIB
#include "Libraries/Base.h"
#include "Libraries/EnvManager.hpp"
#include <providers/dml/dml_provider_factory.h>
#include <thread>
#include <ranges>
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Space_Begin

std::unordered_map<std::wstring, std::shared_ptr<Ort::Session>> GlobalOrtModelCache;
std::unordered_map<std::wstring, std::shared_ptr<DragonianLibOrtEnv>> GlobalOrtEnvCache;

const char* logger_id = "DragonianLib-OnnxRuntime";

void DragonianLibOrtLoggingFn(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location,
	const char* message)
{
	std::string ort_message = severity == ORT_LOGGING_LEVEL_ERROR ? "[Error" : severity == ORT_LOGGING_LEVEL_WARNING ? "[Warning" : severity == ORT_LOGGING_LEVEL_INFO ? "[Info" : severity == ORT_LOGGING_LEVEL_VERBOSE ? "[Verbose" : severity == ORT_LOGGING_LEVEL_FATAL ? "[Fatal" : "[Unknown";
	ort_message += "; @OnnxRuntime::";
	ort_message += code_location;
	ort_message += "]: ";
	ort_message += message;
	LogMessage(UTF8ToWideString(ort_message));
}

static const std::vector GlobalOrtCUDAOptionKeys{
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
static std::vector GlobalOrtCUDAOptionValues{
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
static std::vector<std::string> GlobalOrtCUDAOptionValueStrings{
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

DragonianLibOrtEnv::~DragonianLibOrtEnv()
{
	auto Message =
		L"Destory OnnxRuntime Env With [Provider: " + std::to_wstring(_MyProvider) +
		L" DeviceID: " + std::to_wstring(_MyDeviceID) +
		L" ThreadCount: " + std::to_wstring(_MyThreadCount) + L']';
	LogInfo(Message);
	auto ID = std::to_wstring(uint64_t(this));
	for (const auto& it : GlobalOrtModelCache | std::ranges::views::keys)
		if (it.contains(ID))
			GlobalOrtModelCache.erase(it);
}

DragonianLibOrtEnv::DragonianLibOrtEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
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

void DragonianLibOrtEnv::Load(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
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

void DragonianLibOrtEnv::Create(unsigned ThreadCount_, unsigned DeviceID_, unsigned ExecutionProvider_)
{
	LogInfo(
		L"Creating Env With Provider:[" +
		std::to_wstring(ExecutionProvider_) +
		L"] DeviceID:[" +
		std::to_wstring(DeviceID_)
		+ L']'
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

	LogInfo(L"Env Created");
}

std::shared_ptr<Ort::Session>& DragonianLibOrtEnv::RefOrtCachedModel(
	const std::wstring& Path_,
	const DragonianLibOrtEnv& Env_
)
{
	const auto ID = std::to_wstring(uint64_t(&Env_)) + L" EP:" + std::to_wstring(Env_.GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_.GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_.GetCurThreadCount()) +
		L" PATH:" + Path_;
	auto Iter = GlobalOrtModelCache.find(ID);
	if (Iter != GlobalOrtModelCache.end())
		return Iter->second;
	return GlobalOrtModelCache[ID] = std::make_shared<Ort::Session>(*Env_.GetEnv(), Path_.c_str(), *Env_.GetSessionOptions());
}

void DragonianLibOrtEnv::UnRefOrtCachedModel(
	const std::wstring& Path_,
	const DragonianLibOrtEnv& Env_
)
{
	const auto ID = std::to_wstring(uint64_t(&Env_)) + L" EP:" + std::to_wstring(Env_.GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_.GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_.GetCurThreadCount()) +
		L" PATH:" + Path_;
	auto Iter = GlobalOrtModelCache.find(ID);
	if (Iter != GlobalOrtModelCache.end())
		GlobalOrtModelCache.erase(Iter);
}

void DragonianLibOrtEnv::ClearModelCache()
{
	GlobalOrtModelCache.clear();
}

void DragonianLibOrtEnv::SetCUDAOption(
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

std::shared_ptr<DragonianLibOrtEnv>& DragonianLibOrtEnv::CreateEnv(
	unsigned ThreadCount, unsigned DeviceID, unsigned Provider
)
{
	const auto ID = L"EP:" + std::to_wstring(Provider) +
		L" DEVICE:" + std::to_wstring(DeviceID) +
		L" THREAD:" + std::to_wstring(ThreadCount);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		return Iter->second;
	return GlobalOrtEnvCache[ID] = std::shared_ptr<DragonianLibOrtEnv>(
		new DragonianLibOrtEnv(ThreadCount, DeviceID, Provider)
	);
}

void DragonianLibOrtEnv::DestroyEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
{
	const auto ID = L"EP:" + std::to_wstring(Provider) +
		L" DEVICE:" + std::to_wstring(DeviceID) +
		L" THREAD:" + std::to_wstring(ThreadCount);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		GlobalOrtEnvCache.erase(Iter);
}

_D_Dragonian_Lib_Space_End

#endif
