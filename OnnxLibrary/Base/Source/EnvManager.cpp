#ifdef DRAGONIANLIB_ONNXRT_LIB
#include <thread>
#include <ranges>
#include <providers/dml/dml_provider_factory.h>

#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"
#include "OnnxLibrary/Base/EnvManager.hpp"

_D_Dragonian_Lib_Onnx_Runtime_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		*_D_Dragonian_Lib_Namespace GetDefaultLogger(),
		L"OnnxRuntime"
	);
	return _MyLogger;
}

std::unordered_map<std::wstring, std::shared_ptr<OnnxRuntimeEnvironmentBase>> GlobalOrtEnvCache;

void DragonianLibOrtLoggingFn(
	void*, 
	OrtLoggingLevel severity, 
	const char* category, 
	const char* logid,
	const char* code_location,
	const char* message
)
{
	std::string ort_message =
		severity == ORT_LOGGING_LEVEL_ERROR ? "[Error" :
		severity == ORT_LOGGING_LEVEL_WARNING ? "[Warning" :
		severity == ORT_LOGGING_LEVEL_INFO ? "[Info" :
		severity == ORT_LOGGING_LEVEL_VERBOSE ? "[Verbose" :
		severity == ORT_LOGGING_LEVEL_FATAL ? "[Fatal" :
		"[Unknown";

	ort_message += "; @OnnxRuntime";
	if (logid != nullptr && logid[0] != '\0')
	{
		ort_message += "::";
		ort_message += logid;
	}

	if (category != nullptr && category[0] != '\0')
	{
		ort_message += "; ";
		ort_message += category;
	}

	if (code_location != nullptr && code_location[0] != '\0')
	{
		ort_message += "; ";
		ort_message += code_location;
	}

	if (message != nullptr && message[0] != '\0')
	{
		ort_message += "]: ";
		ort_message += message;
	}

	_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->LogMessage(UTF8ToWideString(ort_message));
}

OnnxRuntimeEnvironmentBase::~OnnxRuntimeEnvironmentBase()
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << this;
		wss >> HexPtr;
	}

	_MyStaticLogger->LogMessage(
		L"Destroying Envireoment: Instance[PTR:" + HexPtr + 
		L", Provider:" + std::to_wstring(static_cast<int>(GetProvider())) + 
		L", DeviceID:" + std::to_wstring(GetDeviceID()) + 
		L", InterOpNumThreads:" + std::to_wstring(GetInterOpNumThreads()) +
		L", IntraOpNumThreads:" + std::to_wstring(GetIntraOpNumThreads()) +
		L", LoggerId:" + UTF8ToWideString(_MyLoggerId) +
		L", LoggingLevel:" + std::to_wstring(_MyLoggingLevel) +
		L']');

	GlobalOrtModelCache.clear();

	_MyStaticLogger->LogMessage(
		L"Envireoment Destroyed: Instance[PTR:" + HexPtr + L']'
	);
}

OnnxRuntimeEnvironmentBase::OnnxRuntimeEnvironmentBase(const OnnxEnvironmentOptions& Options)
{
	_D_Dragonian_Lib_Rethrow_Block(Load(Options););
}

void OnnxRuntimeEnvironmentBase::Load(const OnnxEnvironmentOptions& Options)
{
	_D_Dragonian_Lib_Rethrow_Block(Create(Options););
}

void OnnxRuntimeEnvironmentBase::Create(const OnnxEnvironmentOptions& Options)
{
	_MyIntraOpNumThreads = Options.IntraOpNumThreads;
	_MyInterOpNumThreads = Options.InterOpNumThreads;
	_MyDeviceID = Options.DeviceID;
	_MyProvider = Options.Provider;
	_MyLoggingLevel = Options.LoggingLevel;
	_MyLoggerId = Options.LoggerId;
	_MyCUDAOptions = Options.CUDAOptions;

	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	_MyStaticLogger->LogInfo(
		L"Creating Envireoment With Provider:[" +
		std::to_wstring(static_cast<int>(_MyProvider)) +
		L"], DeviceID:[" +
		std::to_wstring(_MyDeviceID)
		+ L"], IntraOpNumThreads:[" +
		std::to_wstring(_MyIntraOpNumThreads) +
		L"], InterOpNumThreads:[" +
		std::to_wstring(_MyInterOpNumThreads) +
		L"], LoggerId:[" +
		UTF8ToWideString(_MyLoggerId) +
		L"], LoggingLevel:[" +
		std::to_wstring(_MyLoggingLevel) +
		L"]"
	);

	if (_MyIntraOpNumThreads < 1)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid Thread Count, expected: [1, " +
			std::to_string(std::thread::hardware_concurrency()) +
			"], got: " + std::to_string(_MyIntraOpNumThreads)
		);
	if (_MyIntraOpNumThreads > std::thread::hardware_concurrency())
		_D_Dragonian_Lib_Throw_Exception("Invalid Thread Count, expected: [1, " +
			std::to_string(std::thread::hardware_concurrency()) + "], got: " +
			std::to_string(_MyIntraOpNumThreads)
		);

	if (_MyInterOpNumThreads < 1)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid Thread Count, expected: [1, " +
			std::to_string(std::thread::hardware_concurrency()) +
			"], got: " + std::to_string(_MyInterOpNumThreads)
		);
	if (_MyInterOpNumThreads > std::thread::hardware_concurrency())
		_D_Dragonian_Lib_Throw_Exception("Invalid Thread Count, expected: [1, " +
			std::to_string(std::thread::hardware_concurrency()) + "], got: " +
			std::to_string(_MyInterOpNumThreads)
		);

	const OrtApi& GlobalOrtApi = Ort::GetApi();

	_MyOrtSessionOptions = std::make_shared<Ort::SessionOptions>();

	if (_MyProvider == Device::CPU)
	{
		_MyOrtEnv = std::make_shared<Ort::Env>(
			_MyLoggingLevel,
			_MyLoggerId.c_str(),
			DragonianLibOrtLoggingFn,
			nullptr
		);
		_MyOrtSessionOptions->SetIntraOpNumThreads(
			static_cast<int>(_MyIntraOpNumThreads)
		);
		_MyOrtSessionOptions->SetInterOpNumThreads(
			static_cast<int>(_MyInterOpNumThreads)
		);
		_MyOrtSessionOptions->SetGraphOptimizationLevel(
			ORT_ENABLE_ALL
		);
		_MyOrtSessionOptions->EnableMemPattern();
		_MyOrtSessionOptions->EnableCpuMemArena();
		_MyOrtSessionOptions->SetExecutionMode(ORT_PARALLEL);
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
		);
	}
	else if (_MyProvider == Device::CUDA)
	{
		const auto AvailableProviders = Ort::GetAvailableProviders();
		bool Found = true;
		for (const auto& it : AvailableProviders)
			if (it.find("CUDA") != std::string::npos)
				Found = false;
		if (Found)
			_D_Dragonian_Lib_Throw_Exception("CUDA Provider Not Found");

		OrtCUDAProviderOptionsV2* TmpCudaProviderOptionsV2 = nullptr;
		GlobalOrtApi.CreateCUDAProviderOptions(&TmpCudaProviderOptionsV2);
		_MyCudaOptionsV2 = std::shared_ptr<OrtCUDAProviderOptionsV2>(
			TmpCudaProviderOptionsV2,
			GlobalOrtApi.ReleaseCUDAProviderOptions
		);

		std::vector<const char*> OrtCUDAOptionKeys;
		std::vector<const char*> OrtCUDAOptionValues;

		_MyCUDAOptions["device_id"] = std::to_string(_MyDeviceID);

		for (const auto& it : _MyCUDAOptions)
		{
			OrtCUDAOptionKeys.emplace_back(it.first.c_str());
			OrtCUDAOptionValues.emplace_back(it.second.c_str());
		}
		
		GlobalOrtApi.UpdateCUDAProviderOptions(
			_MyCudaOptionsV2.get(),
			OrtCUDAOptionKeys.data(),
			OrtCUDAOptionValues.data(),
			OrtCUDAOptionKeys.size()
		);

		_MyOrtEnv = std::make_shared<Ort::Env>(
			_MyLoggingLevel,
			_MyLoggerId.c_str(),
			DragonianLibOrtLoggingFn,
			nullptr
		);
		_MyOrtSessionOptions->SetIntraOpNumThreads(
			static_cast<int>(_MyIntraOpNumThreads)
		);
		_MyOrtSessionOptions->SetInterOpNumThreads(
			static_cast<int>(_MyInterOpNumThreads)
		);
		_MyOrtSessionOptions->SetGraphOptimizationLevel(
			ORT_ENABLE_ALL
		);
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
		);
	}
	else if (_MyProvider == Device::DIRECTX)
	{
		const auto AvailableProviders = Ort::GetAvailableProviders();
		std::string ret;
		for (const auto& it : AvailableProviders)
			if (it.find("Dml") != std::string::npos)
				ret = it;
		if (ret.empty())
			_D_Dragonian_Lib_Throw_Exception("DML Provider Not Found");

		const OrtDmlApi* OrtDmlApi = nullptr;
		GlobalOrtApi.GetExecutionProviderApi(
			"DML",
			ORT_API_VERSION,
			reinterpret_cast<const void**>(&OrtDmlApi)
		);

		Ort::ThreadingOptions ThreadingOptions;
		ThreadingOptions.SetGlobalInterOpNumThreads(static_cast<int>(_MyInterOpNumThreads));
		ThreadingOptions.SetGlobalIntraOpNumThreads(static_cast<int>(_MyIntraOpNumThreads));

		_MyOrtEnv = std::make_shared<Ort::Env>(
			ThreadingOptions,
			DragonianLibOrtLoggingFn,
			nullptr,
			_MyLoggingLevel,
			_MyLoggerId.c_str()
		);

		_MyOrtEnv->DisableTelemetryEvents();
		OrtDmlApi->SessionOptionsAppendExecutionProvider_DML(
			*_MyOrtSessionOptions,
			int(_MyDeviceID)
		);
		_MyOrtSessionOptions->SetGraphOptimizationLevel(
			ORT_ENABLE_ALL
		);
		_MyOrtSessionOptions->SetExecutionMode(
			ORT_SEQUENTIAL
		);
		_MyOrtSessionOptions->DisablePerSessionThreads();
		_MyOrtSessionOptions->DisableMemPattern();
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(
				OrtDeviceAllocator,
				OrtMemTypeCPU
			)
		);
	}
	else
	{
		_D_Dragonian_Lib_Throw_Exception("Invalid Execution Provider");
	}

	_MyStaticLogger->LogInfo(
		L"Envireoment Created With Provider:[" +
		std::to_wstring(static_cast<int>(_MyProvider)) +
		L"], DeviceID:[" +
		std::to_wstring(_MyDeviceID)
		+ L"], IntraOpNumThreads:[" +
		std::to_wstring(_MyIntraOpNumThreads) +
		L"], InterOpNumThreads:[" +
		std::to_wstring(_MyInterOpNumThreads) +
		L"], LoggerId:[" +
		UTF8ToWideString(_MyLoggerId) +
		L"], LoggingLevel:[" +
		std::to_wstring(_MyLoggingLevel) +
		L"]"
	);
}

OnnxRuntimeModel& OnnxRuntimeEnvironmentBase::RefOnnxRuntimeModel(const std::wstring& ModelPath)
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();

	std::wstring EnvPtr;
	{
		std::wstringstream wss;
		wss << std::hex << this;
		wss >> EnvPtr;
	}
	auto Iter = GlobalOrtModelCache.find(ModelPath);
	if (Iter != GlobalOrtModelCache.end())
	{
		std::wstring HexPtr;
		{
			std::wstringstream wss;
			wss << std::hex << Iter->second.get();
			wss >> HexPtr;
		}
		_MyStaticLogger->LogInfo(
			L"Referencing Model: Instance[PTR:" + HexPtr +
			L", PATH:\"" + ModelPath +
			L"\"], Current Referece Count: " + std::to_wstring(Iter->second.use_count())
		);
		return Iter->second;
	}
	try
	{
		_MyStaticLogger->LogInfo(
			L"Loading Model: \"" + ModelPath +
			L"\" With OnnxEnvironment: Instance[PTR:" + EnvPtr + L"], Current Referece Count: 1"
		);
		auto _DeleterLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
		return GlobalOrtModelCache[ModelPath] = std::shared_ptr<Ort::Session>(
			new Ort::Session(*GetEnvironment(), ModelPath.c_str(), *GetSessionOptions()),
			[_DeleterLogger, ModelPath](const Ort::Session* Ptr)
			{
				std::wstring HexPtr;
				{
					std::wstringstream wss;
					wss << std::hex << Ptr;
					wss >> HexPtr;
				}
				delete Ptr;
				_DeleterLogger->LogInfo(L"Model Unloaded: Instance[PTR:" + HexPtr + L", PATH:\"" + ModelPath + L"\"], Current Referece Count: 0");
			}
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

void OnnxRuntimeEnvironmentBase::UnRefOnnxRuntimeModel(const std::wstring& ModelPath)
{
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	
	auto Iter = GlobalOrtModelCache.find(ModelPath);
	if (Iter != GlobalOrtModelCache.end())
	{
		std::wstring HexPtr;
		{
			std::wstringstream wss;
			wss << std::hex << Iter->second.get();
			wss >> HexPtr;
		}
		_MyStaticLogger->LogInfo(
			L"UnReference Model: Instance[PTR:" + HexPtr +
			L", PATH:\"" + ModelPath + L"\"], Current Referece Count: "
			+ std::to_wstring(Iter->second.use_count() - 1)
		);
		GlobalOrtModelCache.erase(Iter);
	}
	else
		_MyStaticLogger->LogWarn(
			L"Failed to UnReference Model: PATH:\"" + ModelPath + L"\""
		);
}

void OnnxRuntimeEnvironmentBase::ClearOnnxRuntimeModel()
{
	GlobalOrtModelCache.clear();
}

OnnxRuntimeEnvironment& OnnxRuntimeEnvironmentBase::CreateEnv(const OnnxEnvironmentOptions& Options)
{
	auto ID = L"EP:" + std::to_wstring(static_cast<int>(Options.Provider)) +
		L" DEVICE:" + std::to_wstring(Options.DeviceID) +
		L" INTER:" + std::to_wstring(Options.InterOpNumThreads) +
		L" INTRA:" + std::to_wstring(Options.IntraOpNumThreads) +
		L" LLEVEL:" + std::to_wstring(Options.LoggingLevel) +
		L" LID:" + UTF8ToWideString(Options.LoggerId);
	for (const auto& [Key, Value] : Options.CUDAOptions)
		ID += L" " + UTF8ToWideString(Key) + L":" + UTF8ToWideString(Value);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		return Iter->second;
	return GlobalOrtEnvCache[ID] = OnnxRuntimeEnvironment(
		new OnnxRuntimeEnvironmentBase(Options)
	);
}

void OnnxRuntimeEnvironmentBase::DestroyEnv(const OnnxEnvironmentOptions& Options)
{
	auto ID = L"EP:" + std::to_wstring(static_cast<int>(Options.Provider)) +
		L" DEVICE:" + std::to_wstring(Options.DeviceID) +
		L" INTER:" + std::to_wstring(Options.InterOpNumThreads) +
		L" INTRA:" + std::to_wstring(Options.IntraOpNumThreads) +
		L" LLEVEL:" + std::to_wstring(Options.LoggingLevel) +
		L" LID:" + UTF8ToWideString(Options.LoggerId);
	for (const auto& [Key, Value] : Options.CUDAOptions)
		ID += L" " + UTF8ToWideString(Key) + L":" + UTF8ToWideString(Value);
	auto Iter = GlobalOrtEnvCache.find(ID);
	if (Iter != GlobalOrtEnvCache.end())
		GlobalOrtEnvCache.erase(Iter);
}

_D_Dragonian_Lib_Onnx_Runtime_End

#endif
