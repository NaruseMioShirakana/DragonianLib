#ifdef DRAGONIANLIB_ONNXRT_LIB
#include <thread>
#include <ranges>

#ifdef DRAGONIANLIB_ENABLEDML
#include <providers/dml/dml_provider_factory.h>
#endif

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

Ort::AllocatorWithDefaultOptions& GetDefaultOrtAllocator()
{
	static Ort::AllocatorWithDefaultOptions Allocator;
	return Allocator;
}

static void DragonianLibOrtLoggingFn(
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

OnnxRuntimeModel::OnnxRuntimeModel(OnnxRuntimeModelPointer Model)
	: _MyModel(std::move(Model))
{
	if (!_MyModel)
		return;
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << _MyModel.get();
		wss >> HexPtr;
	}
	_MyStaticLogger->LogMessage(L"Loaded Model: Instance[PTR:" + HexPtr + L"], Current Referece Count: " + std::to_wstring(_MyModel.use_count()));
}

OnnxRuntimeModel::~OnnxRuntimeModel()
{
	if (!_MyModel)
		return;
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << _MyModel.get();
		wss >> HexPtr;
	}
	_MyStaticLogger->LogMessage(L"UnReference Model: Instance[PTR:" + HexPtr + L"], Current Referece Count: " + std::to_wstring(_MyModel.use_count() - 1));
}

OnnxRuntimeModel::OnnxRuntimeModel(const OnnxRuntimeModel& _Left) : _MyModel(_Left._MyModel)
{
	if (!_MyModel)
		return;
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << _MyModel.get();
		wss >> HexPtr;
	}
	_MyStaticLogger->LogMessage(L"Reference Model: Instance[PTR:" + HexPtr + L"], Current Referece Count: " + std::to_wstring(_MyModel.use_count()));
}

OnnxRuntimeModel& OnnxRuntimeModel::operator=(const OnnxRuntimeModel& _Left)
{
	if (this == &_Left)
		return *this;
	_MyModel = _Left._MyModel;
	if (!_MyModel)
		return *this;
	static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
	std::wstring HexPtr;
	{
		std::wstringstream wss;
		wss << std::hex << _MyModel.get();
		wss >> HexPtr;
	}
	_MyStaticLogger->LogMessage(L"Reference Model: Instance[PTR:" + HexPtr + L"], Current Referece Count: " + std::to_wstring(_MyModel.use_count()));
	return *this;
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

	if (_MyProvider == Device::CPU)
		_MyDeviceID = 0;
	if (_MyProvider == Device::CUDA || _MyProvider == Device::DIRECTX)
	{
		_MyIntraOpNumThreads = 1;
		_MyInterOpNumThreads = 1;
	}

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
	if (std::cmp_greater(_MyIntraOpNumThreads, std::thread::hardware_concurrency()))
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
	if (std::cmp_greater(_MyInterOpNumThreads, std::thread::hardware_concurrency()))
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
		_MyOrtSessionOptions->SetExecutionMode(
			ORT_SEQUENTIAL
		);
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
		);
	}
#ifdef DRAGONIANLIB_ENABLECUDA
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
			1
		);
		_MyOrtSessionOptions->SetInterOpNumThreads(
			1
		);
		_MyOrtSessionOptions->SetGraphOptimizationLevel(
			ORT_ENABLE_ALL
		);
		_MyOrtSessionOptions->AppendExecutionProvider_CUDA_V2(
			*_MyCudaOptionsV2
		);
		_MyOrtSessionOptions->SetExecutionMode(
			ORT_SEQUENTIAL
		);
		_MyOrtMemoryInfo = std::make_shared<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
		);
	}
#endif
#ifdef DRAGONIANLIB_ENABLEDML
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
		ThreadingOptions.SetGlobalInterOpNumThreads(
			1
		);
		ThreadingOptions.SetGlobalIntraOpNumThreads(
			1
		);

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
#endif
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
		return Iter->second;
	try
	{
		_MyStaticLogger->LogInfo(
			L"Loading Model: \"" + ModelPath +
			L"\" With OnnxEnvironment: Instance[PTR:" + EnvPtr + L"], Current Referece Count: 0"
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
		GlobalOrtModelCache.erase(Iter);
	else
		_MyStaticLogger->LogWarn(
			L"Failed to UnReference Model: PATH:\"" + ModelPath + L"\""
		);
}

void OnnxRuntimeEnvironmentBase::ClearOnnxRuntimeModel()
{
	GlobalOrtModelCache.clear();
}

void OnnxRuntimeEnvironmentBase::EnableMemPattern(bool Enable) const
{
	if (Enable)
		_MyOrtSessionOptions->EnableMemPattern();
	else
		_MyOrtSessionOptions->DisableMemPattern();
}

void OnnxRuntimeEnvironmentBase::EnableCpuMemArena(bool Enable) const
{
	if (Enable)
		_MyOrtSessionOptions->EnableCpuMemArena();
	else
		_MyOrtSessionOptions->DisableCpuMemArena();
}

void OnnxRuntimeEnvironmentBase::EnableProfiling(bool Enable, const std::wstring& FilePath) const
{
	if (Enable)
	{
		_MyOrtSessionOptions->EnableProfiling(FilePath.c_str());
	}
	else
	{
		_MyOrtSessionOptions->DisableProfiling();
	}
}

void OnnxRuntimeEnvironmentBase::SetIntraOpNumThreads(Int64 Threads)
{
	_MyIntraOpNumThreads = Threads;
	_MyOrtSessionOptions->SetIntraOpNumThreads(static_cast<int>(Threads));
}

void OnnxRuntimeEnvironmentBase::SetInterOpNumThreads(Int64 Threads)
{
	_MyInterOpNumThreads = Threads;
	_MyOrtSessionOptions->SetInterOpNumThreads(static_cast<int>(Threads));
}

void OnnxRuntimeEnvironmentBase::SetExecutionMode(ExecutionMode Mode) const
{
	_MyOrtSessionOptions->SetExecutionMode(Mode);
}

void OnnxRuntimeEnvironmentBase::SetGraphOptimizationLevel(GraphOptimizationLevel Level) const
{
	_MyOrtSessionOptions->SetGraphOptimizationLevel(Level);
}

void OnnxRuntimeEnvironmentBase::SetLogLevel(OrtLoggingLevel Level)
{
	_MyOrtSessionOptions->SetLogSeverityLevel(Level);
	_MyLoggingLevel = Level;
}

OnnxRuntimeEnvironment OnnxRuntimeEnvironmentBase::CreateEnv(const OnnxEnvironmentOptions& Options)
{
	return OnnxRuntimeEnvironment(
		new OnnxRuntimeEnvironmentBase(Options)
	);
}

_D_Dragonian_Lib_Onnx_Runtime_End

#endif
