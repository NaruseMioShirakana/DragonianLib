#include "../Header/ModelBase.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

LibTTSModule::LibTTSModule(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_) :
	OrtApiEnv(std::make_shared<DragonianLibOrtEnv>(ThreadCount_, DeviceID_, (unsigned)ExecutionProvider_))
{
	ModelExecutionProvider = ExecutionProvider_;
	OnnxEnv = OrtApiEnv->GetEnv();
	MemoryInfo = OrtApiEnv->GetMemoryInfo();
	SessionOptions = OrtApiEnv->GetSessionOptions();
}

LibTTSModule::~LibTTSModule()
{
	OnnxEnv = nullptr;
	MemoryInfo = nullptr;
	SessionOptions = nullptr;
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End