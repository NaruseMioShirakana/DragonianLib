#include "../../header/Models/ModelBase.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

LibSvcModule::LibSvcModule(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_) :
	OrtApiEnv(DragonianLibOrtEnv::CreateEnv(ThreadCount_, DeviceID_, (unsigned)ExecutionProvider_))
{
	ModelExecutionProvider = ExecutionProvider_;
	OnnxEnv = OrtApiEnv->GetEnv();
	MemoryInfo = OrtApiEnv->GetMemoryInfo();
	SessionOptions = OrtApiEnv->GetSessionOptions();
}

LibSvcModule::LibSvcModule(const std::shared_ptr<DragonianLibOrtEnv>& Env_) :
	OrtApiEnv(Env_)
{
	ModelExecutionProvider = static_cast<ExecutionProviders>(Env_->GetCurProvider());
	OnnxEnv = OrtApiEnv->GetEnv();
	MemoryInfo = OrtApiEnv->GetMemoryInfo();
	SessionOptions = OrtApiEnv->GetSessionOptions();
}

LibSvcModule::~LibSvcModule()
{
	OnnxEnv = nullptr;
	MemoryInfo = nullptr;
	SessionOptions = nullptr;
}

/*
int OnnxModule::InsertMessageToEmptyEditBox(std::wstring& _inputLens)
{
#ifdef WIN32
	std::vector<TCHAR> szFileName(MaxPath);
	std::vector<TCHAR> szTitleName(MaxPath);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lpstrFile = szFileName.data();
	ofn.nMaxFile = MaxPath;
	ofn.lpstrFileTitle = szTitleName.data();
	ofn.nMaxFileTitle = MaxPath;
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = nullptr;

	constexpr TCHAR szFilter[] = TEXT("Audio (*.wav;*.mp3;*.ogg;*.flac;*.aac)\0*.wav;*.mp3;*.ogg;*.flac;*.aac\0");
	ofn.lpstrFilter = szFilter;
	ofn.lpstrTitle = nullptr;
	ofn.lpstrDefExt = TEXT("wav");
	ofn.Flags = OFN_HIDEREADONLY | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT | OFN_EXPLORER;
	if (GetOpenFileName(&ofn))
	{
		auto filePtr = szFileName.data();
		std::wstring preFix = filePtr;
		filePtr += preFix.length() + 1;
		if (!*filePtr)
			_inputLens = preFix;
		else
		{
			preFix += L'\\';
			while (*filePtr != 0)
			{
				std::wstring thisPath(filePtr);
				_inputLens += preFix + thisPath + L'\n';
				filePtr += thisPath.length() + 1;
			}
		}
	}
	else
		return -2;
	return 0;
#else
#endif
}
 */

 /*
 void OnnxModule::ChangeDevice(Device _dev)
 {
	 if (_dev == device_)
		 return;
	 device_ = _dev;
	 delete session_options;
	 delete env;
	 delete memory_info;
	 env = nullptr;
	 session_options = nullptr;
	 memory_info = nullptr;
	 switch (_dev)
	 {
		 case Device::CUDA:
		 {
			 const auto AvailableProviders = Ort::GetAvailableProviders();
			 bool ret = true;
			 for (const auto& it : AvailableProviders)
				 if (it.find("CUDA") != std::string::npos)
					 ret = false;
			 if (ret)
				 LibDLVoiceCodecThrow("CUDA Provider Not Found");
			 OrtCUDAProviderOptions cuda_option;
			 cuda_option.device_id = int(__MoeVSGPUID);
			 session_options = new Ort::SessionOptions;
			 env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
			 session_options->AppendExecutionProvider_CUDA(cuda_option);
			 session_options->SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
			 session_options->SetIntraOpNumThreads(1);
			 memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
			 break;
		 }
 #ifdef MOEVSDMLPROVIDER
		 case Device::DML:
		 {
			 const auto AvailableProviders = Ort::GetAvailableProviders();
			 std::string ret;
			 for (const auto& it : AvailableProviders)
				 if (it.find("Dml") != std::string::npos)
					 ret = it;
			 if (ret.empty())
				 LibDLVoiceCodecThrow("DML Provider Not Found");
			 const OrtApi& ortApi = Ort::GetApi();
			 const OrtDmlApi* ortDmlApi = nullptr;
			 ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));

			 const Ort::ThreadingOptions threadingOptions;
			 env = new Ort::Env(threadingOptions, ORT_LOGGING_LEVEL_VERBOSE, "");
			 env->DisableTelemetryEvents();
			 session_options = new Ort::SessionOptions;
			 ortDmlApi->SessionOptionsAppendExecutionProvider_DML(*session_options, int(__MoeVSGPUID));
			 session_options->SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
			 session_options->DisablePerSessionThreads();
			 session_options->SetExecutionMode(ORT_SEQUENTIAL);
			 session_options->DisableMemPattern();
			 memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU));
			 break;
		 }
 #endif
		 default:
		 {
			 session_options = new Ort::SessionOptions;
			 env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
			 session_options->SetIntraOpNumThreads(static_cast<int>(__MoeVSNumThreads));
			 session_options->SetGraphOptimizationLevel(ORT_ENABLE_ALL);
			 memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
			 break;
		 }
	 }
 }
  */

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End