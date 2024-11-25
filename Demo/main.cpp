#include "Libraries/AvCodec/AvCodec.h"
#include "TensorRT/SingingVoiceConversion/VitsSvc.hpp"
#include <iostream>

auto MyLastTime = std::chrono::high_resolution_clock::now();
size_t TotalStep = 0;
void ShowProgressBar(size_t progress) {
	int barWidth = 70;
	float progressRatio = static_cast<float>(progress) / float(TotalStep);
	int pos = static_cast<int>(float(barWidth) * progressRatio);

	std::cout << "\r";
	std::cout.flush();
	auto TimeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - MyLastTime).count();
	MyLastTime = std::chrono::high_resolution_clock::now();
	std::cout << "[Speed: " << 1000.0f / static_cast<float>(TimeUsed) << " it/s] ";
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
}

void ProgressCb(size_t a, size_t b)
{
	if (a == 0)
		TotalStep = b;
	ShowProgressBar(a);
}

template <typename _Function>
std::enable_if_t<std::is_invocable_v<_Function>>
WithTimer(const _Function& _Func)
{
	auto start = std::chrono::high_resolution_clock::now();
	_Func();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

enum class ModelType
{
	float32,
	float16,
	bfloat16,
	int8
};	

int main(int argc, char** argv)
{
	++argv;
	--argc;

	std::wstring ModelName = L"RVC";
	std::wstring OnnxModelPath = LR"(D:\VSGIT\MoeSS - Release\Retrieval-based-Voice-Conversion-WebUI-main\kikiV1-s.onnx)";
	std::wstring OnnxHubertPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-256-layer-9.onnx)";
	std::wstring TrtModelPath = LR"(D:\VSGIT\MoeSS - Release\Retrieval-based-Voice-Conversion-WebUI-main\kikiV1-s.trt)";
	std::wstring TrtHubertPath = LR"(D:\VSGIT\MoeVoiceStudioSvc - Core - Cmd\x64\Debug\hubert\vec-256-layer-9.trt)";
	int64_t ClusterCenterSize = 10000;
	std::wstring ClusterType;
	std::wstring ClusterPath;
	int DLACore = 0;
	bool EnableFallback = true;
	ModelType ModelPrecision = ModelType::float16;
	nvinfer1::ILogger::Severity Severity = nvinfer1::ILogger::Severity::kWARNING;
	int OptimizationLevel = 5;
	int SamplingRate = 40000;
	int HopSize = 320;
	int FeatureDimention = 256;
	int SpeakerCount = 1;
	bool EnableSpeakerMix = false;
	bool EnableVolumeEmbed = false;

	if (argc == 0 || (argc == 1 && strcmp(*argv, "--help") == 0))
	{
		std::cout << "Usage: " << " [options] means this argument is optional\n";
		std::cout << "Options:\n\n";
		std::cout << "  [--model-name] <name> (default: RVC)\n";
		std::cout << "     |- Optional Items: [\"RVC\", \"SoVits\"]\n";
		std::cout << '\n';
		std::cout << "  --onnx-model-path <path>\n";
		std::cout << "     |- Path Of Onnx Runtime Vits Model\n";
		std::cout << "     |- If TensorRT Engine Is Not Exist, This Argument Must Be Set With\n";
		std::cout << "     |  An Exist Onnx Model Path! If TensorRT Engine Is Exist, This Ar-\n";
		std::cout << "     |  gument Must Be Set With \"Vits\".\n";
		std::cout << '\n';
		std::cout << "  --onnx-hubert-path <path>\n";
		std::cout << "     |- Path Of Onnx Runtime Hubert Model\n";
		std::cout << "     |- If TensorRT Engine Is Not Exist, This Argument Must Be Set With\n";
		std::cout << "     |  An Exist Onnx Model Path! If TensorRT Engine Is Exist, This Ar-\n";
		std::cout << "     |  gument Must Be Set With \"Hubert\".\n";
		std::cout << '\n';
		std::cout << "  --trt-model-path <path>\n";
		std::cout << "     |- Path Of TensorRT Vits Engine\n";
		std::cout << "     |- If TensorRT Engine Is Not Exist, It Will Be Created Automatically\n";
		std::cout << "     |  This Argument Is The Path Of The Created TensorRT Engine. If The\n";
		std::cout << "     |  Path Is An Exist TensorRT Engine Path, It Will Be Loaded Directly.\n";
		std::cout << '\n';
		std::cout << "  --trt-hubert-path <path>\n";
		std::cout << "     |- Path Of TensorRT Hubert Engine\n";
		std::cout << "     |- If TensorRT Engine Is Not Exist, It Will Be Created Automatically\n";
		std::cout << "     |  This Argument Is The Path Of The Created TensorRT Engine. If The\n";
		std::cout << "     |  Path Is An Exist TensorRT Engine Path, It Will Be Loaded Directly.\n";
		std::cout << '\n';
		std::cout << "  [--cluster-center-size] <integer> (default: 10000)\n";
		std::cout << "     |- The Size Of Cluster Center\n";
		std::cout << '\n';
		std::cout << "  [--cluster-type] <string> (default: None)\n";
		std::cout << "     |- Type Of Cluster\n";
		std::cout << "     |- Optional Items: [\"KMeans\", \"Index\"]\n";
		std::cout << '\n';
		std::cout << "  [--cluster-path] <string> (default: None)\n";
		std::cout << "     |- Path Of Cluster File\n";
		std::cout << '\n';
		std::cout << "  [--dla-core] <integer> (default: 0)\n";
		std::cout << "     |- The Count Of The DLA Core, 0 Means Not Use DLA Core\n";
		std::cout << '\n';
		std::cout << "  [--disable-fallback] (default: true)\n";
		std::cout << "     |- Disable Operator Fallback\n";
		std::cout << '\n';
		std::cout << "  [--model-precision] <precision> (default: float16)\n";
		std::cout << "     |- The Precision Of The Model\n";
		std::cout << "     |- Optional Items: [\"float32\", \"float16\", \"bfloat16\", \"int8\"]\n";
		std::cout << '\n';
		std::cout << "  [--severity] <severity> (default: kWARNING)\n";
		std::cout << "     |- The Severity Of The Logger\n";
		std::cout << "     |- Optional Items: [\"kERROR\", \"kWARNING\", \"kINFO\", \"kVERBOSE\"]\n";
		std::cout << '\n';
		std::cout << "  [--optimization-level] <integer> (default: 5)\n";
		std::cout << "     |- The Optimization Level Of The TensorRT Engine\n";
		std::cout << "     |- Level 0: This enables the fastest compilation by disabling dynamic\n";
		std::cout << "     |  kernel generation and selecting the first tactic that succeeds in\n";
		std::cout << "     |  execution. This will also not respect a timing cache.\n";
		std::cout << "     |- Level 1: Available tactics are sorted by heuristics, but only the \n";
		std::cout << "     |  top are tested to select the best. If a dynamic kernel is gener-\n";
		std::cout << "     |  ated its compile optimization is low.\n";
		std::cout << "     |- Level 2: Available tactics are sorted by heuristics, but only the \n";
		std::cout << "     |  fastest tactics are tested to select the best.\n";
		std::cout << "     |- Level 3: Apply heuristics to see if a static precompiled kernel is\n";
		std::cout << "     |  applicable or if a new one has to be compiled dynamically.\n";
		std::cout << "     |- Level 4: Always compiles a dynamic kernel.\n";
		std::cout << "     |- Level 5: Always compiles a dynamic kernel and compares it to static\n";
		std::cout << "     |  kernels.\n";
		std::cout << '\n';
		std::cout << "  [--sampling-rate] <integer> (default: 40000)\n";
		std::cout << "     |- The Sampling Rate Of The Model\n";
		std::cout << '\n';
		std::cout << "  [--hop-size] <integer> (default: 320)\n";
		std::cout << "     |- The Hop Size Of The Model\n";
		std::cout << '\n';
		std::cout << "  [--feature-dimention] <integer> (default: 256)\n";
		std::cout << "     |- The Feature Dimention Of The Model\n";
		std::cout << '\n';
		std::cout << "  [--speaker-count] <integer> (default: 1)\n";
		std::cout << "     |- The Count Of Speaker\n";
		std::cout << '\n';
		std::cout << "  [--enable-speaker-mix] (default: false)\n";
		std::cout << "     |- Enable Speaker Mix\n";
		std::cout << '\n';
		std::cout << "  [--enable-volume-embed] (default: false)\n";
		std::cout << "     |- Enable Volume Embed\n\n\n";
		return 0;
	}

	if (argc == 1 && strcmp(*argv, "--debug") == 0) 
	{
		argv++;
		argc--;
	}
	else
	{
		OnnxModelPath.clear();
		OnnxHubertPath.clear();
		TrtModelPath.clear();
		TrtHubertPath.clear();
	}

	while (argc)
	{
		
		if (strcmp(*argv, "--model-name") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --model-name\n";
				return 1;
			}
			ModelName = DragonianLib::UTF8ToWideString(*argv);
			if (ModelName != L"RVC" && ModelName != L"SoVits")
			{
				std::cerr << "Error: Unknown Model Name " << *argv << '\n';
				return 1;
			}
		}
		else if (strcmp(*argv, "--onnx-model-path") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --onnx-model-path\n";
				return 1;
			}
			OnnxModelPath = DragonianLib::UTF8ToWideString(*argv);
			if (OnnxModelPath.empty())
			{
				std::cerr << "Error: Onnx Model Path Must Be Set\n";
				return 1;
			}
			if (OnnxHubertPath == OnnxModelPath)
			{
				std::cerr << "Error: Onnx Hubert Path Must Be Different From Onnx Model Path\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--onnx-hubert-path") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --onnx-hubert-path\n";
				return 1;
			}
			OnnxHubertPath = DragonianLib::UTF8ToWideString(*argv);
			if (OnnxHubertPath.empty())
			{
				std::cerr << "Error: Onnx Hubert Path Must Be Set\n";
				return 1;
			}
			if (OnnxHubertPath == OnnxModelPath)
			{
				std::cerr << "Error: Onnx Hubert Path Must Be Different From Onnx Model Path\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--trt-model-path") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --trt-model-path\n";
				return 1;
			}
			TrtModelPath = DragonianLib::UTF8ToWideString(*argv);
			if (TrtModelPath.empty())
			{
				std::cerr << "Error: TensorRT Model Path Must Be Set\n";
				return 1;
			}
			if (TrtHubertPath == TrtModelPath)
			{
				std::cerr << "Error: TensorRT Hubert Path Must Be Different From TensorRT Model Path\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--trt-hubert-path") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --trt-hubert-path\n";
				return 1;
			}
			TrtHubertPath = DragonianLib::UTF8ToWideString(*argv);
			if (TrtHubertPath.empty())
			{
				std::cerr << "Error: TensorRT Hubert Path Must Be Set\n";
				return 1;
			}
			if (TrtHubertPath == TrtModelPath)
			{
				std::cerr << "Error: TensorRT Hubert Path Must Be Different From TensorRT Model Path\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--cluster-center-size") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --cluster-center-size\n";
				return 1;
			}
			ClusterCenterSize = std::stoll(*argv);
			if (ClusterCenterSize <= 0)
			{
				std::cerr << "Error: Cluster Center Size Must Be Greater Than 0\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--cluster-type") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --cluster-type\n";
				return 1;
			}
			ClusterType = DragonianLib::UTF8ToWideString(*argv);
			if (ClusterType != L"KMeans" && ClusterType != L"Index")
			{
				std::cerr << "Error: Unknown Cluster Type " << *argv << '\n';
				return 1;
			}
		}
		else if (strcmp(*argv, "--cluster-path") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --cluster-path\n";
				return 1;
			}
			ClusterPath = DragonianLib::UTF8ToWideString(*argv);
		}
		else if (strcmp(*argv, "--dla-core") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --dla-core\n";
				return 1;
			}
			DLACore = std::stoi(*argv);
			if (DLACore < 0)
			{
				std::cerr << "Error: DLA Core Must Be Greater Than Or Equal To 0\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--disable-fallback") == 0)
		{
			EnableFallback = false;
		}
		else if (strcmp(*argv, "--model-precision") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --model-precision\n";
				return 1;
			}
			auto Precision = DragonianLib::UTF8ToWideString(*argv);
			if (Precision == L"float32")
				ModelPrecision = ModelType::float32;
			else if (Precision == L"float16")
				ModelPrecision = ModelType::float16;
			else if (Precision == L"bfloat16")
				ModelPrecision = ModelType::bfloat16;
			else if (Precision == L"int8")
				ModelPrecision = ModelType::int8;
			else
			{
				std::cerr << "Error: Unknown Model Precision " << *argv << '\n';
				return 1;
			}
		}
		else if (strcmp(*argv, "--severity") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --severity\n";
				return 1;
			}
			auto SeverityStr = DragonianLib::UTF8ToWideString(*argv);
			if (SeverityStr == L"kERROR")
				Severity = nvinfer1::ILogger::Severity::kERROR;
			else if (SeverityStr == L"kWARNING")
				Severity = nvinfer1::ILogger::Severity::kWARNING;
			else if (SeverityStr == L"kINFO")
				Severity = nvinfer1::ILogger::Severity::kINFO;
			else if (SeverityStr == L"kVERBOSE")
				Severity = nvinfer1::ILogger::Severity::kVERBOSE;
			else
			{
				std::cerr << "Error: Unknown Severity " << *argv << '\n';
				return 1;
			}
		}
		else if (strcmp(*argv, "--optimization-level") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --optimization-level\n";
				return 1;
			}
			OptimizationLevel = std::stoi(*argv);
			if (OptimizationLevel < 0 || OptimizationLevel > 5)
			{
				std::cerr << "Error: Optimization Level Must Be In Range [0, 5]\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--sampling-rate") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --sampling-rate\n";
				return 1;
			}
			SamplingRate = std::stoi(*argv);
			if (SamplingRate < 16000)
			{
				std::cerr << "Error: Sampling Rate Must Be GreaterEqual Than 16000\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--hop-size") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --hop-size\n";
				return 1;
			}
			HopSize = std::stoi(*argv);
			if (HopSize < 1)
			{
				std::cerr << "Error: Hop Size Must Be Greater Than 0\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--feature-dimention") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --feature-dimention\n";
				return 1;
			}
			FeatureDimention = std::stoi(*argv);
			if (FeatureDimention % 256)
			{
				std::cerr << "Error: Feature Dimention Must Be Multiple Of 256\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--speaker-count") == 0)
		{
			++argv;
			--argc;
			if (!argc)
			{
				std::cerr << "Error: Missing Argument Value For --speaker-count\n";
				return 1;
			}
			SpeakerCount = std::stoi(*argv);
			if (SpeakerCount < 1)
			{
				std::cerr << "Error: Speaker Count Must Be Greater Than 0\n";
				return 1;
			}
		}
		else if (strcmp(*argv, "--enable-speaker-mix") == 0)
		{
			EnableSpeakerMix = true;
		}
		else if (strcmp(*argv, "--enable-volume-embed") == 0)
		{
			EnableVolumeEmbed = true;
		}
		else
			std::cerr << "Error: Unknown Argument " << *argv << '\n';
		++argv;
		--argc;
	}

	DragonianLib::TensorRTLib::SingingVoiceConversion::VitsSvcConfig MyConfig{
			OnnxModelPath,
			OnnxHubertPath,
			ModelName,
			nullptr,
			DragonianLib::TensorRTLib::SingingVoiceConversion::ClusterConfig{
				ClusterCenterSize,
				ClusterPath,
				ClusterType
			},
			DragonianLib::TensorRTLib::TrtConfig{
				{
					{
						OnnxModelPath,
						TrtModelPath
					},
					{
						OnnxHubertPath,
						TrtHubertPath
					}
				},
				{},
				DLACore,
				EnableFallback,
				ModelPrecision == ModelType::float16,
				ModelPrecision == ModelType::bfloat16,
				ModelPrecision == ModelType::int8,
				Severity,
				OptimizationLevel
			},
			SamplingRate,
			HopSize,
			FeatureDimention,
			SpeakerCount,
			EnableSpeakerMix,
			EnableVolumeEmbed
	};
	auto DynaSetting = DragonianLib::TensorRTLib::SingingVoiceConversion::VitsSvc::VitsSvcDefaultsDynaSetting;
	DynaSetting[1].Max.d[2] = MyConfig.HiddenUnitKDims;
	DynaSetting[1].Min.d[2] = MyConfig.HiddenUnitKDims;
	DynaSetting[1].Opt.d[2] = MyConfig.HiddenUnitKDims;
	MyConfig.TrtSettings.DynaSetting = std::move(DynaSetting);
	DragonianLib::TensorRTLib::SingingVoiceConversion::VitsSvc Model{
		MyConfig,
		ProgressCb
	};

	auto SourceSamplingRate = 48000;
	std::cout << "\nPress Target Sampling Rate: > ";
	std::cin >> SourceSamplingRate;
	std::cin.ignore();
	bool Continue = true;
	while (Continue)
	{
		wchar_t SourcePath[1024];
		std::cout << "Press Source Path: > ";
		std::wcin.getline(SourcePath, 1024);

		DragonianLibSTL::Vector<float> Audio;
		try
		{
			Audio = DragonianLib::AvCodec::AvCodec().DecodeFloat(
				DragonianLib::UnicodeToAnsi(SourcePath).c_str(),
				SourceSamplingRate
			);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << "\n\n";
			continue;
		}

		const auto AudioSeconds = static_cast<double>(Audio.Size()) / static_cast<double>(SourceSamplingRate);
		std::cout << "\n\n//******************************************Start Inference******************************************//\n\n";

		try
		{
			MyLastTime = std::chrono::high_resolution_clock::now();
			auto TimeBegin = std::chrono::high_resolution_clock::now();
			Audio = Model.InferenceAudio(
				Audio,
				DragonianLib::TensorRTLib::SingingVoiceConversion::InferenceParams{},
				SourceSamplingRate,
				4,
				true
			);
			const auto UsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TimeBegin).count();
			auto InferenceDurations = static_cast<double>(UsedTime) / 1000.0;
			std::cout << "\n\nInference Use Times: " << InferenceDurations << "s\n";
			std::cout << "Audio Durations: " << AudioSeconds << "s\n";
			std::cout << "Inference Speed: " << AudioSeconds / InferenceDurations << "x\n";
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << '\n';
		}

		bool Output = true;
		while (Output)
		{
			wchar_t OutputPath[1024];
			std::cout << "\nPress Output Path: > ";
			std::wcin.getline(OutputPath, 1024);
			try
			{
				DragonianLib::AvCodec::WritePCMData(
					OutputPath,
					Audio,
					SourceSamplingRate
				);
				std::cout << "Output Audio Saved To: " << DragonianLib::UnicodeToAnsi(OutputPath) << '\n';
				Output = false;
			}
			catch (const std::exception& e)
			{
				std::cout << e.what() << '\n';
			}
		}

		std::cout << "\n//*******************************************End Inference*******************************************//\n";
		std::cout << "\nContinue? [0/1] ";
		std::cin >> Continue;
		std::cin.ignore();
	}

	
	return 0;
}