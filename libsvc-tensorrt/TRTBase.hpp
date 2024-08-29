#include <onnxruntime_cxx_api.h>
#include "Base.h"
#include "Models/Params.hpp"

namespace tlibsvc 
{
	using namespace libsvc;
	using ProgressCallback = std::function<void(size_t, size_t)>;

	struct DiffusionSvcPaths
	{
		std::wstring Encoder;
		std::wstring Denoise;
		std::wstring Pred;
		std::wstring After;
		std::wstring Alpha;
		std::wstring Naive;

		std::wstring DiffSvc;
	};

	struct ReflowSvcPaths
	{
		std::wstring Encoder;
		std::wstring VelocityFn;
		std::wstring After;
	};

	struct VitsSvcPaths
	{
		std::wstring VitsSvc;
	};

	struct ClusterConfig
	{
		int64_t ClusterCenterSize = 10000;
		std::wstring Path;
		/**
		 * \brief Type Of Cluster : "KMeans" "Index"
		 */
		std::wstring Type;
	};

	struct Hparams
	{
		/**
		 * \brief Model Version
		 * For VitsSvc : "SoVits2.0" "SoVits3.0" "SoVits4.0" "SoVits4.0-DDSP" "RVC"
		 * For DiffusionSvc : "DiffSvc" "DiffusionSvc"
		 */
		std::wstring TensorExtractor = L"DiffSvc";
		/**
		 * \brief Path Of Hubert Model
		 */
		std::wstring HubertPath;
		/**
		 * \brief Path Of DiffusionSvc Model
		 */
		DiffusionSvcPaths DiffusionSvc;
		/**
		 * \brief Path Of VitsSvc Model
		 */
		VitsSvcPaths VitsSvc;
		/**
		 * \brief Path Of ReflowSvc Model
		 */
		ReflowSvcPaths ReflowSvc;
		/**
		 * \brief Config Of Cluster
		 */
		ClusterConfig Cluster;

		long SamplingRate = 22050;

		int HopSize = 320;
		int64_t HiddenUnitKDims = 256;
		int64_t SpeakerCount = 1;
		bool EnableCharaMix = false;
		bool EnableVolume = false;
		bool VaeMode = true;

		int64_t MelBins = 128;
		int64_t Pndms = 100;
		int64_t MaxStep = 1000;
		float SpecMin = -12;
		float SpecMax = 2;
		float Scale = 1000.f;
	};
}