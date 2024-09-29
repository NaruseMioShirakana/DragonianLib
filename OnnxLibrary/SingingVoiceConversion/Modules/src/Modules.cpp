#include "../header/Modules.hpp"
#include "F0Extractor/DioF0Extractor.hpp"
#include "F0Extractor/F0ExtractorManager.hpp"
#include "../header/InferTools/TensorExtractor/TensorExtractor.hpp"
#include "Cluster/ClusterManager.hpp"
#include "Cluster/KmeansCluster.hpp"
#include "F0Extractor/HarvestF0Extractor.hpp"
#include "../header/InferTools/Sampler/SamplerManager.hpp"
#include "../header/InferTools/Sampler/Samplers.hpp"
#include "F0Extractor/NetF0Predictors.hpp"
#include "Cluster/IndexCluster.hpp"

#define RegisterF0ConstructorImp(__RegisterName, __ClassName) DragonianLib::RegisterF0Extractor(__RegisterName,   \
	[](int32_t sampling_rate, int32_t hop_size,																			\
	int32_t n_f0_bins, double max_f0, double min_f0)																	\
	-> DragonianLib::F0Extractor																					\
	{																													\
		return std::make_shared<DragonianLib::__ClassName>(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0);					\
	})

#define RegisterTensorConstructorImp(__RegisterName, __ClassName) LibSvcSpace RegisterTensorExtractor(__RegisterName,\
	[](uint64_t _srcsr, uint64_t _sr, uint64_t _hop,													        \
		bool _smix, bool _volume, uint64_t _hidden_size,													    \
		uint64_t _nspeaker,																	                    \
		const LibSvcSpace LibSvcTensorExtractor::Others& _other)                             \
		->LibSvcSpace TensorExtractor													            \
	{																										    \
		return std::make_shared<LibSvcSpace __ClassName>(_srcsr, _sr, _hop, _smix, _volume,						\
			_hidden_size, _nspeaker, _other);													                \
	})

#define RegisterClusterImp(__RegisterName, __ClassName) DragonianLib::RegisterCluster(__RegisterName,\
	[](const std::wstring& _path, size_t hidden_size, size_t KmeansLen)												 \
		->DragonianLib::ClusterWrp																		 \
	{																												 \
		return std::make_shared<DragonianLib::__ClassName>(_path, hidden_size, KmeansLen);								 \
	})

#define RegisterSamplerImp(__RegisterName, __ClassName) LibSvcSpace RegisterSampler(__RegisterName,		 \
	[](Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins,								 \
		const LibSvcSpace BaseSampler::ProgressCallback& _ProgressCallback,								 	 \
		Ort::MemoryInfo* memory) -> LibSvcSpace SamplerWrp														 \
	{																												 \
		return std::make_shared<LibSvcSpace __ClassName>(alpha, dfn, pred, Mel_Bins, _ProgressCallback, memory);				 \
	})

#define RegisterReflowSamplerImp(__RegisterName, __ClassName) LibSvcSpace RegisterReflowSampler(__RegisterName,		 \
	[](Ort::Session* velocity, int64_t Mel_Bins,								 \
		const LibSvcSpace BaseSampler::ProgressCallback& _ProgressCallback,								 	 \
		Ort::MemoryInfo* memory) -> LibSvcSpace ReflowSamplerWrp														 \
	{																												 \
		return std::make_shared<LibSvcSpace __ClassName>(velocity, Mel_Bins, _ProgressCallback, memory);				 \
	})

LibSvcHeader

bool KernelSetup = false;
std::unordered_map<std::wstring, DlCodecStft::Mel*> MelOperators;

void SetupKernel()
{
	if (KernelSetup)
		return;
	RegisterF0ConstructorImp(L"Dio", DioF0Extractor);
	RegisterF0ConstructorImp(L"Harvest", HarvestF0Extractor);
	RegisterF0ConstructorImp(L"RMVPE", RMVPEF0Extractor);
	RegisterF0ConstructorImp(L"FCPE", MELPEF0Extractor);
	RegisterTensorConstructorImp(L"SoVits2.0", SoVits2TensorExtractor);
	RegisterTensorConstructorImp(L"SoVits3.0", SoVits3TensorExtractor);
	RegisterTensorConstructorImp(L"SoVits4.0", SoVits4TensorExtractor);
	RegisterTensorConstructorImp(L"SoVits4.0-DDSP", SoVits4DDSPTensorExtractor);
	RegisterTensorConstructorImp(L"RVC", RVCTensorExtractor);
	RegisterTensorConstructorImp(L"DiffSvc", DiffSvcTensorExtractor);
	RegisterTensorConstructorImp(L"DiffusionSvc", DiffusionSvcTensorExtractor);
	RegisterTensorConstructorImp(L"ReflowSvc", DiffusionSvcTensorExtractor);
	RegisterTensorConstructorImp(L"UnionSvc", DiffusionSvcTensorExtractor);
	RegisterClusterImp(L"KMeans", KMeansCluster);
	RegisterClusterImp(L"Index", IndexCluster);
	RegisterSamplerImp(L"Pndm", PndmSampler);
	RegisterSamplerImp(L"DDim", DDimSampler);
	RegisterReflowSamplerImp(L"Eular", ReflowEularSampler);
	RegisterReflowSamplerImp(L"Rk4", ReflowRk4Sampler);
	RegisterReflowSamplerImp(L"Heun", ReflowHeunSampler);
	RegisterReflowSamplerImp(L"Pecece", ReflowPececeSampler);
	KernelSetup = true;
}

DlCodecStft::Mel& GetMelOperator(
	int32_t _SamplingRate,
	int32_t _Hopsize,
	int32_t _MelBins
)
{
	const std::wstring _Name = L"S" +
		std::to_wstring(_SamplingRate) +
		L"H" + std::to_wstring(_Hopsize) +
		L"M" + std::to_wstring(_MelBins);
	if (!MelOperators.contains(_Name))
	{
		if (MelOperators.size() > 10)
		{
			delete MelOperators.begin()->second;
			MelOperators.erase(MelOperators.begin());
		}
		MelOperators[_Name] = new DlCodecStft::Mel(_Hopsize * 4, _Hopsize, _SamplingRate, _MelBins);
	}
	return *MelOperators[_Name];
}

LibSvcEnd
