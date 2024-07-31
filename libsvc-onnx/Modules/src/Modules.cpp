#include "Modules.hpp"
#include "F0Extractor/DioF0Extractor.hpp"
#include "F0Extractor/F0ExtractorManager.hpp"
#include "InferTools/TensorExtractor/TensorExtractor.hpp"
#include "Cluster/ClusterManager.hpp"
#include "Cluster/KmeansCluster.hpp"
#include "F0Extractor/HarvestF0Extractor.hpp"
#include "InferTools/Sampler/SamplerManager.hpp"
#include "InferTools/Sampler/Samplers.hpp"
#include "F0Extractor/NetF0Predictors.hpp"
#include "Cluster/IndexCluster.hpp"

#define RegisterF0ConstructorImp(__RegisterName, __ClassName) DragonianLib::RegisterF0Extractor(__RegisterName,   \
	[](int32_t sampling_rate, int32_t hop_size,																			\
	int32_t n_f0_bins, double max_f0, double min_f0)																	\
	-> DragonianLib::F0Extractor																					\
	{																													\
		return new DragonianLib::__ClassName(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0);					\
	})

#define RegisterTensorConstructorImp(__RegisterName, __ClassName) libsvc::RegisterTensorExtractor(__RegisterName,\
	[](uint64_t _srcsr, uint64_t _sr, uint64_t _hop,													        \
		bool _smix, bool _volume, uint64_t _hidden_size,													    \
		uint64_t _nspeaker,																	                    \
		const libsvc::LibSvcTensorExtractor::Others& _other)                             \
		->libsvc::TensorExtractor													            \
	{																										    \
		return new libsvc::__ClassName(_srcsr, _sr, _hop, _smix, _volume,						\
			_hidden_size, _nspeaker, _other);													                \
	})

#define RegisterClusterImp(__RegisterName, __ClassName) DragonianLib::RegisterCluster(__RegisterName,\
	[](const std::wstring& _path, size_t hidden_size, size_t KmeansLen)												 \
		->DragonianLib::ClusterWrp																		 \
	{																												 \
		return new DragonianLib::__ClassName(_path, hidden_size, KmeansLen);								 \
	})

#define RegisterSamplerImp(__RegisterName, __ClassName) libsvc::RegisterSampler(__RegisterName,		 \
	[](Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins,								 \
		const libsvc::BaseSampler::ProgressCallback& _ProgressCallback,								 	 \
		Ort::MemoryInfo* memory) -> libsvc::SamplerWrp														 \
	{																												 \
		return new libsvc::__ClassName(alpha, dfn, pred, Mel_Bins, _ProgressCallback, memory);				 \
	})

#define RegisterReflowSamplerImp(__RegisterName, __ClassName) libsvc::RegisterReflowSampler(__RegisterName,		 \
	[](Ort::Session* velocity, int64_t Mel_Bins,								 \
		const libsvc::BaseSampler::ProgressCallback& _ProgressCallback,								 	 \
		Ort::MemoryInfo* memory) -> libsvc::ReflowSamplerWrp														 \
	{																												 \
		return new libsvc::__ClassName(velocity, Mel_Bins, _ProgressCallback, memory);				 \
	})

namespace libsvc
{
	libsvc::SingingVoiceConversion* UnionSvcModel::GetPtr() const
	{
		if (Diffusion_) return Diffusion_;
		return Reflow_;
	}

	DragonianLibSTL::Vector<int16_t> UnionSvcModel::InferPCMData(const DragonianLibSTL::Vector<int16_t>& _PCMData, long _SrcSamplingRate, const libsvc::InferenceParams& _Params) const
	{
		DragonianLibSTL::Vector<int16_t> Audio;
		if (Diffusion_) Audio = Diffusion_->InferPCMData(_PCMData, _SrcSamplingRate, _Params);
		else Audio = Reflow_->InferPCMData(_PCMData, _SrcSamplingRate, _Params);
		Audio.Resize(_PCMData.Size(), 0i16);
		return Audio;
	}

	DragonianLibSTL::Vector<int16_t> UnionSvcModel::ShallowDiffusionInference(
		DragonianLibSTL::Vector<float>& _16KAudioHubert,
		const libsvc::InferenceParams& _Params,
		std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
		const DragonianLibSTL::Vector<float>& _SrcF0,
		const DragonianLibSTL::Vector<float>& _SrcVolume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
		size_t& Process,
		int64_t SrcSize
	)const
	{
		if (Diffusion_) return Diffusion_->ShallowDiffusionInference(_16KAudioHubert, _Params, _Mel, _SrcF0, _SrcVolume, _SrcSpeakerMap, Process, SrcSize);
		return Reflow_->ShallowDiffusionInference(_16KAudioHubert, _Params, _Mel, _SrcF0, _SrcVolume, _SrcSpeakerMap, Process, SrcSize);
	}

	DragonianLibSTL::Vector<int16_t> UnionSvcModel::SliceInference(const libsvc::SingleSlice& _Slice, const libsvc::InferenceParams& _Params, size_t& _Process) const
	{
		if (Diffusion_) return Diffusion_->SliceInference(_Slice, _Params, _Process);
		return Reflow_->SliceInference(_Slice, _Params, _Process);
	}

	UnionSvcModel::UnionSvcModel(const libsvc::Hparams& Config, const libsvc::LibSvcModule::ProgressCallback& Callback, int ProviderID, int NumThread, int DeviceID)
	{
		if ((Config.TensorExtractor.find(L"Diff") != std::wstring::npos) || Config.TensorExtractor.find(L"diff") != std::wstring::npos)
			Diffusion_ = new libsvc::DiffusionSvc(
				Config, Callback,
				libsvc::LibSvcModule::ExecutionProviders(ProviderID),
				DeviceID, NumThread
			);
		else
			Reflow_ = new libsvc::ReflowSvc(
				Config, Callback,
				libsvc::LibSvcModule::ExecutionProviders(ProviderID),
				DeviceID, NumThread
			);
	}

	UnionSvcModel::~UnionSvcModel()
	{
		delete Diffusion_;
		delete Reflow_;
		Diffusion_ = nullptr;
		Reflow_ = nullptr;
	}

	int64_t UnionSvcModel::GetMaxStep() const
	{
		if (Diffusion_) return Diffusion_->GetMaxStep();
		return Reflow_->GetMaxStep();
	}

	bool UnionSvcModel::OldVersion() const
	{
		if (Diffusion_) return Diffusion_->OldVersion();
		return false;
	}

	const std::wstring& UnionSvcModel::GetDiffSvcVer() const
	{
		if (Diffusion_) return Diffusion_->GetDiffSvcVer();
		return Reflow_->GetReflowSvcVer();
	}

	int64_t UnionSvcModel::GetMelBins() const
	{
		if (Diffusion_) return Diffusion_->GetMelBins();
		return Reflow_->GetMelBins();
	}

	int UnionSvcModel::GetHopSize() const
	{
		if (Diffusion_) return Diffusion_->GetHopSize();
		return Reflow_->GetHopSize();
	}

	int64_t UnionSvcModel::GetHiddenUnitKDims() const
	{
		if (Diffusion_) return Diffusion_->GetHiddenUnitKDims();
		return Reflow_->GetHiddenUnitKDims();
	}

	int64_t UnionSvcModel::GetSpeakerCount() const
	{
		if (Diffusion_) return Diffusion_->GetSpeakerCount();
		return Reflow_->GetSpeakerCount();
	}

	bool UnionSvcModel::CharaMixEnabled() const
	{
		if (Diffusion_) return Diffusion_->SpeakerMixEnabled();
		return Reflow_->SpeakerMixEnabled();
	}

	long UnionSvcModel::GetSamplingRate() const
	{
		if (Diffusion_) return Diffusion_->GetSamplingRate();
		return Reflow_->GetSamplingRate();
	}

	void UnionSvcModel::NormMel(DragonianLibSTL::Vector<float>& MelSpec) const
	{
		if (Diffusion_) return Diffusion_->NormMel(MelSpec);
		return Reflow_->NormMel(MelSpec);
	}

	bool UnionSvcModel::IsDiffusion() const
	{
		return Diffusion_;
	}

	DragonianLib::DragonianLibOrtEnv& UnionSvcModel::GetDlEnv()
	{
		if (Diffusion_) return Diffusion_->GetDlEnv();
		return Reflow_->GetDlEnv();
	}

	const DragonianLib::DragonianLibOrtEnv& UnionSvcModel::GetDlEnv() const
	{
		if (Diffusion_) return Diffusion_->GetDlEnv();
		return Reflow_->GetDlEnv();
	}

}

namespace libsvc
{
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
}