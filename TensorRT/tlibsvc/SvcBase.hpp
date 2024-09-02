#pragma once
#include "../TRTBase.hpp"

namespace tlibsvc
{
	using namespace TensorRTLib;

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

	struct VitsSvcConfig
	{
		std::wstring ModelPath;
		std::wstring HubertPath;
		std::wstring TensorExtractor = L"SoVits4.0";
		std::shared_ptr<TrtModel> HubertModel;
		ClusterConfig Cluster;
		TrtConfig TrtSettings;

		long SamplingRate = 22050;
		int HopSize = 320;
		int64_t HiddenUnitKDims = 256;
		int64_t SpeakerCount = 1;
		bool EnableCharaMix = false;
		bool EnableVolume = false;
	};

	struct SingleSlice
	{
		DragonianLibSTL::Vector<int16_t> Audio;
		DragonianLibSTL::Vector<float> F0;
		DragonianLibSTL::Vector<float> Volume;
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> Speaker;
		int32_t OrgLen = 0;
		bool IsNotMute = false;
	};

	struct SingleAudio
	{
		DragonianLibSTL::Vector<SingleSlice> Slices;
		std::wstring Path;
	};

	struct InferenceParams
	{
		float NoiseScale = 0.3f;                           //噪声修正因子          0-10
		int64_t Seed = 52468;                              //种子
		int64_t SpeakerId = 0;                             //角色ID
		uint64_t SrcSamplingRate = 48000;                  //源采样率
		int64_t SpkCount = 2;                              //模型角色数
		float IndexRate = 0.f;                             //索引比               0-1
		float ClusterRate = 0.f;                           //聚类比               0-1
		float DDSPNoiseScale = 0.8f;                       //DDSP噪声修正因子      0-10
		float Keys = 0.f;                                  //升降调               -64-64
		size_t MeanWindowLength = 2;                       //均值滤波器窗口大小     1-20
		size_t Pndm = 1;                                   //Diffusion加速倍数    1-200
		size_t Step = 100;                                 //Diffusion总步数      1-1000
		float TBegin = 0.f;
		float TEnd = 1.f;
		std::wstring Sampler = L"Pndm";                    //Diffusion采样器
		std::wstring ReflowSampler = L"Eular";             //Reflow采样器
		std::wstring F0Method = L"Dio";                    //F0提取算法
		bool UseShallowDiffusion = false;                  //使用浅扩散
		void* VocoderModel = nullptr;
		void* ShallowDiffusionModel = nullptr;
		bool ShallowDiffusionUseSrcAudio = true;
		int VocoderHopSize = 512;
		int VocoderMelBins = 128;
		int VocoderSamplingRate = 44100;
		int64_t ShallowDiffuisonSpeaker = 0;
	};

	struct TensorData
	{
		DragonianLibSTL::Vector<float> HiddenUnit;
		DragonianLibSTL::Vector<float> F0;
		DragonianLibSTL::Vector<float> Volume;
		DragonianLibSTL::Vector<float> SpkMap;
		DragonianLibSTL::Vector<float> DDSPNoise;
		DragonianLibSTL::Vector<float> Noise;
		DragonianLibSTL::Vector<int64_t> Alignment;
		DragonianLibSTL::Vector<float> UnVoice;
		DragonianLibSTL::Vector<int64_t> NSFF0;
		int64_t Length[1] = { 0 };
		int64_t Speaker[1] = { 0 };

		static inline nvinfer1::Dims OneShape{ 1, {1,0,0,0,0,0,0,0} };
	};

	struct TensorXData
	{
		TensorData Data;
		DragonianLibSTL::Vector<TrtTensor> Tensors;
	};

	//获取换算为0-255的f0
	[[nodiscard]] DragonianLibSTL::Vector<int64_t> GetNSFF0(const DragonianLibSTL::Vector<float>&);

	//将F0中0值单独插值
	DragonianLibSTL::Vector<float> GetInterpedF0(const DragonianLibSTL::Vector<float>&);

	//
	DragonianLibSTL::Vector<float> InterpUVF0(const DragonianLibSTL::Vector<float>&, size_t PaddedIndex = size_t(-1));

	//获取UnVoiceMask
	DragonianLibSTL::Vector<float> GetUV(const DragonianLibSTL::Vector<float>&);

	//获取对齐矩阵
	DragonianLibSTL::Vector<int64_t> GetAligments(size_t, size_t);

	//线性组合
	template <typename T>
	void LinearCombination(DragonianLibSTL::Vector<DragonianLibSTL::Vector<T>>& _data, size_t default_id, T Value = T(1.0))
	{
		if (_data.Empty())
			return;
		if (default_id > _data.Size())
			default_id = 0;

		for (size_t i = 0; i < _data[0].Size(); ++i)
		{
			T Sum = T(0.0);
			for (size_t j = 0; j < _data.Size(); ++j)
				Sum += _data[j][i];
			if (Sum < T(0.0001))
			{
				for (size_t j = 0; j < _data.Size(); ++j)
					_data[j][i] = T(0);
				_data[default_id][i] = T(1);
				continue;
			}
			Sum *= T(Value);
			for (size_t j = 0; j < _data.Size(); ++j)
				_data[j][i] /= Sum;
		}
	}

	//将F0中0值单独插值（可设置是否取log）
	[[nodiscard]] DragonianLibSTL::Vector<float> GetInterpedF0log(const DragonianLibSTL::Vector<float>&, bool);

	//获取正确的角色混合数据
	[[nodiscard]] DragonianLibSTL::Vector<float> GetCurrectSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, int64_t curspk, int64_t _NSpeaker);

	//获取正确的角色混合数据
	[[nodiscard]] DragonianLibSTL::Vector<float> GetSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, size_t spk_count);
}
