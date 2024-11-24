/**
 * FileName: SvcBase.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "../TensorRTBase/TRTBase.hpp"
#include "F0Extractor/F0ExtractorManager.hpp"
#include "Cluster/ClusterManager.hpp"

#define _D_Dragonian_Lib_TRT_Svc_Space_Header _D_Dragonian_TensorRT_Lib_Space_Header namespace SingingVoiceConversion {
#define _D_Dragonian_Lib_TRT_Svc_Space_End } _D_Dragonian_TensorRT_Lib_Space_End

_D_Dragonian_Lib_TRT_Svc_Space_Header

static inline std::vector<DynaShapeSlice> HubertDynaSetting{
	{"source", nvinfer1::Dims3(1, 1, 3200)	, nvinfer1::Dims3(1, 1, 32000), nvinfer1::Dims3(1, 1, 320000)}
};

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
	std::shared_ptr<TrtModel> HubertModel = nullptr;
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
	DragonianLibSTL::Vector<float> Audio;
	DragonianLibSTL::Vector<float> F0;
	DragonianLibSTL::Vector<float> Volume;
	DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> Speaker;
	int32_t OrgLen = 0;
	int32_t SamplingRate = 32000;
	bool IsNotMute = false;
};

struct SingleAudio
{
	DragonianLibSTL::Vector<SingleSlice> Slices;
	std::wstring Path;
};

struct InferenceParams
{
	float NoiseScale = 0.3f;							//噪声修正因子          0-10
	int64_t Seed = 52468;								//种子
	int64_t SpeakerId = 0;								//角色ID
	int64_t SpkCount = 2;								//模型角色数
	float IndexRate = 0.f;								//索引比               0-1
	float ClusterRate = 0.f;							//聚类比               0-1
	float DDSPNoiseScale = 0.8f;						//DDSP噪声修正因子      0-10
	float Keys = 0.f;									//升降调               -64-64
	size_t MeanWindowLength = 2;						//均值滤波器窗口大小     1-20
	size_t Pndm = 1;									//Diffusion加速倍数    1-200
	size_t Step = 100;									//Diffusion总步数      1-1000
	double Threshold = -60.;							//音量阈值             -100-20
	long MuteCheckHopSize = 512;						//静音检测帧长
	float TBegin = 0.f;
	float TEnd = 1.f;
	std::wstring Sampler = L"Pndm";						//Diffusion采样器
	std::wstring ReflowSampler = L"Eular";				//Reflow采样器
	std::wstring F0Method = L"Dio";						//F0提取算法
	void* VocoderModel = nullptr;
	void* UserParameters = nullptr;
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
	std::vector<void*> InputData;
	std::vector<ITensorInfo> Tensors;
};

class SvcBase
{
public:
	SvcBase() = default;
	virtual ~SvcBase() = default;
	virtual DragonianLibSTL::Vector<float> SliceInference(const SingleSlice& _Slice, const InferenceParams& _Params) = 0;
	virtual void EmptyCache() = 0;
	int64_t GetSamplingRate() const { return MySamplingRate; }
	int64_t GetHopSize() const { return HopSize; }
	int64_t GetHiddenUnitKDims() const { return HiddenUnitKDims; }
	int64_t GetSpeakerCount() const { return SpeakerCount; }
	int64_t GetClusterCenterSize() const { return ClusterCenterSize; }
	bool IsVolumeEnabled() const { return EnableVolume; }
	bool IsSpeakerMixEnabled() const { return EnableCharaMix; }
	bool IsClusterEnabled() const { return EnableCluster; }
	SvcBase& operator=(const SvcBase&) = delete;
	SvcBase& operator=(SvcBase&&) = delete;
	SvcBase(const SvcBase&) = delete;
	SvcBase(SvcBase&&) = delete;

	DragonianLibSTL::Vector<float> InferenceAudio(
		const DragonianLibSTL::Vector<float>& _Audio,
		const InferenceParams& _Params,
		int64_t _SourceSamplingRate,
		size_t _SliceTime,
		bool _Refersh
	);

	template <typename T>
	static void LinearCombination(DragonianLibSTL::Vector<DragonianLibSTL::Vector<T>>& _data, size_t default_id, T Value = T(1.0))
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

protected:
	int64_t MySamplingRate = 32000, HopSize = 320, HiddenUnitKDims = 256, SpeakerCount = 1, ClusterCenterSize = 10000;
	bool EnableVolume = false, EnableCharaMix = false, EnableCluster = false;
	ProgressCallback ProgressFn;
	Cluster::Cluster Cluster;

public:
	static DragonianLibSTL::Vector<float> GetInterpedF0log(
		const DragonianLibSTL::Vector<float>&,
		bool
	);
	static DragonianLibSTL::Vector<float> GetCurrectSpkMixData(
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
		size_t dst_len, int64_t curspk, int64_t _NSpeaker
	);
	static DragonianLibSTL::Vector<float> GetSpkMixData(
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
		size_t dst_len, size_t spk_count
	);
	static DragonianLibSTL::Vector<float> ExtractVolume(
		const DragonianLibSTL::Vector<float>& _Audio, int _HopSize
	);
	static SingleAudio GetAudioSlice(
		const DragonianLibSTL::Vector<float>& _InputPCM,
		const DragonianLibSTL::Vector<size_t>& _SlicePos,
		double Threshold
	);
	static void PreProcessAudio(
		SingleAudio& _Input, int _SamplingRate, int _HopSize,
		const std::wstring& _F0Method, const void* UserParameter
	);
	static DragonianLibSTL::Vector<int64_t> GetNSFF0(
		const DragonianLibSTL::Vector<float>&
	);
	static DragonianLibSTL::Vector<float> GetInterpedF0(
		const DragonianLibSTL::Vector<float>&
	);
	static DragonianLibSTL::Vector<float> GetUV(
		const DragonianLibSTL::Vector<float>&
	);
	static DragonianLibSTL::Vector<int64_t> GetAligments(
		size_t, size_t
	);
};

_D_Dragonian_Lib_TRT_Svc_Space_End
