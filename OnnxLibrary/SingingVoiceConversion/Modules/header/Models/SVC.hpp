/**
 * FileName: SVC.hpp
 * Note: MoeVoiceStudioCore OnnxSvc 模型基类
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
 * date: 2022-10-17 Create
*/

#pragma once
#include "ModelBase.hpp"
#include "../InferTools/TensorExtractor/TensorExtractorManager.hpp"
#include "Cluster/ClusterManager.hpp"

LibSvcHeader

using OrtTensors = std::vector<Ort::Value>;
/*
class OnnxModule
{
public:
	enum class Device
	{
		CPU = 0,
		CUDA = 1,
#ifdef MOEVSDMLPROVIDER
		DML = 2
#endif
	};
	using callback = std::function<void(size_t, size_t)>;
	using int64 = int64_t;
	using MTensor = Ort::Value;

	OnnxModule();
	virtual ~OnnxModule();
	void ChangeDevice(Device _dev);

	static std::vector<std::wstring> CutLens(const std::wstring& input);

	[[nodiscard]] long GetSamplingRate() const
	{
		return _samplingRate;
	}

	template <typename T = float>
	static void LinearCombination(std::vector<T>& _data, T Value = T(1.0))
	{
		if(_data.empty())
		{
			_data = std::vector<T>(1, Value);
			return;
		}
		T Sum = T(0.0);
		for(const auto& i : _data)
			Sum += i;
		if (Sum < T(0.0001))
		{
			_data = std::vector<T>(_data.size(), T(0.0));
			_data[0] = Value;
			return;
		}
		Sum *= T(Value);
		for (auto& i : _data)
			i /= Sum;
	}
protected:
	Ort::Env* env = nullptr;
	Ort::SessionOptions* session_options = nullptr;
	Ort::MemoryInfo* memory_info = nullptr;

	modelType _modelType = modelType::SoVits;
	Device device_ = Device::CPU;

	long _samplingRate = 22050;

	callback _callback;

	static constexpr long MaxPath = 8000l;
	std::wstring _outputPath = GetCurrentFolder() + L"\\outputs";
};
 */

class SingingVoiceConversion : public LibSvcModule
{
public:
	SingingVoiceConversion(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_ = 0);

	/**
	 * \brief 推理一个切片
	 * \param _Slice 切片数据
	 * \param _Params 推理参数
	 * \param _Process 推理进度指针
	 * \return 推理结果（PCM signed-int16 单声道）
	 */
	[[nodiscard]] virtual DragonianLibSTL::Vector<int16_t> SliceInference(
		const SingleSlice& _Slice,
		const InferenceParams& _Params,
		size_t& _Process
	) const;

	/**
	 * \brief 推理一个音频（使用PCM）
	 * \param _PCMData 输入的PCM数据（signed-int16 单声道）
	 * \param _SrcSamplingRate 输入PCM的采样率
	 * \param _Params 推理参数
	 * \return 推理结果（PCM signed-int16 单声道）
	 */
	[[nodiscard]] virtual DragonianLibSTL::Vector<int16_t> InferPCMData(
		const DragonianLibSTL::Vector<int16_t>& _PCMData,
		long _SrcSamplingRate, 
		const InferenceParams& _Params
	) const;

	//提取音量
	[[nodiscard]] static DragonianLibSTL::Vector<float> ExtractVolume(
		const DragonianLibSTL::Vector<int16_t>& _Audio,
		int _HopSize
	);

	//提取音量
	[[nodiscard]] DragonianLibSTL::Vector<float> ExtractVolume(
		const DragonianLibSTL::Vector<double>& _Audio
	) const;

	//获取HopSize
	[[nodiscard]] int GetHopSize() const
	{
		return HopSize;
	}

	//获取HiddenUnitKDims
	[[nodiscard]] int64_t GetHiddenUnitKDims() const
	{
		return HiddenUnitKDims;
	}

	//获取角色数量
	[[nodiscard]] int64_t GetSpeakerCount() const
	{
		return SpeakerCount;
	}

	//获取角色混合启用状态
	[[nodiscard]] bool SpeakerMixEnabled() const
	{
		return EnableCharaMix;
	}

	/**
	 * \brief 切片一个音频
	 * \param _InputPCM 输入的PCM数据（signed-int16 单声道）
	 * \param _SlicePos 切片位置（单位为采样）
	 * \param _SlicerConfig 切片机设置
	 * \return 音频数据
	 */
	[[nodiscard]] static SingleAudio GetAudioSlice(
		const DragonianLibSTL::Vector<int16_t>& _InputPCM,
		const DragonianLibSTL::Vector<size_t>& _SlicePos,
		const SlicerSettings& _SlicerConfig
	);

	/**
	 * \brief 预处理音频数据
	 * \param _Input 完成切片的音频数据
	 * \param _SamplingRate 采样率
	 * \param _HopSize HopSize
	 * \param _F0Method F0算法
	 */
	static void PreProcessAudio(
		const SingleAudio& _Input,
		int _SamplingRate = 48000, 
		int _HopSize = 512, 
		const std::wstring& _F0Method = L"Dio"
	);

	~SingingVoiceConversion() override;

protected:
	TensorExtractor _TensorExtractor;
	Ort::Session* hubert = nullptr;

	int HopSize = 320;
	int64_t HiddenUnitKDims = 256;
	int64_t SpeakerCount = 1;
	bool EnableCharaMix = false;
	bool EnableVolume = false;

	DragonianLib::ClusterWrp Cluster;

	int64_t ClusterCenterSize = 10000;
	bool EnableCluster = false;
	bool EnableIndex = false;

	const std::vector<const char*> hubertOutput = { "embed" };
	const std::vector<const char*> hubertInput = { "source" };
private:
	SingingVoiceConversion& operator=(SingingVoiceConversion&&) = delete;
	SingingVoiceConversion& operator=(const SingingVoiceConversion&) = delete;
	SingingVoiceConversion(const SingingVoiceConversion&) = delete;
	SingingVoiceConversion(SingingVoiceConversion&&) = delete;
};

LibSvcEnd