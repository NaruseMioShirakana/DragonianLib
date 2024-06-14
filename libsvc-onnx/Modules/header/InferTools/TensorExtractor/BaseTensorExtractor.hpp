/**
 * FileName: MoeVoiceStudioTensorExtractor.hpp
 * Note: MoeVoiceStudioCore 张量预处理基类
 *
 * Copyright (C) 2022-2023 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of MoeVoiceStudioCore library.
 * MoeVoiceStudioCore library is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * MoeVoiceStudioCore library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include "InferTools/InferTools.hpp"
#include "onnxruntime_cxx_api.h"

LibSvcHeader

class LibSvcTensorExtractor
{
public:
	
	struct Tensors
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

		DragonianLibSTL::Vector<int64_t> HiddenUnitShape;
		DragonianLibSTL::Vector<int64_t> FrameShape;
		DragonianLibSTL::Vector<int64_t> SpkShape;
		DragonianLibSTL::Vector<int64_t> DDSPNoiseShape;
		DragonianLibSTL::Vector<int64_t> NoiseShape;
		int64_t OneShape[1] = { 1 };
	};

	struct InferParams
	{
		float NoiseScale = 0.3f;
		float DDSPNoiseScale = 1.0f;
		int Seed = 520468;
		size_t AudioSize = 0;
		int64_t Chara = 0;
		float upKeys = 0.f;
		void* Other = nullptr;
		size_t Padding = size_t(-1);
	};

	struct Others
	{
		int f0_bin = 256;
		float f0_max = 1100.0;
		float f0_min = 50.0;
		OrtMemoryInfo* Memory = nullptr;
		void* Other = nullptr;
	};

	using Params = const InferParams&;

	struct Inputs
	{
		Tensors Data;
		std::vector<Ort::Value> Tensor;
		const char* const* InputNames = nullptr;
		const char* const* OutputNames = nullptr;
		size_t InputCount = 1;
		size_t OutputCount = 1;
	};

	/**
	 * \brief 构造张量预处理器
	 * \param _srcsr 原始采样率
	 * \param _sr 目标采样率
	 * \param _hop HopSize
	 * \param _smix 是否启用角色混合
	 * \param _volume 是否启用音量emb
	 * \param _hidden_size hubert的维数
	 * \param _nspeaker 角色数
	 * \param _other 其他参数，其中的memoryInfo必须为你当前模型的memoryInfo
	 */
	LibSvcTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other);
	virtual ~LibSvcTensorExtractor() = default;
	LibSvcTensorExtractor(const LibSvcTensorExtractor&) = delete;
	LibSvcTensorExtractor(LibSvcTensorExtractor&&) = delete;
	LibSvcTensorExtractor operator=(const LibSvcTensorExtractor&) = delete;
	LibSvcTensorExtractor operator=(LibSvcTensorExtractor&&) = delete;

	/**
	 * \brief 预处理张量
	 * \param HiddenUnit HiddenUnit
	 * \param F0 F0
	 * \param Volume 音量
	 * \param SpkMap 角色混合数据
	 * \param params 参数
	 * \return 完成预处理的张量（请将张量接管的所有Vector的数据都存储到Tensors Data中，因为ORT创建的张量要求调用方管理内存，如果不存储到这个位置会导致数据提前析构
	 */
	virtual Inputs Extract(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		Params params
	);

	void SetSrcSamplingRates(uint64_t sr) { _SrcSamplingRate = sr; }

	//获取换算为0-255的f0
	[[nodiscard]] DragonianLibSTL::Vector<int64_t> GetNSFF0(const DragonianLibSTL::Vector<float>&) const;

	//将F0中0值单独插值
	static DragonianLibSTL::Vector<float> GetInterpedF0(const DragonianLibSTL::Vector<float>&);

	//
	static DragonianLibSTL::Vector<float> InterpUVF0(const DragonianLibSTL::Vector<float>&, size_t PaddedIndex = size_t(-1));

	//获取UnVoiceMask
	static DragonianLibSTL::Vector<float> GetUV(const DragonianLibSTL::Vector<float>&);

	//获取对齐矩阵
	static DragonianLibSTL::Vector<int64_t> GetAligments(size_t, size_t);

	//线性组合
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

	//将F0中0值单独插值（可设置是否取log）
	[[nodiscard]] static DragonianLibSTL::Vector<float> GetInterpedF0log(const DragonianLibSTL::Vector<float>&, bool);

	//获取正确的角色混合数据
	[[nodiscard]] DragonianLibSTL::Vector<float> GetCurrectSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, int64_t curspk) const;

	//获取正确的角色混合数据
	[[nodiscard]] static DragonianLibSTL::Vector<float> GetSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, size_t spk_count);
protected:
	uint64_t _NSpeaker = 1;
	uint64_t _SrcSamplingRate = 32000;
	uint64_t _SamplingRate = 32000;
	uint64_t _HopSize = 512;
	bool _SpeakerMix = false;
	bool _Volume = false;
	uint64_t _HiddenSize = 256;
	int f0_bin = 256;
	float f0_max = 1100.0;
	float f0_min = 50.0;
	float f0_mel_min = 1127.f * log(1.f + f0_min / 700.f);
	float f0_mel_max = 1127.f * log(1.f + f0_max / 700.f);
	OrtMemoryInfo* Memory = nullptr;
};

LibSvcEnd