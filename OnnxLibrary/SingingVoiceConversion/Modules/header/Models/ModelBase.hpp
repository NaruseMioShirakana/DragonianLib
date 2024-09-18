/**
 * FileName: ModelBase.hpp
 * Note: MoeVoiceStudioCore Onnx 模型基类
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
#include <functional>
#include <thread>
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif
#include "Params.hpp"
#include "EnvManager.hpp"
#include "../InferTools/inferTools.hpp"

LibSvcHeader

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

inline float Clamp(float in, float min = -1.f, float max = 1.f)
{
	if (in > max)
		return max;
	if (in < min)
		return min;
	return in;
}

class LibSvcModule
{
public:
	//进度条回调
	using ProgressCallback = std::function<void(size_t, size_t)>;

	//Provicer
	enum class ExecutionProviders
	{
		CPU = 0,
		CUDA = 1,
		DML = 2
	};

	/**
	 * \brief 构造Onnx模型基类
	 * \param ExecutionProvider_ ExecutionProvider(可以理解为设备)
	 * \param DeviceID_ 设备ID
	 * \param ThreadCount_ 线程数
	 */
	LibSvcModule(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_ = 0);

	virtual ~LibSvcModule();

	/**
	 * \brief 获取采样率
	 * \return 采样率
	 */
	[[nodiscard]] long GetSamplingRate() const
	{
		return _samplingRate;
	}

	[[nodiscard]] DragonianLib::DragonianLibOrtEnv& GetDlEnv() { return OrtApiEnv; }

	[[nodiscard]] const DragonianLib::DragonianLibOrtEnv& GetDlEnv() const { return OrtApiEnv; }
protected:
	//采样率
	long _samplingRate = 22050;
	Ort::Env* env = nullptr;
	Ort::SessionOptions* session_options = nullptr;
	Ort::MemoryInfo* memory_info = nullptr;
	ExecutionProviders _cur_execution_provider = ExecutionProviders::CPU;
	DragonianLib::DragonianLibOrtEnv OrtApiEnv;
	ProgressCallback _callback;
public:
	//*******************删除的函数********************//
	LibSvcModule& operator=(LibSvcModule&&) = delete;
	LibSvcModule& operator=(const LibSvcModule&) = delete;
	LibSvcModule(const LibSvcModule&) = delete;
	LibSvcModule(LibSvcModule&&) = delete;
};

LibSvcEnd