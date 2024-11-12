﻿/**
 * FileName: ModelBase.hpp
 * Note: MoeVoiceStudioCore Onnx Module Base
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
#include "Params.hpp"
#include "EnvManager.hpp"
#include "../InferTools/inferTools.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @struct DiffusionSvcPaths
 * @brief Paths Of DiffusionSvc
 */
struct DiffusionSvcPaths
{
    /**
     * @brief Path to the encoder
     */
    std::wstring Encoder;

    /**
     * @brief Path to the denoise model
     */
    std::wstring Denoise;

    /**
     * @brief Path to the prediction model
     */
    std::wstring Pred;

    /**
     * @brief Path to the after model
     */
    std::wstring After;

    /**
     * @brief Path to the alpha model
     */
    std::wstring Alpha;

    /**
     * @brief Path to the naive model
     */
    std::wstring Naive;

    /**
     * @brief Path to the DiffSvc model
     */
    std::wstring DiffSvc;
};

/**
 * @struct ReflowSvcPaths
 * @brief Paths Of ReflowSvc
 */
struct ReflowSvcPaths
{
    /**
     * @brief Path to the encoder
     */
    std::wstring Encoder;

    /**
     * @brief Path to the velocity function
     */
    std::wstring VelocityFn;

    /**
     * @brief Path to the after model
     */
    std::wstring After;
};

/**
 * @struct VitsSvcPaths
 * @brief Paths Of VitsSvc
 */
struct VitsSvcPaths
{
    /**
     * @brief Path to the VitsSvc model
     */
    std::wstring VitsSvc;
};

/**
 * @struct ClusterConfig
 * @brief Configuration for Clustering
 */
struct ClusterConfig
{
    /**
     * @brief Size of the cluster center
     */
    int64_t ClusterCenterSize = 10000;

    /**
     * @brief Path to the cluster configuration
     */
    std::wstring Path;

    /**
     * @brief Type of cluster: "KMeans" or "Index"
     */
    std::wstring Type;
};

/**
 * @struct Hparams
 * @brief Hyperparameters for the model
 */
struct Hparams
{
    /**
     * @brief Model Version
     * For VitsSvc: "SoVits2.0", "SoVits3.0", "SoVits4.0", "SoVits4.0-DDSP", "RVC"
     * For DiffusionSvc: "DiffSvc", "DiffusionSvc"
     */
    std::wstring TensorExtractor = L"DiffSvc";

    /**
     * @brief Path of Hubert model
     */
    std::wstring HubertPath;

    /**
     * @brief Path of DiffusionSvc model
     */
    DiffusionSvcPaths DiffusionSvc;

    /**
     * @brief Path of VitsSvc model
     */
    VitsSvcPaths VitsSvc;

    /**
     * @brief Path of ReflowSvc model
     */
    ReflowSvcPaths ReflowSvc;

    /**
     * @brief Configuration of cluster
     */
    ClusterConfig Cluster;

    /**
     * @brief Sampling rate
     */
    long SamplingRate = 22050;

    /**
     * @brief Hop size
     */
    int HopSize = 320;

    /**
     * @brief Hidden unit K dimensions
     */
    int64_t HiddenUnitKDims = 256;

    /**
     * @brief Number of speakers
     */
    int64_t SpeakerCount = 1;

    /**
     * @brief Enable character mix
     */
    bool EnableCharaMix = false;

    /**
     * @brief Enable volume
     */
    bool EnableVolume = false;

    /**
     * @brief Enable VAE mode
     */
    bool VaeMode = true;

    /**
     * @brief Number of mel bins
     */
    int64_t MelBins = 128;

    /**
     * @brief Number of PNDMS
     */
    int64_t Pndms = 100;

    /**
     * @brief Maximum step
     */
    int64_t MaxStep = 1000;

    /**
     * @brief Minimum value of the spectrum
     */
    float SpecMin = -12;

    /**
     * @brief Maximum value of the spectrum
     */
    float SpecMax = 2;

    /**
     * @brief Scale factor
     */
    float Scale = 1000.f;
};

/**
 * @brief Clamps a value between a minimum and maximum
 * @param in Input value
 * @param min Minimum value
 * @param max Maximum value
 * @return Clamped value
 */
inline float Clamp(float in, float min = -1.f, float max = 1.f)
{
    if (in > max)
        return max;
    if (in < min)
        return min;
    return in;
}

/**
 * @class LibSvcModule
 * @brief Base class for Onnx models in MoeVoiceStudioCore
 */
class LibSvcModule
{
public:
    /**
     * @typedef ProgressCallback
     * @brief Callback function for progress updates
     */
    using ProgressCallback = std::function<void(size_t, size_t)>;

    /**
     * @enum ExecutionProviders
     * @brief Enum for execution providers (devices)
     */
    enum class ExecutionProviders
    {
        CPU = 0,
        CUDA = 1,
        DML = 2
    };

    /**
     * @brief Constructs the Onnx model base class
     * @param ExecutionProvider_ ExecutionProvider (device)
     * @param DeviceID_ Device ID
     * @param ThreadCount_ Number of threads
     */
    LibSvcModule(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_ = 0);

    /**
     * @brief Destructor for LibSvcModule
     */
    virtual ~LibSvcModule();

    /**
     * @brief Gets the sampling rate
     * @return Sampling rate
     */
    [[nodiscard]] long GetSamplingRate() const
    {
        return ModelSamplingRate;
    }

    /**
     * @brief Gets the DragonianLibOrtEnv
     * @return Reference to DragonianLibOrtEnv
     */
    [[nodiscard]] DragonianLibOrtEnv& GetDlEnv() { return *OrtApiEnv; }

    /**
     * @brief Gets the DragonianLibOrtEnv (const version)
     * @return Const reference to DragonianLibOrtEnv
     */
    [[nodiscard]] const DragonianLibOrtEnv& GetDlEnv() const { return *OrtApiEnv; }
protected:
    /**
     * @brief Sampling rate
     */
    long ModelSamplingRate = 22050;

    /**
     * @brief ONNX environment
     */
    Ort::Env* OnnxEnv = nullptr;

    /**
     * @brief Session options
     */
    Ort::SessionOptions* SessionOptions = nullptr;

    /**
     * @brief Memory info
     */
    Ort::MemoryInfo* MemoryInfo = nullptr;

    /**
     * @brief Execution provider
     */
    ExecutionProviders ModelExecutionProvider = ExecutionProviders::CPU;

    /**
     * @brief Shared pointer to DragonianLibOrtEnv
     */
    std::shared_ptr<DragonianLibOrtEnv> OrtApiEnv;

    /**
     * @brief Progress callback function
     */
    ProgressCallback ProgressCallbackFunction;
public:
    LibSvcModule& operator=(LibSvcModule&&) = default;
    LibSvcModule& operator=(const LibSvcModule&) = default;
    LibSvcModule(const LibSvcModule&) = default;
    LibSvcModule(LibSvcModule&&) = default;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
