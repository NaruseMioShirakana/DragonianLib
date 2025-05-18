/**
 * @file EnvManager.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Manages the ONNX Runtime environment and session options.
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Created From Old Version <
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once

#include "onnxruntime_cxx_api.h"
#include "Libraries/MyTemplateLibrary/Util.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/TypeDef.h"
#include "Libraries/Util/Util.h"

#define _D_Dragonian_Lib_Onnx_Runtime_Header \
	_D_Dragonian_Lib_Space_Begin \
	namespace OnnxRuntime \
	{

#define _D_Dragonian_Lib_Onnx_Runtime_End \
	} \
	_D_Dragonian_Lib_Space_End

#define _D_Dragonian_Lib_Onnx_Runtime_Space \
	_D_Dragonian_Lib_Namespace \
	OnnxRuntime::

_D_Dragonian_Lib_Onnx_Runtime_Header

DLogger& GetDefaultLogger() noexcept;

Ort::AllocatorWithDefaultOptions& GetDefaultOrtAllocator();

class OnnxRuntimeEnvironmentBase;
using OnnxRuntimeEnvironment = std::shared_ptr<OnnxRuntimeEnvironmentBase>;
using OnnxRuntimeModelPointer = std::shared_ptr<Ort::Session>;

struct OnnxEnvironmentOptions
{
    friend class OnnxRuntimeEnvironmentBase;

	OnnxEnvironmentOptions(const OnnxEnvironmentOptions&) = default;
	OnnxEnvironmentOptions(OnnxEnvironmentOptions&&) noexcept = default;
	OnnxEnvironmentOptions& operator=(const OnnxEnvironmentOptions&) = default;
	OnnxEnvironmentOptions& operator=(OnnxEnvironmentOptions&&) noexcept = default;
	~OnnxEnvironmentOptions() = default;

    OnnxEnvironmentOptions(
        Device _Provider = Device::CPU,
        Int64 _DeviceID = 0,
        Int64 _IntraOpNumThreads = 4,
        Int64 _InterOpNumThreads = 2,
        OrtLoggingLevel _LoggingLevel = ORT_LOGGING_LEVEL_WARNING,
        std::string _LoggerId = "DragonianLib"
    ) : Provider(_Provider), DeviceID(_DeviceID), IntraOpNumThreads(_IntraOpNumThreads), InterOpNumThreads(_InterOpNumThreads), LoggingLevel(_LoggingLevel), LoggerId(std::move(_LoggerId))
    {
    }

    Device Provider = Device::CPU; ///< Execution provider (device) of the environment.
    Int64 DeviceID = 0; ///< Device ID of the environment.
    Int64 IntraOpNumThreads = 4; ///< Number of threads for intra-op parallelism.
    Int64 InterOpNumThreads = 2; ///< Number of threads for inter-op parallelism.
    OrtLoggingLevel LoggingLevel = ORT_LOGGING_LEVEL_VERBOSE; ///< Log level of the environment.
    std::string LoggerId = "DragonianLib"; ///< Logger ID of the environment.

	void SetCUDAOptions(const std::string& Key, const std::string& Value)
	{
		CUDAOptions[Key] = Value;
	}

protected:
    std::unordered_map<std::string, std::string> CUDAOptions{
        {"device_id", "0"},
        {"gpu_mem_limit", "4294967296"},
        {"arena_extend_strategy", "kNextPowerOfTwo"},
        {"cudnn_conv_algo_search", "EXHAUSTIVE"},
        {"do_copy_in_default_stream", "0"},
        {"cudnn_conv_use_max_workspace", "1"},
        {"cudnn_conv1d_pad_to_nc1d", "1"},
        {"enable_cuda_graph", "0"},
    };
};

class OnnxRuntimeModel
{
public:
	OnnxRuntimeModel() = default;
    OnnxRuntimeModel(OnnxRuntimeModelPointer Model);
    ~OnnxRuntimeModel();
    OnnxRuntimeModel(const OnnxRuntimeModel&);
    OnnxRuntimeModel(OnnxRuntimeModel&&) noexcept = default;
    OnnxRuntimeModel& operator=(const OnnxRuntimeModel&);
    OnnxRuntimeModel& operator=(OnnxRuntimeModel&&) noexcept = default;
    OnnxRuntimeModel(nullptr_t) noexcept
    {
        _MyModel = nullptr;
    }
    OnnxRuntimeModel& operator=(nullptr_t) noexcept
    {
        _MyModel = nullptr;
        return *this;
    }

    auto Get() const noexcept
    {
        return _MyModel.get();
    }
    decltype(auto) operator*() const noexcept
    {
        return *_MyModel;
    }
    decltype(auto) operator->() const noexcept
    {
        return _MyModel.operator->();
    }
	long UseCount() const noexcept
	{
		return _MyModel.use_count();
	}
	operator bool() const noexcept
	{
		return _MyModel.operator bool();
	}

private:
    OnnxRuntimeModelPointer _MyModel = nullptr;
};

/**
 * @class OnnxRuntimeEnvironmentBase
 * @brief Manages the ONNX Runtime environment and session options.
 */
class OnnxRuntimeEnvironmentBase : public std::enable_shared_from_this<OnnxRuntimeEnvironmentBase>
{
public:
    friend std::shared_ptr<OnnxRuntimeEnvironmentBase>;

    ~OnnxRuntimeEnvironmentBase();
    OnnxRuntimeEnvironmentBase(const OnnxRuntimeEnvironmentBase&) = default;
    OnnxRuntimeEnvironmentBase(OnnxRuntimeEnvironmentBase&&) noexcept = default;
    OnnxRuntimeEnvironmentBase& operator=(const OnnxRuntimeEnvironmentBase&) = default;
    OnnxRuntimeEnvironmentBase& operator=(OnnxRuntimeEnvironmentBase&&) noexcept = default;

    /**
     * @brief Gets the ONNX Runtime environment.
     * @return Pointer to the ONNX Runtime environment.
     */
    [[nodiscard]] Ort::Env* GetEnvironment() const { return _MyOrtEnv.get(); }

    /**
     * @brief Gets the session options for the ONNX Runtime.
     * @return Pointer to the session options.
     */
    [[nodiscard]] Ort::SessionOptions* GetSessionOptions() const { return _MyOrtSessionOptions.get(); }

    /**
     * @brief Gets the memory info for the ONNX Runtime.
     * @return Pointer to the memory info.
     */
    [[nodiscard]] Ort::MemoryInfo* GetMemoryInfo() const { return _MyOrtMemoryInfo.get(); }

    /**
	 * @brief Gets the InterOpNumThreads.
	 * @return Number of threads for inter-op parallelism.
     */
    [[nodiscard]] Int64 GetInterOpNumThreads() const { return _MyInterOpNumThreads; }

    /**
	 * @brief Gets the IntraOpNumThreads.
	 * @return Number of threads for intra-op parallelism.
     */
    [[nodiscard]] Int64 GetIntraOpNumThreads() const { return _MyIntraOpNumThreads; }

    /**
     * @brief Gets the current device ID.
     * @return Current device ID.
     */
    [[nodiscard]] Int64 GetDeviceID() const { return _MyDeviceID; }

    /**
     * @brief Gets the current provider.
     * @return Current provider.
     */
    [[nodiscard]] Device GetProvider() const { return _MyProvider; }

    /**
     * @brief References a ONNX model.
     * @param ModelPath Path to the model.
     * @return Shared pointer to the ONNX session.
     */
	OnnxRuntimeModel& RefOnnxRuntimeModel(const std::wstring& ModelPath);

    /**
     * @brief Unreferences a ONNX model.
     * @param ModelPath Path to the model.
     */
	void UnRefOnnxRuntimeModel(const std::wstring& ModelPath);

    /**
	 * @brief Clears all global ONNX models loaded by this environment.
     */
	void ClearOnnxRuntimeModel();

    void EnableMemPattern(bool Enable) const;

    void EnableCpuMemArena(bool Enable) const;

	void EnableProfiling(bool Enable, const std::wstring& FilePath) const;

    void SetIntraOpNumThreads(Int64 Threads);

    void SetInterOpNumThreads(Int64 Threads);

	void SetExecutionMode(ExecutionMode Mode) const;

	void SetGraphOptimizationLevel(GraphOptimizationLevel Level) const;

	void SetLoggingLevel(OrtLoggingLevel Level);

    void ClearCache();

    /**
     * @brief Creates an ONNX Runtime environment.
	 * @param Options ONNX Runtime environment options.
     * @return Shared pointer to the ONNX Runtime environment.
     */
    static OnnxRuntimeEnvironment CreateEnv(const OnnxEnvironmentOptions& Options);

protected:
    /**
     * @brief Constructor to initialize the ONNX Runtime environment.
	 * @param Options ONNX Runtime environment options.
     */
    OnnxRuntimeEnvironmentBase(const OnnxEnvironmentOptions& Options);

private:
    /**
     * @brief Loads the ONNX Runtime environment.
	 * @param Options ONNX Runtime environment options.
     */
    void Load(const OnnxEnvironmentOptions& Options);

    /**
     * @brief Creates the ONNX Runtime environment.
	 * @param Options ONNX Runtime environment options.
     */
    void Create(const OnnxEnvironmentOptions& Options);
    
    std::unordered_map<std::wstring, OnnxRuntimeModel> GlobalOrtModelCache; ///< Global ONNX model cache.
    std::shared_ptr<Ort::Env> _MyOrtEnv = nullptr; ///< Pointer to the ONNX Runtime environment.
    std::shared_ptr<Ort::SessionOptions> _MyOrtSessionOptions = nullptr; ///< Pointer to the session options.
    std::shared_ptr<Ort::MemoryInfo> _MyOrtMemoryInfo = nullptr; ///< Pointer to the memory info.
    std::shared_ptr<OrtCUDAProviderOptionsV2> _MyCudaOptionsV2 = nullptr; ///< CUDA provider options.
    std::shared_ptr<OrtTensorRTProviderOptionsV2> _MyTensorRTOptionsV2 = nullptr; ///< CUDA provider options.
    Int64 _MyIntraOpNumThreads = 4; ///< Number of threads for intra-op parallelism.
    Int64 _MyInterOpNumThreads = 2; ///< Number of threads for inter-op parallelism.
    Int64 _MyDeviceID = 0; ///< Current device ID.
    Device _MyProvider = Device::CPU; ///< Current provider.
    OrtLoggingLevel _MyLoggingLevel = ORT_LOGGING_LEVEL_VERBOSE; ///< Log level.
    std::string _MyLoggerId = "DragonianLib"; ///< Logger ID.
    std::unordered_map<std::string, std::string> _MyCUDAOptions{
        {"device_id", "0"},
        {"gpu_mem_limit", "2147483648"},
        {"arena_extend_strategy", "kNextPowerOfTwo"},
        {"cudnn_conv_algo_search", "EXHAUSTIVE"},
        {"do_copy_in_default_stream", "0"},
        {"cudnn_conv_use_max_workspace", "1"},
        {"cudnn_conv1d_pad_to_nc1d", "1"},
        {"enable_cuda_graph", "0"}
    };
};

/**
 * @brief References a ONNX model, if it is not loaded, it will be loaded.
 * @param _ModelPath Path to the model.
 * @param _Environment ONNX Runtime environment.
 * @return Shared pointer to the ONNX session.
 */
inline OnnxRuntimeModel& RefOnnxRuntimeModel(const std::wstring& _ModelPath, const OnnxRuntimeEnvironment& _Environment)
{
    return _Environment->RefOnnxRuntimeModel(_ModelPath);
}

/**
 * @brief Unreferences a ONNX model.
 * @param _ModelPath Path of the model when it was loaded.
 * @param _Environment ONNX Runtime environment.
 */
inline void UnrefOnnxRuntimeModel(const std::wstring& _ModelPath, const OnnxRuntimeEnvironment& _Environment)
{
	_Environment->UnRefOnnxRuntimeModel(_ModelPath);
}

/**
 * @brief Unreferences all global ONNX models loaded by this environment.
 * @param _Environment ONNX Runtime environment.
 */
inline void UnrefAllOnnxRuntimeModel(const OnnxRuntimeEnvironment& _Environment)
{
    _Environment->ClearOnnxRuntimeModel();
}

inline OnnxRuntimeEnvironment CreateOnnxRuntimeEnvironment(const OnnxEnvironmentOptions& _Options)
{
    return OnnxRuntimeEnvironmentBase::CreateEnv(_Options);
}

_D_Dragonian_Lib_Onnx_Runtime_End
