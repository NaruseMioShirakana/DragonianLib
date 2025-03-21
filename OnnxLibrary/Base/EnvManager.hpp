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
#include "Libraries/Util/Logger.h"
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

class OnnxRuntimeEnviromentBase;

using OnnxRuntimeEnviroment = std::shared_ptr<OnnxRuntimeEnviromentBase>;
using OnnxRuntimeModel = std::shared_ptr<Ort::Session>;

/**
 * @class OnnxRuntimeEnviromentBase
 * @brief Manages the ONNX Runtime environment and session options.
 */
class OnnxRuntimeEnviromentBase
{
public:
    friend std::shared_ptr<OnnxRuntimeEnviromentBase>;

    ~OnnxRuntimeEnviromentBase();
    OnnxRuntimeEnviromentBase(const OnnxRuntimeEnviromentBase&) = default;
    OnnxRuntimeEnviromentBase(OnnxRuntimeEnviromentBase&&) noexcept = default;
    OnnxRuntimeEnviromentBase& operator=(const OnnxRuntimeEnviromentBase&) = default;
    OnnxRuntimeEnviromentBase& operator=(OnnxRuntimeEnviromentBase&&) noexcept = default;

    /**
     * @brief Gets the ONNX Runtime environment.
     * @return Pointer to the ONNX Runtime environment.
     */
    [[nodiscard]] Ort::Env* GetEnv() const { return _MyOrtEnv.get(); }

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
     * @brief Gets the current thread count.
     * @return Current thread count.
     */
    [[nodiscard]] int GetCurThreadCount() const { return (int)_MyThreadCount; }

    /**
     * @brief Gets the current device ID.
     * @return Current device ID.
     */
    [[nodiscard]] int GetCurDeviceID() const { return (int)_MyDeviceID; }

    /**
     * @brief Gets the current provider.
     * @return Current provider.
     */
    [[nodiscard]] int GetCurProvider() const { return (int)_MyProvider; }

    /**
     * @brief References a cached ONNX model.
     * @param Path_ Path to the model.
     * @param Env_ ONNX Runtime environment.
     * @return Shared pointer to the ONNX session.
     */
    static OnnxRuntimeModel& RefOrtCachedModel(const std::wstring& Path_, const OnnxRuntimeEnviroment& Env_);

    /**
     * @brief Unreferences a cached ONNX model.
     * @param Path_ Path to the model.
     * @param Env_ ONNX Runtime environment.
     */
    static void UnRefOrtCachedModel(const std::wstring& Path_, const OnnxRuntimeEnviroment& Env_);

    /**
     * @brief Clears the model cache.
     */
    static void ClearModelCache(const OnnxRuntimeEnviroment& Env_);

    /**
     * @brief Sets a CUDA option.
     * @param Key Key of the option.
     * @param Value Value of the option.
     */
    static void SetCUDAOption(
        const std::string& Key,
        const std::string& Value
    );

    /**
     * @brief Creates an ONNX Runtime environment.
     * @param ThreadCount Number of threads to use.
     * @param DeviceID ID of the device to use.
     * @param Provider Execution provider to use.
     * @return Shared pointer to the ONNX Runtime environment.
     */
    static OnnxRuntimeEnviroment& CreateEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

    /**
     * @brief Destroys an ONNX Runtime environment.
     * @param ThreadCount Number of threads to use.
     * @param DeviceID ID of the device to use.
     * @param Provider Execution provider to use.
     */
    static void DestroyEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

protected:
    /**
     * @brief Constructor to initialize the ONNX Runtime environment.
     * @param ThreadCount Number of threads to use.
     * @param DeviceID ID of the device to use.
     * @param Provider Execution provider to use.
     */
    OnnxRuntimeEnviromentBase(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

private:
    /**
     * @brief Loads the ONNX Runtime environment.
     * @param ThreadCount Number of threads to use.
     * @param DeviceID ID of the device to use.
     * @param Provider Execution provider to use.
     */
    void Load(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

    /**
     * @brief Creates the ONNX Runtime environment.
     * @param ThreadCount_ Number of threads to use.
     * @param DeviceID_ ID of the device to use.
     * @param ExecutionProvider_ Execution provider to use.
     */
    void Create(unsigned ThreadCount_, unsigned DeviceID_, unsigned ExecutionProvider_);

    std::unordered_map<std::wstring, OnnxRuntimeModel> GlobalOrtModelCache; ///< Global ONNX model cache.
    std::shared_ptr<Ort::Env> _MyOrtEnv = nullptr; ///< Pointer to the ONNX Runtime environment.
    std::shared_ptr<Ort::SessionOptions> _MyOrtSessionOptions = nullptr; ///< Pointer to the session options.
    std::shared_ptr<Ort::MemoryInfo> _MyOrtMemoryInfo = nullptr; ///< Pointer to the memory info.
    std::shared_ptr <OrtCUDAProviderOptionsV2> _MyCudaOptionsV2 = nullptr; ///< CUDA provider options.
    unsigned _MyThreadCount = 0; ///< Current thread count.
    unsigned _MyDeviceID = 0; ///< Current device ID.
    unsigned _MyProvider = 0; ///< Current provider.
};

/**
 * @brief References a ONNX model, if it is not loaded, it will be loaded.
 * @param _ModelPath Path to the model.
 * @param _Enviroment ONNX Runtime environment.
 * @return Shared pointer to the ONNX session.
 */
inline OnnxRuntimeModel& RefOnnxRuntimeModel(const std::wstring& _ModelPath, const OnnxRuntimeEnviroment& _Enviroment)
{
    return OnnxRuntimeEnviromentBase::RefOrtCachedModel(_ModelPath, _Enviroment);
}

/**
 * @brief Unreferences a ONNX model.
 * @param _ModelPath Path of the model when it was loaded.
 * @param _Enviroment ONNX Runtime environment.
 */
inline void UnrefOnnxRuntimeModel(const std::wstring& _ModelPath, const OnnxRuntimeEnviroment& _Enviroment)
{
    OnnxRuntimeEnviromentBase::UnRefOrtCachedModel(_ModelPath, _Enviroment);
}

/**
 * @brief Unreferences all global ONNX models loaded by this environment.
 * @param _Enviroment ONNX Runtime environment.
 */
inline void UnrefAllOnnxRuntimeModel(const OnnxRuntimeEnviroment& _Enviroment)
{
    OnnxRuntimeEnviromentBase::ClearModelCache(_Enviroment);
}

inline OnnxRuntimeEnviroment& CreateOnnxRuntimeEnviroment(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider)
{
    return OnnxRuntimeEnviromentBase::CreateEnv(_ThreadCount, _DeviceID, _Provider);
}

_D_Dragonian_Lib_Onnx_Runtime_End
