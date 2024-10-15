﻿/**
 * FileName: EnvManager.hpp
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

#ifdef DRAGONIANLIB_ONNXRT_LIB
#include "onnxruntime_cxx_api.h"

namespace DragonianLib {

    /**
     * @class DragonianLibOrtEnv
     * @brief Manages the ONNX Runtime environment and session options.
     */
    class DragonianLibOrtEnv
    {
    public:
        /**
         * @brief Constructor to initialize the ONNX Runtime environment.
         * @param ThreadCount Number of threads to use.
         * @param DeviceID ID of the device to use.
         * @param Provider Execution provider to use.
         */
        DragonianLibOrtEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
        {
            Load(ThreadCount, DeviceID, Provider);
        }

        /**
         * @brief Destructor to clean up the ONNX Runtime environment.
         */
        ~DragonianLibOrtEnv() { Destory(); }

        DragonianLibOrtEnv(const DragonianLibOrtEnv&) = delete;
        DragonianLibOrtEnv(DragonianLibOrtEnv&&) = delete;
        DragonianLibOrtEnv operator=(const DragonianLibOrtEnv&) = delete;
        DragonianLibOrtEnv operator=(DragonianLibOrtEnv&&) = delete;

        /**
         * @brief Destroys the ONNX Runtime environment.
         */
        void Destory();

        /**
         * @brief Gets the ONNX Runtime environment.
         * @return Pointer to the ONNX Runtime environment.
         */
        [[nodiscard]] Ort::Env* GetEnv() const { return GlobalOrtEnv; }

        /**
         * @brief Gets the session options for the ONNX Runtime.
         * @return Pointer to the session options.
         */
        [[nodiscard]] Ort::SessionOptions* GetSessionOptions() const { return GlobalOrtSessionOptions; }

        /**
         * @brief Gets the memory info for the ONNX Runtime.
         * @return Pointer to the memory info.
         */
        [[nodiscard]] Ort::MemoryInfo* GetMemoryInfo() const { return GlobalOrtMemoryInfo; }

        /**
         * @brief Gets the current thread count.
         * @return Current thread count.
         */
        [[nodiscard]] int GetCurThreadCount() const { return (int)CurThreadCount; }

        /**
         * @brief Gets the current device ID.
         * @return Current device ID.
         */
        [[nodiscard]] int GetCurDeviceID() const { return (int)CurDeviceID; }

        /**
         * @brief Gets the current provider.
         * @return Current provider.
         */
        [[nodiscard]] int GetCurProvider() const { return (int)CurProvider; }

        /**
         * @brief References a cached ONNX model.
         * @param Path_ Path to the model.
         * @param Env_ ONNX Runtime environment.
         * @return Shared pointer to the ONNX session.
         */
        static std::shared_ptr<Ort::Session>& RefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_);

        /**
         * @brief Unreferences a cached ONNX model.
         * @param Path_ Path to the model.
         * @param Env_ ONNX Runtime environment.
         */
        static void UnRefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_);

        /**
         * @brief Clears the model cache.
         */
        static void ClearModelCache();

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

        Ort::Env* GlobalOrtEnv = nullptr; ///< Pointer to the ONNX Runtime environment.
        Ort::SessionOptions* GlobalOrtSessionOptions = nullptr; ///< Pointer to the session options.
        Ort::MemoryInfo* GlobalOrtMemoryInfo = nullptr; ///< Pointer to the memory info.
        unsigned CurThreadCount = unsigned(-1); ///< Current thread count.
        unsigned CurDeviceID = unsigned(-1); ///< Current device ID.
        unsigned CurProvider = unsigned(-1); ///< Current provider.
        OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr; ///< CUDA provider options.
    };

	/**
	 * @brief References a cached ONNX model.
	 * @param Path_ Path to the model.
	 * @param Env_ ONNX Runtime environment.
	 * @return Shared pointer to the ONNX session.
	 */
    inline std::shared_ptr<Ort::Session>& RefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_)
    {
        return DragonianLibOrtEnv::RefOrtCachedModel(Path_, Env_);
    }

	/**
	 * @brief Unreferences a cached ONNX model.
	 * @param Path_ Path to the model.
	 * @param Env_ ONNX Runtime environment.
	 */
    inline void UnRefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_)
    {
        DragonianLibOrtEnv::UnRefOrtCachedModel(Path_, Env_);
    }

	/**
	 * @brief Clears the model cache.
	 */
    inline void ClearModelCache()
    {
        DragonianLibOrtEnv::ClearModelCache();
    }

}

#endif
