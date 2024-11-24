/**
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
        friend std::shared_ptr<DragonianLibOrtEnv>;

        ~DragonianLibOrtEnv();
        DragonianLibOrtEnv(const DragonianLibOrtEnv&) = default;
        DragonianLibOrtEnv(DragonianLibOrtEnv&&) = default;
        DragonianLibOrtEnv& operator=(const DragonianLibOrtEnv&) = default;
        DragonianLibOrtEnv& operator=(DragonianLibOrtEnv&&) = default;

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
		static std::shared_ptr<DragonianLibOrtEnv>& CreateEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

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
        DragonianLibOrtEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);

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

        std::shared_ptr<Ort::Env> _MyOrtEnv = nullptr; ///< Pointer to the ONNX Runtime environment.
        std::shared_ptr<Ort::SessionOptions> _MyOrtSessionOptions = nullptr; ///< Pointer to the session options.
        std::shared_ptr<Ort::MemoryInfo> _MyOrtMemoryInfo = nullptr; ///< Pointer to the memory info.
        std::shared_ptr <OrtCUDAProviderOptionsV2> _MyCudaOptionsV2 = nullptr; ///< CUDA provider options.
		unsigned _MyThreadCount = 0; ///< Current thread count.
		unsigned _MyDeviceID = 0; ///< Current device ID.
		unsigned _MyProvider = 0; ///< Current provider.
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
