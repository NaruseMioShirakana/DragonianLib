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

	class DragonianLibOrtEnv
	{
	public:
		DragonianLibOrtEnv(unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
		{
			Load(ThreadCount, DeviceID, Provider);
		}
		~DragonianLibOrtEnv() { Destory(); }
		DragonianLibOrtEnv(const DragonianLibOrtEnv&) = delete;
		DragonianLibOrtEnv(DragonianLibOrtEnv&&) = delete;
		DragonianLibOrtEnv operator=(const DragonianLibOrtEnv&) = delete;
		DragonianLibOrtEnv operator=(DragonianLibOrtEnv&&) = delete;
		void Destory();
		[[nodiscard]] Ort::Env* GetEnv() const { return GlobalOrtEnv; }
		[[nodiscard]] Ort::SessionOptions* GetSessionOptions() const { return GlobalOrtSessionOptions; }
		[[nodiscard]] Ort::MemoryInfo* GetMemoryInfo() const { return GlobalOrtMemoryInfo; }
		[[nodiscard]] int GetCurThreadCount() const { return (int)CurThreadCount; }
		[[nodiscard]] int GetCurDeviceID() const { return (int)CurDeviceID; }
		[[nodiscard]] int GetCurProvider() const { return (int)CurProvider; }
        static std::shared_ptr<Ort::Session>& RefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_);
        static void UnRefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_);
        static void ClearModelCache();
	private:
		void Load(unsigned ThreadCount, unsigned DeviceID, unsigned Provider);
		void Create(unsigned ThreadCount_, unsigned DeviceID_, unsigned ExecutionProvider_);
		Ort::Env* GlobalOrtEnv = nullptr;
		Ort::SessionOptions* GlobalOrtSessionOptions = nullptr;
		Ort::MemoryInfo* GlobalOrtMemoryInfo = nullptr;
		unsigned CurThreadCount = unsigned(-1);
		unsigned CurDeviceID = unsigned(-1);
		unsigned CurProvider = unsigned(-1);
		OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr;
	};

    inline std::shared_ptr<Ort::Session>& RefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_)
    {
        return DragonianLibOrtEnv::RefOrtCachedModel(Path_, Env_);
    }

    inline void UnRefOrtCachedModel(const std::wstring& Path_, const DragonianLibOrtEnv& Env_)
    {
        DragonianLibOrtEnv::UnRefOrtCachedModel(Path_, Env_);
    }

    inline void ClearModelCache()
    {
        DragonianLibOrtEnv::ClearModelCache();
    }

}

#endif
