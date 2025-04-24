/**
 * @file PluginBase.h
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
 * @brief Plugin interface for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Util/Logger.h"
#include <unordered_map>

_D_Dragonian_Lib_Space_Begin

namespace Plugin
{
	DLogger& GetDefaultLogger() noexcept;

	class MPlugin final
	{
	public:
		using GetInstanceFunc = void* (*)(const void*);
		using DestoryInstanceFunc = void (*)(void*);
		MPlugin(const std::wstring& RelativePath);
		~MPlugin() = default;

		void* GetInstance(const void* UserParameter) const;
		void DestoryInstance(void* Instance) const;
		void* GetFunction(const std::string& FunctionName, bool Restrict = false) const;
		void* GetFunction(const char* FunctionName, bool Restrict = false) const;
		void* GetFunction(const std::string& FunctionName, bool Restrict = false);
		void* GetFunction(const char* FunctionName, bool Restrict = false);

		template <typename ReturnType, typename ...ArgTypes>
		ReturnType Invoke(const std::string& FunctionName, ArgTypes... Args)
		{
			auto Iter = _MyFunctions.find(FunctionName);
			void* Function;
			if (Iter == _MyFunctions.end())
				Function = GetFunction(FunctionName, true);
			else
				Function = Iter->second;

			return reinterpret_cast<ReturnType(*)(ArgTypes...)>(Function)(Args...);
		}

		template <typename ReturnType, typename ...ArgTypes>
		ReturnType Invoke(const char* FunctionName, ArgTypes... Args) const
		{
			auto Iter = _MyFunctions.find(FunctionName);
			if (Iter == _MyFunctions.end())
				_D_Dragonian_Lib_Throw_Exception("Failed to find function: " + std::string(FunctionName));
			return reinterpret_cast<ReturnType(*)(ArgTypes...)>(Iter->second)(Args...);
		}

	protected:
		std::shared_ptr<void> _MyLibrary = nullptr;
		GetInstanceFunc _MyGetInstance = nullptr;
		DestoryInstanceFunc _MyDestoryInstance = nullptr;
		std::unordered_map<std::string, void*> _MyFunctions;

	private:
		MPlugin() = delete;
		MPlugin(const MPlugin&) = delete;
		MPlugin(MPlugin&&) = delete;
		MPlugin& operator=(const MPlugin&) = delete;
		MPlugin& operator=(MPlugin&&) = delete;
		static void Free(void* Pointer);
		static void* MyLoadLibrary(const std::wstring& RelativePath);
	};

	using Plugin = std::shared_ptr<MPlugin>;

}

_D_Dragonian_Lib_Space_End