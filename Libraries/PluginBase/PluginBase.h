#pragma once
#include "Base.h"

_D_Dragonian_Lib_Space_Begin

namespace Plugin
{
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
			void* Function = nullptr;
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