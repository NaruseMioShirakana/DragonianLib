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

	protected:
		std::shared_ptr<void> _MyLibrary = nullptr;
		GetInstanceFunc _MyGetInstance = nullptr;
		DestoryInstanceFunc _MyDestoryInstance = nullptr;
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

	Plugin LoadPlugin(const std::wstring& RelativePath);

	void UnloadPlugin(const std::wstring& RelativePath);

}

_D_Dragonian_Lib_Space_End