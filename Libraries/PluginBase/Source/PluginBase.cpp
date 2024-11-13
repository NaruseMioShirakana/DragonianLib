#include "../PluginBase.h"
#include "Util/StringPreprocess.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

_D_Dragonian_Lib_Space_Begin

namespace Plugin
{

	MPlugin::MPlugin(const std::wstring& RelativePath) : _MyLibrary(MyLoadLibrary(RelativePath), Free)
	{
		_MyGetInstance = (GetInstanceFunc)GetFunction("CreateInstance", true);
		_MyDestoryInstance = (DestoryInstanceFunc)GetFunction("DestoryInstance", true);
	}

	void MPlugin::Free(void* Pointer)
	{
#ifdef _WIN32
		if (Pointer)
			FreeLibrary(HINSTANCE(Pointer));
#else
		if (Pointer)
			dlclose(Pointer);
#endif
	}

	void* MPlugin::MyLoadLibrary(const std::wstring& RelativePath)
	{
#ifdef _WIN32
		void* Library = LoadLibraryW(RelativePath.c_str());
		if (!Library)
			_D_Dragonian_Lib_Throw_Exception("Failed to load library: " + WideStringToUTF8(RelativePath));
#else
		void* Library = dlopen(RelativePath.c_str(), RTLD_LAZY);
		auto ErrorMessage = dlerror();
		if (ErrorMessage)
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
#endif
		return Library;
	}

	void* MPlugin::GetInstance(const void* UserParameter) const
	{
		return _MyGetInstance(UserParameter);
	}

	void MPlugin::DestoryInstance(void* Instance) const
	{
		_MyDestoryInstance(Instance);
	}

	void* MPlugin::GetFunction(const char* FunctionName, bool Restrict) const
	{
#ifdef _WIN32
		auto Function = GetProcAddress(HINSTANCE(_MyLibrary.get()), FunctionName);
		if (!Function && Restrict)
			_D_Dragonian_Lib_Throw_Exception("Failed to get function: " + std::string(FunctionName));
		return (void*)Function;
#else
		auto Function = dlsym(_MyLibrary, FunctionName);
		auto ErrorMessage = dlerror();
		if (ErrorMessage && Restrict)
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		return Function;
#endif
	}

	void* MPlugin::GetFunction(const std::string& FunctionName, bool Restrict) const
	{
		return GetFunction(FunctionName.c_str(), Restrict);
	}

	static std::unordered_map<std::wstring, std::shared_ptr<MPlugin>> _MyPlugins;

	std::shared_ptr<MPlugin> LoadPlugin(const std::wstring& RelativePath)
	{
		auto PluginIt = _MyPlugins.find(RelativePath);
		if (PluginIt != _MyPlugins.end())
			return PluginIt->second;
		return _MyPlugins[RelativePath] = std::make_shared<MPlugin>(RelativePath);
	}

	void UnloadPlugin(const std::wstring& RelativePath)
	{
		auto PluginIt = _MyPlugins.find(RelativePath);
		if (PluginIt != _MyPlugins.end())
			_MyPlugins.erase(PluginIt);
	}

}

_D_Dragonian_Lib_Space_End