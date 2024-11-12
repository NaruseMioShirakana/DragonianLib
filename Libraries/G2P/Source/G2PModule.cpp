#include "../G2PModule.hpp"
#ifdef _WIN32
#include <Windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

_D_Dragonian_Lib_G2P_Header

using GetG2PModuleFn = std::function<G2PModule(const void*)>;
std::vector<std::wstring> G2PModulesList;
std::unordered_map<std::wstring, GetG2PModuleFn> RegisteredG2PModules;

G2PModule GetG2P(
	const std::wstring& Name,
	const void* Parameter
)
{
	const auto G2PModuleIt = RegisteredG2PModules.find(Name);
	try
	{
		if (G2PModuleIt != RegisteredG2PModules.end())
			return G2PModuleIt->second(Parameter);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available G2PModule");
}

void RegisterPlugin(const std::wstring& _PluginRootDirectory, const std::wstring& _PluginName)
{
	if (RegisteredG2PModules.contains(_PluginName))
		return;
	const auto PluginPath = _PluginRootDirectory + L"\\" + _PluginName;
	try
	{
		auto Plugin = Plugin::LoadPlugin(PluginPath);
		RegisteredG2PModules.emplace(_PluginName, [Plugin](const void* UserParameter) -> G2PModule {
			return std::make_shared<BasicG2P>(UserParameter, Plugin);
			});
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	G2PModulesList.emplace_back(_PluginName);
}

void RegisterG2PModule(const std::wstring& _PluginRootDirectory)
{
	if (_PluginRootDirectory.empty())
		return;
#ifdef _WIN32
	WIN32_FIND_DATAW FindFileData;
	HANDLE hFind = FindFirstFileW((_PluginRootDirectory + L"\\*.dll").c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
		return;
	do
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue;
		if (FindFileData.cFileName[0] == L'.')
			continue;
		const auto PluginName = FindFileData.cFileName;
		try
		{
			RegisterPlugin(_PluginRootDirectory, PluginName);
		}
		catch (std::exception& e)
		{
			FindClose(hFind);
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	} while (FindNextFileW(hFind, &FindFileData));
	FindClose(hFind);
#else
	DIR* dir;
	struct dirent* ptr;
	if ((dir = opendir(_PluginRootDirectory.c_str())) == nullptr)
		return;

	while ((ptr = readdir(dir)) != nullptr)
	{
		if (ptr->d_type == DT_DIR)
			continue;
		const auto PluginName = ptr->d_name;
		try {
			RegisterPlugin(_PluginRootDirectory, PluginName);
		}
		catch (std::exception& e)
		{
			closedir(dir);
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	}
	closedir(dir);
#endif
}

const std::vector<std::wstring>& GetG2PModuleList()
{
	return G2PModulesList;
}

_D_Dragonian_Lib_G2P_End