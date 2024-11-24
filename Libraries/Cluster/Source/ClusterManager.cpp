#include <map>
#include "Libraries/Base.h"
#include "Libraries/Cluster/KmeansCluster.hpp"
#include "Libraries/Cluster/IndexCluster.hpp"
#include "Libraries/Cluster/PluginBasedCluster.hpp"
#include "Libraries/Cluster/ClusterManager.hpp"
#include <functional>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

_D_Dragonian_Lib_Cluster_Namespace_Begin

using GetClusterFn = std::function<Cluster(const std::wstring&, size_t, size_t)>;

std::map<std::wstring, GetClusterFn> RegisteredCluster;
std::vector<std::wstring> ClusterList;

Cluster GetCluster(const std::wstring& ClusterName, const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize)
{
	const auto f_ClusterFn = RegisteredCluster.find(ClusterName);
	try
	{
		if (f_ClusterFn != RegisteredCluster.end())
			return f_ClusterFn->second(ClusterFile, ClusterDimension, ClusterSize);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	throw std::runtime_error("Unable To Find An Available Cluster");
}

void RegisterPlugin(const std::wstring& _PluginRootDirectory, const std::wstring& _PluginName)
{
	if (RegisteredCluster.contains(_PluginName))
		return;
	const auto PluginPath = _PluginRootDirectory + L"\\" + _PluginName;
	const auto _PluginFileName = _PluginName.substr(0, _PluginName.find_last_of('.'));
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(PluginPath);
		RegisteredCluster.emplace(_PluginFileName, [Plugin](const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize) -> Cluster {
			return std::make_shared<PluginCluster>(
				Plugin, PluginClusterInfo{ ClusterFile, ClusterDimension, ClusterSize }
			);
			});
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	ClusterList.emplace_back(_PluginFileName);
}

void RegisterCluster(const std::wstring& _PluginRootDirectory)
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

const std::vector<std::wstring>& GetClusterList()
{
	return ClusterList;
}

struct Init
{
	Init()
	{
		ClusterList.clear();
		RegisteredCluster.clear();
		ClusterList.emplace_back(L"KMeans");
		RegisteredCluster.emplace(L"KMeans", [](const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize) -> Cluster {
			return std::make_shared<KMeansCluster>(ClusterFile, ClusterDimension, ClusterSize);
			});
		ClusterList.emplace_back(L"Index");
		RegisteredCluster.emplace(L"Index", [](const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize) -> Cluster {
			return std::make_shared<IndexCluster>(ClusterFile, ClusterDimension, ClusterSize);
			});
		RegisterCluster(GetCurrentFolder() + L"/Plugins/Cluster");
	}
};
Init _Valdef_Init;

_D_Dragonian_Lib_Cluster_Namespace_End
