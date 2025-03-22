#include <map>
#include "Libraries/Base.h"
#include "Libraries/Cluster/KmeansCluster.hpp"
#include "Libraries/Cluster/IndexCluster.hpp"
#include "Libraries/Cluster/PluginBasedCluster.hpp"
#include "Libraries/Cluster/ClusterManager.hpp"
#include <functional>

_D_Dragonian_Lib_Cluster_Namespace_Begin

std::unordered_map<std::wstring, Constructor> _GlobalRegisteredCluster;
std::vector<std::wstring> _GlobalClusterList;

void RegisterPlugin(
	const std::wstring& _PluginPath,
	const std::wstring& _PluginName
)
{
	if (_PluginPath.empty() || _PluginName.empty())
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Could not register plugin: " + _PluginName + L" at " + _PluginPath, L"ClusterManager");
		return;
	}

	if (_GlobalRegisteredCluster.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" at " + _PluginPath + L" already registered", L"ClusterManager");
		return;
	}
	try
	{
		auto Plugin = std::make_shared<Plugin::MPlugin>(_PluginPath);
		_GlobalRegisteredCluster.emplace(
			_PluginName,
			[Plugin](const std::wstring& ClusterFile, Int64 ClusterDimension, Int64 ClusterSize) -> Cluster {
				return std::make_shared<PluginCluster>(Plugin, PluginClusterInfo{ ClusterFile, ClusterDimension, ClusterSize });
			}
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_GlobalClusterList.emplace_back(_PluginName);
}

void RegisterClusters(
	const std::wstring& _PluginRootDirectory
)
{
	if (_PluginRootDirectory.empty())
		return;
	std::filesystem::path PluginRootDirectory(_PluginRootDirectory);
	if (!std::filesystem::exists(PluginRootDirectory))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin root directory: " + _PluginRootDirectory + L" does not exist", L"ClusterManager");
		return;
	}
	for (const auto& PluginDirectoryEntry : std::filesystem::directory_iterator(PluginRootDirectory))
	{
		if (PluginDirectoryEntry.is_regular_file())
		{
			const auto Extension = PluginDirectoryEntry.path().extension().wstring();
			if (Extension != L".dll" && Extension != L".so" && Extension != L".dylib")
				continue;
			const auto PluginName = PluginDirectoryEntry.path().stem().wstring();
			RegisterPlugin(PluginDirectoryEntry.path().wstring(), PluginName);
		}
		else if (PluginDirectoryEntry.is_directory())
		{
			const auto PluginName = PluginDirectoryEntry.path().filename().wstring();
			const auto PluginPath = PluginDirectoryEntry.path() / (PluginName + (_WIN32 ? L".dll" : L".so"));
			RegisterPlugin(PluginPath.wstring(), PluginName);
		}
	}
}

void RegisterCluster(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
)
{
	if (_GlobalRegisteredCluster.contains(_PluginName))
	{
		Plugin::GetDefaultLogger()->LogWarn(L"Plugin: " + _PluginName + L" already registered", L"ClusterManager");
		return;
	}
	_GlobalRegisteredCluster.emplace(_PluginName, _Constructor);
	_GlobalClusterList.emplace_back(_PluginName);
}

Cluster New(
	const std::wstring& ClusterName,
	const std::wstring& ClusterFile,
	Int64 ClusterDimension,
	Int64 ClusterSize
)
{
	const auto f_ClusterFn = _GlobalRegisteredCluster.find(ClusterName);
	try
	{
		if (f_ClusterFn != _GlobalRegisteredCluster.end())
			return f_ClusterFn->second(ClusterFile, ClusterDimension, ClusterSize);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Unable To Find An Available Cluster");
}

const std::vector<std::wstring>& GetClusterList()
{
	return _GlobalClusterList;
}

struct Init
{
	Init()
	{
		_GlobalClusterList.clear();
		_GlobalRegisteredCluster.clear();

		RegisterCluster(
			L"KMeans",
			[](const std::wstring& ClusterFile, Int64 ClusterDimension, Int64 ClusterSize) -> Cluster {
				return std::make_shared<KMeansCluster>(ClusterFile, ClusterDimension, ClusterSize);
			}
		);
		RegisterCluster(
			L"Index",
			[](const std::wstring& ClusterFile, Int64 ClusterDimension, Int64 ClusterSize) -> Cluster {
				return std::make_shared<IndexCluster>(ClusterFile, ClusterDimension, ClusterSize);
			}
		);
		RegisterClusters(GetCurrentFolder() + L"/Plugins/Cluster");
	}
};
Init _Valdef_Init;

_D_Dragonian_Lib_Cluster_Namespace_End
