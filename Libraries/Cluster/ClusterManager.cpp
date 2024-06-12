#include "ClusterManager.hpp"
#include <map>
#include <stdexcept>
#include "Base.h"
#include "Util/Logger.h"

namespace DragonianLib {

	std::map<std::wstring, GetClusterFn> RegisteredCluster;

	ClusterWrp GetCluster(const std::wstring& _name, const std::wstring& _path, size_t hidden_size, size_t KmeansLen)
	{
		const auto f_ClusterFn = RegisteredCluster.find(_name);
		if (f_ClusterFn != RegisteredCluster.end())
			return f_ClusterFn->second(_path, hidden_size, KmeansLen);
		throw std::runtime_error("Unable To Find An Available Cluster");
	}

	void RegisterCluster(const std::wstring& _name, const GetClusterFn& _constructor_fn)
	{
		if (RegisteredCluster.contains(_name))
		{
			DragonianLibLogMessage(L"[Warn] ClusterNameConflict");
			return;
		}
		RegisteredCluster[_name] = _constructor_fn;
	}

}
