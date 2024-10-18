#include "ClusterManager.hpp"
#include <map>
#include <stdexcept>
#include "Base.h"
#include "Util/Logger.h"

namespace DragonianLib {

	std::map<std::wstring, GetClusterFn> RegisteredCluster;

	ClusterWrp GetCluster(const std::wstring& ClusterName, const std::wstring& ClusterFile, size_t ClusterDimension, size_t ClusterSize)
	{
		const auto f_ClusterFn = RegisteredCluster.find(ClusterName);
		if (f_ClusterFn != RegisteredCluster.end())
			return f_ClusterFn->second(ClusterFile, ClusterDimension, ClusterSize);
		throw std::runtime_error("Unable To Find An Available Cluster");
	}

	void RegisterCluster(const std::wstring& ClusterName, const GetClusterFn& Constructor)
	{
		if (RegisteredCluster.contains(ClusterName))
		{
			LogWarn(L"Name Of Cluster Already Registered");
			return;
		}
		RegisteredCluster[ClusterName] = Constructor;
	}

}
