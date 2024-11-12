#pragma once
#include "BaseCluster.hpp"
#include "PluginBase/PluginBase.h"

_D_Dragonian_Lib_Cluster_Namespace_Begin

struct PluginClusterInfo
{
	std::wstring ClusterFile;
	size_t ClusterDimension;
	size_t ClusterSize;
};

class PluginCluster : public BaseCluster
{
public:
	using SearchFunctionType = void(*)(void*, const float*, long, int64_t, float*); ///< Search function type(Instance, Point, Sid, Count, Output)

	PluginCluster(const Plugin::Plugin& Plugin, const PluginClusterInfo& Params);
	~PluginCluster() override;

	DragonianLibSTL::Vector<float> Search(float* Point, long SpeakerId, int64_t Count) override;

protected:
	void* _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;
	size_t _MyClusterDimension;
	size_t _MyClusterSize;
	SearchFunctionType _MySearchFunction = nullptr; ///< "void ClusterSearch(void*, const float*, long, int64_t, float*)"
private:
	PluginCluster(const PluginCluster&) = delete;
	PluginCluster& operator=(const PluginCluster&) = delete;
	PluginCluster(PluginCluster&&) = delete;
	PluginCluster& operator=(PluginCluster&&) = delete;
};

_D_Dragonian_Lib_Cluster_Namespace_End