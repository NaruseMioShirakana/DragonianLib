#include "../PluginBasedCluster.hpp"

_D_Dragonian_Lib_Cluster_Namespace_Begin

PluginCluster::PluginCluster(const Plugin::Plugin& Plugin, const PluginClusterInfo& Params)
	:_MyInstance(Plugin->GetInstance(&Params)), _MyPlugin(Plugin),
	_MyClusterDimension(Params.ClusterDimension), _MyClusterSize(Params.ClusterSize),
	_MySearchFunction((SearchFunctionType)Plugin->GetFunction("ClusterSearch", true)) {}

PluginCluster::~PluginCluster()
{
	_MyPlugin->DestoryInstance(_MyInstance);
}

DragonianLibSTL::Vector<float> PluginCluster::Search(float* Point, long SpeakerId, int64_t Count)
{
	DragonianLibSTL::Vector<float> Result(Count * _MyClusterDimension);
	_MySearchFunction(_MyInstance, Point, SpeakerId, Count, Result.Data());
	return Result;
}

_D_Dragonian_Lib_Cluster_Namespace_End