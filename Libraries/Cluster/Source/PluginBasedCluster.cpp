#include "Libraries/Cluster/PluginBasedCluster.hpp"
#include "Libraries/MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Cluster_Namespace_Begin

PluginCluster::PluginCluster(const Plugin::Plugin& Plugin, const PluginClusterInfo& Params)
	: BaseCluster(Params.ClusterDimension), _MyInstance(Plugin->GetInstance(&Params)), _MyPlugin(Plugin),
	_MySearchFunction((SearchFunctionType)Plugin->GetFunction("ClusterSearch", true))
{

}

PluginCluster::~PluginCluster()
{
	_MyPlugin->DestoryInstance(_MyInstance);
}

Tensor<Float32, 2, Device::CPU> PluginCluster::Search(Float32* Points, Long CodebookID, Int64 Count)
{
	DragonianLibSTL::Vector<float> Result(Count * _MyDimension);
	_MySearchFunction(_MyInstance, Points, CodebookID, Count, Result.Data());
	auto Allocator = Result.GetAllocator();
	auto [Data, Size] = Result.Release();
	return Tensor<Float32, 2, Device::CPU>::FromBuffer(
		Dimensions<2>{Count, _MyDimension }, Data, Size, Allocator
	);
}

_D_Dragonian_Lib_Cluster_Namespace_End