#include "../BaseCluster.hpp"

_D_Dragonian_Lib_Cluster_Namespace_Begin

Tensor<Float32, 2, Device::CPU> BaseCluster::Search(Float32*, Long, Int64)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

Tensor<Float32, 2, Device::CPU> BaseCluster::Search(const Tensor<Float32, 2, Device::CPU>& Points, Long CodebookID)
{
	const auto Count = Points.Size(0);
	const auto InputDim = Points.Size(1);
	const auto PointsCont = Points.Continuous().Evaluate();
	if (InputDim != _MyDimension)
		_D_Dragonian_Lib_Throw_Exception("Dimension mismatch");
	return Search(PointsCont.Data(), CodebookID, Count);
}

_D_Dragonian_Lib_Cluster_Namespace_End