#include "Libraries/Cluster/KmeansCluster.hpp"
#include "Libraries/NumpySupport/NumpyFileFormat.h"

_D_Dragonian_Lib_Cluster_Namespace_Begin

Tensor<Float32, 2, Device::CPU> KMeansCluster::Search(Float32* Points, Long CodebookID, Int64 Count)
{
	if (size_t(CodebookID) < _MyTrees.size())
	{
		DragonianLibSTL::Vector<float> Result;
		Result.Reserve(_MyDimension * Count);
		for (int64_t PId = 0; PId < Count; ++PId)
		{
			auto Output = _MyTrees[CodebookID].nearest_point(
				{ Points + PId * _MyDimension, Points + (PId + 1) * _MyDimension }
			);
			Result.Insert(Result.End(), Output.data(), Output.data() + Output.size());
		}
		auto Allocator = Result.GetAllocator();
		auto [Data, Size] = Result.Release();
		return Tensor<Float32, 2, Device::CPU>::FromBuffer(
			Dimensions<2>{Count, _MyDimension }, Data, Size, Allocator
		);
	}
	_D_Dragonian_Lib_Throw_Exception("CodebookID out of range");
}

KMeansCluster::KMeansCluster(const std::wstring& ClusterRootPath, Int64 Dimension, Int64 ClusterSize)
	:BaseCluster(Dimension)
{
	auto [Shape, Data] = NumpyFileFormat::LoadNumpyFile(ClusterRootPath + L"/KMeans.npy");
	if (Shape.Size() != 3 || Shape[1] != ClusterSize || Shape[2] != Dimension)
		_D_Dragonian_Lib_Throw_Exception("Invalid cluster file");
	for (SizeType i = 0; i < Shape[0]; ++i)
	{
		KDTree::pointVec PointVec;
		for (SizeType j = 0; j < ClusterSize; ++j)
			PointVec.emplace_back(
				((float*)Data.Data()) + i * ClusterSize * Dimension + j * Dimension,
				((float*)Data.Data()) + i * ClusterSize * Dimension + (j + 1) * Dimension
			);
		_MyTrees.emplace_back(std::move(PointVec));
	}
}

_D_Dragonian_Lib_Cluster_Namespace_End