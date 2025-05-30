﻿#include "Libraries/Cluster/IndexCluster.hpp"
#include "Libraries/MyTemplateLibrary/Vector.h"
#include <filesystem>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

_D_Dragonian_Lib_Cluster_Namespace_Begin

class IndexClusterCore
{
public:
	IndexClusterCore() = delete;
	IndexClusterCore(const char* _path);
	~IndexClusterCore() = default;
	IndexClusterCore(const IndexClusterCore&) = default;
	IndexClusterCore(IndexClusterCore&& move) noexcept = default;
	IndexClusterCore& operator=(const IndexClusterCore&) = default;
	IndexClusterCore& operator=(IndexClusterCore&& move) noexcept = default;
	DragonianLibSTL::Vector<float> find(const float* points, faiss::idx_t n_points, faiss::idx_t n_searched_points = 8);
	float* GetVec(faiss::idx_t index);
private:
	std::shared_ptr<faiss::Index> IndexPtr = nullptr;
	faiss::idx_t Dim = 0;
	DragonianLibSTL::Vector<float> IndexsVector;
};


IndexClusterCore::IndexClusterCore(const char* _path) : IndexPtr(faiss::read_index(_path))
{
	IndexsVector = DragonianLibSTL::Vector(IndexPtr->ntotal * IndexPtr->d, 0.f);
	IndexPtr->reconstruct_n(0, IndexPtr->ntotal, IndexsVector.Data());
	Dim = IndexPtr->d;
}

float* IndexClusterCore::GetVec(faiss::idx_t index)
{
	return IndexsVector.Data() + index * Dim;
}

DragonianLibSTL::Vector<float> IndexClusterCore::find(const float* points, faiss::idx_t n_points, faiss::idx_t n_searched_points)
{
	DragonianLibSTL::Vector<float> result(Dim * n_points);
	DragonianLibSTL::Vector<float> distances(n_searched_points * n_points);
	DragonianLibSTL::Vector<faiss::idx_t> labels(n_searched_points * n_points);
	IndexPtr->search(n_points, points, n_searched_points, distances.Data(), labels.Data());
	for (faiss::idx_t pt = 0; pt < n_points; ++pt)
	{
		DragonianLibSTL::Vector result_pt(Dim, 0.f);
		const auto idx_vec = labels.Data() + pt * n_searched_points;     // SIZE:[n_searched_points]
		const auto dis_vec = distances.Data() + pt * n_searched_points;  // SIZE:[n_searched_points]
		float sum = 0.f;                                                       // dis_vec[i] / sum = pGetVec(idx_vec[i])
		for (faiss::idx_t spt = 0; spt < n_searched_points; ++spt)             // result_pt = GetVec(idx_vec[i])
		{
			if (idx_vec[spt] < 0)
				continue;
			dis_vec[spt] = (1 / dis_vec[spt]) * (1 / dis_vec[spt]);
			sum += dis_vec[spt];
		}
		if (sum == 0.f) sum = 1.f;
		for (faiss::idx_t spt = 0; spt < n_searched_points; ++spt)
		{
			if (idx_vec[spt] < 0)
				continue;
			const auto sedpt = GetVec(idx_vec[spt]);
			const auto pcnt = (dis_vec[spt] / sum);
			for (faiss::idx_t sptp = 0; sptp < Dim; ++sptp)
				result_pt[sptp] += pcnt * sedpt[sptp];
		}
		memcpy(result.Data() + pt * Dim, result_pt.Data(), Dim * sizeof(float));
	}
	return result;
}

IndexCluster::IndexCluster(const std::wstring& RootPath, Int64 Dimension, Int64)
	:BaseCluster(Dimension)
{
	const auto RawPath = RootPath + L"/Index-";
	size_t idx = 0;
	while (true)
	{
		std::filesystem::path IndexPath = RawPath + std::to_wstring(idx++) + L".index";
		if (!exists(IndexPath)) break;
		Indexs.emplace_back(std::make_shared<IndexClusterCore>(IndexPath.string().c_str()));
	}
	if (Indexs.empty())
		_D_Dragonian_Lib_Throw_Exception("Could not find any index file");
}

Tensor<Float32, 2, Device::CPU> IndexCluster::Search(Float32* Point, Long CodebookID, Int64 PointCount)
{
	if (size_t(CodebookID) < Indexs.size())
	{
		auto Result = Indexs[CodebookID]->find(Point, PointCount);
		auto Allocator = Result.GetAllocator();
		auto [Data, Size] = Result.Release();
		return Tensor<Float32, 2, Device::CPU>::FromBuffer(
			Dimensions<2>{PointCount, _MyDimension }, Data, Size, Allocator
		);
	}
	_D_Dragonian_Lib_Throw_Exception("CodebookID out of range");
}

_D_Dragonian_Lib_Cluster_Namespace_End
