#include "../KmeansCluster.hpp"
#include "Libraries/Base.h"

_D_Dragonian_Lib_Cluster_Namespace_Begin

DragonianLibSTL::Vector<float> KMeansCluster::Search(float* Point, long SpeakerId, int64_t Count)
{
	if (size_t(SpeakerId) < _tree.size())
	{
		DragonianLibSTL::Vector<float> res;
		res.Reserve(dims * Count * 2);
		for (int64_t pt = 0; pt < Count; ++pt)
		{
			auto tmp = _tree[SpeakerId].nearest_point({ Point + pt * dims,Point + (pt + 1) * dims });
			res.Insert(res.End(), tmp.data(), tmp.data() + tmp.size());
		}
		return res;
	}
	return { Point, Point + dims * Count };
}

KMeansCluster::KMeansCluster(const std::wstring& _RootPath, size_t Dims, size_t KMeansLen)
{
	dims = Dims;
	FILE* file = nullptr;
	_wfopen_s(&file, (_RootPath + L"/KMeans.npy").c_str(), L"rb");
	if (!file)
		_D_Dragonian_Lib_Throw_Exception("KMeansFileNotExist");
	constexpr long idx = 128;
	fseek(file, idx, SEEK_SET);
	std::vector<float> tmpData(Dims);
	const size_t ec = size_t(Dims) * sizeof(float);
	std::vector<std::vector<float>> _tmp;
	_tmp.reserve(KMeansLen);
	while (fread(tmpData.data(), 1, ec, file) == ec)
	{
		_tmp.emplace_back(tmpData);
		if (_tmp.size() == KMeansLen)
		{
			_tree.emplace_back(_tmp);
			_tmp.clear();
		}
	}
}

_D_Dragonian_Lib_Cluster_Namespace_End