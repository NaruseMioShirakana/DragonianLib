#include "Libraries/Cluster/BaseCluster.hpp"

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
#ifdef _DEBUG
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
	auto Result = Search(PointsCont.Data(), CodebookID, Count);
	static auto _MyLogger = _D_Dragonian_Lib_Namespace GetDefaultLogger();
	_MyLogger->LogInfo(
		L"Search Codebook ID: " +
		std::to_wstring(CodebookID) +
		L", Input Shape: [" +
		std::to_wstring(Count) +
		L", " +
		std::to_wstring(InputDim) +
		L"], Cost Time: " +
		std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TimeBegin).count()) +
		L"ms",
		L"Cluster"
	);
	return Result;
#else
	return Search(PointsCont.Data(), CodebookID, Count);
#endif

}

_D_Dragonian_Lib_Cluster_Namespace_End