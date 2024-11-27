#pragma once
#include "Libraries/Base.h"
#include "ncnn/net.h"

#define _D_Dragonian_NCNN_Lib_Space_Header _D_Dragonian_Lib_Space_Begin namespace NCNNLib{
#define _D_Dragonian_NCNN_Lib_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_NCNN_Lib_Space_Header

struct Tensor final
{
	void* Buffer = nullptr;
	int Shape[4];
	int Rank = 0;
	size_t BufferSize = 0;
};

class NCNNModel final
{
public:
	NCNNModel() = delete;
	NCNNModel(
		const std::wstring& _Path,
		unsigned _DeviceID,
		unsigned _ThreadCount,
		bool _UseVulkan = true,
		FloatPrecision _Precision = FloatPrecision::Float16,
		bool _AddCache = false
	);

	std::vector<Tensor> Run(
		const std::vector<Tensor>& _Input,
		const std::vector<std::wstring>& _InputNames,
		const std::vector<std::wstring>& _OutputNames
	) const;

protected:
	std::shared_ptr<ncnn::Net> m_NCNNNet;
	bool m_UseVulkan;
	unsigned m_DeviceID;
	unsigned m_ThreadCount;

public:
	static void UnloadCachedModel(
		const std::wstring& _Path,
		unsigned _DeviceID,
		unsigned _ThreadCount,
		bool _UseVulkan = true,
		FloatPrecision _Precision = FloatPrecision::Float16
	);
};

_D_Dragonian_NCNN_Lib_Space_End