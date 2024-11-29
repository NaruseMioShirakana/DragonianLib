#include "../NCNNBase.h"

_D_Dragonian_NCNN_Lib_Space_Header

std::unordered_map<std::wstring, std::shared_ptr<ncnn::Net>> GlobalNCNNModelCache;

NCNNModel::NCNNModel(
	const std::wstring& _Path,
	unsigned _DeviceID,
	unsigned _ThreadCount,
	bool _UseVulkan,
	FloatPrecision _Precision,
	bool _AddCache
) : m_UseVulkan(_UseVulkan), m_DeviceID(_DeviceID), m_ThreadCount(_ThreadCount)
{
	const auto ModelID =
		_Path + L"_" + std::to_wstring(_DeviceID) + L"_" +
		std::to_wstring(_ThreadCount) + L"_" +
		std::to_wstring(_UseVulkan) + L"_" +
		std::to_wstring(static_cast<unsigned long long>(_Precision));
	auto Iter = GlobalNCNNModelCache.find(ModelID);
	if (Iter != GlobalNCNNModelCache.end())
		m_NCNNNet = Iter->second;
	else
	{
		m_NCNNNet = std::make_shared<ncnn::Net>();
		if (_UseVulkan)
		{
			m_NCNNNet->opt.use_vulkan_compute = _UseVulkan;
			m_NCNNNet->set_vulkan_device(static_cast<int>(_DeviceID));
		}
		m_NCNNNet->opt.num_threads = static_cast<int>(_ThreadCount);
		m_NCNNNet->opt.use_bf16_storage = _Precision == FloatPrecision::BFloat16;
		m_NCNNNet->opt.use_fp16_storage = _Precision == FloatPrecision::Float16;
		m_NCNNNet->opt.use_fp16_arithmetic = _Precision == FloatPrecision::Float16;
		m_NCNNNet->opt.use_fp16_packed = _Precision == FloatPrecision::Float16;
		FileGuard PARAM, BIN;
		try
		{
			PARAM.Open(_Path + L"/model.param", L"rb");
			PARAM.Open(_Path + L"/model.bin", L"rb");
		}
		catch (const std::exception& _Exception)
		{
			_D_Dragonian_Lib_Throw_Exception(_Exception.what());
		}
		if (!PARAM.Enabled())
			_D_Dragonian_Lib_Throw_Exception("Failed to open model.param");
		if (!BIN.Enabled())
			_D_Dragonian_Lib_Throw_Exception("Failed to open model.bin");
		if (m_NCNNNet->load_param(PARAM))
			_D_Dragonian_Lib_Throw_Exception("Failed to load model.param");
		if (m_NCNNNet->load_model(BIN))
			_D_Dragonian_Lib_Throw_Exception("Failed to load model.bin");
		if (_AddCache)
			GlobalNCNNModelCache[ModelID] = m_NCNNNet;
	}
}

std::vector<Tensor> NCNNModel::Run(
	const std::vector<Tensor>& _Input
) const
{
	auto Context = m_NCNNNet->create_extractor();
	auto OutputIndices = m_NCNNNet->output_indexes();
	{
		int Index = 0;
		for (const auto& Input : _Input)
		{
			ncnn::Mat Mat;
			switch (Input.Rank)
			{
			case 1:
				Mat = ncnn::Mat(Input.Shape[0], 4ull);
				std::memcpy(Mat.data, Input.Buffer, Input.BufferSize);
				break;
			case 2:
				Mat = ncnn::Mat(Input.Shape[0], Input.Shape[1], 4ull);
				std::memcpy(Mat.data, Input.Buffer, Input.BufferSize);
				break;
			case 3:
				Mat = ncnn::Mat(Input.Shape[0], Input.Shape[1], Input.Shape[2], 4ull);
				std::memcpy(Mat.data, Input.Buffer, Input.BufferSize);
				break;
			case 4:
				Mat = ncnn::Mat(Input.Shape[0], Input.Shape[1], Input.Shape[2], Input.Shape[3], 4ull);
				std::memcpy(Mat.data, Input.Buffer, Input.BufferSize);
				break;
			default:
				_D_Dragonian_Lib_Throw_Exception("Invalid rank");
			}
			if (Context.input(Index++, Mat))
				_D_Dragonian_Lib_Throw_Exception("Failed to set input");
		}
	}
	std::vector<Tensor> Output;
	Output.reserve(OutputIndices.size());
	for(const auto Index : OutputIndices)
	{
		ncnn::Mat Mat;
		if(Context.extract(Index, Mat))
			_D_Dragonian_Lib_Throw_Exception("Failed to extract output");
		switch (Mat.dims)
		{
		case 1:
			Output.emplace_back(Mat.data, { Mat.w }, 1, Mat.w * 4ull, Mat);
			break;
		case 2:
			Output.emplace_back(Mat.data, { Mat.h, Mat.w }, 2, 4ull * Mat.w * Mat.h, Mat);
			break;
		case 3:
			Output.emplace_back(Mat.data, { Mat.c, Mat.h, Mat.w }, 3, 4ull * Mat.w * Mat.h * Mat.c, Mat);
			break;
		case 4:
			Output.emplace_back(Mat.data, { Mat.c, Mat.d, Mat.h, Mat.w }, 4, 4ull * Mat.w * Mat.h * Mat.c * Mat.d, Mat);
			break;
		default:
			_D_Dragonian_Lib_Throw_Exception("Invalid rank");
		}
	}
	return Output;
}


void NCNNModel::UnloadCachedModel(
	const std::wstring& _Path,
	unsigned _DeviceID,
	unsigned _ThreadCount,
	bool _UseVulkan,
	FloatPrecision _Precision
)
{
	const auto ModelID =
		_Path + L"_" + std::to_wstring(_DeviceID) + L"_" +
		std::to_wstring(_ThreadCount) + L"_" +
		std::to_wstring(_UseVulkan) + L"_" +
		std::to_wstring(static_cast<unsigned long long>(_Precision));
	auto Iter = GlobalNCNNModelCache.find(ModelID);
	if (Iter != GlobalNCNNModelCache.end())
		GlobalNCNNModelCache.erase(Iter);
}

_D_Dragonian_NCNN_Lib_Space_End