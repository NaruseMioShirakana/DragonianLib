#include "NCNNLibrary/NCNNBase/NCNNBase.h"
#include "ncnn/net.h"
#include "omp.h"

_D_Dragonian_NCNN_Lib_Space_Header

std::unordered_map<UInt64, std::shared_ptr<ncnn::Net>> GlobalNCNNModelCache;  // NOLINT(misc-use-internal-linkage)

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->GetLoggerId() + L"::NCNN",
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

ncnn::Option Cvt2Opt(const NCNNOptions& m_Options)
{
	ncnn::Option Opt;
	Opt.lightmode = m_Options.lightmode;
	Opt.use_shader_pack8 = m_Options.use_shader_pack8;
	Opt.use_subgroup_ops = m_Options.use_subgroup_ops;
	Opt.use_reserved_0 = m_Options.use_reserved_0;
	Opt.num_threads = m_Options.num_threads;
	Opt.blob_allocator = m_Options.blob_allocator;
	Opt.workspace_allocator = m_Options.workspace_allocator;
#if NCNN_VULKAN
	Opt.blob_vkallocator = m_Options.blob_vkallocator;
	Opt.workspace_vkallocator = m_Options.workspace_vkallocator;
	Opt.staging_vkallocator = m_Options.staging_vkallocator;
	Opt.pipeline_cache = m_Options.pipeline_cache;
#endif
	Opt.openmp_blocktime = m_Options.openmp_blocktime;
	Opt.use_winograd_convolution = m_Options.use_winograd_convolution;
	Opt.use_sgemm_convolution = m_Options.use_sgemm_convolution;
	Opt.use_int8_inference = m_Options.use_int8_inference;
	Opt.use_vulkan_compute = m_Options.use_vulkan_compute;
	Opt.use_bf16_storage = m_Options.use_bf16_storage;
	Opt.use_fp16_packed = m_Options.use_fp16_packed;
	Opt.use_fp16_storage = m_Options.use_fp16_storage;
	Opt.use_fp16_arithmetic = m_Options.use_fp16_arithmetic;
	Opt.use_int8_packed = m_Options.use_int8_packed;
	Opt.use_int8_storage = m_Options.use_int8_storage;
	Opt.use_int8_arithmetic = m_Options.use_int8_arithmetic;
	Opt.use_packing_layout = m_Options.use_packing_layout;
	Opt.vulkan_device_index = m_Options.vulkan_device_index;
	Opt.use_reserved_1 = m_Options.use_reserved_1;
	Opt.use_image_storage = m_Options.use_image_storage;
	Opt.use_tensor_storage = m_Options.use_tensor_storage;
	Opt.use_reserved_2 = m_Options.use_reserved_2;
	Opt.flush_denormals = m_Options.flush_denormals;
	Opt.use_local_pool_allocator = m_Options.use_local_pool_allocator;
	Opt.use_shader_local_memory = m_Options.use_shader_local_memory;
	Opt.use_cooperative_matrix = m_Options.use_cooperative_matrix;
	Opt.use_winograd23_convolution = m_Options.use_winograd23_convolution;
	Opt.use_winograd43_convolution = m_Options.use_winograd43_convolution;
	Opt.use_winograd63_convolution = m_Options.use_winograd63_convolution;
	Opt.use_a53_a55_optimized_kernel = m_Options.use_a53_a55_optimized_kernel;
	Opt.use_fp16_uniform = m_Options.use_fp16_uniform;
	Opt.use_int8_uniform = m_Options.use_int8_uniform;
	Opt.use_reserved_9 = m_Options.use_reserved_9;
	Opt.use_reserved_10 = m_Options.use_reserved_10;
	Opt.use_reserved_11 = m_Options.use_reserved_11;
	return Opt;
}

NCNNModel::NCNNModel(
	const std::wstring& _Path,
	const NCNNOptions& Options,
	bool _AddCache,
	DLogger _Logger
) : m_Options(Options), m_Logger(std::move(_Logger))
{
	using m_bytes_t = const char[sizeof(NCNNOptions)];
	const auto ModelID = Hash::Hash(*((m_bytes_t*)(&m_Options))) + Hash::Hash(_Path);
	auto Iter = GlobalNCNNModelCache.find(ModelID);
	if (Iter != GlobalNCNNModelCache.end())
		m_NCNNNet = Iter->second;
	else
	{
		m_NCNNNet = std::make_shared<ncnn::Net>();
		m_NCNNNet->register_custom_layer(
			"nn.InstanceNorm1d", InstanceNorm1D_layer_creator, InstanceNorm1D_layer_destroyer
		);
		m_NCNNNet->opt = Cvt2Opt(m_Options);
		if (m_Options.use_vulkan_compute)
			m_NCNNNet->set_vulkan_device(m_Options.vulkan_device_index);
		
		FileGuard PARAM, BIN;
		try
		{
			PARAM.Open(_Path + L"/model.param", L"rb");
			BIN.Open(_Path + L"/model.bin", L"rb");
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

void NCNNModel::UnloadCachedModel(
	const std::wstring& _Path,
	const NCNNOptions& Options
)
{
	using m_bytes_t = const char[sizeof(NCNNOptions)];
	const auto ModelID = Hash::Hash(*((m_bytes_t*)(&Options))) + Hash::Hash(_Path);
	auto Iter = GlobalNCNNModelCache.find(ModelID);
	if (Iter != GlobalNCNNModelCache.end())
		GlobalNCNNModelCache.erase(Iter);
}

class InstanceNorm1D : public ncnn::Layer
{
public:
	InstanceNorm1D()
	{
		one_blob_only = true;
		support_inplace = true;
	}

	int load_param(const ncnn::ParamDict& pd) override
	{
		affine = pd.get(0, 0);
		eps = pd.get(1, 1e-5f);
		if (affine)
			channels = pd.get(2, 0);
		return 0;
	}

	int load_model(const ncnn::ModelBin& mb) override
	{
		if (affine)
		{
			beta = mb.load(channels, 1);
			scale = mb.load(channels, 1);
			return scale.empty() || beta.empty();
		}
		return 0;
	}

	int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const override
	{
		if (channels != bottom_top_blob.h)
		{
			NCNN_LOGE("num_feature mismatch, requested [%d] but got [%d]", channels, bottom_top_blob.h);
			return -1;
		}
		if (opt.use_fp16_storage && bottom_top_blob.elemsize / bottom_top_blob.elempack == 2)
			return forward_inplace_f16(bottom_top_blob, opt);
		return forward_inplace_f32(bottom_top_blob, opt);
	}

	int forward_inplace_f16(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
	{
		return -1;
	}

	int forward_inplace_f32(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
	{
		int w = bottom_top_blob.w;

#pragma omp parallel for num_threads(opt.num_threads)
		for (int c = 0; c < channels; ++c)
		{
			float* x = bottom_top_blob.row(c);

			float mean = 0.f;
			float s = 0.f;
			for (int i = 0; i < w; ++i)
			{
				auto v = x[i];
				mean += v;
				s += v * v;
			}
			mean /= static_cast<float>(w);
			s /= static_cast<float>(w);

			if (affine)
			{
				auto a = scale[c] / sqrt(s + eps);
				auto b = beta[c];

				//gamma[c] / sqrt(s + eps) * (x - mean) + beta[c]
				//a * (x - mean) + b
				for (int i = 0; i < w; ++i)
					x[i] = a * (x[i] - mean) + b;
			}
			else
			{
				auto a = sqrt(s + eps);
				for (int i = 0; i < w; ++i)
					x[i] = a * (x[i] - mean);
			}
		}

		return 0;
	}

private:
	int channels = 0;
	float eps = 0;
	bool affine = false;
	ncnn::Mat scale;
	ncnn::Mat beta;
};

DEFINE_LAYER_CREATOR(InstanceNorm1D);
DEFINE_LAYER_DESTROYER(InstanceNorm1D);

_D_Dragonian_NCNN_Lib_Space_End