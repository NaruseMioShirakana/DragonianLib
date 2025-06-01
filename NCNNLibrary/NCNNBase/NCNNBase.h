#pragma once
#include "TensorLib/Include/Base/Tensor/Functional.h"

#define _D_Dragonian_NCNN_Lib_Space_Header _D_Dragonian_Lib_Space_Begin namespace NCNN{
#define _D_Dragonian_NCNN_Lib_Space_End } _D_Dragonian_Lib_Space_End
#define _D_Dragonian_Lib_NCNN_Space \
	_D_Dragonian_Lib_Namespace \
	NCNN::

namespace ncnn { class Net; class Layer; class Allocator; class VkAllocator; class PipelineCache; }

_D_Dragonian_NCNN_Lib_Space_Header

DLogger& GetDefaultLogger() noexcept;

struct NCNNOptions
{
    // light mode
    // intermediate blob will be recycled when enabled,
    // enabled by default
    bool lightmode = true;

    // use pack8 shader
    bool use_shader_pack8 = false;

    // enable subgroup in shader
    bool use_subgroup_ops = false;

    bool use_reserved_0 = false;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());

    // blob memory allocator
    ncnn::Allocator* blob_allocator = nullptr;

    // workspace memory allocator
    ncnn::Allocator* workspace_allocator = nullptr;

    // blob memory allocator
    ncnn::VkAllocator* blob_vkallocator = nullptr;

    // workspace memory allocator
    ncnn::VkAllocator* workspace_vkallocator = nullptr;

    // staging memory allocator
    ncnn::VkAllocator* staging_vkallocator = nullptr;

    // pipeline cache
    ncnn::PipelineCache* pipeline_cache = nullptr;

    // the time openmp threads busy-wait for more work before going to sleep
    // default value is 20ms to keep the cores enabled
    // without too much extra power consumption afterward
    int openmp_blocktime = 20;

    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution = true;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution = true;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_int8_inference = true;

    // enable vulkan compute
    bool use_vulkan_compute = false;

    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    bool use_bf16_storage = false;

    // enable options for gpu inference
    bool use_fp16_packed = false;
    bool use_fp16_storage = false;
    bool use_fp16_arithmetic = false;
    bool use_int8_packed = false;
    bool use_int8_storage = false;
    bool use_int8_arithmetic = false;

    // enable simd-friendly packed memory layout
    // improve all operator performance on all arm devices, will consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_packing_layout = true;

    // the vulkan device
    int vulkan_device_index = -1;

    bool use_reserved_1 = false;

    // turn on for adreno
    bool use_image_storage = false;
    bool use_tensor_storage = false;

    bool use_reserved_2 = false;

    // enable DAZ(Denormals-Are-Zero) and FTZ(Flush-To-Zero)
    // default value is 3
    // 0 = DAZ OFF, FTZ OFF
    // 1 = DAZ ON , FTZ OFF
    // 2 = DAZ OFF, FTZ ON
    // 3 = DAZ ON,  FTZ ON
    int flush_denormals = 3;

    bool use_local_pool_allocator = true;

    // enable local memory optimization for gpu inference
    bool use_shader_local_memory = true;

    // enable cooperative matrix optimization for gpu inference
    bool use_cooperative_matrix = true;

    // more fine-grained control of winograd convolution
    bool use_winograd23_convolution = true;
    bool use_winograd43_convolution = true;
    bool use_winograd63_convolution = true;

    // this option is turned on for A53/A55 automatically, 
    // but you can force this on/off if you wish
    bool use_a53_a55_optimized_kernel = false;

    // enable options for shared variables in gpu shader
    bool use_fp16_uniform = true;
    bool use_int8_uniform = true;

    bool use_reserved_9 = false;
    bool use_reserved_10 = false;
    bool use_reserved_11 = false;
};

class NCNNModel
{
public:
	NCNNModel() = delete;
	NCNNModel(
		const std::wstring& _Path,
        const NCNNOptions& Options,
		bool _AddCache = false,
		DLogger _Logger = _D_Dragonian_Lib_NCNN_Space GetDefaultLogger()
	);

protected:
	std::shared_ptr<ncnn::Net> m_NCNNNet;
    NCNNOptions m_Options;
	DLogger m_Logger;

public:
    static void UnloadCachedModel(
        const std::wstring& _Path,
        const NCNNOptions& Options
    );
};

::ncnn::Layer* InstanceNorm1D_layer_creator(void*);
void InstanceNorm1D_layer_destroyer(::ncnn::Layer* layer, void*);

_D_Dragonian_NCNN_Lib_Space_End