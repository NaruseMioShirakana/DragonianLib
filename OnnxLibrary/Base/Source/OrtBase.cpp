#include "../OrtBase.hpp"

_D_Dragonian_Lib_Onnx_Runtime_Header

OnnxRuntimeEnviroment CreateEnvironment(
	ExecutionProviders _Provider,
	UInt64 _DeviceId,
	UInt64 _IntraOpNumThreads
)
{
	_D_Dragonian_Lib_Rethrow_Block(return CreateOnnxRuntimeEnviroment(
		static_cast<unsigned>(_IntraOpNumThreads),
		static_cast<unsigned>(_DeviceId),
		static_cast<unsigned>(_Provider)
	););
}

Ort::AllocatorWithDefaultOptions& GetDefaultOrtAllocator()
{
	static Ort::AllocatorWithDefaultOptions Allocator;
	return Allocator;
}

_D_Dragonian_Lib_Onnx_Runtime_End