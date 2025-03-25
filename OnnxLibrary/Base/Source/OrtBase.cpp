#include "../OrtBase.hpp"

_D_Dragonian_Lib_Onnx_Runtime_Header

OnnxRuntimeEnvironment CreateEnvironment(
	const OnnxEnvironmentOptions& Options
)
{
	_D_Dragonian_Lib_Rethrow_Block(return CreateOnnxRuntimeEnvironment(
		Options
	););
}

_D_Dragonian_Lib_Onnx_Runtime_End