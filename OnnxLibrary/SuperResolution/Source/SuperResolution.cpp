#include "OnnxLibrary/SuperResolution/SuperResolution.hpp"

_D_Dragonian_Lib_Lib_Super_Resolution_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		*_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger(),
		L"SuperResolution"
	);
	return _MyLogger;
}

_D_Dragonian_Lib_Lib_Super_Resolution_End