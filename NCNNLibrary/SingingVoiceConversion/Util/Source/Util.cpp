﻿#include "NCNNLibrary/SingingVoiceConversion/Util/Util.hpp"

_D_Dragonian_Lib_NCNN_Singing_Voice_Conversion_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_NCNN_Space GetDefaultLogger()->GetLoggerId() + L"::SingingVoiceConversion",
		_D_Dragonian_Lib_NCNN_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

namespace PreDefinedF0PreprocessMethod
{
	Tensor<Float32, 3, Device::CPU> Log2(
		const Tensor<Float32, 3, Device::CPU>& F0,
		void*
	)
	{
		return F0.Log2();
	}
}

_D_Dragonian_Lib_NCNN_Singing_Voice_Conversion_End