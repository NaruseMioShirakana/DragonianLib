﻿#include "SuperResolution.hpp"
#include "Libraries/Base.h"
#include "Libraries/Util/StringPreprocess.h"

namespace DragonianLib
{
	namespace LibSuperResolution
	{
		ImageVideo::Image& SuperResolution::Infer(ImageVideo::Image& _Image, int64_t _BatchSize) const
		{
			_D_Dragonian_Lib_Not_Implemented_Error;
		}

		SuperResolution::SuperResolution(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider, ProgressCallback _Callback)
			: Env_(DragonianLibOrtEnv::CreateEnv(_ThreadCount, _DeviceID, _Provider)), Callback_(std::move(_Callback))
		{

		}
	}
}