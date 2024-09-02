#include "SuperResolution.hpp"
#include "Base.h"
#include "Util/StringPreprocess.h"

namespace DragonianLib
{
	namespace LibSuperResolution
	{
		DragonianLib::Image& SuperResolution::Infer(DragonianLib::Image& _Image, int64_t _BatchSize) const
		{
			DragonianLibNotImplementedError;
		}

		SuperResolution::SuperResolution(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider, ProgressCallback _Callback)
			: Env_(_ThreadCount, _DeviceID, _Provider), Callback_(std::move(_Callback))
		{

		}
	}
}