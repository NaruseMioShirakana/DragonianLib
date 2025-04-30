#include "Libraries/Util/Util.h"

#ifdef _WIN32
#include <Windows.h>
#endif

_D_Dragonian_Lib_Space_Begin

HResult::operator bool() const
{
#ifdef _WIN32
	return SUCCEEDED(Value);
#else
	return Value == 0;
#endif
}

uint32_t Cvt2tagCOINIT(ComInitializeFlag _Flag)
{
#ifdef _WIN32
	switch (_Flag)
	{
	case ComInitializeFlag::COINIT_APARTMENTTHREADED:
		return COINIT_APARTMENTTHREADED;
	case ComInitializeFlag::COINIT_MULTITHREADED:
		return COINIT_MULTITHREADED;
	case ComInitializeFlag::COINIT_DISABLE_OLE1DDE:
		return COINIT_DISABLE_OLE1DDE;
	case ComInitializeFlag::COINIT_SPEED_OVER_MEMORY:
		return COINIT_SPEED_OVER_MEMORY;
	}
	return COINIT_MULTITHREADED;
#else
	return 0;
#endif
}

HResult ComInitialize(uint32_t _Flag)
{
#ifdef _WIN32
	return { static_cast<int64_t>(CoInitializeEx(nullptr, _Flag)) };
#else
	return { 0ll };
#endif
}

void ComUninitialize()
{
#ifdef _WIN32
	CoUninitialize();
#endif
}

_D_Dragonian_Lib_Space_End