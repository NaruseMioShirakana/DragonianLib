#pragma once
#ifdef DRAGONIANLIB_IMPORT
#define _Dragonian_Lib_Svc_Api __declspec(dllimport)
#else
#ifdef DRAGONIANLIB_EXPORT
#define _Dragonian_Lib_Svc_Api __declspec(dllexport)
#else
#define _Dragonian_Lib_Svc_Api
#endif
#endif