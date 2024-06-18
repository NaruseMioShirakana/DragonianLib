#pragma once
#ifdef DRAGONIANLIB_IMPORT
#define LibSvcApi __declspec(dllexport)
#else
#ifdef DRAGONIANLIB_EXPORT
#define LibSvcApi __declspec(dllimport)
#else
#define LibSvcApi
#endif
#endif