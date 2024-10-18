#pragma once
#ifdef DRAGONIANLIB_IMPORT
#define LibSvcApi __declspec(dllimport)
#else
#ifdef DRAGONIANLIB_EXPORT
#define LibSvcApi __declspec(dllexport)
#else
#define LibSvcApi
#endif
#endif