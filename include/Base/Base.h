#pragma once
#include <cstdint>
#include <filesystem>
#include "ggml-alloc.h"
#include "Util/Logger.h"

#if LIBSVC_ALLOC_ALIG > 1
#if _MSC_VER
#define LIBSVC_MALLOC(size) _aligned_malloc(size, LIBSVC_ALLOC_ALIG)
#define LIBSVC_FREE(ptr) _aligned_free(ptr)
#else
#error
#endif
#else
#define LIBSVC_MALLOC(size) malloc(size)
#define LIBSVC_FREE(size) free(size)
#endif

#ifndef UNUSED
#define UNUSED(x) void(x)
#endif

#define LibSvcBegin namespace libsvc {
#define LibSvcEnd }
#define LIBSVCND [[nodiscard]]

#define LibSvcThrowImpl(message, exception_type) {\
	const std::string __LibSvc__Message__ = message;\
	const std::string __LibSvc__Message__Prefix__ =\
	std::string("[In File: \"") + std::filesystem::path(__FILE__).filename().string() + "\", " +\
	"Function: \"" + __FUNCSIG__ + "\", " +\
	"Line: " + std::to_string(__LINE__) + " ] ";\
	if (__LibSvc__Message__.substr(0, __LibSvc__Message__Prefix__.length()) != __LibSvc__Message__Prefix__)\
		throw exception_type((__LibSvc__Message__Prefix__ + __LibSvc__Message__).c_str());\
	throw exception_type(__LibSvc__Message__.c_str());\
}

#define LibSvcThrow(message) LibSvcThrowImpl(message, std::exception)

#define LibSvcNotImplementedError LibSvcThrow("NotImplementedError!")

#define RegLayer(ModuleName, MemberName, ...) ModuleName MemberName{this, #MemberName, __VA_ARGS__}

#define LogMessage(message) libsvc::GetLogger().log(message)

#define ErrorMessage(message) libsvc::GetLogger().error(message)

#define GetGGMLUnusedMemorySize(ctx) (ggml_get_mem_size(ctx) - ggml_used_mem(ctx)) 

LibSvcBegin

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using float32 = float;
using float64 = double;
using byte = unsigned char;
using lpvoid = void*;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

#ifdef _WIN32
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
struct WeightHeader
{
	int64 Shape[8] = { 0,0,0,0,0,0,0,0 };
	char LayerName[GGML_MAX_NAME] = { 0 };
	char Type[16] = { 0 };
};
#ifdef _WIN32
#pragma pack(pop)
#else
#pragma pack()
#endif

struct WeightData
{
	WeightHeader Header_;
	std::vector<byte> Data_;
	std::vector<int64> Shape_;
	std::string Type_, LayerName_;
};

class FileGuard
{
public:
	FileGuard() = default;
	~FileGuard();
	FileGuard(const FileGuard& _Left) = delete;
	FileGuard& operator=(const FileGuard& _Left) = delete;
	FileGuard(FileGuard&& _Right) noexcept;
	FileGuard& operator=(FileGuard&& _Right) noexcept;
	void Open(const std::wstring& _Path, const std::wstring& _Mode);
	void Close();
	operator FILE* () const;
	LIBSVCND bool Enabled() const;
private:
	FILE* file_ = nullptr;
};

LibSvcEnd