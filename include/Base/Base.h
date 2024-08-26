#pragma once
#include <cstdint>
#include <filesystem>

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

#define DragonianLibSpaceBegin namespace DragonianLib {
#define DragonianLibSpaceEnd }
#define DragonianLibNDIS [[nodiscard]]

#define DragonianLibThrowImpl(message, exception_type) do {\
	const std::string __DragonianLib__Message__ = message;\
	const std::string __DragonianLib__Message__Prefix__ =\
	std::string("[In File: \"") + std::filesystem::path(__FILE__).filename().string() + "\", " +\
	"Function: \"" + __FUNCSIG__ + "\", " +\
	"Line: " + std::to_string(__LINE__) + " ] ";\
	if (__DragonianLib__Message__.substr(0, __DragonianLib__Message__Prefix__.length()) != __DragonianLib__Message__Prefix__)\
		throw exception_type((__DragonianLib__Message__Prefix__ + __DragonianLib__Message__).c_str());\
	throw exception_type(__DragonianLib__Message__.c_str());\
} while(0)

#define DragonianLibThrow(message) DragonianLibThrowImpl(message, std::exception)

#define DragonianLibNotImplementedError DragonianLibThrow("NotImplementedError!")

#define DragonianLibRegLayer(ModuleName, MemberName, ...) ModuleName MemberName{this, #MemberName, __VA_ARGS__}

#define DragonianLibLogMessage(message) DragonianLib::GetLogger().log(message)

#define DragonianLibErrorMessage(message) DragonianLib::GetLogger().error(message)

#define GetGGMLUnusedMemorySize(ctx) (ggml_get_mem_size(ctx) - ggml_used_mem(ctx)) 

DragonianLibSpaceBegin

struct float16_t
{
	unsigned char Val[2];
	float16_t(float _Val)
	{
		auto Ptr = (unsigned short*)Val;
		uint32_t inu = *((uint32_t*)&_Val);

		uint32_t t1 = inu & 0x7fffffffu;                 // Non-sign bits
		uint32_t t2 = inu & 0x80000000u;                 // Sign bit
		uint32_t t3 = inu & 0x7f800000u;                 // Exponent

		t1 >>= 13u;                             // Align mantissa on MSB
		t2 >>= 16u;                             // Shift sign bit into position

		t1 -= 0x1c000;                         // Adjust bias

		t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
		t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
		t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

		t1 |= t2;                              // Re-insert sign bit

		*(Ptr) = t1;
	}
	float16_t& operator=(float _Val)
	{
		auto Ptr = (unsigned short*)Val;
		uint32_t inu = *((uint32_t*)&_Val);

		uint32_t t1 = inu & 0x7fffffffu;                 // Non-sign bits
		uint32_t t2 = inu & 0x80000000u;                 // Sign bit
		uint32_t t3 = inu & 0x7f800000u;                 // Exponent

		t1 >>= 13u;                             // Align mantissa on MSB
		t2 >>= 16u;                             // Shift sign bit into position

		t1 -= 0x1c000;                         // Adjust bias

		t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
		t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
		t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

		t1 |= t2;                              // Re-insert sign bit

		*(Ptr) = t1;
		return *this;
	}
	operator float() const
	{
		float out;
		auto in = *(const uint16_t*)Val;

		uint32_t t1 = in & 0x7fffu;                       // Non-sign bits
		uint32_t t2 = in & 0x8000u;                       // Sign bit
		uint32_t t3 = in & 0x7c00u;                       // Exponent

		t1 <<= 13u;                              // Align mantissa on MSB
		t2 <<= 16u;                              // Shift sign bit into position

		t1 += 0x38000000;                       // Adjust bias

		t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

		t1 |= t2;                               // Re-insert sign bit

		*((uint32_t*)&out) = t1;
		return out;
	}
};

struct float8_t
{
	unsigned char Val;
	float8_t(float _Val)
	{
		Val = 0ui8;
		DragonianLibNotImplementedError;
	}
	float8_t& operator=(float _Val)
	{
		Val = 0ui8;
		DragonianLibNotImplementedError;
	}
	operator float() const
	{
		DragonianLibNotImplementedError;
	}
};

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using float8 = float8_t;
using float16 = float16_t;
using float32 = float;
using float64 = double;
using byte = unsigned char;
using lpvoid = void*;
using cpvoid = const void*;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
struct NoneType {};
static constexpr NoneType None;

using DictType = int;

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
struct WeightHeader
{
	int64 Shape[8] = { 0,0,0,0,0,0,0,0 };
	char LayerName[DRAGONIANLIB_NAME_MAX_SIZE];
	char Type[16];
};
#ifdef _MSC_VER
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

std::wstring GetCurrentFolder();

void SetGlobalEnvDir(const std::wstring& _Folder);

class FileGuard
{
public:
	FileGuard() = default;
	~FileGuard();
	FileGuard(const std::wstring& _Path, const std::wstring& _Mode);
	FileGuard(const FileGuard& _Left) = delete;
	FileGuard& operator=(const FileGuard& _Left) = delete;
	FileGuard(FileGuard&& _Right) noexcept;
	FileGuard& operator=(FileGuard&& _Right) noexcept;
	void Open(const std::wstring& _Path, const std::wstring& _Mode);
	void Close();
	operator FILE* () const;
	DragonianLibNDIS bool Enabled() const;
private:
	FILE* file_ = nullptr;
};

DragonianLibSpaceEnd