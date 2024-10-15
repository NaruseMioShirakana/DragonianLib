/**
 * FileName: Base.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <cstdint>
#include <filesystem>
#include <unordered_map>

// Define UNUSED macro for unused variables
#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

// Define namespace macros
#define DragonianLibSpaceBegin namespace DragonianLib {
#define DragonianLibSpaceEnd }
#define DragonianLibNDIS [[nodiscard]]

// Define exception throwing macro
#define DragonianLibThrowImpl(message, exception_type) do {\
	const std::string __DragonianLib__Message__ = message;\
	const std::string __DragonianLib__Message__Prefix__ =\
	std::string("[In File: \"") + std::filesystem::path(__FILE__).filename().string() + "\", " +\
	"Function: \"" + __FUNCSIG__ + "\", " +\
	"Line: " + std::to_string(__LINE__) + " ]; \n";\
	if (__DragonianLib__Message__.substr(0, __DragonianLib__Message__Prefix__.length()) != __DragonianLib__Message__Prefix__)\
		throw exception_type((__DragonianLib__Message__Prefix__ + __DragonianLib__Message__).c_str());\
	throw exception_type(__DragonianLib__Message__.c_str());\
} while(0)

// Define general exception throwing macro
#define DragonianLibThrow(message) DragonianLibThrowImpl(message, std::exception)

// Define not implemented error macro
#define DragonianLibNotImplementedError DragonianLibThrow("NotImplementedError!")

// Define fatal error macro
#define DragonianLibFatalError DragonianLibThrow("FatalError!")

// Define registration layer macro
#define DragonianLibRegLayer(ModuleName, MemberName, ...) ModuleName MemberName{this, #MemberName, __VA_ARGS__}

// Define log message macro
#define DragonianLibLogMessage(message) DragonianLib::GetLogger().log(message)

// Define error message macro
#define DragonianLibErrorMessage(message) DragonianLib::GetLogger().error(message)

// Define macro to get unused memory size
#define GetGGMLUnusedMemorySize(ctx) (ggml_get_mem_size(ctx) - ggml_used_mem(ctx)) 

DragonianLibSpaceBegin

// Define float16_t struct
struct float16_t {
	float16_t(float _Val);
	float16_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val[2];
	static uint16_t float32_to_float16(uint32_t f32);
	static uint32_t float16_to_float32(uint16_t f16);
};

// Define float8_t struct
struct float8_t
{
	float8_t(float _Val);
	float8_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val;
};

// Define bfloat16_t struct
struct bfloat16_t
{
	bfloat16_t(float _Val);
	bfloat16_t& operator=(float _Val);
	operator float() const;
private:
	unsigned char Val[2];
};

// Type aliases
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using float8 = float8_t;
using bfloat16 = bfloat16_t;
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

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
// Define WeightHeader struct
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

// Define WeightData struct
struct WeightData
{
	WeightHeader Header_;
	std::vector<byte> Data_;
	std::vector<int64> Shape_;
	std::string Type_, LayerName_;
};

// Type alias for dictionary
using DictType = std::unordered_map<std::string, WeightData>;

/**
 * @brief Get global enviroment folder
 * @return global enviroment folder
 */
std::wstring GetCurrentFolder();

/**
 * @brief Set global enviroment folder
 * @param _Folder Folder to set
 */
void SetGlobalEnvDir(const std::wstring& _Folder);

/**
 * @class FileGuard
 * @brief RAII File
 */
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

	/**
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const std::wstring& _Path, const std::wstring& _Mode);

	/**
	 * @brief Close file
	 */
	void Close();

	/**
	 * @brief Get file pointer
	 * @return file pointer
	 */
	operator FILE* () const;

	/**
	 * @brief Check if file is enabled
	 * @return true if file is enabled
	 */
	DragonianLibNDIS bool Enabled() const;

private:
	FILE* file_ = nullptr;
};
DragonianLibSpaceEnd
