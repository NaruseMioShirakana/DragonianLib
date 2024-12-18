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
#include <unordered_map>
#include "Util/TypeTraits.h"

_D_Dragonian_Lib_Space_Begin

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
	FileGuard(const std::wstring& _Path, const wchar_t* _Mode);
	FileGuard(const wchar_t* _Path, const wchar_t* _Mode);
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
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const std::wstring& _Path, const wchar_t* _Mode);

	/**
	 * @brief Open file
	 * @param _Path file path
	 * @param _Mode file mode
	 */
	void Open(const wchar_t* _Path, const wchar_t* _Mode);

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
	_D_Dragonian_Lib_No_Discard bool Enabled() const;

	void Seek(long _Offset, int _Origin) const;
	size_t Tell() const;

	size_t Read(void* _Buffer, size_t _BufferSize, size_t _ElementSize, size_t _Count = 1) const;
	size_t Write(const void* _Buffer, size_t _ElementSize, size_t _Count = 1) const;

	FileGuard& operator<<(const std::string& _Str);
	FileGuard& operator<<(const std::wstring& _Str);
	FileGuard& operator<<(const char* _Str);
	FileGuard& operator<<(const wchar_t* _Str);
	FileGuard& operator<<(char _Ch);
	FileGuard& operator<<(wchar_t _Ch);

private:
	FILE* file_ = nullptr;
};

template <typename _Func>
struct TidyGuard
{
	TidyGuard() = delete;
	TidyGuard(_Func Fn) : Fn_(Fn) {}
	~TidyGuard() { Fn_(); }
private:
	_Func Fn_;
	TidyGuard(const TidyGuard&) = delete;
	TidyGuard& operator=(const TidyGuard&) = delete;
	TidyGuard(TidyGuard&&) = delete;
	TidyGuard& operator=(TidyGuard&&) = delete;
};

enum class FloatPrecision
{
	BFloat16,
	Float16,
	Float32
};

template <typename _Type>
decltype(auto) CvtToString(const _Type& _Value)
{
	if constexpr (TypeTraits::IsComplexValue<_Type>)
		return std::to_string(_Value.real()) + " + " + std::to_string(_Value.imag()) + "i";
	else if constexpr (TypeTraits::IsArithmeticValue<_Type>)
		return std::to_string(_Value);
	else if constexpr (TypeTraits::IsStringValue<_Type>)
		return _Value;
	else if constexpr (TypeTraits::CouldBeConvertedFromValue<std::string, _Type> ||
		TypeTraits::CouldBeConvertedFromValue<const char*, _Type>)
		return std::string(_Value);
	else if constexpr (TypeTraits::CouldBeConvertedFromValue<std::wstring, _Type> ||
		TypeTraits::CouldBeConvertedFromValue<const wchar_t*, _Type>)
		return std::wstring(_Value);
	else
		return _Value.to_string();
}

_D_Dragonian_Lib_Space_End
