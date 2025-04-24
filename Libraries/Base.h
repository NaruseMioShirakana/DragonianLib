/**
 * @file Base.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Base of DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <unordered_map>
#include <functional>
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
	FileGuard(FILE* _FileStream) : file_(_FileStream) {}
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

	FILE* Release();
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

enum class FloatPrecision : UInt8
{
	BFloat16,
	Float16,
	Float32
};

class IOStream
{
public:
	IOStream() = delete;
	~IOStream();
	IOStream(const IOStream&) = delete;
	IOStream(IOStream&& _Right) noexcept;
	IOStream& operator=(const IOStream&) = delete;
	IOStream& operator=(IOStream&& _Right) noexcept;
	void _Tidy();

private:
	FILE* _MyFile = nullptr;
	Byte* _MyBuffer = nullptr;
	Byte* _MyBufferEnd = nullptr;
	Byte* _MyIter = nullptr;

public:
	IOStream(const std::wstring& _Path, const std::wstring& _Mode);
	IOStream(FILE* _FileStream);
	IOStream(size_t _BufferSize);

	bool IsFile() const noexcept { return _MyFile; }
	bool IsBuffer() const noexcept { return _MyBuffer; }

	FILE* ReleaseFile() noexcept;
	Byte* ReleaseBuffer() noexcept;

	FILE* GetFile() const noexcept { return _MyFile; }
	Byte* Data() const noexcept { return _MyBuffer; }

	size_t Size() const noexcept { return _MyIter - _MyBuffer; }
	size_t Capacity() const noexcept { return _MyBufferEnd - _MyBuffer; }

	Byte* Begin() const noexcept { return _MyBuffer; }
	Byte* End() const noexcept { return _MyIter; }
	
	Byte* begin() const noexcept { return _MyBuffer; }
	Byte* end() const noexcept { return _MyIter; }

	bool Enabled() const noexcept { return IsFile() xor IsBuffer(); }

	void Seek(long _Offset, int _Origin) noexcept;
	size_t Tell() const noexcept;

	void Reserve(size_t _Size);
};

template <typename _FunTy>
class SharedScopeExit
{
public:
	static_assert(std::is_invocable_v<_FunTy>, "_FunTy is not invocable!");

	SharedScopeExit() = delete;
	SharedScopeExit(_FunTy _Fn) : _MyFun(std::move(_Fn)) {}
	~SharedScopeExit() { if (_MyFun) _MyFun(); }
	SharedScopeExit(SharedScopeExit&&) noexcept = default;
	SharedScopeExit(const SharedScopeExit&) = default;
	SharedScopeExit& operator=(SharedScopeExit&&) noexcept = default;
	SharedScopeExit& operator=(const SharedScopeExit&) = default;

private:
	_FunTy _MyFun;
};

template <typename _FunTy>
class OnStartUP
{
public:
	static_assert(std::is_invocable_v<_FunTy>, "_FunTy is not invocable!");

	OnStartUP() = delete;
	OnStartUP(_FunTy _Fn) { if (_Fn) _Fn(); }
};

_D_Dragonian_Lib_Space_End
