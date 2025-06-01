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
#include <functional>
#include <mutex>

#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Space_Begin

enum class FloatPrecision : UInt8
{
	BFloat16,
	Float16,
	Float32
};

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

class SharedMutex
{
public:
	SharedMutex() : _MyMutex(std::make_shared<std::mutex>()) {}

	void lock() const;
	void Lock() const { lock(); }
	bool try_lock() const noexcept;
	bool TryLock() const noexcept { return try_lock(); }
	void unlock() const noexcept;
	void Unlock() const noexcept { unlock(); }

	std::mutex* get() const noexcept { return _MyMutex.get(); }
	std::mutex* Get() const noexcept { return _MyMutex.get(); }
	std::mutex* operator->() const noexcept { return _MyMutex.get(); }
private:
	std::shared_ptr<std::mutex> _MyMutex;
};

template <typename _Type>
class Atomic
{
public:
	Atomic() = delete;


protected:
	std::shared_ptr<_Type> _MyVal;
	SharedMutex _MyMutex;
};

class ByteBuffer
{
public:
	ByteBuffer() = default;
	ByteBuffer(StdSize _Size) : _MyBuffer(new Byte[_Size]), _MySize(_Size) {}
	~ByteBuffer() = default;
	ByteBuffer(const ByteBuffer&) = delete;
	ByteBuffer(ByteBuffer&& _Right) noexcept = default;
	ByteBuffer& operator=(const ByteBuffer&) = delete;
	ByteBuffer& operator=(ByteBuffer&& _Right) noexcept = default;

	void Reallocate(StdSize _Size)
	{
		if (_MyBuffer)
		{
			auto OldBuffer = std::move(_MyBuffer);
			_MyBuffer = std::make_unique<Byte[]>(_Size);
			const auto Overlap = std::min(_Size, _MySize);
			memcpy(_MyBuffer.get(), OldBuffer.get(), Overlap);
			_MySize = _Size;
		}
		else
			_MyBuffer = std::make_unique<Byte[]>(_Size);
	}

	const Byte* CBegin() const noexcept { return _MyBuffer.get(); }
	const Byte* CEnd() const noexcept { return _MyBuffer.get() + _MySize; }
	Byte* Begin() const noexcept { return _MyBuffer.get(); }
	Byte* End() const noexcept { return _MyBuffer.get() + _MySize; }
	Byte* Data() const noexcept { return _MyBuffer.get(); }
	Byte* Get() const noexcept { return _MyBuffer.get(); }

	const Byte* cbegin() const noexcept { return _MyBuffer.get(); }
	const Byte* cend() const noexcept { return _MyBuffer.get() + _MySize; }
	Byte* begin() const noexcept { return _MyBuffer.get(); }
	Byte* end() const noexcept { return _MyBuffer.get() + _MySize; }
	Byte* data() const noexcept { return _MyBuffer.get(); }
	Byte* get() const noexcept { return _MyBuffer.get(); }

	StdSize Size() const noexcept { return _MySize; }

	Byte* Release() noexcept { _MySize = 0; return _MyBuffer.release(); }
	
	Byte& operator[](StdSize _Index)
	{
#ifdef _DEBUG
		if (_Index >= _MySize)
			_D_Dragonian_Lib_Throw_Exception("Index out of range");
#endif
		return _MyBuffer[_Index];
	}
	const Byte& operator[](StdSize _Index) const
	{
#ifdef _DEBUG
		if (_Index >= _MySize)
			_D_Dragonian_Lib_Throw_Exception("Index out of range");
#endif
		return _MyBuffer[_Index];
	}

	operator bool() const noexcept { return _MyBuffer != nullptr; }

private:
	std::unique_ptr<Byte[]> _MyBuffer = nullptr;
	StdSize _MySize = 0;
};

template <typename _StreamType, typename _ValTy>
bool ReadInstanceFromStream(_StreamType&& _Stream, _ValTy& _Val)
	requires(TypeTraits::IsArithmeticValue<_ValTy>)
{
	if constexpr (TypeTraits::IsFloatingPointValue<_ValTy>)
		_Val = _ValTy(std::stod(std::forward<_StreamType>(_Stream).FindNextNumber()));
	else if constexpr (TypeTraits::IsIntegerValue<_ValTy>)
		_Val = _ValTy(std::stoll(std::forward<_StreamType>(_Stream).FindNextNumber()));
	else if constexpr (TypeTraits::IsComplexValue<_ValTy>)
		_Val = _ValTy(std::stod(std::forward<_StreamType>(_Stream).FindNextNumber()),
			std::stod(std::forward<_StreamType>(_Stream).FindNextNumber()));
	return true;
}

class ByteStream
{
public:
	ByteStream() = default;
	virtual ~ByteStream() = default;
	ByteStream(const ByteStream&) = default;
	ByteStream(ByteStream&&) = default;
	ByteStream& operator=(const ByteStream&) = default;
	ByteStream& operator=(ByteStream&&) = default;

	virtual void Open() = 0;
	virtual void Close() = 0;

	virtual bool Enabled() const = 0;
	virtual bool IsFile() const = 0;
	virtual bool IsByteBuffer() const = 0;
	
	virtual Int Seek(Int64 _Offset, Int _Origin) const = 0;
	virtual StdSize Tell() const = 0;

	virtual StdSize Read(void* _Buffer, StdSize _BufferSize, StdSize _ElementSize, StdSize _Count = 1) const = 0;
	virtual StdSize Write(const void* _Buffer, StdSize _ElementSize, StdSize _Count = 1) = 0;

	virtual void* GetHandle() = 0;
	virtual const void* GetHandle() const = 0;
	virtual void* ReleaseHandle() = 0;

	virtual std::string ReadLine() const = 0;
	virtual std::wstring ReadLineW() const = 0;
	virtual bool WriteString(const char* _Str) = 0;
	virtual bool WriteString(const wchar_t* _Str) = 0;

	bool WriteString(const std::string& _Str);
	bool WriteString(const std::wstring& _Str);
	bool WriteStr(const std::string& _Str);

	template <typename _ThisType>
	decltype(auto) operator<<(this _ThisType&& _Self, const char* _Str)
	{
		if (std::forward<_ThisType>(_Self).WriteString(_Str))
			return std::forward<_ThisType>(_Self);
		_D_Dragonian_Lib_Throw_Exception("Error when write string to file");
	}

	template <typename _ThisType>
	decltype(auto) operator<<(this _ThisType&& _Self, const wchar_t* _Str)
	{
		if (std::forward<_ThisType>(_Self).WriteString(_Str))
			return std::forward<_ThisType>(_Self);
		_D_Dragonian_Lib_Throw_Exception("Error when write string to file");
	}

	template <typename _ThisType>
	decltype(auto) operator<<(this _ThisType&& _Self, const std::string& _Str)
	{
		if (std::forward<_ThisType>(_Self).WriteString(_Str))
			return std::forward<_ThisType>(_Self);
		_D_Dragonian_Lib_Throw_Exception("Error when write string to file");
	}

	template <typename _ThisType>
	decltype(auto) operator<<(this _ThisType&& _Self, const std::wstring& _Str)
	{
		if (std::forward<_ThisType>(_Self).WriteString(_Str))
			return std::forward<_ThisType>(_Self);
		_D_Dragonian_Lib_Throw_Exception("Error when write string to file");
	}

	template <typename _ThisType>
	decltype(auto) operator>>(this _ThisType&& _Self, std::string& _Str)
	{
		_Str = std::forward<_ThisType>(_Self).ReadLine();
		return std::forward<_ThisType>(_Self);
	}

	template <typename _ThisType>
	decltype(auto) operator>>(this _ThisType&& _Self, std::wstring& _Str)
	{
		_Str = std::forward<_ThisType>(_Self).ReadLineW();
		return std::forward<_ThisType>(_Self);
	}

	template <typename _ThisType, typename _Type>
	decltype(auto) operator<<(this _ThisType&& _Self, const _Type& _Value)
		requires (!TypeTraits::IsStringValue<_Type>)
	{
		if constexpr (std::is_trivially_copy_assignable_v<_Type>)
		{
			if (_Self._IsByteMode)
			{
				if (_Self.Write(&_Value, sizeof(_Type), 1) != 1)
					_D_Dragonian_Lib_Throw_Exception("Failed to write data to stream");
				return std::forward<_ThisType>(_Self);
			}
		}
		if (!_Self.WriteStr(CvtToString(_Value)))
			_D_Dragonian_Lib_Throw_Exception("Failed to write data to stream");
		return std::forward<_ThisType>(_Self);
	}

	template <typename _ThisType, typename _Type>
	decltype(auto) operator>>(this _ThisType&& _Self, _Type& _Value)
	{
		if constexpr (std::is_trivially_copy_assignable_v<_Type>)
		{
			if (std::forward<_ThisType>(_Self)._IsByteMode)
			{
				if (std::forward<_ThisType>(_Self).Read(&_Value, sizeof(_Type), sizeof(_Type), 1) != 1)
					_D_Dragonian_Lib_Throw_Exception("Failed to read data from stream");
				return std::forward<_ThisType>(_Self);
			}
		}
		if (!ReadInstanceFromStream(std::forward<_ThisType>(_Self), _Value))
			_D_Dragonian_Lib_Throw_Exception("Failed to read data from stream");
		return std::forward<_ThisType>(_Self);
	}

	std::string FindNextNumber() const;

protected:
	bool _IsByteMode = false;
};

class FileStream final : public ByteStream
{
public:
	FileStream() = default;

	explicit FileStream(const std::wstring& _Path, const std::wstring& _Mode);
	explicit FileStream(const std::string& _Path, const std::string& _Mode);

	explicit FileStream(const std::wstring& _Path, const wchar_t* _Mode);
	explicit FileStream(const std::string& _Path, const char* _Mode);

	explicit FileStream(const wchar_t* _Path, const std::wstring& _Mode);
	explicit FileStream(const char* _Path, const std::string& _Mode);

	explicit FileStream(const wchar_t* _Path, const wchar_t* _Mode);
	explicit FileStream(const char* _Path, const char* _Mode);

	void Open(const std::wstring& _Path, const std::wstring& _Mode);
	void Open(const std::wstring& _Path, const wchar_t* _Mode);
	void Open(const wchar_t* _Path, const wchar_t* _Mode);

	void Open() override;
	void Close() override;

	bool Enabled() const override { return _MyFile.get(); }
	bool IsFile() const override { return true; }
	bool IsByteBuffer() const override { return false; }

	Int Seek(Int64 _Offset, Int _Origin) const override;
	StdSize Tell() const override;

	StdSize Read(void* _Buffer, StdSize _BufferSize, StdSize _ElementSize, StdSize _Count = 1) const override;
	StdSize Write(const void* _Buffer, StdSize _ElementSize, StdSize _Count = 1) override;

	void* GetHandle() override;
	const void* GetHandle() const override;
	void* ReleaseHandle() override;

	std::string ReadLine() const override;
	std::wstring ReadLineW() const override;
	bool WriteString(const char* _Str) override;
	bool WriteString(const wchar_t* _Str) override;

	operator FILE* () const { return _MyFile.get(); }
	FILE* Release() const { return _MyFile.get(); }

private:
	void ReOpen(const wchar_t* _Path, const wchar_t* _Mode);
	std::shared_ptr<FILE> _MyFile = nullptr;
	SharedMutex _MyMutex;
};
using FileGuard = FileStream;

class ByteIO final : public ByteStream
{
public:
	ByteIO(bool _Byte = true) { _MyCur = _MyLast = _MyBuffer.Get(); _IsByteMode = _Byte; }

	StdSize RemainderCapacity() const;
	StdSize RemainderSize() const;
	StdSize Remainder() const;
	StdSize Size() const;
	StdSize Capacity() const;

	Byte* End() const { return _MyLast; }
	Byte* Begin() const { return _MyCur; }
	Byte* end() const { return _MyLast; }
	Byte* begin() const { return _MyCur; }
	const Byte* CEnd() const { return _MyLast; }
	const Byte* CBegin() const { return _MyCur; }
	const Byte* cend() const { return _MyLast; }
	const Byte* cbegin() const { return _MyCur; }
	Byte* Data() const { return _MyBuffer.Data(); }
	Byte* data() const { return _MyBuffer.Data(); }
	Byte* Get() const { return _MyBuffer.Get(); }
	Byte* get() const { return _MyBuffer.Get(); }

	void Open() override;
	void Close() override;

	bool Enabled() const override { return _IsOpen; }
	bool IsFile() const override { return true; }
	bool IsByteBuffer() const override { return false; }

	Int Seek(Int64 _Offset, Int _Origin) const override;
	StdSize Tell() const override;

	StdSize Read(void* _Buffer, StdSize _BufferSize, StdSize _ElementSize, StdSize _Count = 1) const override;
	StdSize Write(const void* _Buffer, StdSize _ElementSize, StdSize _Count = 1) override;

	void* GetHandle() override;
	const void* GetHandle() const override;
	void* ReleaseHandle() override;

	std::string ReadLine() const override;
	std::wstring ReadLineW() const override;
	bool WriteString(const char* _Str) override;
	bool WriteString(const wchar_t* _Str) override;

private:
	bool _IsOpen = true;
	SharedMutex _MyMutex;
	ByteBuffer _MyBuffer{ 65535 };
	Byte* _MyLast = nullptr;
	mutable Byte* _MyCur = nullptr;
};

class UniqueScopeExit
{
public:
	friend class SharedScopeExit;

	UniqueScopeExit() = default;

	template <typename _FunTy, typename... _Args>
	UniqueScopeExit(
		_FunTy&& _Function,
		_Args&&... _Arguments
	) requires (std::is_invocable_v<_FunTy, _Args...>) :
		_MyFun(
			std::bind(
				std::forward<_FunTy>(_Function),
				std::forward<_Args>(_Arguments)...
			)
		)
	{

	}

	~UniqueScopeExit() { if (_MyFun) _MyFun(); }
	UniqueScopeExit(UniqueScopeExit&&) noexcept = default;
	UniqueScopeExit(const UniqueScopeExit&) = delete;
	UniqueScopeExit& operator=(UniqueScopeExit&&) noexcept = default;
	UniqueScopeExit& operator=(const UniqueScopeExit&) = delete;

	void Execute()
	{
		if (_MyFun) _MyFun();
		Detach();
	}

	void Detach()
	{
		_MyFun = nullptr;
	}

private:
	std::function<void()> _MyFun;
};

class SharedScopeExit
{
public:
	SharedScopeExit() = default;
	template <typename _FunTy, typename... _Args>
	SharedScopeExit(
		_FunTy&& _Function,
		_Args&&... _Arguments
	) requires (std::is_invocable_v<_FunTy, _Args...>) :
		_MyFun(
			std::make_shared<UniqueScopeExit>(
				std::forward<_FunTy>(_Function),
				std::forward<_Args>(_Arguments)...
			)
		)
	{

	}

	SharedScopeExit(
		UniqueScopeExit&& _Right
	) : _MyFun(std::make_shared<UniqueScopeExit>(std::move(_Right)))
	{

	}

	void Execute() const
	{
		_MyFun->Execute();
	}

	void Reset()
	{
		_MyFun = nullptr;
	}

	void Detach() const
	{
		_MyFun->Detach();
	}

private:
	std::shared_ptr<UniqueScopeExit> _MyFun;
};

class OnConstruct
{
public:
	template <typename _FunTy, typename... _Args>
	OnConstruct(
		_FunTy&& _Function,
		_Args&&... _Arguments
	) requires (std::is_invocable_v<_FunTy, _Args...>)
	{
		std::forward<_FunTy>(_Function)(std::forward<_Args>(_Arguments)...);
	}
};

using OnDeConstruct = UniqueScopeExit;

template <typename StreamType>
class DefaultProgressCallback
{
public:
	DefaultProgressCallback(StreamType& _Stream) : _MyStream(&_Stream) {}

	void operator()(bool _Cond, Int64 _Cur) const
	{
		if (_Cond)
		{
			_MyLastTime = std::chrono::high_resolution_clock::now();
			_Total = _Cur;
		}
		else
			ProgressBarFn(_Cur);
	}

	void ProgressBarFn(Int64 _Progress) const
	{
		if (!_MyStream)
			return;

		int BarWidth = 70;
		float progressRatio = static_cast<float>(_Progress) / float(_Total);
		int Pos = static_cast<int>(float(BarWidth) * progressRatio);

		(*_MyStream) << "\r";
		(*_MyStream).flush();
		auto TimeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - _MyLastTime
		).count();
		_MyLastTime = std::chrono::high_resolution_clock::now();
		(*_MyStream) << "[Speed: " << 1000.0f / static_cast<float>(TimeUsed) << " it/s] ";
		(*_MyStream) << "[";
		for (int i = 0; i < BarWidth; ++i) {
			if (i < Pos) (*_MyStream) << "=";
			else if (i == Pos) (*_MyStream) << ">";
			else (*_MyStream) << " ";
		}
		(*_MyStream) << "] " << int(progressRatio * 100.0) << "%  \r";
		_MyLastTime = std::chrono::high_resolution_clock::now();
	}

	operator std::function<void(bool, Int64)>(this DefaultProgressCallback&& Self)
	{
		return [It = std::move(Self)](bool _Cond, Int64 _Cur)
			{
				It(_Cond, _Cur);
			};
	}

	operator std::function<void(bool, Int64)>() const
	{
		return [this](bool _Cond, Int64 _Cur)
			{
				operator()(_Cond, _Cur);
			};
	}
private:
	mutable Int64 _Total = 0;
	mutable decltype(std::chrono::high_resolution_clock::now()) _MyLastTime;
	StreamType* _MyStream = nullptr;
};

template <typename _MyModule>
class ModulePointer : public std::shared_ptr<_MyModule>
{
public:
	using _MyBase = std::shared_ptr<_MyModule>;
	template <typename... _ArgumentTypes>
	_D_Dragonian_Lib_Constexpr_Force_Inline ModulePointer(_ArgumentTypes&&... _Arguments)
		requires(TypeTraits::ConstructibleFrom<_MyBase, _ArgumentTypes...>)
	: _MyBase(std::forward<_ArgumentTypes>(_Arguments)...)
	{

	}

	template <typename... _ArgumentTypes>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(_ArgumentTypes&&... _Arguments)
		requires(requires(_ArgumentTypes&&... _Args) { this->get()->operator()(std::forward<_ArgumentTypes>(_Args)...); })
	{
		return this->get()->operator()(std::forward<_ArgumentTypes>(_Arguments)...);
	}
};

_D_Dragonian_Lib_Space_End

namespace Dlib
{
	using namespace ::_D_Dragonian_Lib;
}
