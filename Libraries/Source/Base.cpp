#ifdef _WIN32
#include "Windows.h"
#endif

#include "Libraries/Base.h"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Space_Begin

std::wstring GlobalEnvDir;  // NOLINT(misc-use-internal-linkage)

std::wstring GetCurrentFolder()
{
	if (!GlobalEnvDir.empty())
		return GlobalEnvDir;

	wchar_t path[1024];
#ifdef _WIN32
	GetModuleFileNameW(nullptr, path, 1024);
	std::wstring _curPath = path;
	_curPath = _curPath.substr(0, _curPath.rfind(L'\\'));
	return _curPath;
#else
	if (GlobalEnvDir.empty())
		_D_Dragonian_Lib_Throw_Exception("GlobalEnvDir Is Empty!");
#endif
}

void SetGlobalEnvDir(const std::wstring& _Folder)
{
	GlobalEnvDir = _Folder;
}

void SharedMutex::lock() const
{
	return _MyMutex->lock();
}

bool SharedMutex::try_lock() const noexcept
{
	return _MyMutex->try_lock();
}

void SharedMutex::unlock() const noexcept
{
	_MyMutex->unlock();
}

bool ByteStream::WriteString(const std::string& _Str)
{
	return WriteString(_Str.c_str());
}

bool ByteStream::WriteString(const std::wstring& _Str)
{
	return WriteString(_Str.c_str());
}

bool ByteStream::WriteStr(const std::string& _Str)
{
	return WriteString(_Str);
}

std::string ByteStream::FindNextNumber() const
{
	std::string _Str;
	char Ch = 0;
	bool Real = true;
	while (true)
	{
		if (Read(&Ch, sizeof(char), 1) == 0)
			break;
		if ((Ch >= '0' && Ch <= '9') || (Ch == '-' && _Str.empty()))
			_Str += Ch;  // NOLINT(bugprone-branch-clone)
		else if (Real && Ch == '.')
		{
			Real = false;
			if (_Str.empty() || _Str.back() == '-')
				_Str += '0';
			_Str += '.';
		}
		else
			break;
	}
	if (_Str.back() == '.' || _Str.back() == '-')
		_Str += '0';
	if (_Str.empty())
		_D_Dragonian_Lib_Throw_Exception("Error when get number!");
	return _Str;
}

void FileStream::ReOpen(const wchar_t* _Path, const wchar_t* _Mode)
{
	std::lock_guard lg(_MyMutex);
	FILE* MFile = nullptr;
#if _MSC_VER
	_wfopen_s(&MFile, _Path, _Mode);
#else
	MFile = _wfopen(_Path, _Mode);
#endif
	if (!MFile)
		_D_Dragonian_Lib_Throw_Exception("Failed to open file.");

	_MyFile = std::shared_ptr<FILE>(
		MFile,
		fclose
	);
	const auto Len = wcslen(_Mode);
	if (std::ranges::find(_Mode, _Mode + Len, L'b'))
		_IsByteMode = true;
}

void FileStream::Open(const std::wstring& _Path, const std::wstring& _Mode)
{
	ReOpen(_Path.c_str(), _Mode.c_str());
}

void FileStream::Open(const std::wstring& _Path, const wchar_t* _Mode)
{
	ReOpen(_Path.c_str(), _Mode);
}

void FileStream::Open(const wchar_t* _Path, const wchar_t* _Mode)
{
	ReOpen(_Path, _Mode);
}

FileStream::FileStream(const std::wstring& _Path, const std::wstring& _Mode)
{
	ReOpen(_Path.c_str(), _Mode.c_str());
}

FileStream::FileStream(const std::wstring& _Path, const wchar_t* _Mode)
{
	ReOpen(_Path.c_str(), _Mode);
}

FileStream::FileStream(const wchar_t* _Path, const std::wstring& _Mode)
{
	ReOpen(_Path, _Mode.c_str());
}

FileStream::FileStream(const wchar_t* _Path, const wchar_t* _Mode)
{
	ReOpen(_Path, _Mode);
}

FileStream::FileStream(const std::string& _Path, const std::string& _Mode)
{
	Open(UTF8ToWideString(_Path), UTF8ToWideString(_Mode));
}

FileStream::FileStream(const std::string& _Path, const char* _Mode)
{
	Open(UTF8ToWideString(_Path), UTF8ToWideString(_Mode));
}

FileStream::FileStream(const char* _Path, const std::string& _Mode)
{
	Open(UTF8ToWideString(_Path), UTF8ToWideString(_Mode));
}

FileStream::FileStream(const char* _Path, const char* _Mode)
{
	Open(UTF8ToWideString(_Path), UTF8ToWideString(_Mode));
}

void FileStream::Open()
{
	if (!Enabled())
		_D_Dragonian_Lib_Throw_Exception("File not enabled!");
}

void FileStream::Close()
{
	_MyFile.reset();
}

Int FileStream::Seek(Int64 _Offset, Int _Origin) const
{
	if (!Enabled())
		_D_Dragonian_Lib_Throw_Exception("File not enabled!");
#if _MSC_VER
	return _fseeki64(_MyFile.get(), _Offset, _Origin);
#else
	return fseeko64(_MyFile.get(), _Offset, _Origin);
#endif
}

StdSize FileStream::Tell() const
{
	if (!Enabled())
		_D_Dragonian_Lib_Throw_Exception("File not enabled!");
#if _MSC_VER
	return _ftelli64(_MyFile.get());
#else
	return ftello64(_MyFile.get());
#endif
}

StdSize FileStream::Read(void* _Buffer, StdSize _BufferSize, StdSize _ElementSize, StdSize _Count) const
{
	if (!Enabled())
		_D_Dragonian_Lib_Throw_Exception("File not enabled!");
#if _MSC_VER
	return fread_s(_Buffer, _BufferSize, _ElementSize, _Count, _MyFile.get());
#else
	return fread(_Buffer, _ElementSize, _Count, _MyFile.get());
#endif
}

StdSize FileStream::Write(const void* _Buffer, StdSize _ElementSize, StdSize _Count)
{
	if (!Enabled())
		_D_Dragonian_Lib_Throw_Exception("File not enabled!");
	return fwrite(_Buffer, _ElementSize, _Count, _MyFile.get());
}

void* FileStream::GetHandle()
{
	return _MyFile.get();
}

const void* FileStream::GetHandle() const
{
	return _MyFile.get();
}

void* FileStream::ReleaseHandle()
{
	return _MyFile.get();
}

std::string FileStream::ReadLine() const
{
	if (Enabled())
	{
		std::string Ret;
		char Ch = 0;
		while (Ch != '\n' && Ch != '\r')
		{
			if (Read(&Ch, sizeof(char), 1) == 0)
				break;
			if (Ch != '\n' && Ch != '\r')
				Ret += Ch;
		}
		return Ret;
	}
	_D_Dragonian_Lib_Throw_Exception("File not enabled!");
}

std::wstring FileStream::ReadLineW() const
{
	if (Enabled())
	{
		std::wstring Ret;
		wchar_t Ch = 0;
		while (Ch != L'\n' && Ch != '\r')
		{
			if (Read(&Ch, sizeof(wchar_t), 1) == 0)
				break;
			if (Ch != L'\n' && Ch != '\r')
				Ret += Ch;
		}
		return Ret;
	}
	_D_Dragonian_Lib_Throw_Exception("File not enabled!");
}

bool FileStream::WriteString(const char* _Str)
{
	if (!_Str)
		return false;
	const auto StrLen = strlen(_Str);
	if (StrLen == 0)
		return false;
	if (Enabled() && Write(_Str, sizeof(char), StrLen) == StrLen)
		return true;
	return false;
}

bool FileStream::WriteString(const wchar_t* _Str)
{
	if (!_Str)
		return false;
	const auto StrLen = wcslen(_Str);
	if (StrLen == 0)
		return false;
	if (Enabled() && Write(_Str, sizeof(wchar_t), StrLen) == StrLen)
		return true;
	return false;
}

StdSize ByteIO::RemainderCapacity() const
{
	return _MyBuffer.End() - _MyLast;
}

StdSize ByteIO::RemainderSize() const
{
	return _MyBuffer.End() - _MyCur;
}

StdSize ByteIO::Remainder() const
{
	return _MyLast - _MyCur;
}

StdSize ByteIO::Size() const
{
	return _MyLast - _MyCur;
}

StdSize ByteIO::Capacity() const
{
	return _MyBuffer.Size();
}

void ByteIO::Open()
{
	if (!_MyBuffer)
		_MyBuffer = { 65535 };
	_IsOpen = true;
	_MyCur = _MyLast = _MyBuffer.Begin();
}

void ByteIO::Close()
{
	_IsOpen = false;
	_MyCur = _MyLast = nullptr;
}

Int ByteIO::Seek(Int64 _Offset, Int _Origin) const
{
	Byte* NewCur;
	if (_Origin == SEEK_SET)
		NewCur = _MyBuffer.Begin() + _Offset;
	else if (_Origin == SEEK_CUR)
		NewCur = _MyCur + _Offset;
	else if (_Origin == SEEK_END)
		NewCur = _MyLast + _Offset;
	else
		return -1;

	_MyCur = std::clamp(NewCur, _MyBuffer.Begin(), _MyLast);
	return static_cast<Int>(_MyCur - _MyBuffer.Begin());
}

StdSize ByteIO::Tell() const
{
	return _MyCur - _MyBuffer.Begin();
}

void* ByteIO::GetHandle()
{
	return _MyBuffer.Get();
}

const void* ByteIO::GetHandle() const
{
	return _MyBuffer.Get();
}

void* ByteIO::ReleaseHandle()
{
	_MyCur = _MyLast = nullptr;
	return _MyBuffer.Release();
}

StdSize ByteIO::Read(void* _Buffer, StdSize _BufferSize, StdSize _ElementSize, StdSize _Count) const
{
	if (Enabled())
	{
		const auto _BufferCount = _BufferSize / _ElementSize;
		const auto _RemainderCount = Remainder() / _ElementSize;
		_Count = std::min(_BufferCount, _Count);
		_Count = std::min(_RemainderCount, _Count);
		const auto _TgrSize = _ElementSize * _Count;
		if (_TgrSize == 0)
			return 0;
#if _MSC_VER
		memcpy_s(_Buffer, _TgrSize, _MyCur, _TgrSize);
#else
		memcpy(_Buffer, _MyCur, _TgrSize);
#endif
		_MyCur += _TgrSize;
		return _TgrSize / _ElementSize;
	}
	_D_Dragonian_Lib_Throw_Exception("ByteIO not enabled!");
}

StdSize ByteIO::Write(const void* _Buffer, StdSize _ElementSize, StdSize _Count)
{
	if (Enabled())
	{
		const auto _RemainderCapacity = RemainderSize();
		const auto _TgrSize = _ElementSize * _Count;
		if (_TgrSize == 0)
			return 0;
		if (_RemainderCapacity < _TgrSize)
			_MyBuffer.Reallocate((Tell() + _TgrSize) << 1);
#if _MSC_VER
		memcpy_s(_MyCur, _TgrSize, _Buffer, _TgrSize);
#else
		memcpy(_MyCur, _Buffer, _TgrSize);
#endif
		_MyCur += _TgrSize;
		_MyLast = std::max(_MyCur, _MyLast);

		return _Count;
	}
	_D_Dragonian_Lib_Throw_Exception("ByteIO not enabled!");
}

std::string ByteIO::ReadLine() const
{
	if (Enabled())
	{
		std::string Ret;
		char Ch = 0;
		while (Ch != '\n')
		{
			if (Read(&Ch, sizeof(char), 1) == 0)
				break;
			if (Ch != '\n')
				Ret += Ch;
		}
		return Ret;
	}
	_D_Dragonian_Lib_Throw_Exception("ByteIO not enabled!");
}

std::wstring ByteIO::ReadLineW() const
{
	if (Enabled())
	{
		std::wstring Ret;
		wchar_t Ch = 0;
		while (Ch != L'\n' && Ch != '\r')
		{
			if (Read(&Ch, sizeof(wchar_t), 1) == 0)
				break;
			if (Ch != L'\n' && Ch != '\r')
				Ret += Ch;
		}
		return Ret;
	}
	_D_Dragonian_Lib_Throw_Exception("ByteIO not enabled!");
}

bool ByteIO::WriteString(const char* _Str)
{
	if (!_Str)
		return false;
	const auto StrLen = strlen(_Str);
	if (StrLen == 0)
		return false;
	if (Enabled() && Write(_Str, sizeof(char), StrLen) == StrLen)
		return true;
	return false;
}

bool ByteIO::WriteString(const wchar_t* _Str)
{
	if (!_Str)
		return false;
	const auto StrLen = wcslen(_Str);
	if (StrLen == 0)
		return false;
	if (Enabled() && Write(_Str, sizeof(wchar_t), StrLen) == StrLen)
		return true;
	return false;
}

_D_Dragonian_Lib_Space_End