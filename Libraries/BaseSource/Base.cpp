#include "Libraries/Base.h"
#ifdef _WIN32
#include "Windows.h"
#endif

_D_Dragonian_Lib_Space_Begin

std::wstring GlobalEnvDir;

std::wstring GetCurrentFolder()
{
	if (!GlobalEnvDir.empty())
		return GlobalEnvDir;

	wchar_t path[1024];
#ifdef _WIN32
	GetModuleFileName(nullptr, path, 1024);
	std::wstring _curPath = path;
	_curPath = _curPath.substr(0, _curPath.rfind(L'\\'));
	return _curPath;
#else
	if (GlobalEnvDir.empty())
		DragonianLibThrow("GlobalEnvDir Is Empty!");
#endif
}

void SetGlobalEnvDir(const std::wstring& _Folder)
{
	GlobalEnvDir = _Folder;
}

FileGuard::FileGuard(const std::wstring& _Path, const std::wstring& _Mode)
{
	Open(_Path, _Mode);
}

FileGuard::FileGuard(const std::wstring& _Path, const wchar_t* _Mode)
{
	Open(_Path, _Mode);
}

FileGuard::FileGuard(const wchar_t* _Path, const wchar_t* _Mode)
{
	Open(_Path, _Mode);
}

FileGuard::~FileGuard()
{
	Close();
}

FileGuard::FileGuard(FileGuard&& _Right) noexcept
{
	file_ = _Right.file_;
	_Right.file_ = nullptr;
}

FileGuard& FileGuard::operator=(FileGuard&& _Right) noexcept
{
	file_ = _Right.file_;
	_Right.file_ = nullptr;
	return *this;
}

void FileGuard::Open(const std::wstring& _Path, const std::wstring& _Mode)
{
	Close();
#ifdef _WIN32
	_wfopen_s(&file_, _Path.c_str(), _Mode.c_str());
#else
	file_ = _wfopen(_Path.c_str(), _Mode.c_str());
#endif
}

void FileGuard::Open(const std::wstring& _Path, const wchar_t* _Mode)
{
	Close();
#ifdef _WIN32
	_wfopen_s(&file_, _Path.c_str(), _Mode);
#else
	file_ = _wfopen(_Path.c_str(), _Mode);
#endif
}

void FileGuard::Open(const wchar_t* _Path, const wchar_t* _Mode)
{
	Close();
#ifdef _WIN32
	_wfopen_s(&file_, _Path, _Mode);
#else
	file_ = _wfopen(_Path, _Mode);
#endif
}

FileGuard::operator FILE* () const
{
	return file_;
}

bool FileGuard::Enabled() const
{
	return file_;
}

void FileGuard::Close()
{
	if (file_)
		fclose(file_);
	file_ = nullptr;
}

void FileGuard::Seek(long _Offset, int _Origin) const
{
	if (file_)
		fseek(file_, _Offset, _Origin);
}

size_t FileGuard::Tell() const
{
	if (file_)
		return ftell(file_);
	return 0;
}

size_t FileGuard::Read(void* _Buffer, size_t _BufferSize, size_t _ElementSize, size_t _Count) const
{
	if (file_)
		return fread_s(_Buffer, _BufferSize, _ElementSize, _Count, file_);
	return 0;
}

size_t FileGuard::Write(const void* _Buffer, size_t _ElementSize, size_t _Count) const
{
	if (file_)
		return fwrite(_Buffer, _ElementSize, _Count, file_);
	return 0;
}

FileGuard& FileGuard::operator<<(char _Ch)
{
	if (file_)
		fputc(_Ch, file_);
	return *this;
}

FileGuard& FileGuard::operator<<(const char* _Str)
{
	if (file_)
		fputs(_Str, file_);
	return *this;
}

FileGuard& FileGuard::operator<<(const std::string& _Str)
{
	if (file_)
		fputs(_Str.c_str(), file_);
	return *this;
}

FileGuard& FileGuard::operator<<(wchar_t _Ch)
{
	if (file_)
		fputwc(_Ch, file_);
	return *this;
}

FileGuard& FileGuard::operator<<(const wchar_t* _Str)
{
	if (file_)
		fputws(_Str, file_);
	return *this;
}

FileGuard& FileGuard::operator<<(const std::wstring& _Str)
{
	if (file_)
		fputws(_Str.c_str(), file_);
	return *this;
}

FILE* FileGuard::Release()
{
	auto _File = file_;
	file_ = nullptr;
	return _File;
}

IOStream::~IOStream()
{
	_Tidy();
}

IOStream::IOStream(IOStream&& _Right) noexcept
{
	_MyFile = _Right._MyFile;
	_MyBuffer = _Right._MyBuffer;
	_MyBufferEnd = _Right._MyBufferEnd;
	_MyIter = _Right._MyIter;
	_Right._MyFile = nullptr;
	_Right._MyBuffer = nullptr;
	_Right._MyBufferEnd = nullptr;
	_Right._MyIter = nullptr;
}

IOStream& IOStream::operator=(IOStream&& _Right) noexcept
{
	_MyFile = _Right._MyFile;
	_MyBuffer = _Right._MyBuffer;
	_MyBufferEnd = _Right._MyBufferEnd;
	_MyIter = _Right._MyIter;
	_Right._MyFile = nullptr;
	_Right._MyBuffer = nullptr;
	_Right._MyBufferEnd = nullptr;
	_Right._MyIter = nullptr;
	return *this;
}

void IOStream::_Tidy()
{
	if (_MyFile)
		fclose(_MyFile);
	delete[] _MyBuffer;

	_MyFile = nullptr;
	_MyBuffer = nullptr;
	_MyBufferEnd = nullptr;
	_MyIter = nullptr;
}

IOStream::IOStream(const std::wstring& _Path, const std::wstring& _Mode)
{
#ifdef _WIN32
	if (_wfopen_s(&_MyFile, _Path.c_str(), _Mode.c_str()))
		_D_Dragonian_Lib_Throw_Exception("Failed to open file.");
#else
	_MyFile = _wfopen(_Path.c_str(), _Mode.c_str());
	if (!_MyFile)
		_D_Dragonian_Lib_Throw_Exception("Failed to open file.");
#endif
}

IOStream::IOStream(FILE* _FileStream) : _MyFile(_FileStream) {}

IOStream::IOStream(size_t _BufferSize)
{
	_MyBuffer = new Byte[_BufferSize];
	_MyBufferEnd = _MyBuffer + _BufferSize;
	_MyIter = _MyBuffer;
}

FILE* IOStream::ReleaseFile() noexcept
{
	auto _File = _MyFile;
	_MyFile = nullptr;
	return _File;
}

Byte* IOStream::ReleaseBuffer() noexcept
{
	auto _Buffer = _MyBuffer;
	_MyBuffer = nullptr;
	_MyBufferEnd = nullptr;
	_MyIter = nullptr;
	return _Buffer;
}

size_t IOStream::Tell() const noexcept
{
	if (_MyFile)
		return ftell(_MyFile);
	return Capacity();
}

void IOStream::Seek(long _Offset, int _Origin) noexcept
{
	if (_MyFile)
		fseek(_MyFile, _Offset, _Origin);
	else
	{
		auto Temp = _MyIter;
		if (_Origin == SEEK_SET)
			_MyIter = _MyBuffer + _Offset;
		else if (_Origin == SEEK_CUR)
			_MyIter += _Offset;
		else if (_Origin == SEEK_END)
			_MyIter = _MyBufferEnd + _Offset;
		if (_MyIter < _MyBuffer || _MyIter > _MyBufferEnd)
			_MyIter = Temp;
	}
}

void IOStream::Reserve(size_t _Size)
{
	if (Enabled())
		_D_Dragonian_Lib_Throw_Exception("IOStream is enabled.");

	if (IsBuffer() && Capacity() != _Size)
	{
		auto Tmp = new Byte[_Size];
		auto Offset = static_cast<size_t>(_MyIter - _MyBuffer);
		memcpy(Tmp, _MyBuffer, std::min(Capacity(), _Size));
		delete[] _MyBuffer;
		_MyBuffer = Tmp;
		_MyBufferEnd = _MyBuffer + _Size;
		_MyIter = _MyBuffer + std::min(Offset, _Size);
	}
}

_D_Dragonian_Lib_Space_End