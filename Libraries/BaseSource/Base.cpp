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

_D_Dragonian_Lib_Space_End