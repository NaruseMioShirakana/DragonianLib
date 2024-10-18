#include "Base.h"
#include "Util/StringPreprocess.h"
#ifdef _WIN32
#include "Windows.h"
#endif

DragonianLibSpaceBegin

std::string DragonianLibThrowFunctionImpl(const std::string& Message, const char* Path, const char* Function, int Line)
{
	const std::string Prefix =
		std::string("[@file: \"") + std::filesystem::path(Path).filename().string() + "\"; " +
		"function: \"" + Function + "\"; " +
		"line: " + std::to_string(Line) + "]:";
	if (Message.substr(0, 2) == "[@")
	{
		if (Message.substr(0, Prefix.length()) == Prefix)
			return Message;
		return Prefix.substr(0, Prefix.length() - 2) + "\n " + Message.substr(1);
	}
	return Prefix + ' ' + Message;
}

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

DragonianLibSpaceEnd