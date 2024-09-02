#include "Base.h"
#ifdef _WIN32
#include "Windows.h"
#endif

DragonianLibSpaceBegin

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

FileGuard::operator FILE*() const
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

DObject::~DObject()
{
	if (MyRefCountPtr_ && RemoveRef() <= 0)
		_Tidy();
}

DObject::DObject(std::atomic_int64_t* RefCountPtr_)
	: MyRefCountPtr_(RefCountPtr_)
{
	if (!RefCountPtr_)
		DragonianLibThrow("Ref Count Pointr Can Not Be Nullptr!");
	AddRef();
}

void DObject::_Tidy()
{
	Destory();
	delete MyRefCountPtr_;
	MyRefCountPtr_ = nullptr;
}

void DObject::AddRef(int64_t Count) const
{
	std::atomic_int64_t& MyRefCount = *MyRefCountPtr_;
	MyRefCount += Count;
}

int64_t DObject::RemoveRef(int64_t Count) const
{
	std::atomic_int64_t& MyRefCount = *MyRefCountPtr_;
	MyRefCount -= Count;
	return MyRefCount;
}

DObject::DObject(const DObject& _Left)
{
	MyRefCountPtr_ = _Left.MyRefCountPtr_;
	if (MyRefCountPtr_)
		AddRef();
}

DObject::DObject(DObject&& _Right) noexcept
{
	MyRefCountPtr_ = _Right.MyRefCountPtr_;
	_Right.MyRefCountPtr_ = nullptr;
}

DObject& DObject::operator=(const DObject& _Left)
{
	if (&_Left == this)
		return *this;
	MyRefCountPtr_ = _Left.MyRefCountPtr_;
	if (MyRefCountPtr_)
		AddRef();
	return *this;
}

DObject& DObject::operator=(DObject&& _Right) noexcept
{
	MyRefCountPtr_ = _Right.MyRefCountPtr_;
	_Right.MyRefCountPtr_ = nullptr;
	return *this;
}

DragonianLibSpaceEnd