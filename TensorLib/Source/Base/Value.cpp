#include "Value.h"

DragonianLibSpaceBegin
Value::~Value()
{
}

Value& Value::load(const std::wstring& _Path, bool _Strict)
{
	FileGuard DictFile;
	DictFile.Open(_Path, L"rb");
	if (!DictFile.Enabled())
		DragonianLibThrow("Failed To Open File!");
	DictFile.Close();

	//loadData(WeightData_, _Strict);
	return *this;
}

Value& Value::save(const std::wstring& _Path)
{
	FileGuard file;
	file.Open(_Path, L"rb");
	if (!file.Enabled())
		DragonianLibThrow("Failed to open file!");
	saveData(file);
	return *this;
}

void Value::loadData(const DictType& _WeightDict, bool _Strict)
{
	DragonianLibNotImplementedError;
}

void Value::saveData(FileGuard& _File)
{
	DragonianLibNotImplementedError;
}
DragonianLibSpaceEnd