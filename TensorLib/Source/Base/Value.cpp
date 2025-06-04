#include "TensorLib/Include/Base/Value.h"

_D_Dragonian_Lib_Space_Begin

DlibValue& DlibValue::Load(const std::wstring& _Path, bool _Strict)
{
	FileGuard DictFile;
	DictFile.Open(_Path, L"rb");
	if (!DictFile.Enabled())
		_D_Dragonian_Lib_Throw_Exception("Failed To Open File!");
	DictFile.Close();

	//loadData(WeightData_, _Strict);
	return *this;
}

DlibValue& DlibValue::Save(const std::wstring& _Path)
{
	FileGuard file;
	file.Open(_Path, L"rb");
	if (!file.Enabled())
		_D_Dragonian_Lib_Throw_Exception("Failed to open file!");
	SaveData(file);
	return *this;
}

void DlibValue::LoadData(const DictType& _WeightDict, bool _Strict)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

void DlibValue::SaveData(FileGuard& _File)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

_D_Dragonian_Lib_Space_End