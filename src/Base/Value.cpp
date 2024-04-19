#include "Value.h"

LibSvcBegin
Value::~Value()
{
	if (WeightData_) ggml_free(WeightData_);
	WeightData_ = nullptr;
	if (WeightDict_) gguf_free(WeightDict_);
	WeightDict_ = nullptr;
}

Value& Value::load(const std::wstring& _Path, bool _Strict)
{
	FileGuard DictFile;
	DictFile.Open(_Path, L"rb");
	if (!DictFile.Enabled())
		LibSvcThrow("Failed To Open File!");
	DictFile.Close();

	const gguf_init_params Params{ false,&WeightData_ };
	WeightDict_ = gguf_init_from_file(to_byte_string(_Path).c_str(), Params);
	if (!WeightDict_)
		LibSvcThrow("Failed To Load Dict!");
	LogMessage(L"[Data Loader] Loading Static Dict From " + _Path);
	loadData(WeightData_, _Strict);
	return *this;
}

Value& Value::save(const std::wstring& _Path)
{
	FileGuard file;
	file.Open(_Path, L"rb");
	if (!file.Enabled())
		LibSvcThrow("Failed to open file!");
	saveData(file);
	return *this;
}

void Value::loadData(ggml_context* _WeightDict, bool _Strict)
{
	LibSvcNotImplementedError;
}

void Value::saveData(FileGuard& _File)
{
	LibSvcNotImplementedError;
}
LibSvcEnd