#pragma once
#include "Base.h"

LibSvcBegin
class Value
{
public:
	Value() = default;
	Value(const Value& _Left) = delete;
	virtual ~Value();

protected:
	std::wstring RegName_;
	ggml_context* WeightData_ = nullptr;
	gguf_context* WeightDict_ = nullptr;

public:
	virtual Value& load(const std::wstring& _Path, bool _Strict = false);
	virtual Value& save(const std::wstring& _Path);
	virtual void loadData(ggml_context* _WeightDict, bool _Strict = false);
	virtual void saveData(FileGuard& _File);
};
LibSvcEnd