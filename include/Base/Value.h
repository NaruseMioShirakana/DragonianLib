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

public:
	virtual Value& load(const std::wstring& _Path, bool _Strict = false);
	virtual Value& save(const std::wstring& _Path);
	virtual void loadData(const DictType& _WeightDict, bool _Strict = false);
	virtual void saveData(FileGuard& _File);
};
LibSvcEnd