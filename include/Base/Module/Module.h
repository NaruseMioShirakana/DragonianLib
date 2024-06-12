#pragma once
#include "Tensor/Tensor.h"

DragonianLibSpaceBegin

class Module : public Value
{
public:
	Module(Module* _Parent, const std::wstring& _Name);
	~Module() override = default;
private:
	std::unordered_map<std::wstring, Value*> Layers_;

public:
	void loadData(const DictType& _WeightDict, bool _Strict) override;
	void saveData(FileGuard& _File) override;
};

class Parameter : public Module
{
public:
	using Ty = float;
	Parameter(Module* _Parent, const std::wstring& _Name, const std::vector<int64>& Shape);
	~Parameter() override = default;
	

protected:

public:
	void loadData(const DictType& _WeightDict, bool _Strict) override;
	void saveData(FileGuard& _File) override;

};

DragonianLibSpaceEnd