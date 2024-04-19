#include <ranges>
#include "Module/Module.h"

LibSvcBegin

Module::Module(Module* _Parent, const std::wstring& _Name)
{
	if (_Parent != nullptr)
	{
		RegName_ = _Parent->RegName_ + L"." + _Name;
		_Parent->Layers_[RegName_] = this;
	}
	else
		RegName_ = _Name;
}

void Module::loadData(ggml_context* _WeightDict, bool _Strict)
{
	for (const auto& it : Layers_ | std::views::values)
		it->loadData(_WeightDict, _Strict);
}

void Module::saveData(FileGuard& _File)
{
	for (const auto& it : Layers_ | std::views::values)
		it->saveData(_File);
}

Parameter::Parameter(Module* _Parent, const std::wstring& _Name, const std::vector<int64>& Shape) : Module(_Parent, _Name)
{
	
}
LibSvcEnd