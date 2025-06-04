#include <ranges>
#include "TensorLib/Include/Base/Module/Module.h"
#include "Libraries/Util/StringPreprocess.h"

_D_Dragonian_Lib_Graph_Space_Begin

Module::Module(Module* _Parent, const std::wstring& _Name)
{
	if (_Parent != nullptr)
	{
		if (!_Parent->RegName_.empty())
			RegName_ = _Parent->RegName_ + L"." + _Name;
		else
			RegName_ = _Name;
		_Parent->Layers_[RegName_] = this;
	}
	else
		RegName_ = _Name;
}

void Module::LoadData(const DictType& _WeightDict, bool _Strict)
{
	for (const auto& it : Layers_ | std::views::values)
		it->LoadData(_WeightDict, _Strict);
}

void Module::SaveData(FileGuard& _File)
{
	for (const auto& it : Layers_ | std::views::values)
		it->SaveData(_File);
}

std::optional<std::any> Module::operator()(const std::any&)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

void Module::ReCalcLayerName(const std::wstring& _Parent)
{
	std::unordered_map<std::wstring, Module*> Lay;
	for (auto i : Layers_ | std::ranges::views::values)
	{
		i->ReCalcLayerName(_Parent);
		i->RegName_ = _Parent + L"." + i->RegName_;
		Lay[i->RegName_] = i;
	}
	Layers_ = std::move(Lay);
}

std::wstring Module::DumpLayerNameInfo()
{
	std::wstring Ret;
	DumpLayerNameInfoImpl(Ret, 0);
	return Ret;
}

void Module::DumpLayerNameInfoImpl(std::wstring& _Tmp, int TabCount)
{
	for (int i = 0; i < TabCount; ++i) _Tmp += L"  ";
	DumpCurrentLayerInfo(_Tmp);
	if (!Layers_.empty())
	{
		if (!_Tmp.empty())
			_Tmp += L" = {";
		else
			_Tmp += L"{";
		_Tmp += '\n';
		for (auto i : Layers_ | std::ranges::views::values)
			i->DumpLayerNameInfoImpl(_Tmp, TabCount + 1);
		for (int i = 0; i < TabCount; ++i) _Tmp += L"  ";
		_Tmp += L'}';
		_Tmp += '\n';
	}
	else
		_Tmp += '\n';
}

void Module::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += RegName_;
}

Sequential::Sequential(Module* _Parent, const std::wstring& _Name, const std::initializer_list<std::shared_ptr<Module>>& _Input)
	:Module(_Parent, _Name)
{
	for (const auto& i : _Input)
		Append(i);
}

Sequential& Sequential::operator=(const std::initializer_list<std::shared_ptr<Module>>& _Input)
{
	for (const auto& i : _Input)
		Append(i);
	return *this;
}

std::optional<std::any> Sequential::operator()(const std::any& _Input)
{
	if (_Items.empty())
		return std::nullopt;
	auto Output = (*_Items[0])(_Input);
	for (size_t i = 1; i < _Items.size(); ++i)
		Output = (*_Items[i])(Output);
	return Output;
}

void Sequential::Append(const std::shared_ptr<Module>& _Module)
{
	_Module->Name() = RegName_ + L"." + std::to_wstring(_Items.size());
	_Module->ReCalcLayerName(_Module->Name());
	_Items.emplace_back(_Module);
	Layers_[_Module->Name()] = _Module.get();
}

void Sequential::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += RegName_;
	if (_Items.empty())
		_Tmp += L" = (empty)";
}

std::optional<std::any> ModuleList::operator()(const std::any& _Input)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

_D_Dragonian_Lib_Graph_Space_End