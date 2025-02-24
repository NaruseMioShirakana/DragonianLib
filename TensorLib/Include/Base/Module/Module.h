/**
 * FileName: Module.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <any>
#include "../Tensor/Tensor.h"

#define _D_Dragonian_Lib_Graph_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Graph {
#define _D_Dragonian_Lib_Graph_Space_End _D_Dragonian_Lib_Space_End }

#define DragonianLibRegisterLayer(MemberName, ...) MemberName(this, L#MemberName, __VA_ARGS__)
#define DragonianLibLayerItem(TypeName, ...) std::make_shared<TypeName>(nullptr, L"", __VA_ARGS__)

_D_Dragonian_Lib_Graph_Space_Begin

class Module : public DlibValue
{
public:
	Module(Module* _Parent, const std::wstring& _Name);
	~Module() override = default;

	Module(const Module&) = default;
	Module(Module&&) noexcept = default;
	Module& operator=(const Module&) = default;
	Module& operator=(Module&&) noexcept = default;

	virtual std::optional<std::any> operator()(const std::any& _Input);
	std::wstring& Name() { return RegName_; }
	std::wstring DumpLayerNameInfo();

private:
	void DumpLayerNameInfoImpl(std::wstring& _Tmp, int TabCount);
	virtual void DumpCurrentLayerInfo(std::wstring& _Tmp);

protected:
	std::unordered_map<std::wstring, Module*> Layers_;

public:
	void LoadData(const DictType& _WeightDict, bool _Strict) override;
	void SaveData(FileGuard& _File) override;
	void ReCalcLayerName(const std::wstring& _Parent);
};

template <typename _Type, size_t _NRank, Device _MyDevice>
class Parameter : public Module
{
public:
	Parameter(Module* _Parent, const std::wstring& _Name) :
		Module(_Parent, _Name) {}
	Parameter(Module* _Parent, const std::wstring& _Name, const Dimensions<_NRank>& _Shape) :
		Module(_Parent, _Name)
	{
		Weight_ = Tensor<_Type, _NRank, _MyDevice>::New(_Shape);
	}
	~Parameter() override = default;

	Parameter(const Parameter&) = default;
	Parameter(Parameter&&) noexcept = default;
	Parameter& operator=(const Parameter&) = default;
	Parameter& operator=(Parameter&&) noexcept = default;

	operator const Tensor<_Type, _NRank, _MyDevice>& () const { return Weight_; }
	Tensor<_Type, _NRank, _MyDevice>* operator->() { return &Weight_; }

protected:
	Tensor<_Type, _NRank, _MyDevice> Weight_;

public:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += (RegName_ + L" (Size[");
		for (auto i : Weight_.Shape())
			_Tmp += (std::to_wstring(i) + L", ");
		if (!Weight_.Shape().Empty())
		{
			_Tmp.pop_back();
			_Tmp.pop_back();
		}
		_Tmp += L"])";
	}

	void LoadData(const DictType& _WeightDict, bool _Strict) override
	{
		//TODO
	}

	void SaveData(FileGuard& _File) override
	{
		//TODO
	}

	void ChangeShape(const Dimensions<_NRank>& _Shape)
	{
		Weight_ = Tensor<_Type, _NRank, _MyDevice>::New(_Shape);
	}
};

class Sequential : public Module
{
public:
	Sequential(Module* _Parent, const std::wstring& _Name) : Module(_Parent, _Name) {}
	~Sequential() override = default;

	Sequential(const Sequential&) = default;
	Sequential(Sequential&&) noexcept = default;
	Sequential& operator=(const Sequential&) = default;
	Sequential& operator=(Sequential&&) noexcept = default;

	Sequential(Module* _Parent, const std::wstring& _Name, const std::initializer_list<std::shared_ptr<Module>>& _Input);
	Sequential& operator=(const std::initializer_list<std::shared_ptr<Module>>& _Input);

	std::optional<std::any> operator()(const std::any& _Input) override;
	void Append(const std::shared_ptr<Module>& _Module);

protected:
	std::vector<std::shared_ptr<Module>> _Items;

public:
	std::shared_ptr<Module>* begin() { return _Items.data(); }
	std::shared_ptr<Module>* end() { return _Items.data() + _Items.size(); }
	std::shared_ptr<Module> operator[](int64_t _Index) const { return _Items[_Index]; }

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
};

class ModuleList : public Sequential
{
public:
	ModuleList(Module* _Parent, const std::wstring& _Name) : Sequential(_Parent, _Name) {}
	~ModuleList() override = default;

	ModuleList(const ModuleList&) = default;
	ModuleList(ModuleList&&) noexcept = default;
	ModuleList& operator=(const ModuleList&) = default;
	ModuleList& operator=(ModuleList&&) noexcept = default;

	ModuleList(Module* _Parent, const std::wstring& _Name, const std::initializer_list<std::shared_ptr<Module>>& _Input)
		: Sequential(_Parent, _Name, _Input) {}

private:
	std::optional<std::any> operator()(const std::any& _Input) override;
};

_D_Dragonian_Lib_Graph_Space_End