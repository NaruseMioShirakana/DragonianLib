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
#include "Tensor/Tensor.h"

#define DragonianLibGraphBegin namespace Graph {
#define DragonianLibGraphEnd }
#define DragonianLibRegisterLayer(MemberName, ...) MemberName(this, L#MemberName, __VA_ARGS__)
#define DragonianLibLayerItem(TypeName, ...) new TypeName(nullptr, L"", __VA_ARGS__)

DragonianLibSpaceBegin
DragonianLibGraphBegin

class Module : public Value
{
public:
	Module(Module* _Parent, const std::wstring& _Name);
	~Module() override = default;
	virtual std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false);
	std::wstring& Name() { return RegName_; }
	std::wstring DumpLayerNameInfo();

private:
	void DumpLayerNameInfoImpl(std::wstring& _Tmp, int TabCount);
	virtual void DumpCurrentLayerInfo(std::wstring& _Tmp);
	Module(const Module&) = delete;
	Module(Module&&) = delete;
	Module operator=(const Module&) = delete;
	Module operator=(Module&&) = delete;

protected:
	std::unordered_map<std::wstring, Module*> Layers_;

public:
	void loadData(const DictType& _WeightDict, bool _Strict) override;
	void saveData(FileGuard& _File) override;
	void ReCalcLayerName(const std::wstring& _Parent);
};

class Parameter : public Module
{
public:
	using Ty = float;
	Parameter(Module* _Parent, const std::wstring& _Name, const std::vector<SizeType>& _Shape, TensorType _DType = TensorType::Float32);
	~Parameter() override = default;
	operator const Tensor& () const;
	Tensor* operator->() { return &Weight_; }
	void ChangeShape(const std::vector<SizeType>& _Shape);

protected:
	Tensor Weight_;

public:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	void loadData(const DictType& _WeightDict, bool _Strict) override;
	void saveData(FileGuard& _File) override;

private:
	Parameter(const Parameter&) = delete;
	Parameter(Parameter&&) = delete;
	Parameter operator=(const Parameter&) = delete;
	Parameter operator=(Parameter&&) = delete;
};

class Sequential : public Module
{
public:
	Sequential(Module* _Parent, const std::wstring& _Name);
	~Sequential() override;
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void Append(Module* _Module);
	virtual Sequential& operator=(const std::initializer_list<Module*>& _Input);
	Module** begin() { return _Items.data(); }
	Module** end() { return _Items.data() + _Items.size(); }
	inline Module* operator[](int64_t _Index) const { return *(_Items.data() + _Index); }

private:
	std::vector<Module*> _Items;
	Sequential(const Sequential&) = delete;
	Sequential(Sequential&&) = delete;
	Sequential operator=(const Sequential&) = delete;
	Sequential operator=(Sequential&&) = delete;
};

class ModuleList : public Sequential
{
public:
	ModuleList(Module* _Parent, const std::wstring& _Name);
	ModuleList(Module* _Parent, const std::wstring& _Name, const std::initializer_list<Module*>& _Input);
	~ModuleList() override;
	ModuleList& operator=(const std::initializer_list<Module*>& _Input) override;
	
private:
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	ModuleList(const ModuleList&) = delete;
	ModuleList(ModuleList&&) = delete;
	ModuleList operator=(const ModuleList&) = delete;
	ModuleList operator=(ModuleList&&) = delete;
};

//**************************************************Operator***********************************************

class Conv1D : public Module
{
public:
	struct ConvParam
	{
		SizeType InChannels;
		SizeType OutChannels;
		SizeType KernelSize;
		int Stride = 1;
		int Padding = 0;
		int Dilation = 1;
		int Groups = 1;
		bool Bias = true;
		PaddingType PaddingMode = PaddingType::Zero;
	};
	Conv1D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const ConvParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	Parameter bias;
	ConvParam params;
};

class Conv2D : public Module
{
public:
	struct ConvParam
	{
		SizeType InChannels;
		SizeType OutChannels;
		SizeType KernelSize[2];
		int Stride[2] = { 1, 1 };
		int Padding[2] = { 0, 0 };
		int Dilation[2] = { 1, 1 };
		int Groups = 1;
		bool Bias = true;
		PaddingType PaddingMode = PaddingType::Zero;
	};
	Conv2D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const ConvParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	Parameter bias;
	ConvParam params;
};

class ConvTranspose1D : public Module
{
public:
	struct ConvParam
	{
		SizeType InChannels;
		SizeType OutChannels;
		SizeType KernelSize;
		int Stride = 1;
		int Padding = 0;
		int OutPutPadding = 0;
		int Dilation = 1;
		int Groups = 1;
		bool Bias = true;
		PaddingType PaddingMode = PaddingType::Zero;
	};
	ConvTranspose1D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const ConvParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	Parameter bias;
	ConvParam params;
};

class ConvTranspose2D : public Module
{
public:
	struct ConvParam
	{
		SizeType InChannels;
		SizeType OutChannels;
		SizeType KernelSize[2];
		int Stride[2] = { 1, 1 };
		int Padding[2] = { 0, 0 };
		int OutPutPadding[2] = { 0, 0 };
		int Dilation[2] = { 1, 1 };
		int Groups = 1;
		bool Bias = true;
		PaddingType PaddingMode = PaddingType::Zero;
	};
	ConvTranspose2D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const ConvParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	Parameter bias;
	ConvParam params;
};

class Linear : public Module
{
public:
	struct LinearParam
	{
		SizeType InFeatures;
		SizeType OutFeatures;
		bool Bias = true;
	};
	Linear(Module* _Parent, const std::wstring& _Name, const LinearParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const LinearParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	Parameter bias;
	LinearParam params;
};

class Embedding : public Module
{
public:
	struct EmbeddingParam
	{
		int NumEmbeddings;
		int EmbeddingDim;
		int PaddingIdx = INT_MAX;
		float MaxNorm = NAN;
		float NormType = 2.f;
		bool ScaleGradByFreq = false;
		bool Sparse = false;
		bool Freeze = false;
	};
	Embedding(Module* _Parent, const std::wstring& _Name, const EmbeddingParam& _Params);
	std::any operator()(const std::any& _Input, ThreadPool* _Thp = nullptr, bool _Inplace = false) override;
	void ChangeParam(const EmbeddingParam& _Params);
private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override;
	Parameter weight;
	//Parameter bias;
	EmbeddingParam params;
};

DragonianLibGraphEnd
DragonianLibSpaceEnd