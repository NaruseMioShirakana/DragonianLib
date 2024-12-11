#pragma once
#include "Module.h"

_D_Dragonian_Lib_Graph_Space_Begin

struct Conv1DParam
{
	SizeType InChannels;
	SizeType OutChannels;
	SizeType KernelSize;
	SizeType Stride = 1;
	SizeType Padding = 0;
	SizeType Dilation = 1;
	SizeType Groups = 1;
	bool Bias = true;
	PaddingType PaddingMode = PaddingType::Zero;
};

struct Conv2DParam
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

struct ConvTranspose1DParam
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

struct ConvTranspose2DParam
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

template <typename _Type, Device _MyDevice>
class Conv1D : public Module
{
public:
	Conv1D(Module* _Parent, const std::wstring& _Name, const Conv1DParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(weight, { _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize }),
		DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
		params(_Params)
	{
		if (!_Params.Bias)
			Layers_.erase(bias.Name());
	}

	std::optional<std::any> operator()(const std::any& _Input) override
	{
		return std::nullopt;
		//TODO
	}

	void ChangeParam(const Conv1DParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize });
		bias.ChangeShape({ _Params.OutChannels });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} Conv1D(in_channel[{}], out_channel[{}], kernel_size[{}])",
			RegName_,
			params.InChannels,
			params.OutChannels,
			params.KernelSize
		);
	}

	Parameter<_Type, 3, _MyDevice> weight;
	Parameter<_Type, 1, _MyDevice> bias;
	Conv1DParam params;
};

template <typename _Type, Device _MyDevice>
class Conv2D : public Module
{
public:
	Conv2D(Module* _Parent, const std::wstring& _Name, const Conv2DParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(
			weight,
			{ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] }
		),
		DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
		params(_Params)
	{
		if (!_Params.Bias)
			Layers_.erase(bias.Name());
	}

	std::optional<std::any> operator()(const std::any& _Input) override
	{
		return std::nullopt;
		//TODO
	}

	void ChangeParam(const Conv2DParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] });
		bias.ChangeShape({ _Params.OutChannels });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} Conv2D(in_channel[{}], out_channel[{}], kernel_size[{}, {}])",
			RegName_,
			params.InChannels,
			params.OutChannels,
			params.KernelSize[0],
			params.KernelSize[1]
		);
	}

	Parameter<_Type, 4, _MyDevice> weight;
	Parameter<_Type, 1, _MyDevice> bias;
	Conv2DParam params;
};

template <typename _Type, Device _MyDevice>
class ConvTranspose1D : public Module
{
public:
	ConvTranspose1D(Module* _Parent, const std::wstring& _Name, const ConvTranspose1DParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(weight, { _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize }),
		DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
		params(_Params)
	{
		if (_Params.PaddingMode != PaddingType::Zero)
			_D_Dragonian_Lib_Throw_Exception("Only `Zeros` Padding Mode Is Supported For ConvTranspose1D");
		if (!_Params.Bias)
			Layers_.erase(bias.Name());
	}

	std::optional<std::any> operator()(const std::any& _Input) override
	{
		return std::nullopt;
		//TODO
	}

	void ChangeParam(const ConvTranspose1DParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize });
		bias.ChangeShape({ _Params.OutChannels });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} ConvTranspose1D(in_channel[{}], out_channel[{}], kernel_size[{}])",
			RegName_,
			params.InChannels,
			params.OutChannels,
			params.KernelSize
		);
	}

	Parameter<_Type, 3, _MyDevice> weight;
	Parameter<_Type, 1, _MyDevice> bias;
	ConvTranspose1DParam params;
};

template <typename _Type, Device _MyDevice>
class ConvTranspose2D : public Module
{
public:
	ConvTranspose2D(Module* _Parent, const std::wstring& _Name, const ConvTranspose2DParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(
			weight,
			{ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] }
		),
		DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
		params(_Params)
	{
		if (!_Params.Bias)
			Layers_.erase(bias.Name());
	}

	std::optional<std::any> operator()(const std::any& _Input) override
	{
		return std::nullopt;
		//TODO
	}

	void ChangeParam(const ConvTranspose2DParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] });
		bias.ChangeShape({ _Params.OutChannels });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} ConvTranspose2D(in_channel[{}], out_channel[{}], kernel_size[{}, {}])",
			RegName_,
			params.InChannels,
			params.OutChannels,
			params.KernelSize[0],
			params.KernelSize[1]
		);
	}

	Parameter<_Type, 4, _MyDevice> weight;
	Parameter<_Type, 1, _MyDevice> bias;
	ConvTranspose2DParam params;
};

_D_Dragonian_Lib_Graph_Space_End