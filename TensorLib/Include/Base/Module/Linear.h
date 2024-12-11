#pragma once
#include "Module.h"

_D_Dragonian_Lib_Graph_Space_Begin

struct LinearParam
{
	SizeType InFeatures;
	SizeType OutFeatures;
	bool Bias = true;
};

template <typename _Type, Device _MyDevice>
class Linear : public Module
{
public:
	Linear(Module* _Parent, const std::wstring& _Name, const LinearParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(weight, { _Params.OutFeatures, _Params.InFeatures }),
		DragonianLibRegisterLayer(bias, { _Params.OutFeatures }),
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

	void ChangeParam(const LinearParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.OutFeatures, _Params.InFeatures });
		bias.ChangeShape({ _Params.OutFeatures });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} Linear(in_feature[{}], out_feature[{}])",
			RegName_,
			params.InFeatures,
			params.OutFeatures
		);
	}

	Parameter<_Type, 2, _MyDevice> weight;
	Parameter<_Type, 1, _MyDevice> bias;
	LinearParam params;
};

_D_Dragonian_Lib_Graph_Space_End