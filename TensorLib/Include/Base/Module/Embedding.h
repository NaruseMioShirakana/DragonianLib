#pragma once
#include "Module.h"

_D_Dragonian_Lib_Graph_Space_Begin

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

template <typename _Type, Device _MyDevice>
class Embedding : public Module
{
public:
	Embedding(Module* _Parent, const std::wstring& _Name, const EmbeddingParam& _Params) :
		Module(_Parent, _Name),
		DragonianLibRegisterLayer(weight, { _Params.NumEmbeddings, _Params.EmbeddingDim }),
		params(_Params) {}

	std::optional<std::any> operator()(const std::any& _Input) override
	{
		return std::nullopt;
		//TODO
	}

	void ChangeParam(const EmbeddingParam& _Params)
	{
		params = _Params;
		weight.ChangeShape({ _Params.NumEmbeddings, _Params.EmbeddingDim });
	}

private:
	void DumpCurrentLayerInfo(std::wstring& _Tmp) override
	{
		_Tmp += std::format(
			L"{} Embedding(num_embeddings[{}], embedding_dim[{}])",
			RegName_,
			params.NumEmbeddings,
			params.EmbeddingDim
		);
	}

	Parameter<_Type, 2, _MyDevice> weight;
	//Parameter bias;
	EmbeddingParam params;
};

_D_Dragonian_Lib_Graph_Space_End