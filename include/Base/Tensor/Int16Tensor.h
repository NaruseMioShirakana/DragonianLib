#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Int16
{
	using ThisType = int16;

	ThisType CastFrom(TensorType _Type, void* _Val);

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType);

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid);

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB);

	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma);
}
LibSvcEnd