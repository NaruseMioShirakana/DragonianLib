#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Int8
{
	using ThisType = int8;

	ThisType CastFrom(TensorType _Type, void* _Val);

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType);

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid);
}
LibSvcEnd