#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Float16
{
	using ThisType = uint16;

	ThisType CastFrom(TensorType _Type, void* _Val);

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType);

}
LibSvcEnd