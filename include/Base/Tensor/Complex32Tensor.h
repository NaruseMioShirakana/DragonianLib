#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Complex32
{
	using ThisType = std::complex<float>;

	ThisType CastFrom(TensorType _Type, void* _Val);

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType);

}
LibSvcEnd