#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Float64
{
	using ThisType = float64;

	ThisType CastFrom(TensorType _Type, void* _Val);

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType, ThreadPool* _ThreadPool);

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool);

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool);

	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool);
}
LibSvcEnd