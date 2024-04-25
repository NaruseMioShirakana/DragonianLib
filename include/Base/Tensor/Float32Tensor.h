#pragma once
#include "Tensor/Tensor.h"
LibSvcBegin
namespace Float32
{
	using ThisType = float32;

	ThisType CastFrom(TensorType _Type, cpvoid _Val);

	void AssignValue(const Tensor& _Input, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool);

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool);

	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool);

	Tensor Gather(const Tensor& _Input, const Tensor& _Indices, ThreadPool* _ThreadPool);

	void Cast(const Tensor& _Dst, const Tensor& _Src, ThreadPool* _ThreadPool);
}
LibSvcEnd