#pragma once
#include "Tensor/TensorOperator.h"
DragonianLibSpaceBegin
namespace Float64
{
	using ThisType = float64;

	ThisType CastFrom(TensorType _Type, cpvoid _Val);

	void AssignValue(const Tensor& _Input, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool);
	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool);
	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool);

	Tensor Gather(const Tensor& _Input, const Tensor& _IndicesInp, SizeType _Axis, ThreadPool* _ThreadPool);
	void Cast(const Tensor& _Dst, const Tensor& _Src, ThreadPool* _ThreadPool);

	Tensor Add(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Sub(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Mul(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Div(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Pow(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Add(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Sub(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Mul(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Div(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Pow(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);

	void AddInplace(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	void SubInplace(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	void MulInplace(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	void DivInplace(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	void PowInplace(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	void AddInplace(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	void SubInplace(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	void MulInplace(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	void DivInplace(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	void PowInplace(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);

	DragonianLibMonoOperatorFunctionDef(Abs);
	DragonianLibMonoOperatorFunctionDef(Sin);
	DragonianLibMonoOperatorFunctionDef(Sinh);
	DragonianLibMonoOperatorFunctionDef(Cos);
	DragonianLibMonoOperatorFunctionDef(Cosh);
	DragonianLibMonoOperatorFunctionDef(Tan);
	DragonianLibMonoOperatorFunctionDef(Tanh);
	DragonianLibMonoOperatorFunctionDef(ASin);
	DragonianLibMonoOperatorFunctionDef(ACos);
	DragonianLibMonoOperatorFunctionDef(ATan);
	DragonianLibMonoOperatorFunctionDef(ASinh);
	DragonianLibMonoOperatorFunctionDef(ACosh);
	DragonianLibMonoOperatorFunctionDef(ATanh);
	DragonianLibMonoOperatorFunctionDef(Exp);
	DragonianLibMonoOperatorFunctionDef(Exp2);
	DragonianLibMonoOperatorFunctionDef(Exp10);
	DragonianLibMonoOperatorFunctionDef(Log);
	DragonianLibMonoOperatorFunctionDef(Log2);
	DragonianLibMonoOperatorFunctionDef(Log10);

	Tensor Less(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Greater(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Equal(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor LessEqual(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor GreaterEqual(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor NotEqual(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool);
	Tensor Less(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Greater(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor Equal(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor LessEqual(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor GreaterEqual(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);
	Tensor NotEqual(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool);

	DragonianLibMonoOperatorFunctionDef(Floor);
	DragonianLibMonoOperatorFunctionDef(Ceil);
	DragonianLibMonoOperatorFunctionDef(Round);

	Tensor Sum(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool);
	Tensor CumSum(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool);
	Tensor CumProd(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool);
	void CumSumImpl(const Tensor& _Dst, SizeType CurDims);
	void CumProdImpl(const Tensor& _Dst, SizeType CurDims);
}
DragonianLibSpaceEnd