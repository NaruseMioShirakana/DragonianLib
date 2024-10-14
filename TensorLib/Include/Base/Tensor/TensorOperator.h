/**
 * FileName: TensorOperator.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "Tensor/Tensor.h"
#include <random>
#include "Util/Avx.h"
#include "Tensor/OperatorMacro.h"

DragonianLibSpaceBegin

template <typename GType>
void GatherImp(
	const SizeType* __restrict _Shape,
	const SizeType* __restrict _Stride,
	const SizeType* __restrict _BeginPtr,
	const SizeType* __restrict _Step,
	GType* _Dst,
	const GType* _Src,
	SizeType NDims,
	bool IsContinuous,
	SizeType TotalSize
)
{
	if (IsContinuous)
	{
		DragonianLibMemCpy(_Dst, _Src, TotalSize);
		return;
	}

	if (NDims > 5)
	{
		Vector<SizeType> CurIndice(NDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			_Shape,
			NDims,
			{
				SizeType Index = 0;
				for (SizeType i = 0; i < NDims; ++i)
					Index += ((IndicesPtr[i] * _Stride[i]) + _BeginPtr[i]) * _Step[i];
				*(_Dst++) = _Src[Index];
			}
		);
		return;
	}

	if (NDims == 5)
	{
		for (SizeType i = 0; i < _Shape[0]; ++i)
		{
			const auto Index0 = ((i * _Stride[0]) + _BeginPtr[0]) * _Step[0];
			for (SizeType j = 0; j < _Shape[1]; ++j)
			{
				const auto Index1 = Index0 +
					((j * _Stride[1]) + _BeginPtr[1]) * _Step[1];
				for (SizeType k = 0; k < _Shape[2]; ++k)
				{
					const auto Index2 = Index1 +
						((k * _Stride[2]) + _BeginPtr[2]) * _Step[2];
					for (SizeType l = 0; l < _Shape[3]; ++l)
					{
						const auto Index3 = Index2 +
							((l * _Stride[3]) + _BeginPtr[3]) * _Step[3];
						for (SizeType m = 0; m < _Shape[4]; ++m)
						{
							const auto Index4 = Index3 +
								((m * _Stride[4]) + _BeginPtr[4]) * _Step[4];
							*(_Dst++) = _Src[Index4];
						}
					}
				}
			}
		}
	}
	else if (NDims == 4)
	{
		for (SizeType i = 0; i < _Shape[0]; ++i)
		{
			const auto Index0 = ((i * _Stride[0]) + _BeginPtr[0]) * _Step[0];
			for (SizeType j = 0; j < _Shape[1]; ++j)
			{
				const auto Index1 = Index0 +
					((j * _Stride[1]) + _BeginPtr[1]) * _Step[1];
				for (SizeType k = 0; k < _Shape[2]; ++k)
				{
					const auto Index2 = Index1 +
						((k * _Stride[2]) + _BeginPtr[2]) * _Step[2];
					for (SizeType l = 0; l < _Shape[3]; ++l)
					{
						const auto Index3 = Index2 +
							((l * _Stride[3]) + _BeginPtr[3]) * _Step[3];
						*(_Dst++) = _Src[Index3];
					}
				}
			}
		}
	}
	else if (NDims == 3)
	{
		for (SizeType i = 0; i < _Shape[0]; ++i)
		{
			const auto Index0 = ((i * _Stride[0]) + _BeginPtr[0]) * _Step[0];
			for (SizeType j = 0; j < _Shape[1]; ++j)
			{
				const auto Index1 = Index0 +
					((j * _Stride[1]) + _BeginPtr[1]) * _Step[1];
				for (SizeType k = 0; k < _Shape[2]; ++k)
				{
					const auto Index2 = Index1 +
						((k * _Stride[2]) + _BeginPtr[2]) * _Step[2];
					*(_Dst++) = _Src[Index2];
				}
			}
		}
	}
	else if (NDims == 2)
	{
		for (SizeType i = 0; i < _Shape[0]; ++i)
		{
			const auto Index0 = ((i * _Stride[0]) + _BeginPtr[0]) * _Step[0];
			for (SizeType j = 0; j < _Shape[1]; ++j)
			{
				const auto Index1 = Index0 +
					((j * _Stride[1]) + _BeginPtr[1]) * _Step[1];
				*(_Dst++) = _Src[Index1];
			}
		}
	}
	else if (NDims == 1)
	{
		for (SizeType i = 0; i < _Shape[0]; ++i)
			*(_Dst++) = _Src[((i * _Stride[0]) + _BeginPtr[0]) * _Step[0]];
	}
}

template<typename _Type>
Vector<_Type> VectorArangeImpl(SizeType Begin, SizeType End, _Type Div = (_Type)(1.))
{
	Vector<_Type> Ret;
	Ret.reserve(End);
	while (Begin != End)
		Ret.emplace_back((_Type)(Begin++) / Div);
	return Ret;
}

template<typename _Type>
bool Greater(const _Type& _A, const _Type& _B)
{
	return _A > _B;
}

template<typename _Type>
bool Less(const _Type& _A, const _Type& _B)
{
	return _A < _B;
}

template<typename _Type>
bool Equal(const _Type& _A, const _Type& _B)
{
	if constexpr (std::is_floating_point_v<_Type>)
		return fabs(double(_A) - double(_B)) < 1e-6;
	return _A == _B;
}

template<typename _Type>
bool GreaterEqual(const _Type& _A, const _Type& _B)
{
	return _A >= _B;
}

template<typename _Type>
bool LessEqual(const _Type& _A, const _Type& _B)
{
	return _A <= _B;
}

template<typename _Type>
bool NotEqual(const _Type& _A, const _Type& _B)
{
	return !Equal(_A, _B);
}

template<typename _1Ty, typename _2Ty>
void CastFrom(
	const Tensor& _InputA,
	const Tensor& _InputB,
	const SizeType CurDims,
	const SliceOptions& _SlicePos
)
{
	_1Ty* DataPtr1 = (_1Ty*)_InputA.Data();
	const _2Ty* DataPtr2 = (_2Ty*)_InputB.Data();

	if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous(_SlicePos) && _InputB.IsContinuous(_SlicePos))
	{
		DataPtr1 = (_1Ty*)_InputA.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_2Ty*)_InputB.Data(GetBeginIndices(_SlicePos));
		const size_t BufferSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + BufferSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = (_1Ty)(*(DataPtr2++));
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6], __BINDEX[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _InputA.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _InputA.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _InputA.Strides().data();
		const SizeType* __restrict __StepPtr2 = _InputB.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _InputB.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _InputB.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _InputA.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_1Ty);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_2Ty);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							DataPtr1[IndexAxis5A] = (_1Ty)DataPtr2[IndexAxis5B];
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MonoOperators(
	const Tensor& _Dst,
	const Tensor& _Src,
	const SizeType CurDims,
	const SliceOptions& _SlicePos,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src.Data();

	if (!_Dst.IsBroadCasted() && !_Src.IsBroadCasted() && _Dst.IsContinuous(_SlicePos) && _Src.IsContinuous(_SlicePos))
	{
		DataPtr1 = (_Type*)_Dst.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_Type*)_Src.Data(GetBeginIndices(_SlicePos));
		const size_t DataSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++));
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6], __BINDEX[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __StepPtr2 = _Src.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							DataPtr1[IndexAxis5A] = (_Type)_Func(DataPtr2[IndexAxis5B]);
						}
					}
				}
			}
		}
	}
}

template<typename _Type>
void TensorLoopIterator(
	const Tensor& _Src,
	const SliceOptions& _SlicePos,
	void (*_Func)(_Type*, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType)
)
{
	if (!_Func)
		DragonianLibFatalError;

	const SizeType CurDims = _Src.DimCount();
	_Type* DataPtr2 = (_Type*)_Src.Data();

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6], __BINDEX[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr2 = _Src.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Src.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							_Func(DataPtr2 + IndexAxis5B, i, j, k, l, m, n);
						}
					}
				}
			}
		}
	}
}

template<typename _Type>
void TensorLoopIterator(
	const Tensor& _Src,
	void (*_Func)(_Type*, SizeType, SizeType, SizeType, SizeType, SizeType, SizeType)
)
{
	TensorLoopIterator(_Src, _Src.GetDefaultSliceVector(), _Func);
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MultiOperators(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const Tensor& _Src2,
	const SizeType CurDims,
	const SliceOptions& _SlicePos,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();
	const _Type* DataPtr3 = (_Type*)_Src2.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && !_Src2.IsBroadCasted() && _Dst.IsContinuous(_SlicePos) && _Src1.IsContinuous(_SlicePos) && _Src2.IsContinuous(_SlicePos))
	{
		DataPtr1 = (_Type*)_Dst.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_Type*)_Src1.Data(GetBeginIndices(_SlicePos));
		DataPtr3 = (_Type*)_Src2.Data(GetBeginIndices(_SlicePos));
		const size_t DataSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataPtr3, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), *(DataPtr3++));
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __BINDEX[6],
	__STEP2[6], __BEGIN2[6], __STRIDE2[6],
	__STEP3[6], __BEGIN3[6], __STRIDE3[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __StepPtr2 = _Src1.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src1.Strides().data();
		const SizeType* __restrict __StepPtr3 = _Src2.StepsBack().data();
		const SizeType* __restrict __BeginsPtr3 = _Src2.SliceBegins().data();
		const SizeType* __restrict __StridesPtr3 = _Src2.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__STEP3[i] = 1;
			__BEGIN3[i] = 0;
			__STRIDE3[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__STEP3[i] = __StepPtr3[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN3[i] = __BeginsPtr3[ContPtr[CurIndex]];
			__STRIDE3[i] = __StridesPtr3[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict StepPtr3 = __STEP3;
	const SizeType* __restrict BeginsPtr3 = __BEGIN3;
	const SizeType* __restrict StridesPtr3 = __STRIDE3;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		const auto IndexAxis0C = ((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			const auto IndexAxis1C = IndexAxis0C +
				((j * StridesPtr3[1]) + BeginsPtr3[1]) * StepPtr3[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				const auto IndexAxis2C = IndexAxis1C +
					((k * StridesPtr3[2]) + BeginsPtr3[2]) * StepPtr3[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					const auto IndexAxis3C = IndexAxis2C +
						((l * StridesPtr3[3]) + BeginsPtr3[3]) * StepPtr3[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						const auto IndexAxis4C = IndexAxis3C +
							((m * StridesPtr3[4]) + BeginsPtr3[4]) * StepPtr3[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							const auto IndexAxis5C = IndexAxis4C +
								((n * StridesPtr3[5]) + BeginsPtr3[5]) * StepPtr3[5];
							DataPtr1[IndexAxis5A] = (_Type)_Func(DataPtr2[IndexAxis5B], DataPtr3[IndexAxis5C]);
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MultiOperatorsScalar(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const _Type& _Src2,
	const SizeType CurDims,
	const SliceOptions& _SlicePos,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && _Dst.IsContinuous(_SlicePos) && _Src1.IsContinuous(_SlicePos))
	{
		DataPtr1 = (_Type*)_Dst.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_Type*)_Src1.Data(GetBeginIndices(_SlicePos));
		const size_t DataSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, _Src2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), _Src2);
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6], __BINDEX[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __StepPtr2 = _Src1.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src1.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							DataPtr1[IndexAxis5A] = (_Type)_Func(DataPtr2[IndexAxis5B], _Src2);
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _Fn>
void CompareOperators(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const Tensor& _Src2,
	const SizeType CurDims,
	const SliceOptions& _SlicePos,
	_Fn _Func = nullptr
)
{
	bool* DataPtr1 = (bool*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();
	const _Type* DataPtr3 = (_Type*)_Src2.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && !_Src2.IsBroadCasted() && _Dst.IsContinuous(_SlicePos) && _Src1.IsContinuous(_SlicePos) && _Src2.IsContinuous(_SlicePos))
	{
		DataPtr1 = (bool*)_Dst.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_Type*)_Src1.Data(GetBeginIndices(_SlicePos));
		DataPtr3 = (_Type*)_Src2.Data(GetBeginIndices(_SlicePos));
		const size_t DataSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + DataSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = _Func(*(DataPtr2++), *(DataPtr3++));
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __BINDEX[6],
	__STEP2[6], __BEGIN2[6], __STRIDE2[6],
	__STEP3[6], __BEGIN3[6], __STRIDE3[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __StepPtr2 = _Src1.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src1.Strides().data();
		const SizeType* __restrict __StepPtr3 = _Src2.StepsBack().data();
		const SizeType* __restrict __BeginsPtr3 = _Src2.SliceBegins().data();
		const SizeType* __restrict __StridesPtr3 = _Src2.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__STEP3[i] = 1;
			__BEGIN3[i] = 0;
			__STRIDE3[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__STEP3[i] = __StepPtr3[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN3[i] = __BeginsPtr3[ContPtr[CurIndex]];
			__STRIDE3[i] = __StridesPtr3[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict StepPtr3 = __STEP3;
	const SizeType* __restrict BeginsPtr3 = __BEGIN3;
	const SizeType* __restrict StridesPtr3 = __STRIDE3;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		const auto IndexAxis0C = ((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			const auto IndexAxis1C = IndexAxis0C +
				((j * StridesPtr3[1]) + BeginsPtr3[1]) * StepPtr3[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				const auto IndexAxis2C = IndexAxis1C +
					((k * StridesPtr3[2]) + BeginsPtr3[2]) * StepPtr3[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					const auto IndexAxis3C = IndexAxis2C +
						((l * StridesPtr3[3]) + BeginsPtr3[3]) * StepPtr3[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						const auto IndexAxis4C = IndexAxis3C +
							((m * StridesPtr3[4]) + BeginsPtr3[4]) * StepPtr3[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							const auto IndexAxis5C = IndexAxis4C +
								((n * StridesPtr3[5]) + BeginsPtr3[5]) * StepPtr3[5];
							DataPtr1[IndexAxis5A] = _Func(DataPtr2[IndexAxis5B], DataPtr3[IndexAxis5C]);
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _Fn>
void CompareOperatorsScalar(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const _Type& _Src2,
	const SizeType CurDims,
	const SliceOptions& _SlicePos,
	_Fn _Func = nullptr
)
{
	bool* DataPtr1 = (bool*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && _Dst.IsContinuous(_SlicePos) && _Src1.IsContinuous(_SlicePos))
	{
		DataPtr1 = (bool*)_Dst.Data(GetBeginIndices(_SlicePos));
		DataPtr2 = (_Type*)_Src1.Data(GetBeginIndices(_SlicePos));
		const size_t DataSize = VectorMul(_SlicePos);
		const auto DataEnd = DataPtr1 + DataSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = _Func(*(DataPtr2++), _Src2);
		return;
	}

	if (CurDims > 6)
		DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6], __BINDEX[6];

	{
		const Range* __restrict __ShapePtr = _SlicePos.data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __StepPtr2 = _Src1.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src1.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			__BINDEX[i] = 0;
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(bool);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;
	const SizeType* __restrict IndexBeginPtr = __BINDEX;

	for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							DataPtr1[IndexAxis5A] = _Func(DataPtr2[IndexAxis5B], _Src2);
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _FnType>
void PtrOperatorSingle(
	const Tensor& _Dst,
	const Tensor& _Src,
	SizeType _TotalDim,
	const SizeType _BackDim,
	bool _EnableTranspose,
	_FnType _Func
)
{
	using _OperatorFnPtr = void(*)(_Type*, const SizeType* __restrict, bool, _Type*, const SizeType* __restrict, bool);
	using _OperatorFnTy = void(_Type*, const SizeType* __restrict, bool, _Type*, const SizeType* __restrict, bool);
	using _OperatorFnStd = std::function<_OperatorFnTy>;
	if constexpr (
		std::is_same_v<_FnType, std::function<_Type(_Type)>> ||
		std::is_same_v<_FnType, _Type(*)(_Type)> ||
		std::is_same_v<_FnType, _Type(_Type)>
		) 
	{
		if (_BackDim == 0)
			MonoOperators<_Type>(_Dst, _Src, _TotalDim, _Func, nullptr);
		else
			DragonianLibThrow("Back Dim Must Be Zero!");
	}
	else if constexpr (
		std::is_same_v<_FnType, _OperatorFnPtr> ||
		std::is_same_v<_FnType, _OperatorFnTy> ||
		std::is_same_v<_FnType, _OperatorFnStd>
		)
	{
		bool DstIsTransposed = _EnableTranspose && _Dst.IsTransposed(sizeof(_Type));
		bool SrcIsTransposed = _EnableTranspose && _Src.IsTransposed(sizeof(_Type));

		_TotalDim -= _BackDim;

		if ((!_Dst.IsContinuous(_TotalDim) && !DstIsTransposed) ||
			(!_Src.IsContinuous(_TotalDim) && !SrcIsTransposed))
			DragonianLibThrow("Input Of This Operator Must Be Continuous!");

		_Type* DataPtrDst = (_Type*)_Dst.Data();
		_Type* DataPtrSrc = (_Type*)_Src.Data();

		if (_TotalDim > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

		{
			const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
			const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
			const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
			const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
			const SizeType* __restrict __StepPtr2 = _Src.StepsBack().data();
			const SizeType* __restrict __BeginsPtr2 = _Src.SliceBegins().data();
			const SizeType* __restrict __StridesPtr2 = _Src.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - _TotalDim;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP1[i] = 1;
				__BEGIN1[i] = 0;
				__STRIDE1[i] = 1;
				__STEP2[i] = 1;
				__BEGIN2[i] = 0;
				__STRIDE2[i] = 1;
				++i;
			}
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[CurIndex];
				__STEP1[i] = __StepPtr1[CurIndex] / (SizeType)sizeof(_Type);
				__BEGIN1[i] = __BeginsPtr1[CurIndex];
				__STRIDE1[i] = __StridesPtr1[CurIndex];
				__STEP2[i] = __StepPtr2[CurIndex] / (SizeType)sizeof(_Type);
				__BEGIN2[i] = __BeginsPtr2[CurIndex];
				__STRIDE2[i] = __StridesPtr2[CurIndex];
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr1 = __STEP1;
		const SizeType* __restrict BeginsPtr1 = __BEGIN1;
		const SizeType* __restrict StridesPtr1 = __STRIDE1;
		const SizeType* __restrict StepPtr2 = __STEP2;
		const SizeType* __restrict BeginsPtr2 = __BEGIN2;
		const SizeType* __restrict StridesPtr2 = __STRIDE2;

		auto StepsDst = _Dst.StepsBack();
		for (auto& i : StepsDst)
			i /= sizeof(_Type);
		auto StepsSrc = _Src.StepsBack();
		for (auto& i : StepsSrc)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtrDst = _Dst.Shape().data();
		const SizeType* __restrict ShapePtrSrc = _Src.Shape().data();
		const SizeType* __restrict StepPtrDst = StepsDst.data();
		const SizeType* __restrict StepPtrSrc = StepsSrc.data();
		const SizeType* __restrict BeginsPtrDst = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtrSrc = _Src.SliceBegins().data();

		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
			const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
			for (SizeType j = 0; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
				for (SizeType k = 0; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
					for (SizeType l = 0; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
						for (SizeType m = 0; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5A = IndexAxis4A +
									((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
								const auto IndexAxis5B = IndexAxis4B +
									((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
								_Func(
									DataPtrDst + IndexAxis5A + BeginsPtrDst[_TotalDim] * StepPtrDst[_TotalDim],
									ShapePtrDst + _TotalDim,
									DstIsTransposed,
									DataPtrSrc + IndexAxis5B + BeginsPtrSrc[_TotalDim] * StepPtrSrc[_TotalDim],
									ShapePtrSrc + _TotalDim,
									SrcIsTransposed
								);
								//DataPtr1[IndexAxis5A] = (_Type)_Func(DataPtr2[IndexAxis5B]);
							}
						}
					}
				}
			}
		}
	}
	else
		DragonianLibThrow("Function Type Error!");
}

//using _Type = float;
template<typename _Type>
void LinearInterpolate(
	_Type* _DstPtr,
	const SizeType* __restrict _DstShape,
	bool _DstT,
	_Type* _SrcPtr,
	const SizeType* __restrict _SrcShape,
	bool _SrcT
)
{
	if (_DstT || _SrcT)
		DragonianLibThrow("Input Must Be Continuous!");
	SizeType SrcCount = _SrcShape[0], DstCount = _DstShape[0];

	DragonianLibSTL::Vector<float> Xi;
	if (SrcCount == 1)
		Xi = DragonianLibSTL::Vector(DstCount, 0.f);
	else
		Xi = DragonianLibSTL::Arange(0.f, float(SrcCount - 1), float(SrcCount - 1) / (float)DstCount);

	auto Y0 = _SrcPtr, Yi = _DstPtr;
	size_t Index = 0;
	for (auto i : Xi)
	{
		auto Front = (size_t)ceil(i);
		auto Back = (size_t)floor(i);
		auto Offset = i - (float)Front;
		Yi[Index++] = _Type(Offset * (float)(Y0[Back] - Y0[Front]) + (float)Y0[Front]);
	}
}

template<typename _Type>
void NearestInterpolate1D(
	_Type* _DstPtr,
	const SizeType* __restrict _DstShape,
	bool _DstT,
	_Type* _SrcPtr,
	const SizeType* __restrict _SrcShape,
	bool _SrcT
)
{
	if (_DstT || _SrcT)
		DragonianLibThrow("Input Must Be Continuous!");
	SizeType SrcCount = _SrcShape[0], DstCount = _DstShape[0];

	DragonianLibSTL::Vector<float> Xi;
	if (SrcCount == 1)
		Xi = DragonianLibSTL::Vector(DstCount, 0.f);
	else
		Xi = DragonianLibSTL::Arange(0.f, float(SrcCount - 1), float(SrcCount - 1) / (float)DstCount);

	auto Y0 = _SrcPtr, Yi = _DstPtr;
	size_t Index = 0;
	for (auto i : Xi)
	{
		auto Offset = size_t(round(i));
		Yi[Index++] = Y0[Offset];
	}
}

template<typename _Type>
void NearestInterpolate2D(
	_Type* _DstPtr,
	const SizeType* __restrict _DstShape,
	bool _DstT,
	_Type* _SrcPtr,
	const SizeType* __restrict _SrcShape,
	bool _SrcT
)
{
	SizeType SrcCountHeight = _SrcShape[0], DstCountHeight = _DstShape[0],
		SrcCountWidth = _SrcShape[1], DstCountWidth = _DstShape[1],
		SStrideH = _SrcT ? 1 : _SrcShape[1], SStrideW = _SrcT ? _SrcShape[0] : 1,
		DStrideH = _DstT ? 1 : _DstShape[1], DStrideW = _DstT ? _DstShape[0] : 1;


	DragonianLibSTL::Vector<float> XiH, XiW;
	if (SrcCountHeight == 1)
		XiH = DragonianLibSTL::Vector(DstCountHeight, 0.f);
	else
		XiH = DragonianLibSTL::Arange(0.f, float(SrcCountHeight - 1), float(SrcCountHeight - 1) / (float)DstCountHeight);
	if (SrcCountWidth == 1)
		XiW = DragonianLibSTL::Vector(DstCountWidth, 0.f);
	else
		XiW = DragonianLibSTL::Arange(0.f, float(SrcCountWidth - 1), float(SrcCountWidth - 1) / (float)DstCountWidth);

	auto Y0 = _SrcPtr, Yi = _DstPtr;
	size_t IndexH = 0;
	for (auto i : XiH)
	{
		size_t IndexW = 0;
		auto OffsetH = size_t(round(i)) * SStrideH;
		for(auto j : XiW)
		{
			auto OffsetW = size_t(round(j)) * SStrideW;
			Yi[IndexH + IndexW] = Y0[OffsetH + OffsetW];
			IndexW += DStrideW;
		}
		IndexH += DStrideH;
	}
}

void PtrOperatorMono();

DragonianLibSpaceEnd