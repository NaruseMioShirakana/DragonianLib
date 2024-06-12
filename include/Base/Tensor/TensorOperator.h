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
void CastFrom(const Tensor& _InputA, const Tensor& _InputB, const SizeType CurDims)
{
	_1Ty* DataPtr1 = (_1Ty*)_InputA.Data();
	const _2Ty* DataPtr2 = (_2Ty*)_InputB.Data();

	if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
	{
		DataPtr1 = (_1Ty*)_InputA.GetPtr();
		DataPtr2 = (_2Ty*)_InputB.GetPtr();
		const size_t BufferSize = VectorMul(_InputA.Shape());
		const auto DataEnd = DataPtr1 + BufferSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = (_1Ty)(*(DataPtr2++));
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _InputA.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(_1Ty);
		auto Steps2 = _InputB.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_2Ty);
		const SizeType* __restrict ShapePtr = _InputA.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _InputA.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _InputB.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _InputA.Strides().data();
		const SizeType* __restrict StridesPtr2 = _InputB.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_1Ty)DataPtr2[Index2];
			}
		);
		return;
	}

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

	{
		const SizeType* __restrict __ShapePtr = _InputA.Shape().data();
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
			++i;
		}
		const auto Cont = _InputA.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_1Ty);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_2Ty);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;

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
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src.Data();

	if (!_Dst.IsBroadCasted() && !_Src.IsBroadCasted() && _Dst.IsContinuous() && _Src.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++));
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(_Type);
		auto Steps2 = _Src.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtr = _Dst.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2]);
			}
		);
		return;
	}

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
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;

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
							DataPtr1[IndexAxis5A] = (_Type)_Func(DataPtr2[IndexAxis5B]);
						}
					}
				}
			}
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MultiOperators(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const Tensor& _Src2,
	const SizeType CurDims,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();
	const _Type* DataPtr3 = (_Type*)_Src2.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && !_Src2.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous() && _Src2.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		DataPtr3 = (_Type*)_Src2.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataPtr3, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), *(DataPtr3++));
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(_Type);
		auto Steps2 = _Src1.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_Type);
		auto Steps3 = _Src2.StepsBack();
		for (auto& i : Steps3)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtr = _Dst.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict StepPtr3 = Steps3.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict BeginsPtr3 = _Src2.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();
		const SizeType* __restrict StridesPtr3 = _Src2.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				SizeType Index3 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
					Index3 += ((IndicesPtr[i] * StridesPtr3[i]) + BeginsPtr3[i]) * StepPtr3[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2], DataPtr3[Index3]);
			}
		);

		return;
	}

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6],
	__STEP2[6], __BEGIN2[6], __STRIDE2[6],
	__STEP3[6], __BEGIN3[6], __STRIDE3[6];

	{
		const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
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
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__STEP3[i] = __StepPtr3[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN3[i] = __BeginsPtr3[ContPtr[CurIndex]];
			__STRIDE3[i] = __StridesPtr3[ContPtr[CurIndex]];
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

	for (SizeType i = 0; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		const auto IndexAxis0C = ((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0];
		for (SizeType j = 0; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			const auto IndexAxis1C = IndexAxis0C +
				((j * StridesPtr3[1]) + BeginsPtr3[1]) * StepPtr3[1];
			for (SizeType k = 0; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				const auto IndexAxis2C = IndexAxis1C +
					((k * StridesPtr3[2]) + BeginsPtr3[2]) * StepPtr3[2];
				for (SizeType l = 0; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					const auto IndexAxis3C = IndexAxis2C +
						((l * StridesPtr3[3]) + BeginsPtr3[3]) * StepPtr3[3];
					for (SizeType m = 0; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						const auto IndexAxis4C = IndexAxis3C +
							((m * StridesPtr3[4]) + BeginsPtr3[4]) * StepPtr3[4];
						for (SizeType n = 0; n < ShapePtr[5]; ++n)
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
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, _Src2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), _Src2);
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(_Type);
		auto Steps2 = _Src1.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtr = _Dst.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2], _Src2);
			}
		);
		return;
	}

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

	{
		const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
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
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;

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
	_Fn _Func = nullptr
)
{
	bool* DataPtr1 = (bool*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();
	const _Type* DataPtr3 = (_Type*)_Src2.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && !_Src2.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous() && _Src2.IsContinuous())
	{
		DataPtr1 = (bool*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		DataPtr3 = (_Type*)_Src2.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = _Func(*(DataPtr2++), *(DataPtr3++));
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(bool);
		auto Steps2 = _Src1.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_Type);
		auto Steps3 = _Src2.StepsBack();
		for (auto& i : Steps3)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtr = _Dst.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict StepPtr3 = Steps3.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict BeginsPtr3 = _Src2.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();
		const SizeType* __restrict StridesPtr3 = _Src2.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				SizeType Index3 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
					Index3 += ((IndicesPtr[i] * StridesPtr3[i]) + BeginsPtr3[i]) * StepPtr3[i];
				}
				DataPtr1[Index1] = _Func(DataPtr2[Index2], DataPtr3[Index3]);
			}
		);

		return;
	}

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6],
		__STEP2[6], __BEGIN2[6], __STRIDE2[6],
		__STEP3[6], __BEGIN3[6], __STRIDE3[6];

	{
		const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
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
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(bool);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
			__STEP3[i] = __StepPtr3[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN3[i] = __BeginsPtr3[ContPtr[CurIndex]];
			__STRIDE3[i] = __StridesPtr3[ContPtr[CurIndex]];
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

	for (SizeType i = 0; i < ShapePtr[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		const auto IndexAxis0C = ((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0];
		for (SizeType j = 0; j < ShapePtr[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			const auto IndexAxis1C = IndexAxis0C +
				((j * StridesPtr3[1]) + BeginsPtr3[1]) * StepPtr3[1];
			for (SizeType k = 0; k < ShapePtr[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				const auto IndexAxis2C = IndexAxis1C +
					((k * StridesPtr3[2]) + BeginsPtr3[2]) * StepPtr3[2];
				for (SizeType l = 0; l < ShapePtr[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					const auto IndexAxis3C = IndexAxis2C +
						((l * StridesPtr3[3]) + BeginsPtr3[3]) * StepPtr3[3];
					for (SizeType m = 0; m < ShapePtr[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						const auto IndexAxis4C = IndexAxis3C +
							((m * StridesPtr3[4]) + BeginsPtr3[4]) * StepPtr3[4];
						for (SizeType n = 0; n < ShapePtr[5]; ++n)
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
	_Fn _Func = nullptr
)
{
	bool* DataPtr1 = (bool*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous())
	{
		DataPtr1 = (bool*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = _Func(*(DataPtr2++), _Src2);
		return;
	}

	if (CurDims > 6)
	{
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(bool);
		auto Steps2 = _Src1.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(_Type);
		const SizeType* __restrict ShapePtr = _Dst.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = _Func(DataPtr2[Index2], _Src2);
			}
		);
		return;
	}

	SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

	{
		const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
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
			++i;
		}
		const auto Cont = _Dst.CalcContinuous();
		const SizeType* __restrict ContPtr = Cont.data();
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
			__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(bool);
			__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
			__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
			__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(_Type);
			__BEGIN2[i] = __BeginsPtr2[ContPtr[CurIndex]];
			__STRIDE2[i] = __StridesPtr2[ContPtr[CurIndex]];
		}
	}

	const SizeType* __restrict ShapePtr = __SHAPE;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;

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
							DataPtr1[IndexAxis5A] = _Func(DataPtr2[IndexAxis5B], _Src2);
						}
					}
				}
			}
		}
	}
}

template<typename _Type>
void PtrOperatorSingle(
	const Tensor& _Dst,
	const Tensor& _Src,
	SizeType _TotalDim,
	const SizeType _BackDim,
	bool _EnableTranspose,
	void* _FnPtr
)
{
	if(_BackDim == 0)
	{
		using _OperatorFn = _Type(*)(_Type);
		MonoOperators<_Type, _OperatorFn>(_Dst, _Src, _TotalDim, (_OperatorFn)_FnPtr, nullptr);
		return;
	}

	using _OperatorFn = _Type(*)(
		_Type*,
		const SizeType* __restrict,
		bool,
		_Type*,
		const SizeType* __restrict,
		bool
		);

	_OperatorFn _Fn = (_OperatorFn)_FnPtr;

	bool DstIsTransposed = _EnableTranspose && _Dst.IsTransposed(sizeof(_Type));
	bool SrcIsTransposed = _EnableTranspose && _Src.IsTransposed(sizeof(_Type));

	_TotalDim -= _BackDim;

	if ((!_Dst.IsContinuous(_TotalDim) && !DstIsTransposed) ||
		(!_Src.IsContinuous(_TotalDim) && !SrcIsTransposed))
		DragonianLibThrow("Input Of This Operator Must Be Continuous!");

	_Type* DataPtrDst = (_Type*)_Dst.Data();
	_Type* DataPtrSrc = (_Type*)_Src.Data();

	

	if (_TotalDim > 6)
	{
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
		const SizeType* __restrict StridesPtrDst = _Dst.Strides().data();
		const SizeType* __restrict StridesPtrSrc = _Src.Strides().data();
		Vector<SizeType> CurIndice(_TotalDim, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		DragonianLibCycle(
			IndicesPtr,
			ShapePtrDst,
			_TotalDim,
			{
				SizeType IndexDst = 0;
				SizeType IndexSrc = 0;
				for (SizeType i = 0; i < _TotalDim; ++i)
				{
					IndexDst += ((IndicesPtr[i] * StridesPtrDst[i]) + BeginsPtrDst[i]) * StepPtrDst[i];
					IndexSrc += ((IndicesPtr[i] * StridesPtrSrc[i]) + BeginsPtrSrc[i]) * StepPtrSrc[i];
				}
				_Fn(
					DataPtrDst + IndexDst + BeginsPtrDst[_TotalDim] * StepPtrDst[_TotalDim],
					ShapePtrDst + _TotalDim,
					DstIsTransposed,
					DataPtrSrc + IndexSrc + BeginsPtrSrc[_TotalDim] * StepPtrSrc[_TotalDim],
					ShapePtrSrc + _TotalDim,
					SrcIsTransposed
				);
			}
		);
		return;
	}

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
							_Fn(
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

void PtrOperatorMono();

DragonianLibSpaceEnd