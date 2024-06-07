#pragma once
#include "Base.h"
LibSvcBegin
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
		LibSvcMemCpy(_Dst, _Src, TotalSize);
		return;
	}

	if (NDims > 5)
	{
		Vector<SizeType> CurIndice(NDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

template <typename ThisType, typename Tensor, typename _InterpFn>
void InterpImpl(const Tensor& _Dst, const Tensor& _Src, SizeType CurDims, const SizeType _InterpDim, _InterpFn _Fn)
{
	ThisType* DataPtr1 = (ThisType*)_Dst.Data();
	const ThisType* DataPtr2 = (ThisType*)_Src.Data();

	if (CurDims > 6)
	{
		CurDims -= _InterpDim;
		auto Steps1 = _Dst.StepsBack();
		for (auto& i : Steps1)
			i /= sizeof(ThisType);
		auto Steps2 = _Src.StepsBack();
		for (auto& i : Steps2)
			i /= sizeof(ThisType);
		const SizeType* __restrict ShapePtr1 = _Dst.Shape().data();
		const SizeType* __restrict ShapePtr2 = _Src.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _Src.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict StridesPtr2 = _Src.Strides().data();
		Vector<SizeType> CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr1,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				_Fn(
					DataPtr1 + Index1,
					ShapePtr1 + CurDims,
					StridesPtr1 + CurDims,
					BeginsPtr1 + CurDims,
					StepPtr1 + CurDims,
					DataPtr2 + Index2,
					ShapePtr2 + CurDims,
					StridesPtr2 + CurDims,
					BeginsPtr2 + CurDims,
					StepPtr2 + CurDims
				);
			}
		);
		return;
	}

	CurDims = 6 - _InterpDim;

	SizeType __SHAPE1[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __SHAPE2[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

	{
		const SizeType* __restrict __ShapePtr1 = _Dst.Shape().data();
		const SizeType* __restrict __StepPtr1 = _Dst.StepsBack().data();
		const SizeType* __restrict __BeginsPtr1 = _Dst.SliceBegins().data();
		const SizeType* __restrict __StridesPtr1 = _Dst.Strides().data();
		const SizeType* __restrict __ShapePtr2 = _Dst.Shape().data();
		const SizeType* __restrict __StepPtr2 = _Src.StepsBack().data();
		const SizeType* __restrict __BeginsPtr2 = _Src.SliceBegins().data();
		const SizeType* __restrict __StridesPtr2 = _Src.Strides().data();
		SizeType i = 0;
		SizeType Count = 6 - CurDims;
		while (i < Count)
		{
			__SHAPE1[i] = 1;
			__STEP1[i] = 1;
			__BEGIN1[i] = 0;
			__STRIDE1[i] = 1;
			__SHAPE2[i] = 1;
			__STEP2[i] = 1;
			__BEGIN2[i] = 0;
			__STRIDE2[i] = 1;
			++i;
		}
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			__SHAPE1[i] = __ShapePtr1[CurIndex];
			__STEP1[i] = __StepPtr1[CurIndex] / (SizeType)sizeof(ThisType);
			__BEGIN1[i] = __BeginsPtr1[CurIndex];
			__STRIDE1[i] = __StridesPtr1[CurIndex];
			__SHAPE2[i] = __ShapePtr2[CurIndex];
			__STEP2[i] = __StepPtr2[CurIndex] / (SizeType)sizeof(ThisType);
			__BEGIN2[i] = __BeginsPtr2[CurIndex];
			__STRIDE2[i] = __StridesPtr2[CurIndex];
		}
	}

	const SizeType* __restrict ShapePtr1 = __SHAPE1;
	const SizeType* __restrict StepPtr1 = __STEP1;
	const SizeType* __restrict BeginsPtr1 = __BEGIN1;
	const SizeType* __restrict StridesPtr1 = __STRIDE1;
	const SizeType* __restrict ShapePtr2 = __SHAPE2;
	const SizeType* __restrict StepPtr2 = __STEP2;
	const SizeType* __restrict BeginsPtr2 = __BEGIN2;
	const SizeType* __restrict StridesPtr2 = __STRIDE2;

	for (SizeType i = 0; i < ShapePtr1[0]; ++i)
	{
		const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
		const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
		for (SizeType j = 0; j < ShapePtr1[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1];
			for (SizeType k = 0; k < ShapePtr1[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2];
				for (SizeType l = 0; l < ShapePtr1[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3];
					for (SizeType m = 0; m < ShapePtr1[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * StridesPtr1[4]) + BeginsPtr1[4]) * StepPtr1[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * StridesPtr2[4]) + BeginsPtr2[4]) * StepPtr2[4];
						for (SizeType n = 0; n < ShapePtr1[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * StridesPtr1[5]) + BeginsPtr1[5]) * StepPtr1[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
							_Fn(
								DataPtr1 + IndexAxis5A,
								ShapePtr1 + CurDims,
								StridesPtr1 + CurDims,
								BeginsPtr1 + CurDims,
								StepPtr1 + CurDims,
								DataPtr2 + IndexAxis5B,
								ShapePtr2 + CurDims,
								StridesPtr2 + CurDims,
								BeginsPtr2 + CurDims,
								StepPtr2 + CurDims
							);
						}
					}
				}
			}
		}
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

inline void histc(const double* x, int x_length, const double* edges,
	int edges_length, int* index) {
	int count = 1;

	int i = 0;
	for (; i < edges_length; ++i) {
		index[i] = 1;
		if (edges[i] >= x[0]) break;
	}
	for (; i < edges_length; ++i) {
		if (edges[i] < x[count]) {
			index[i] = count;
		}
		else {
			index[i--] = count++;
		}
		if (count == x_length) break;
	}
	count--;
	for (i++; i < edges_length; ++i) index[i] = count;
}

inline void interp1(
	const double* x, const double* y, int x_length, const double* xi,
	int xi_length, double* yi) {
	double* h = new double[x_length - 1];
	int* k = new int[xi_length];

	for (int i = 0; i < x_length - 1; ++i) h[i] = x[i + 1] - x[i];
	for (int i = 0; i < xi_length; ++i) {
		k[i] = 0;
	}

	histc(x, x_length, xi, xi_length, k);

	for (int i = 0; i < xi_length; ++i) {
		double s = (xi[i] - x[k[i] - 1]) / h[k[i] - 1];
		yi[i] = y[k[i] - 1] + s * (y[k[i]] - y[k[i] - 1]);
	}

	delete[] k;
	delete[] h;
}

template<typename _Type>
void Linear1DImpl(
	_Type* _Yi,
	const SizeType* __restrict _XiShape,
	const SizeType* __restrict _XiStride,
	const SizeType* __restrict _XiBeginPtr,
	const SizeType* __restrict _XiStep,
	const _Type* __restrict _Y0,
	const SizeType* __restrict _X0Shape,
	const SizeType* __restrict _X0Stride,
	const SizeType* __restrict _X0BeginPtr,
	const SizeType* __restrict _X0Step
)
{
	const auto XiVec = VectorArangeImpl(0ll, _XiShape[0], (float)_XiShape[0] / (float)_X0Shape[0]);
	const float* __restrict _Xi = XiVec.data();

	auto x_length = _X0Shape[0], xi_length = _XiShape[0];

	double* h = new double[x_length - 1];
	SizeType* k = new SizeType[xi_length];

	for (int i = 0; i < x_length - 1; ++i) h[i] = x[i + 1] - x[i];
	for (int i = 0; i < xi_length; ++i) k[i] = 0;
}

//using _Type = float;
LibSvcEnd
