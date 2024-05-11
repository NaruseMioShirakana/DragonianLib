#pragma once
#include "Base.h"
LibSvcBegin

template <typename ThisType, typename Tensor, typename _InterpFn>
void InterpImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims, _InterpFn _Fn)
{
	ThisType* DataPtr1 = (ThisType*)_Dst.Data();
	const ThisType* DataPtr2 = (ThisType*)_Src.Data();

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

	if (CurDims > 5)
	{
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

	if (CurDims == 5)
	{
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
							_Fn(
								DataPtr1 + IndexAxis4A,
								ShapePtr1 + CurDims,
								StridesPtr1 + CurDims,
								BeginsPtr1 + CurDims,
								StepPtr1 + CurDims,
								DataPtr2 + IndexAxis4B,
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
	else if (CurDims == 4)
	{
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
						_Fn(
							DataPtr1 + IndexAxis3A,
							ShapePtr1 + CurDims,
							StridesPtr1 + CurDims,
							BeginsPtr1 + CurDims,
							StepPtr1 + CurDims,
							DataPtr2 + IndexAxis3B,
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
	else if (CurDims == 3)
	{
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
					_Fn(
						DataPtr1 + IndexAxis2A,
						ShapePtr1 + CurDims,
						StridesPtr1 + CurDims,
						BeginsPtr1 + CurDims,
						StepPtr1 + CurDims,
						DataPtr2 + IndexAxis2B,
						ShapePtr2 + CurDims,
						StridesPtr2 + CurDims,
						BeginsPtr2 + CurDims,
						StepPtr2 + CurDims
					);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
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
				_Fn(
					DataPtr1 + IndexAxis1A,
					ShapePtr1 + CurDims,
					StridesPtr1 + CurDims,
					BeginsPtr1 + CurDims,
					StepPtr1 + CurDims,
					DataPtr2 + IndexAxis1B,
					ShapePtr2 + CurDims,
					StridesPtr2 + CurDims,
					BeginsPtr2 + CurDims,
					StepPtr2 + CurDims
				);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr1[0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0];
			const auto IndexAxis0B = ((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0];
			_Fn(
				DataPtr1 + IndexAxis0A,
				ShapePtr1 + CurDims,
				StridesPtr1 + CurDims,
				BeginsPtr1 + CurDims,
				StepPtr1 + CurDims,
				DataPtr2 + IndexAxis0B,
				ShapePtr2 + CurDims,
				StridesPtr2 + CurDims,
				BeginsPtr2 + CurDims,
				StepPtr2 + CurDims
			);
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
	const auto XiVec = VectorArangeImpl(0ll, _XiShape[0], (double)_XiShape[0] / (double)_X0Shape[0]);
	const double* __restrict _Xi = XiVec.data();
	for (SizeType i = 0; i < _XiShape[0]; ++i)
	{
		const auto _X0BeginVal = floor(_Xi[i]);
		const auto _X0Begin = (SizeType)_X0BeginVal;
		auto _X0End = (SizeType)ceil(_Xi[i]);
		if (_X0End <= _X0Begin) ++_X0End;
		if (_X0End > _X0Shape[0]) break;
		const auto _Y0IdxB = ((_X0Begin * _X0Stride[0]) + _X0BeginPtr[0]) * _X0Step[0];
		const auto _T0IdxE = ((_X0End * _X0Stride[0]) + _X0BeginPtr[0]) * _X0Step[0];
		_Yi[((i * _XiStride[0]) + _XiBeginPtr[0]) * _XiStep[0]] =
			_Y0[_Y0IdxB] + (_Y0[_T0IdxE] - _Y0[_Y0IdxB]) * (_Xi[i] - _X0BeginVal);
	}
}

//using _Type = float;
LibSvcEnd
