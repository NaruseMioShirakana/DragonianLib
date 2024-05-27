#pragma once
#include "Tensor/Tensor.h"

LibSvcBegin

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
	return _A != _B;
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _InputA.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_1Ty)DataPtr2[IndexAxis4B];
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_1Ty)DataPtr2[IndexAxis3B];
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_1Ty)DataPtr2[IndexAxis2B];
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_1Ty)DataPtr2[IndexAxis1B];
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_1Ty)DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]];
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B]);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B]);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B]);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B]);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]]);
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							const auto IndexAxis4C = IndexAxis3C +
								((m * StridesPtr3[Axis4]) + BeginsPtr3[Axis4]) * StepPtr3[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B], DataPtr3[IndexAxis4C]);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B], DataPtr3[IndexAxis3C]);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B], DataPtr3[IndexAxis2C]);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B], DataPtr3[IndexAxis1C]);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]],
					DataPtr3[((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0]]);
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B], _Src2);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B], _Src2);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B], _Src2);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B], _Src2);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]], _Src2);
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							const auto IndexAxis4C = IndexAxis3C +
								((m * StridesPtr3[Axis4]) + BeginsPtr3[Axis4]) * StepPtr3[Axis4];
							DataPtr1[IndexAxis4A] = _Func(DataPtr2[IndexAxis4B], DataPtr3[IndexAxis4C]);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						DataPtr1[IndexAxis3A] = _Func(DataPtr2[IndexAxis3B], DataPtr3[IndexAxis3C]);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					DataPtr1[IndexAxis2A] = _Func(DataPtr2[IndexAxis2B], DataPtr3[IndexAxis2C]);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				DataPtr1[IndexAxis1A] = _Func(DataPtr2[IndexAxis1B], DataPtr3[IndexAxis1C]);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]],
					DataPtr3[((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0]]);
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

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
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

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; m < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = _Func(DataPtr2[IndexAxis4B], _Src2);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = _Func(DataPtr2[IndexAxis3B], _Src2);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = _Func(DataPtr2[IndexAxis2B], _Src2);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = _Func(DataPtr2[IndexAxis1B], _Src2);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]], _Src2);
		}
	}
}

template<typename _TypeName, typename _FnType, typename _BackEndFnType>
void CustomMultiOperatorImpl(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const Tensor& _Src2,
	const SizeType CurDims,
	_FnType _Fn,
	_BackEndFnType _BackEndFn
)
{
	MultiOperators<_TypeName>(
		_Dst,
		_Src1,
		_Src2,
		CurDims,
		_Fn,
		_BackEndFn
	);
}

template<typename _FnType, typename _BackEndFnType>
static Tensor CustomMultiOperator(
	const Tensor& _A,
	const Tensor& _B,
	_FnType _Fn,
	_BackEndFnType _BackEndFn = nullptr,
	ThreadPool* _ThreadPool = nullptr
)
{
	const auto BroadCast = Tensor::BroadCast(_A, _B);
	Tensor Ret(BroadCast.first.Shape(), BroadCast.first.DType());
	const auto InputA = BroadCast.first.Squeeze();
	const auto InputB = BroadCast.second.Squeeze();
	const auto Result = Ret.Squeeze();

	const auto CurDims = (SizeType)InputA.Shape().size();
	const auto& InputShape = InputA.Shape();
	const auto TotalSize = VectorMul(InputShape);

	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NWorkers = _ThreadPool->GetThreadCount();
		const auto SqueezedDims = (SizeType)InputShape.size();

		Vector<Range> Slices;
		for (SizeType i = 0; i < SqueezedDims; ++i)
		{
			if (InputShape[i] < NWorkers)
				Slices.emplace_back(None);
			else
			{
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers);
				for (SizeType j = 0; ; j += Step)
				{
					const auto End = std::min(j + Step, InputShape[i]);
					if (j >= End)
					{
						_ThreadPool->Join();
						return Ret;
					}
					auto ThreadSlices = Slices;
					ThreadSlices.emplace_back(j, End);

					_ThreadPool->Commit(
						CustomMultiOperatorImpl,
						Result.Slice(ThreadSlices),
						InputA.Slice(ThreadSlices),
						InputB.Slice(ThreadSlices),
						CurDims, _Fn, _BackEndFn
					);

					if (End == InputShape[i])
					{
						_ThreadPool->Join();
						return Ret;
					}
				}
			}
		}
		CustomMultiOperatorImpl(Result, InputA, InputB, CurDims, _Fn, _BackEndFn);
	}
	else
		CustomMultiOperatorImpl(Result, InputA, InputB, CurDims, _Fn, _BackEndFn);

	return Ret;
}

LibSvcEnd