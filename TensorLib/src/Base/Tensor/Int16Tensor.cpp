
#include "Tensor/Int16Tensor.h"

DragonianLibSpaceBegin

namespace Int16
{

	ThisType CastFrom(TensorType _Type, cpvoid _Val)
	{
		ThisType Ret;
		DragonianLibTypeSwitch(
			_Type,
			DragonianLibCastImpl(ThisType, Ret, bool, _Val),
			DragonianLibCastImpl(ThisType, Ret, int8, _Val),
			DragonianLibCastImpl(ThisType, Ret, int16, _Val),
			DragonianLibCastImpl(ThisType, Ret, int32, _Val),
			DragonianLibCastImpl(ThisType, Ret, int64, _Val),
			UNUSED(),
			DragonianLibCastImpl(ThisType, Ret, float32, _Val),
			DragonianLibCastImpl(ThisType, Ret, float64, _Val),
			UNUSED()
		);
		return Ret;
	}

	void AssignImpl(
		const Tensor& _Input,
		cpvoid _Val,
		TensorType _ValType,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		const auto _Value = CastFrom(_ValType, _Val);
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (_Input.IsContinuous(_SlicePos))
		{
			DataPtr = (ThisType*)_Input.Data(GetBeginIndices(_SlicePos));
			const size_t BufferSize = VectorMul(_SlicePos) * sizeof(ThisType);
			DragonianLibMemSet(DataPtr, &_Value, BufferSize, sizeof(ThisType));
			return;
		}

		if (CurDims > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6], __BINDEX[6];

		{
			const Range* __restrict __ShapePtr = _SlicePos.data();
			const SizeType* __restrict __StepPtr = _Input.StepsBack().data();
			const SizeType* __restrict __BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict __StridesPtr = _Input.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP[i] = 1;
				__BEGIN[i] = 0;
				__STRIDE[i] = 1;
				__BINDEX[i] = 0;
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
				__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr = __STEP;
		const SizeType* __restrict BeginsPtr = __BEGIN;
		const SizeType* __restrict StridesPtr = __STRIDE;
		const SizeType* __restrict IndexBeginPtr = __BINDEX;

		for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 +
					((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 +
						((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 +
							((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 +
								((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5 = IndexAxis4 +
									((n * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								DataPtr[IndexAxis5] = _Value;
							}
						}
					}
				}
			}
		}
	}

	void AssignBufferImpl(
		const Tensor& _Input,
		const ThisType* __restrict Buffer,
		const ThisType* __restrict BufferEnd,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	) {
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (BufferEnd < Buffer)
			DragonianLibThrow("[Operator] BufferEnd* < Buffer*, Make Sure BufferEnd* > Buffer*");

		if (_Input.IsContinuous(_SlicePos))
		{
			DataPtr = (ThisType*)_Input.Data(GetBeginIndices(_SlicePos));
			const size_t BufferSize = std::min(VectorMul(_SlicePos), (BufferEnd - Buffer)) * sizeof(ThisType);
			DragonianLibMemCpy(DataPtr, Buffer, BufferSize);
			return;
		}

		if (CurDims > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6], __BINDEX[6];

		{
			const Range* __restrict __ShapePtr = _SlicePos.data();
			const SizeType* __restrict __StepPtr = _Input.StepsBack().data();
			const SizeType* __restrict __BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict __StridesPtr = _Input.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP[i] = 1;
				__BEGIN[i] = 0;
				__STRIDE[i] = 1;
				__BINDEX[i] = 0;
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
				__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr = __STEP;
		const SizeType* __restrict BeginsPtr = __BEGIN;
		const SizeType* __restrict StridesPtr = __STRIDE;
		const SizeType* __restrict IndexBeginPtr = __BINDEX;

		for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 +
					((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 +
						((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 +
							((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 +
								((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5 = IndexAxis4 +
									((n * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								DataPtr[IndexAxis5] = *(Buffer++);
								if (Buffer == BufferEnd)
									return;
							}
						}
					}
				}
			}
		}
	}

	void AssignTensorImpl(
		const Tensor& _InputA,
		const Tensor& _InputB,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		ThisType* DataPtr1 = (ThisType*)_InputA.Data();
		const ThisType* DataPtr2 = (ThisType*)_InputB.Data();

		if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous(_SlicePos) && _InputB.IsContinuous(_SlicePos))
		{
			DataPtr1 = (ThisType*)_InputA.Data(GetBeginIndices(_SlicePos));
			DataPtr2 = (ThisType*)_InputB.Data(GetBeginIndices(_SlicePos));
			const size_t BufferSize = VectorMul(_SlicePos) * sizeof(ThisType);
			DragonianLibMemCpy(DataPtr1, DataPtr2, BufferSize);
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
				__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
				__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
				__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
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
								DataPtr1[IndexAxis5A] = DataPtr2[IndexAxis5B];
							}
						}
					}
				}
			}
		}
	}

	void FixWithRandomImpl(
		const Tensor& _Input,
		uint64 _Seed,
		double _Mean,
		double _Sigma,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		std::mt19937_64 RndDevice(_Seed + std::this_thread::get_id()._Get_underlying_id());
		std::normal_distribution NormGen(_Mean, _Sigma);

		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (CurDims > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6], __BINDEX[6];

		{
			const Range* __restrict __ShapePtr = _SlicePos.data();
			const SizeType* __restrict __StepPtr = _Input.StepsBack().data();
			const SizeType* __restrict __BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict __StridesPtr = _Input.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP[i] = 1;
				__BEGIN[i] = 0;
				__STRIDE[i] = 1;
				__BINDEX[i] = 0;
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]].End;
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
				__BINDEX[i] = __ShapePtr[ContPtr[CurIndex]].Begin;
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr = __STEP;
		const SizeType* __restrict BeginsPtr = __BEGIN;
		const SizeType* __restrict StridesPtr = __STRIDE;
		const SizeType* __restrict IndexBeginPtr = __BINDEX;

		for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 +
					((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 +
						((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 +
							((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 +
								((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5 = IndexAxis4 +
									((n * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								DataPtr[IndexAxis5] = (ThisType)NormGen(RndDevice);
							}
						}
					}
				}
			}
		}
	}

	void GatherImpl(
		const Tensor& _Ret,
		const Tensor& _Input,
		const Tensor& _Indices,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		auto InputSteps = _Input.StepsBack();
		for (auto& i : InputSteps)
			i /= _Input.GetAlignSize();

		bool Cont = _Input.IsContinuous();

		SizeType IND__SHAPE[6], IND__STEP[6], IND__BEGIN[6], IND__STRIDE[6];
		SizeType RET__STEP[6], RET__BEGIN[6], RET__STRIDE[6], __BINDEX[6];

		{
			const Range* __restrict IND__ShapePtr = _SlicePos.data();
			const SizeType* __restrict IND__StepPtr = _Indices.StepsBack().data();
			const SizeType* __restrict IND__BeginsPtr = _Indices.SliceBegins().data();
			const SizeType* __restrict IND__StridesPtr = _Indices.Strides().data();
			const SizeType* __restrict RET__StepPtr = _Ret.StepsBack().data();
			const SizeType* __restrict RET__BeginsPtr = _Ret.SliceBegins().data();
			const SizeType* __restrict RET__StridesPtr = _Ret.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				IND__SHAPE[i] = 1;
				IND__STEP[i] = 1;
				IND__BEGIN[i] = 0;
				IND__STRIDE[i] = 1;
				RET__STEP[i] = 1;
				RET__BEGIN[i] = 0;
				RET__STRIDE[i] = 1;
				__BINDEX[i] = 0;
				++i;
			}
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				IND__SHAPE[i] = IND__ShapePtr[CurIndex].End;
				IND__STEP[i] = IND__StepPtr[CurIndex] / _Indices.GetAlignSize();
				IND__BEGIN[i] = IND__BeginsPtr[CurIndex];
				IND__STRIDE[i] = IND__StridesPtr[CurIndex];
				RET__STEP[i] = RET__StepPtr[CurIndex];
				RET__BEGIN[i] = RET__BeginsPtr[CurIndex];
				RET__STRIDE[i] = RET__StridesPtr[CurIndex];
				__BINDEX[i] = IND__ShapePtr[CurIndex].Begin;
			}
		}

		const SizeType* __restrict ShapePtr = IND__SHAPE;
		const SizeType* __restrict StepPtr = IND__STEP;
		const SizeType* __restrict BeginsPtr = IND__BEGIN;
		const SizeType* __restrict StridesPtr = IND__STRIDE;
		const SizeType* __restrict IndexBeginPtr = __BINDEX;

		const SizeType GatherDims = _Input.DimCount() - 1;
		const SizeType* InputShapePtr = _Input.Shape().data();
		const SizeType InputSize = InputShapePtr[0];
		++InputShapePtr;
		const SizeType* InputStepPtr = InputSteps.data();
		const SizeType InputStep = InputStepPtr[0] * _Input.GetAlignSize();
		++InputStepPtr;
		const SizeType* InputBeginsPtr = _Input.SliceBegins().data();
		const SizeType InputBegin = InputBeginsPtr[0];
		++InputBeginsPtr;
		const SizeType* InputStridesPtr = _Input.Strides().data();
		const SizeType InputStride = InputStridesPtr[0];
		++InputStridesPtr;

		SizeType TotalSize = sizeof(ThisType);
		for (SizeType i = 0; i < GatherDims; ++i)
			TotalSize *= InputShapePtr[i];

		const SizeType* const __restrict RetStepPtr = RET__STEP;
		const SizeType* const __restrict RetBeginsPtr = RET__BEGIN;
		const SizeType* const __restrict RetStridesPtr = RET__STRIDE;

		const int32* IndicePtr = (int32*)_Indices.Data();
		const auto RetPtr = _Ret.Data();
		const auto InputPtr = _Input.Data();

		for (SizeType i = IndexBeginPtr[0]; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			const auto RIndexAxis0 = ((i * RetStridesPtr[0]) + RetBeginsPtr[0]) * RetStepPtr[0];
			for (SizeType j = IndexBeginPtr[1]; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 + ((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				const auto RIndexAxis1 = RIndexAxis0 +
					((j * RetStridesPtr[1]) + RetBeginsPtr[1]) * RetStepPtr[1];
				for (SizeType k = IndexBeginPtr[2]; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 + ((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					const auto RIndexAxis2 = RIndexAxis1 +
						((k * RetStridesPtr[2]) + RetBeginsPtr[2]) * RetStepPtr[2];
					for (SizeType l = IndexBeginPtr[3]; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 + ((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						const auto RIndexAxis3 = RIndexAxis2 +
							((l * RetStridesPtr[3]) + RetBeginsPtr[3]) * RetStepPtr[3];
						for (SizeType m = IndexBeginPtr[4]; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 + ((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							const auto RIndexAxis4 = RIndexAxis3 +
								((m * RetStridesPtr[4]) + RetBeginsPtr[4]) * RetStepPtr[4];
							for (SizeType n = IndexBeginPtr[5]; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5 = IndexAxis4 +
									((n * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								const auto RIndexAxis5 = RIndexAxis4 +
									((n * RetStridesPtr[5]) + RetBeginsPtr[5]) * RetStepPtr[5];
								const auto CIndex = SizeType(IndicePtr[IndexAxis5]);
								if (CIndex < InputSize)
								{
									GatherImp<ThisType>(
										InputShapePtr,
										InputStridesPtr,
										InputBeginsPtr,
										InputStepPtr,
										(ThisType*)(RetPtr + RIndexAxis5),
										(ThisType*)(InputPtr + ((CIndex * InputStride) + InputBegin) * InputStep),
										GatherDims,
										Cont,
										TotalSize
									);
								}
								else
									DragonianLibThrow("Index Out Of Range!");
							}
						}
					}
				}
			}
		}
	}

	void CastImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		if (_Src.DType() == TensorType::Boolean || _Src.DType() == TensorType::Int8)
			CastFrom<ThisType, int8>(_Dst, _Src, CurDims, _SlicePos);
		else if (_Src.DType() == TensorType::Int16)
			CastFrom<ThisType, int16>(_Dst, _Src, CurDims, _SlicePos);
		else if (_Src.DType() == TensorType::Int32)
			CastFrom<ThisType, int32>(_Dst, _Src, CurDims, _SlicePos);
		else if (_Src.DType() == TensorType::Int64)
			CastFrom<ThisType, int64>(_Dst, _Src, CurDims, _SlicePos);
		else if (_Src.DType() == TensorType::Float32)
			CastFrom<ThisType, float32>(_Dst, _Src, CurDims, _SlicePos);
		else if (_Src.DType() == TensorType::Float64)
			CastFrom<ThisType, float64>(_Dst, _Src, CurDims, _SlicePos);
		else
			DragonianLibThrow("UnSupported Type!");
		/*else if (_Src.DType() == TensorType::Float16)
			CastFrom<ThisType, uint16>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Complex32)
			CastFrom<ThisType, int8>(_Dst, _Src, CurDims);*/

	}

	void AddImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibAddFn<ThisType>,
			DragonianLibVectorAdd<ThisType>
		);
	}

	void SubImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibSubFn<ThisType>,
			DragonianLibVectorSub<ThisType>
		);
	}

	void MulImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibMulFn<ThisType>,
			DragonianLibVectorMul<ThisType>
		);
	}

	void DivImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibDivFn<ThisType>,
			DragonianLibVectorDiv<ThisType>
		);
	}

	void PowImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			pow<ThisType, ThisType>,
			DragonianLibVectorPow<ThisType>
		);
	}

	void AddImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibAddFn<ThisType>,
			DragonianLibVectorAddScalar<ThisType>
		);
	}

	void SubImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibSubFn<ThisType>,
			DragonianLibVectorSubScalar<ThisType>
		);
	}

	void MulImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibMulFn<ThisType>,
			DragonianLibVectorMulScalar<ThisType>
		);
	}

	void DivImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			DragonianLibDivFn<ThisType>,
			DragonianLibVectorDivScalar<ThisType>
		);
	}

	void PowImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			pow<ThisType, ThisType>,
			DragonianLibVectorPowScalar<ThisType>
		);
	}

	void AbsImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::conditional_t<std::is_integral_v<ThisType>, int64, float64>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			abs,
			DragonianLibVectorAbs<ThisType>
		);
	}

	void SinImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			sin,
			DragonianLibVectorSin<ThisType>
		);
	}

	void SinhImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			sinh,
			DragonianLibVectorSinh<ThisType>
		);
	}

	void CosImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			cos,
			DragonianLibVectorCos<ThisType>
		);
	}

	void CoshImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			cosh,
			DragonianLibVectorCosh<ThisType>
		);
	}

	void TanImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			tan,
			DragonianLibVectorTan<ThisType>
		);
	}

	void TanhImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			tanh,
			DragonianLibVectorTanh<ThisType>
		);
	}

	void ASinImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			asin,
			DragonianLibVectorASin<ThisType>
		);
	}

	void ACosImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			acos,
			DragonianLibVectorACos<ThisType>
		);
	}

	void ATanImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			atan,
			DragonianLibVectorATan<ThisType>
		);
	}

	void ASinhImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			asinh,
			DragonianLibVectorASinh<ThisType>
		);
	}

	void ACoshImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			acosh,
			DragonianLibVectorACosh<ThisType>
		);
	}

	void ATanhImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			atanh,
			DragonianLibVectorATanh<ThisType>
		);
	}

	void ExpImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			exp,
			DragonianLibVectorExp<ThisType>
		);
	}

	void Exp10Impl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		MonoOperators<ThisType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			DragonianLibExp10<ThisType>,
			DragonianLibVectorExp10<ThisType>
		);
	}

	void Exp2Impl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			exp2,
			DragonianLibVectorExp2<ThisType>
		);
	}

	void LogImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			log,
			DragonianLibVectorLog<ThisType>
		);
	}

	void Log2Impl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			log2,
			DragonianLibVectorLog2<ThisType>
		);
	}

	void Log10Impl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			log10,
			DragonianLibVectorLog10<ThisType>
		);
	}

	void CeilImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			ceil,
			DragonianLibVectorCeil<ThisType>
		);
	}

	void RoundImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			round,
			DragonianLibVectorRound<ThisType>
		);
	}

	void FloorImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			_SlicePos,
			floor,
			DragonianLibVectorFloor<ThisType>
		);
	}

	void LessImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Less<ThisType>
		);
	}

	void LessImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Less<ThisType>
		);
	}

	void GreaterImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Greater<ThisType>
		);
	}

	void GreaterImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Greater<ThisType>
		);
	}

	void EqualImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Equal<ThisType>
		);
	}

	void EqualImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::Equal<ThisType>
		);
	}

	void LessEqualImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::LessEqual<ThisType>
		);
	}

	void LessEqualImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::LessEqual<ThisType>
		);
	}

	void GreaterEqualImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::GreaterEqual<ThisType>
		);
	}

	void GreaterEqualImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::GreaterEqual<ThisType>
		);
	}

	void NotEqualImpl(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const Tensor& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::NotEqual<ThisType>
		);
	}

	void NotEqualImplScalar(
		const Tensor& _Dst,
		const Tensor& _Src1,
		const ThisType& _Src2,
		const SizeType CurDims,
		const SliceOptions& _SlicePos
	)
	{
		CompareOperatorsScalar<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			_SlicePos,
			::DragonianLib::NotEqual<ThisType>
		);
	}

	//ImplEnd

	void AssignValue(const Tensor& _Input, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool)
	{
		if (_Input.IsBroadCasted())
			DragonianLibThrow("You Can't Assign To A BroadCasted Tensor!");

		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();
		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);
		const auto OSlice = SqueezedTensor.GetDefaultSliceVector();

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(SqueezedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, SqueezedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return;
						}
						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };
						_ThreadPool->Commit(AssignImpl, SqueezedTensor, _Val, _ValType, CurDims, ThreadSlices);
						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			AssignImpl(SqueezedTensor, _Val, _ValType, CurDims, OSlice);
		}
		else
			AssignImpl(SqueezedTensor, _Val, _ValType, CurDims, OSlice);
	}

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool)
	{
		if (_Input.IsBroadCasted())
			DragonianLibThrow("You Can't Assign To A BroadCasted Tensor!");

		const byte* Buffer = (const byte*)BufferVoid;
		const byte* BufferEnd = (const byte*)BufferEndVoid;
		if ((BufferEnd - Buffer) % sizeof(ThisType))
			DragonianLibThrow("Buffer Size MisMatch!");
		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();

		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);
		const auto OSlice = SqueezedTensor.GetDefaultSliceVector();

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();
			auto BufferSize = BufferEnd - Buffer;

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(SqueezedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, SqueezedShape[i]);
						if (j >= End || Buffer >= BufferEnd)
						{
							_ThreadPool->Join();
							return;
						}

						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };

						const auto SizeRequired = VectorMul(ThreadSlices) * (SizeType)sizeof(ThisType);
						if (BufferSize >= SizeRequired)
						{
							_ThreadPool->Commit(
								AssignBufferImpl,
								SqueezedTensor,
								(const ThisType*)Buffer,
								(const ThisType*)Buffer + SizeRequired,
								CurDims,
								ThreadSlices
							);
							Buffer += SizeRequired;
							BufferSize -= SizeRequired;
						}
						else
						{
							_ThreadPool->Commit(
								AssignBufferImpl,
								SqueezedTensor,
								(const ThisType*)Buffer,
								(const ThisType*)BufferEnd,
								CurDims,
								ThreadSlices
							);
							Buffer = BufferEnd;
							BufferSize = 0;
						}

						if (End == SqueezedShape[i] || Buffer >= BufferEnd)
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			AssignBufferImpl(SqueezedTensor, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims, OSlice);
		}
		else
			AssignBufferImpl(SqueezedTensor, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims, OSlice);
	}

	void AssignTensorBroadCasted(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		const auto SqueezedTensorA = _InputA.Squeeze();
		const auto SqueezedTensorB = _InputB.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensorA.Shape().size();
		const auto& SqueezedShape = SqueezedTensorA.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);
		const auto OSlice = SqueezedTensorA.GetDefaultSliceVector();

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(SqueezedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, SqueezedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return;
						}

						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };

						_ThreadPool->Commit(
							AssignTensorImpl,
							SqueezedTensorA,
							SqueezedTensorB,
							CurDims,
							ThreadSlices
						);

						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			AssignTensorImpl(SqueezedTensorA, SqueezedTensorB, CurDims, OSlice);
		}
		else
			AssignTensorImpl(SqueezedTensorA, SqueezedTensorB, CurDims, OSlice);
	}

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		if (_InputA.DType() != _InputB.DType())
			DragonianLibThrow("Type MisMatch!");

		if (_InputA.GetDevice() != _InputB.GetDevice())
			DragonianLibThrow("Device MisMatch!");

		if (_InputA.IsBroadCasted())
			DragonianLibThrow("You Can't Assign To a BroadCasted Tensor!");

		if (_InputB.IsScalar())
		{
			AssignValue(_InputA, _InputB.GetPtr(), _InputB.DType(), _ThreadPool);
			return;
		}

		const auto BroadCast = _InputA.BroadCast(_InputB);

		AssignTensorBroadCasted(_InputA, BroadCast, _ThreadPool);
	}

	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool)
	{
		if (_Input.IsBroadCasted())
			DragonianLibThrow("You Can't Assign To A BroadCasted Tensor!");

		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();
		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);
		const auto OSlice = SqueezedTensor.GetDefaultSliceVector();

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(SqueezedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, SqueezedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return;
						}

						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };

						_ThreadPool->Commit(
							FixWithRandomImpl,
							SqueezedTensor,
							_Seed,
							_Mean,
							_Sigma,
							CurDims,
							ThreadSlices
						);
						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			FixWithRandomImpl(SqueezedTensor, _Seed, _Mean, _Sigma, CurDims, OSlice);
		}
		else
			FixWithRandomImpl(SqueezedTensor, _Seed, _Mean, _Sigma, CurDims, OSlice);
	}

	Tensor Gather(const Tensor& _Input, const Tensor& _IndicesInp, SizeType _Axis, ThreadPool* _ThreadPool)
	{
		if (_Input.GetDevice() != _IndicesInp.GetDevice())
			DragonianLibThrow("Device MisMatch!");
		if (_Input.DimCount() <= 1)
			DragonianLibThrow("Shape Of Input Should > 1!");

		auto _Indices = _IndicesInp.Cast(TensorType::Int32, _ThreadPool);
		if (!_Indices.IsContinuous())
			_Indices = _Indices.Continuous(_ThreadPool);
		auto _InputShape = _Input.Shape();
		const auto& _IndicesShape = _Indices.Shape();

		ShapeType _NewShape = _IndicesShape;
		_InputShape.erase(_InputShape.begin() + _Axis);
		_NewShape.insert(_NewShape.end(), _InputShape.begin(), _InputShape.end());
		Tensor Ret(_NewShape, _Input.DType(), _Input.GetDevice());

		const auto TotalSize = VectorMul(_NewShape);
		const auto CurDims = _Indices.DimCount();

		auto DPer = VectorArangeImpl(0ll, _Input.DimCount(), 1ll);
		DPer.erase(DPer.begin() + _Axis);
		DPer.insert(DPer.begin(), _Axis);
		const auto InputPPermuted = _Input.Permute(DPer);
		const auto OSlice = _Indices.GetDefaultSliceVector();

		if (CurDims > 6)
			DragonianLibThrow("Gather Operator Not Support Dim > 6!");

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)_IndicesShape.size();

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (_IndicesShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(_IndicesShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, _IndicesShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return Ret;
						}

						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };

						_ThreadPool->Commit(
							GatherImpl,
							Ret,
							InputPPermuted,
							_Indices,
							CurDims,
							ThreadSlices
						);
						if (End == _IndicesShape[i])
						{
							_ThreadPool->Join();
							return Ret;
						}
					}
				}
			}
			GatherImpl(Ret, InputPPermuted, _Indices, CurDims, OSlice);
		}
		else
			GatherImpl(Ret, InputPPermuted, _Indices, CurDims, OSlice);

		return Ret;
	}

	void Cast(const Tensor& _Dst, const Tensor& _Src, ThreadPool* _ThreadPool)
	{
		if (_Dst.GetDevice() != _Src.GetDevice())
			DragonianLibThrow("Device MisMatch!");

		const auto SqueezedTensorA = _Dst.Squeeze();
		const auto SqueezedTensorB = _Src.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensorA.Shape().size();
		const auto& SqueezedShape = SqueezedTensorA.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);
		const auto OSlice = SqueezedTensorA.GetDefaultSliceVector();

		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] >= NWorkers)
				{
					const auto Step = Tensor::Ceil(SqueezedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, SqueezedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return;
						}

						auto ThreadSlices = OSlice;
						ThreadSlices[i] = { j, End };

						_ThreadPool->Commit(
							CastImpl,
							SqueezedTensorA,
							SqueezedTensorB,
							CurDims,
							ThreadSlices
						);

						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			CastImpl(SqueezedTensorA, SqueezedTensorB, CurDims, OSlice);
		}
		else
			CastImpl(SqueezedTensorA, SqueezedTensorB, CurDims, OSlice);
	}

	//Operators

	DragonianLibMultiOperatorFunctionImpl(Add, AddImpl);
	DragonianLibMultiOperatorFunctionImpl(Sub, SubImpl);
	DragonianLibMultiOperatorFunctionImpl(Mul, MulImpl);
	DragonianLibMultiOperatorFunctionImpl(Div, DivImpl);
	DragonianLibMultiOperatorFunctionImpl(Pow, PowImpl);
	DragonianLibMultiOperatorScalarFunctionImpl(Add, AddImplScalar);
	DragonianLibMultiOperatorScalarFunctionImpl(Sub, SubImplScalar);
	DragonianLibMultiOperatorScalarFunctionImpl(Mul, MulImplScalar);
	DragonianLibMultiOperatorScalarFunctionImpl(Div, DivImplScalar);
	DragonianLibMultiOperatorScalarFunctionImpl(Pow, PowImplScalar);
	DragonianLibMultiOperatorInplaceFunctionImpl(AddInplace, AddImpl);
	DragonianLibMultiOperatorInplaceFunctionImpl(SubInplace, SubImpl);
	DragonianLibMultiOperatorInplaceFunctionImpl(MulInplace, MulImpl);
	DragonianLibMultiOperatorInplaceFunctionImpl(DivInplace, DivImpl);
	DragonianLibMultiOperatorInplaceFunctionImpl(PowInplace, PowImpl);
	DragonianLibMultiOperatorScalarInplaceFunctionImpl(AddInplace, AddImplScalar);
	DragonianLibMultiOperatorScalarInplaceFunctionImpl(SubInplace, SubImplScalar);
	DragonianLibMultiOperatorScalarInplaceFunctionImpl(MulInplace, MulImplScalar);
	DragonianLibMultiOperatorScalarInplaceFunctionImpl(DivInplace, DivImplScalar);
	DragonianLibMultiOperatorScalarInplaceFunctionImpl(PowInplace, PowImplScalar);

	DragonianLibMonoOperatorFunctionImpl(Abs, AbsImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Abs, AbsImpl);
	DragonianLibMonoOperatorFunctionImpl(Sin, SinImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Sin, SinImpl);
	DragonianLibMonoOperatorFunctionImpl(Sinh, SinhImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Sinh, SinhImpl);
	DragonianLibMonoOperatorFunctionImpl(Cos, CosImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Cos, CosImpl);
	DragonianLibMonoOperatorFunctionImpl(Cosh, CoshImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Cosh, CoshImpl);
	DragonianLibMonoOperatorFunctionImpl(Tan, TanImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Tan, TanImpl);
	DragonianLibMonoOperatorFunctionImpl(Tanh, TanhImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Tanh, TanhImpl);
	DragonianLibMonoOperatorFunctionImpl(ASin, ASinImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ASin, ASinImpl);
	DragonianLibMonoOperatorFunctionImpl(ACos, ACosImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ACos, ACosImpl);
	DragonianLibMonoOperatorFunctionImpl(ATan, ATanImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ATan, ATanImpl);
	DragonianLibMonoOperatorFunctionImpl(ASinh, ASinhImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ASinh, ASinhImpl);
	DragonianLibMonoOperatorFunctionImpl(ACosh, ACoshImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ACosh, ACoshImpl);
	DragonianLibMonoOperatorFunctionImpl(ATanh, ATanhImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(ATanh, ATanhImpl);
	DragonianLibMonoOperatorFunctionImpl(Exp, ExpImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Exp, ExpImpl);
	DragonianLibMonoOperatorFunctionImpl(Exp2, Exp2Impl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Exp2, Exp2Impl);
	DragonianLibMonoOperatorFunctionImpl(Exp10, Exp10Impl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Exp10, Exp10Impl);
	DragonianLibMonoOperatorFunctionImpl(Log, LogImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Log, LogImpl);
	DragonianLibMonoOperatorFunctionImpl(Log2, Log2Impl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Log2, Log2Impl);
	DragonianLibMonoOperatorFunctionImpl(Log10, Log10Impl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Log10, Log10Impl);

	DragonianLibCompareOperatorFunctionImpl(Less, LessImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(Less, LessImplScalar);
	DragonianLibCompareOperatorFunctionImpl(Greater, GreaterImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(Greater, GreaterImplScalar);
	DragonianLibCompareOperatorFunctionImpl(Equal, EqualImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(Equal, EqualImplScalar);
	DragonianLibCompareOperatorFunctionImpl(LessEqual, LessEqualImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(LessEqual, LessEqualImplScalar);
	DragonianLibCompareOperatorFunctionImpl(GreaterEqual, GreaterEqualImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(GreaterEqual, GreaterEqualImplScalar);
	DragonianLibCompareOperatorFunctionImpl(NotEqual, NotEqualImpl);
	DragonianLibCompareOperatorScalarFunctionImpl(NotEqual, NotEqualImplScalar);

	DragonianLibMonoOperatorFunctionImpl(Ceil, CeilImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Ceil, CeilImpl);
	DragonianLibMonoOperatorFunctionImpl(Round, RoundImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Round, RoundImpl);
	DragonianLibMonoOperatorFunctionImpl(Floor, FloorImpl);
	DragonianLibMonoOperatorInplaceFunctionImpl(Floor, FloorImpl);

	void SumImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		ThisType* DataPtr1 = (ThisType*)_Dst.Data();
		const ThisType* DataPtr2 = (ThisType*)_Src.Data();

		if (CurDims > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP1[6], __BEGIN1[6], __STRIDE1[6], __STEP2[6], __BEGIN2[6], __STRIDE2[6];

		{
			const SizeType* __restrict __ShapePtr = _Src.Shape().data();
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
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[CurIndex];
				__STEP1[i] = __StepPtr1[CurIndex] / (SizeType)sizeof(ThisType);
				__BEGIN1[i] = __BeginsPtr1[CurIndex];
				__STRIDE1[i] = __StridesPtr1[CurIndex];
				__STEP2[i] = __StepPtr2[CurIndex] / (SizeType)sizeof(ThisType);
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
							DataPtr1[IndexAxis4A] = ThisType(0);
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
							{
								const auto IndexAxis5B = IndexAxis4B +
									((n * StridesPtr2[5]) + BeginsPtr2[5]) * StepPtr2[5];
								DataPtr1[IndexAxis4A] += DataPtr2[IndexAxis5B];
							}
						}
					}
				}
			}
		}
	}

	void CumSumImpl(const Tensor& _Dst, SizeType CurDims)
	{
		ThisType* __restrict DataPtr = (ThisType*)_Dst.Data();

		if (CurDims > 6)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6];

		{
			const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
			const SizeType* __restrict __StepPtr = _Dst.StepsBack().data();
			const SizeType* __restrict __BeginsPtr = _Dst.SliceBegins().data();
			const SizeType* __restrict __StridesPtr = _Dst.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP[i] = 1;
				__BEGIN[i] = 0;
				__STRIDE[i] = 1;
				++i;
			}
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[CurIndex];
				__STEP[i] = __StepPtr[CurIndex] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[CurIndex];
				__STRIDE[i] = __StridesPtr[CurIndex];
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr = __STEP;
		const SizeType* __restrict BeginsPtr = __BEGIN;
		const SizeType* __restrict StridesPtr = __STRIDE;

		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			for (SizeType j = 0; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 +
					((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				for (SizeType k = 0; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 +
						((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					for (SizeType l = 0; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 +
							((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						for (SizeType m = 0; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 +
								((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							for (SizeType ldvar = 1; ldvar < ShapePtr[5]; ++ldvar)
							{
								const auto IndexAxisCur = IndexAxis4 +
									((ldvar * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								const auto IndexAxisLast = IndexAxis4 +
									(((ldvar - 1) * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								DataPtr[IndexAxisCur] += DataPtr[IndexAxisLast];
							}
						}
					}
				}
			}
		}
	}

	void CumProdImpl(const Tensor& _Dst, SizeType CurDims)
	{
		ThisType* __restrict DataPtr = (ThisType*)_Dst.Data();

		if (CurDims > 5)
			DragonianLibThrow("Dim > 6 Could Be Batch Dim, You Can Make A Loop Iterator By Yourself.");

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6];

		{
			const SizeType* __restrict __ShapePtr = _Dst.Shape().data();
			const SizeType* __restrict __StepPtr = _Dst.StepsBack().data();
			const SizeType* __restrict __BeginsPtr = _Dst.SliceBegins().data();
			const SizeType* __restrict __StridesPtr = _Dst.Strides().data();
			SizeType i = 0;
			SizeType Count = 6 - CurDims;
			while (i < Count)
			{
				__SHAPE[i] = 1;
				__STEP[i] = 1;
				__BEGIN[i] = 0;
				__STRIDE[i] = 1;
				++i;
			}
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[CurIndex];
				__STEP[i] = __StepPtr[CurIndex] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[CurIndex];
				__STRIDE[i] = __StridesPtr[CurIndex];
			}
		}

		const SizeType* __restrict ShapePtr = __SHAPE;
		const SizeType* __restrict StepPtr = __STEP;
		const SizeType* __restrict BeginsPtr = __BEGIN;
		const SizeType* __restrict StridesPtr = __STRIDE;

		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			for (SizeType j = 0; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 +
					((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				for (SizeType k = 0; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 +
						((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					for (SizeType l = 0; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 +
							((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						for (SizeType m = 0; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 +
								((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							for (SizeType ldvar = 1; ldvar < ShapePtr[5]; ++ldvar)
							{
								const auto IndexAxisCur = IndexAxis4 +
									((ldvar * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								const auto IndexAxisLast = IndexAxis4 +
									(((ldvar - 1) * StridesPtr[5]) + BeginsPtr[5]) * StepPtr[5];
								DataPtr[IndexAxisCur] *= DataPtr[IndexAxisLast];
							}
						}
					}
				}
			}
		}
	}

	Tensor Sum(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool)
	{
		const auto _Dims = _Src.DimCount();
		//const auto& _Shape = _Src.Shape();
		auto _NewShape = _Src.Shape();
		_Axis = Tensor::CalcIndex(_Axis, _Dims);
		_NewShape[_Axis] = 1;

		//Used By Squeezed Operator
		bool ReqSqu = false;
		if (_NewShape.size() > 1)
		{
			_NewShape.erase(_NewShape.begin() + _Axis);
			ReqSqu = true;
		}

		Tensor Output(_NewShape, _Src.DType(), _Src.GetDevice());

		const auto InputRef = _Src.SwapLastDim(_Axis);
		//auto ReturnRef = Output.SwapLastDim(_Axis);
		auto ReturnRef = Output.CreateView();

		//Used By Squeezed Operator
		if (ReqSqu)
			ReturnRef = ReturnRef.UnSqueeze(-1);

		const auto& PermutedShape = InputRef.Shape();
		const auto TotalSize = VectorMul(PermutedShape);
		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			Vector<Range> Slices;
			for (SizeType i = 0; i < _Dims - 1; ++i)
			{
				if (PermutedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
				{
					const auto Step = Tensor::Ceil(PermutedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, PermutedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return Output;
						}
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(
							SumImpl,
							ReturnRef.Slice(ThreadSlices),
							InputRef.Slice(ThreadSlices),
							_Dims
						);
						if (End == PermutedShape[i])
						{
							_ThreadPool->Join();
							return Output;
						}
					}
				}
			}
		}

		SumImpl(ReturnRef, InputRef, _Dims);

		return Output;
	}

	Tensor CumSum(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool)
	{
		const auto _Dims = _Src.DimCount();
		//const auto& _Shape = _Src.Shape();
		//auto _NewShape = _Src.Shape();
		_Axis = Tensor::CalcIndex(_Axis, _Dims);

		Tensor Output = _Src.Clone(_ThreadPool);

		const auto ReturnRef = Output.SwapLastDim(_Axis);

		const auto& PermutedShape = ReturnRef.Shape();
		const auto TotalSize = VectorMul(PermutedShape);
		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			Vector<Range> Slices;
			for (SizeType i = 0; i < _Dims - 1; ++i)
			{
				if (PermutedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
				{
					const auto Step = Tensor::Ceil(PermutedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, PermutedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return Output;
						}
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(
							CumSumImpl,
							ReturnRef.Slice(ThreadSlices),
							_Dims
						);
						if (End == PermutedShape[i])
						{
							_ThreadPool->Join();
							return Output;
						}
					}
				}
			}
		}

		CumSumImpl(ReturnRef, _Dims);

		return Output;
	}

	Tensor CumProd(const Tensor& _Src, SizeType _Axis, ThreadPool* _ThreadPool)
	{
		const auto _Dims = _Src.DimCount();
		//const auto& _Shape = _Src.Shape();
		//auto _NewShape = _Src.Shape();
		_Axis = Tensor::CalcIndex(_Axis, _Dims);

		Tensor Output = _Src.Clone(_ThreadPool);

		const auto ReturnRef = Output.SwapLastDim(_Axis);

		const auto& PermutedShape = ReturnRef.Shape();
		const auto TotalSize = VectorMul(PermutedShape);
		if (_ThreadPool && TotalSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			Vector<Range> Slices;
			for (SizeType i = 0; i < _Dims - 1; ++i)
			{
				if (PermutedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
				{
					const auto Step = Tensor::Ceil(PermutedShape[i], NWorkers);
					for (SizeType j = 0; ; j += Step)
					{
						const auto End = std::min(j + Step, PermutedShape[i]);
						if (j >= End)
						{
							_ThreadPool->Join();
							return Output;
						}
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(
							CumProdImpl,
							ReturnRef.Slice(ThreadSlices),
							_Dims
						);
						if (End == PermutedShape[i])
						{
							_ThreadPool->Join();
							return Output;
						}
					}
				}
			}
		}

		CumProdImpl(ReturnRef, _Dims);

		return Output;
	}
}

DragonianLibSpaceEnd