#include "Tensor/Float32Tensor.h"

LibSvcBegin

namespace Float32
{

	ThisType CastFrom(TensorType _Type, cpvoid _Val)
	{
		ThisType Ret;
		LibSvcTypeSwitch(
			_Type,
			LibSvcCastImpl(ThisType, Ret, bool, _Val),
			LibSvcCastImpl(ThisType, Ret, int8, _Val),
			LibSvcCastImpl(ThisType, Ret, int16, _Val),
			LibSvcCastImpl(ThisType, Ret, int32, _Val),
			LibSvcCastImpl(ThisType, Ret, int64, _Val),
			UNUSED(),
			LibSvcCastImpl(ThisType, Ret, float32, _Val),
			LibSvcCastImpl(ThisType, Ret, float64, _Val),
			UNUSED()
		);
		return Ret;
	}

	void AssignImpl(const Tensor& _Input, cpvoid _Val, TensorType _ValType, const SizeType CurDims)
	{
		const auto _Value = CastFrom(_ValType, _Val);
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if(_Input.IsContinuous())
		{
			DataPtr = (ThisType*)_Input.GetPtr();
			const size_t BufferSize = VectorMul(_Input.Shape()) * sizeof(ThisType);
			LibSvcMemSet(DataPtr, &_Value, BufferSize, sizeof(ThisType));
			return;
		}

		if (CurDims > 6)
		{
			auto Steps = _Input.StepsBack();
			for (auto& i : Steps)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Input.Shape().data();
			const SizeType* __restrict StepPtr = Steps.data();
			const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict StridesPtr = _Input.Strides().data();
			ShapeType CurIndice(CurDims, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				CurDims,
				{
					SizeType Index = 0;
					for (SizeType i = 0; i < CurDims; ++i)
						Index += ((IndicesPtr[i] * StridesPtr[i]) + BeginsPtr[i]) * StepPtr[i];
					DataPtr[Index] = _Value;
				}
			);
			return;
		}

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6];

		{
			const SizeType* __restrict __ShapePtr = _Input.Shape().data();
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
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
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
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
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
		const SizeType CurDims
	) {
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (BufferEnd < Buffer)
			LibSvcThrow("[Operator] BufferEnd* < Buffer*, Make Sure BufferEnd* > Buffer*");

		if (_Input.IsContinuous())
		{
			DataPtr = (ThisType*)_Input.GetPtr();
			const size_t BufferSize = (BufferEnd - Buffer) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr, Buffer, BufferSize);
			return;
		}

		if (CurDims > 6)
		{
			auto Steps = _Input.StepsBack();
			for (auto& i : Steps)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Input.Shape().data();
			const SizeType* __restrict StepPtr = Steps.data();
			const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict StridesPtr = _Input.Strides().data();
			ShapeType CurIndice(CurDims, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				CurDims,
				{
					SizeType Index = 0;
					for (SizeType i = 0; i < CurDims; ++i)
						Index += ((IndicesPtr[i] * StridesPtr[i]) + BeginsPtr[i]) * StepPtr[i];
					DataPtr[Index] = *(Buffer++);
					if (Buffer == BufferEnd)
						return;
				}
			);
			return;
		}

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6];

		{
			const SizeType* __restrict __ShapePtr = _Input.Shape().data();
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
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
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
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
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

	void AssignTensorImpl(const Tensor& _InputA, const Tensor& _InputB, const SizeType CurDims)
	{
		ThisType* DataPtr1 = (ThisType*)_InputA.Data();
		const ThisType* DataPtr2 = (ThisType*)_InputB.Data();

		if(!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
		{
			DataPtr1 = (ThisType*)_InputA.GetPtr();
			DataPtr2 = (ThisType*)_InputB.GetPtr();
			const size_t BufferSize = VectorMul(_InputA.Shape()) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr1, DataPtr2, BufferSize);
			return;
		}

		if (CurDims > 6)
		{
			auto Steps1 = _InputA.StepsBack();
			for (auto& i : Steps1)
				i /= sizeof(ThisType);
			auto Steps2 = _InputB.StepsBack();
			for (auto& i : Steps2)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _InputA.Shape().data();
			const SizeType* __restrict StepPtr1 = Steps1.data();
			const SizeType* __restrict StepPtr2 = Steps2.data();
			const SizeType* __restrict BeginsPtr1 = _InputA.SliceBegins().data();
			const SizeType* __restrict BeginsPtr2 = _InputB.SliceBegins().data();
			const SizeType* __restrict StridesPtr1 = _InputA.Strides().data();
			const SizeType* __restrict StridesPtr2 = _InputB.Strides().data();
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
					DataPtr1[Index1] = DataPtr2[Index2];
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
				__STEP1[i] = __StepPtr1[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN1[i] = __BeginsPtr1[ContPtr[CurIndex]];
				__STRIDE1[i] = __StridesPtr1[ContPtr[CurIndex]];
				__STEP2[i] = __StepPtr2[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
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
								DataPtr1[IndexAxis5A] = DataPtr2[IndexAxis5B];
							}
						}
					}
				}
			}
		}
	}

	void FixWithRandomImpl(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, const SizeType CurDims)
	{
		std::mt19937_64 RndDevice(_Seed + std::this_thread::get_id()._Get_underlying_id());
		std::normal_distribution NormGen(_Mean, _Sigma);

		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (_Input.IsContinuous())
		{
			DataPtr = (ThisType*)_Input.GetPtr();
			const auto DataSize = VectorMul(_Input.Shape());
			for (auto i = 0; i < DataSize; ++i)
				*(DataPtr++) = (ThisType)NormGen(RndDevice);
			return;
		}

		if (CurDims > 6)
		{
			auto Steps = _Input.StepsBack();
			for (auto& i : Steps)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Input.Shape().data();
			const SizeType* __restrict StepPtr = Steps.data();
			const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
			const SizeType* __restrict StridesPtr = _Input.Strides().data();
			ShapeType CurIndice(CurDims, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				CurDims,
				{
					SizeType Index = 0;
					for (SizeType i = 0; i < CurDims; ++i)
						Index += ((IndicesPtr[i] * StridesPtr[i]) + BeginsPtr[i]) * StepPtr[i];
					DataPtr[Index] = (ThisType)NormGen(RndDevice);
				}
			);
			return;
		}

		SizeType __SHAPE[6], __STEP[6], __BEGIN[6], __STRIDE[6];

		{
			const SizeType* __restrict __ShapePtr = _Input.Shape().data();
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
				++i;
			}
			const auto Cont = _Input.CalcContinuous();
			const SizeType* __restrict ContPtr = Cont.data();
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				__SHAPE[i] = __ShapePtr[ContPtr[CurIndex]];
				__STEP[i] = __StepPtr[ContPtr[CurIndex]] / (SizeType)sizeof(ThisType);
				__BEGIN[i] = __BeginsPtr[ContPtr[CurIndex]];
				__STRIDE[i] = __StridesPtr[ContPtr[CurIndex]];
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
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
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

	void AssignValue(const Tensor& _Input, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool)
	{
		if (_Input.IsBroadCasted())
			LibSvcThrow("You Can't Assign To A BroadCasted Tensor!");

		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();
		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);

		if(_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();
			
			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(AssignImpl, SqueezedTensor.Slice(ThreadSlices), _Val, _ValType, CurDims);
						if(End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			AssignImpl(SqueezedTensor, _Val, _ValType, CurDims);
		}
		else
			AssignImpl(SqueezedTensor, _Val, _ValType, CurDims);
	}

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool)
	{
		if (_Input.IsBroadCasted())
			LibSvcThrow("You Can't Assign To A BroadCasted Tensor!");

		const byte* Buffer = (const byte*)BufferVoid;
		const byte* BufferEnd = (const byte*)BufferEndVoid;
		if ((BufferEnd - Buffer) % sizeof(ThisType))
			LibSvcThrow("Buffer Size MisMatch!");
		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();

		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);

		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();
			auto BufferSize = BufferEnd - Buffer;

			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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

						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						auto Tensor = SqueezedTensor.Slice(ThreadSlices);

						const auto SizeRequired = VectorMul(Tensor.Shape()) * (SizeType)sizeof(ThisType);
						if (BufferSize >= SizeRequired)
						{
							_ThreadPool->Commit(
								AssignBufferImpl,
								std::move(Tensor),
								(const ThisType*)Buffer,
								(const ThisType*)Buffer + SizeRequired,
								CurDims
							);
							Buffer += SizeRequired;
							BufferSize -= SizeRequired;
						}
						else
						{
							_ThreadPool->Commit(
								AssignBufferImpl,
								std::move(Tensor),
								(const ThisType*)Buffer,
								(const ThisType*)BufferEnd,
								CurDims
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
			AssignBufferImpl(SqueezedTensor, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims);
		}
		else
			AssignBufferImpl(SqueezedTensor, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims);
	}

	void AssignTensorBroadCasted(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		const auto SqueezedTensorA = _InputA.Squeeze();
		const auto SqueezedTensorB = _InputB.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensorA.Shape().size();
		const auto& SqueezedShape = SqueezedTensorA.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);

		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);

						_ThreadPool->Commit(
							AssignTensorImpl,
							SqueezedTensorA.Slice(ThreadSlices),
							SqueezedTensorB.Slice(ThreadSlices),
							CurDims
						);

						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			AssignTensorImpl(SqueezedTensorA, SqueezedTensorB, CurDims);
		}
		else
			AssignTensorImpl(SqueezedTensorA, SqueezedTensorB, CurDims);
	}

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		if (_InputA.DType() != _InputB.DType())
			LibSvcThrow("Type MisMatch!");

		if (_InputA.GetDevice() != _InputB.GetDevice())
			LibSvcThrow("Device MisMatch!");

		if (_InputA.IsBroadCasted())
			LibSvcThrow("You Can't Assign To a BroadCasted Tensor!");

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
			LibSvcThrow("You Can't Assign To A BroadCasted Tensor!");

		const auto SqueezedTensor = _Input.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensor.Shape().size();
		const auto& SqueezedShape = SqueezedTensor.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);

		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(
							FixWithRandomImpl,
							SqueezedTensor.Slice(ThreadSlices),
							_Seed,
							_Mean,
							_Sigma,
							CurDims
						);
						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			FixWithRandomImpl(SqueezedTensor, _Seed, _Mean, _Sigma, CurDims);
		}
		else
			FixWithRandomImpl(SqueezedTensor, _Seed, _Mean, _Sigma, CurDims);
	}

	void GatherImpl(const Tensor& _Ret, const Tensor& _Input, const Tensor& _Indices, const SizeType CurDims)
	{
		auto InputSteps = _Input.StepsBack();
		for (auto& i : InputSteps)
			i /= _Input.GetAlignSize();

		bool Cont = _Input.IsContinuous();

		SizeType IND__SHAPE[6], IND__STEP[6], IND__BEGIN[6], IND__STRIDE[6];
		SizeType RET__STEP[6], RET__BEGIN[6], RET__STRIDE[6];

		{
			const SizeType* __restrict IND__ShapePtr = _Indices.Shape().data();
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
				++i;
			}
			for (; i < 6; ++i)
			{
				const auto CurIndex = i - Count;
				IND__SHAPE[i] = IND__ShapePtr[CurIndex];
				IND__STEP[i] = IND__StepPtr[CurIndex] / _Indices.GetAlignSize();
				IND__BEGIN[i] = IND__BeginsPtr[CurIndex];
				IND__STRIDE[i] = IND__StridesPtr[CurIndex];
				RET__STEP[i] = RET__StepPtr[CurIndex];
				RET__BEGIN[i] = RET__BeginsPtr[CurIndex];
				RET__STRIDE[i] = RET__StridesPtr[CurIndex];
			}
		}

		const SizeType* __restrict ShapePtr = IND__SHAPE;
		const SizeType* __restrict StepPtr = IND__STEP;
		const SizeType* __restrict BeginsPtr = IND__BEGIN;
		const SizeType* __restrict StridesPtr = IND__STRIDE;

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

		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			const auto IndexAxis0 = ((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0];
			const auto RIndexAxis0 = ((i * RetStridesPtr[0]) + RetBeginsPtr[0]) * RetStepPtr[0];
			for (SizeType j = 0; j < ShapePtr[1]; ++j)
			{
				const auto IndexAxis1 = IndexAxis0 + ((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1];
				const auto RIndexAxis1 = RIndexAxis0 +
					((j * RetStridesPtr[1]) + RetBeginsPtr[1]) * RetStepPtr[1];
				for (SizeType k = 0; k < ShapePtr[2]; ++k)
				{
					const auto IndexAxis2 = IndexAxis1 + ((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2];
					const auto RIndexAxis2 = RIndexAxis1 +
						((k * RetStridesPtr[2]) + RetBeginsPtr[2]) * RetStepPtr[2];
					for (SizeType l = 0; l < ShapePtr[3]; ++l)
					{
						const auto IndexAxis3 = IndexAxis2 + ((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3];
						const auto RIndexAxis3 = RIndexAxis2 +
							((l * RetStridesPtr[3]) + RetBeginsPtr[3]) * RetStepPtr[3];
						for (SizeType m = 0; m < ShapePtr[4]; ++m)
						{
							const auto IndexAxis4 = IndexAxis3 + ((m * StridesPtr[4]) + BeginsPtr[4]) * StepPtr[4];
							const auto RIndexAxis4 = RIndexAxis3 +
								((m * RetStridesPtr[4]) + RetBeginsPtr[4]) * RetStepPtr[4];
							for (SizeType n = 0; n < ShapePtr[5]; ++n)
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
									LibSvcThrow("Index Out Of Range!");
							}
						}
					}
				}
			}
		}
	}

	Tensor Gather(const Tensor& _Input, const Tensor& _IndicesInp, SizeType _Axis, ThreadPool* _ThreadPool)
	{
		if (_Input.GetDevice() != _IndicesInp.GetDevice())
			LibSvcThrow("Device MisMatch!");
		if (_Input.DimCount() <= 1)
			LibSvcThrow("Shape Of Input Should > 1!");

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

		if (CurDims > 6)
			LibSvcThrow("Gather Operator Not Support Dim > 6!");

		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)_IndicesShape.size();

			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (_IndicesShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);
						_ThreadPool->Commit(
							GatherImpl,
							Ret.Slice(ThreadSlices),
							InputPPermuted,
							_Indices.Slice(ThreadSlices),
							CurDims
						);
						if (End == _IndicesShape[i])
						{
							_ThreadPool->Join();
							return Ret;
						}
					}
				}
			}
			GatherImpl(Ret, InputPPermuted, _Indices, CurDims);
		}
		else
			GatherImpl(Ret, InputPPermuted, _Indices, CurDims);

		return Ret;
	}

	void CastImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		if (_Src.DType() == TensorType::Boolean || _Src.DType() == TensorType::Int8)
			CastFrom<ThisType, int8>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Int16)
			CastFrom<ThisType, int16>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Int32)
			CastFrom<ThisType, int32>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Int64)
			CastFrom<ThisType, int64>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Float32)
			CastFrom<ThisType, float32>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Float64)
			CastFrom<ThisType, float64>(_Dst, _Src, CurDims);
		else
			LibSvcThrow("UnSupported Type!");
		/*else if (_Src.DType() == TensorType::Float16)
			CastFrom<ThisType, uint16>(_Dst, _Src, CurDims);
		else if (_Src.DType() == TensorType::Complex32)
			CastFrom<ThisType, int8>(_Dst, _Src, CurDims);*/
		
	}

	void Cast(const Tensor& _Dst, const Tensor& _Src, ThreadPool* _ThreadPool)
	{
		if (_Dst.GetDevice() != _Src.GetDevice())
			LibSvcThrow("Device MisMatch!");

		const auto SqueezedTensorA = _Dst.Squeeze();
		const auto SqueezedTensorB = _Src.Squeeze();
		const auto CurDims = (SizeType)SqueezedTensorA.Shape().size();
		const auto& SqueezedShape = SqueezedTensorA.Shape();
		const auto TotalSize = VectorMul(SqueezedShape);

		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
		{
			const auto NWorkers = _ThreadPool->GetThreadCount();
			const auto SqueezedDims = (SizeType)SqueezedShape.size();

			Vector<Range> Slices;
			for (SizeType i = 0; i < SqueezedDims; ++i)
			{
				if (SqueezedShape[i] < NWorkers)
					Slices.emplace_back(None);
				else
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
						auto ThreadSlices = Slices;
						ThreadSlices.emplace_back(j, End);

						_ThreadPool->Commit(
							CastImpl,
							SqueezedTensorA.Slice(ThreadSlices),
							SqueezedTensorB.Slice(ThreadSlices),
							CurDims
						);

						if (End == SqueezedShape[i])
						{
							_ThreadPool->Join();
							return;
						}
					}
				}
			}
			CastImpl(SqueezedTensorA, SqueezedTensorB, CurDims);
		}
		else
			CastImpl(SqueezedTensorA, SqueezedTensorB, CurDims);
	}

	//Operators

	void AddImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcAddFn<ThisType>,
			LibSvcVectorAdd<ThisType>
		);
	}

	void SubImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcSubFn<ThisType>,
			LibSvcVectorSub<ThisType>
		);
	}

	void MulImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcMulFn<ThisType>,
			LibSvcVectorMul<ThisType>
		);
	}

	void DivImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcDivFn<ThisType>,
			LibSvcVectorDiv<ThisType>
		);
	}

	void PowImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		MultiOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			pow<ThisType, ThisType>,
			LibSvcVectorPow<ThisType>
		);
	}

	void AddImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcAddFn<ThisType>,
			LibSvcVectorAddScalar<ThisType>
		);
	}

	void SubImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcSubFn<ThisType>,
			LibSvcVectorSubScalar<ThisType>
		);
	}

	void MulImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcMulFn<ThisType>,
			LibSvcVectorMulScalar<ThisType>
		);
	}

	void DivImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			LibSvcDivFn<ThisType>,
			LibSvcVectorDivScalar<ThisType>
		);
	}

	void PowImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		MultiOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			pow<ThisType, ThisType>,
			LibSvcVectorPowScalar<ThisType>
		);
	}

	LibSvcMultiOperatorFunctionImpl(Add, AddImpl);
	LibSvcMultiOperatorFunctionImpl(Sub, SubImpl);
	LibSvcMultiOperatorFunctionImpl(Mul, MulImpl);
	LibSvcMultiOperatorFunctionImpl(Div, DivImpl);
	LibSvcMultiOperatorFunctionImpl(Pow, PowImpl);
	LibSvcMultiOperatorScalarFunctionImpl(Add, AddImplScalar);
	LibSvcMultiOperatorScalarFunctionImpl(Sub, SubImplScalar);
	LibSvcMultiOperatorScalarFunctionImpl(Mul, MulImplScalar);
	LibSvcMultiOperatorScalarFunctionImpl(Div, DivImplScalar);
	LibSvcMultiOperatorScalarFunctionImpl(Pow, PowImplScalar);
	LibSvcMultiOperatorInplaceFunctionImpl(AddInplace, AddImpl);
	LibSvcMultiOperatorInplaceFunctionImpl(SubInplace, SubImpl);
	LibSvcMultiOperatorInplaceFunctionImpl(MulInplace, MulImpl);
	LibSvcMultiOperatorInplaceFunctionImpl(DivInplace, DivImpl);
	LibSvcMultiOperatorInplaceFunctionImpl(PowInplace, PowImpl);
	LibSvcMultiOperatorScalarInplaceFunctionImpl(AddInplace, AddImplScalar);
	LibSvcMultiOperatorScalarInplaceFunctionImpl(SubInplace, SubImplScalar);
	LibSvcMultiOperatorScalarInplaceFunctionImpl(MulInplace, MulImplScalar);
	LibSvcMultiOperatorScalarInplaceFunctionImpl(DivInplace, DivImplScalar);
	LibSvcMultiOperatorScalarInplaceFunctionImpl(PowInplace, PowImplScalar);

	void AbsImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::conditional_t<std::is_integral_v<ThisType>, int64,
			std::enable_if_t<std::is_floating_point_v<ThisType>, float64>>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			abs,
			LibSvcVectorAbs<ThisType>
		);
	}

	void SinImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			sin,
			LibSvcVectorSin<ThisType>
		);
	}

	void SinhImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			sinh,
			LibSvcVectorSinh<ThisType>
		);
	}

	void CosImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			cos,
			LibSvcVectorCos<ThisType>
		);
	}

	void CoshImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			cosh,
			LibSvcVectorCosh<ThisType>
		);
	}

	void TanImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			tan,
			LibSvcVectorTan<ThisType>
		);
	}

	void TanhImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			tanh,
			LibSvcVectorTanh<ThisType>
		);
	}

	void ASinImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			asin,
			LibSvcVectorASin<ThisType>
		);
	}

	void ACosImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			acos,
			LibSvcVectorACos<ThisType>
		);
	}

	void ATanImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			atan,
			LibSvcVectorATan<ThisType>
		);
	}

	void ASinhImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			asinh,
			LibSvcVectorASinh<ThisType>
		);
	}

	void ACoshImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			acosh,
			LibSvcVectorACosh<ThisType>
		);
	}

	void ATanhImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			atanh,
			LibSvcVectorATanh<ThisType>
		);
	}

	void ExpImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			exp,
			LibSvcVectorExp<ThisType>
		);
	}

	void Exp10Impl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		MonoOperators<ThisType>(
			_Dst,
			_Src,
			CurDims,
			LibSvcExp10<ThisType>,
			LibSvcVectorExp10<ThisType>
		);
	}

	void Exp2Impl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			exp2,
			LibSvcVectorExp2<ThisType>
		);
	}

	void LogImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			log,
			LibSvcVectorLog<ThisType>
		);
	}

	void Log2Impl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			log2,
			LibSvcVectorLog2<ThisType>
		);
	}

	void Log10Impl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			log10,
			LibSvcVectorLog10<ThisType>
		);
	}

	LibSvcMonoOperatorFunctionImpl(Abs, AbsImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Abs, AbsImpl);
	LibSvcMonoOperatorFunctionImpl(Sin, SinImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Sin, SinImpl);
	LibSvcMonoOperatorFunctionImpl(Sinh, SinhImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Sinh, SinhImpl);
	LibSvcMonoOperatorFunctionImpl(Cos, CosImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Cos, CosImpl);
	LibSvcMonoOperatorFunctionImpl(Cosh, CoshImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Cosh, CoshImpl);
	LibSvcMonoOperatorFunctionImpl(Tan, TanImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Tan, TanImpl);
	LibSvcMonoOperatorFunctionImpl(Tanh, TanhImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Tanh, TanhImpl);
	LibSvcMonoOperatorFunctionImpl(ASin, ASinImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ASin, ASinImpl);
	LibSvcMonoOperatorFunctionImpl(ACos, ACosImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ACos, ACosImpl);
	LibSvcMonoOperatorFunctionImpl(ATan, ATanImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ATan, ATanImpl);
	LibSvcMonoOperatorFunctionImpl(ASinh, ASinhImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ASinh, ASinhImpl);
	LibSvcMonoOperatorFunctionImpl(ACosh, ACoshImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ACosh, ACoshImpl);
	LibSvcMonoOperatorFunctionImpl(ATanh, ATanhImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(ATanh, ATanhImpl);
	LibSvcMonoOperatorFunctionImpl(Exp, ExpImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Exp, ExpImpl);
	LibSvcMonoOperatorFunctionImpl(Exp2, Exp2Impl);
	LibSvcMonoOperatorInplaceFunctionImpl(Exp2, Exp2Impl);
	LibSvcMonoOperatorFunctionImpl(Exp10, Exp10Impl);
	LibSvcMonoOperatorInplaceFunctionImpl(Exp10, Exp10Impl);
	LibSvcMonoOperatorFunctionImpl(Log, LogImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Log, LogImpl);
	LibSvcMonoOperatorFunctionImpl(Log2, Log2Impl);
	LibSvcMonoOperatorInplaceFunctionImpl(Log2, Log2Impl);
	LibSvcMonoOperatorFunctionImpl(Log10, Log10Impl);
	LibSvcMonoOperatorInplaceFunctionImpl(Log10, Log10Impl);

	void LessImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Less<ThisType>
		);
	}

	void LessImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Less<ThisType>
		);
	}

	void GreaterImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Greater<ThisType>
		);
	}

	void GreaterImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Greater<ThisType>
		);
	}

	void EqualImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Equal<ThisType>
		);
	}

	void EqualImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::Equal<ThisType>
		);
	}

	void LessEqualImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::LessEqual<ThisType>
		);
	}

	void LessEqualImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::LessEqual<ThisType>
		);
	}

	void GreaterEqualImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::GreaterEqual<ThisType>
		);
	}

	void GreaterEqualImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::GreaterEqual<ThisType>
		);
	}

	void NotEqualImpl(const Tensor& _Dst, const Tensor& _Src1, const Tensor& _Src2, const SizeType CurDims)
	{
		CompareOperators<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::NotEqual<ThisType>
		);
	}

	void NotEqualImplScalar(const Tensor& _Dst, const Tensor& _Src1, const ThisType& _Src2, const SizeType CurDims)
	{
		CompareOperatorsScalar<ThisType>(
			_Dst,
			_Src1,
			_Src2,
			CurDims,
			::libsvc::NotEqual<ThisType>
		);
	}

	LibSvcCompareOperatorFunctionImpl(Less, LessImpl);
	LibSvcCompareOperatorScalarFunctionImpl(Less, LessImplScalar);
	LibSvcCompareOperatorFunctionImpl(Greater, GreaterImpl);
	LibSvcCompareOperatorScalarFunctionImpl(Greater, GreaterImplScalar);
	LibSvcCompareOperatorFunctionImpl(Equal, EqualImpl);
	LibSvcCompareOperatorScalarFunctionImpl(Equal, EqualImplScalar);
	LibSvcCompareOperatorFunctionImpl(LessEqual, LessEqualImpl);
	LibSvcCompareOperatorScalarFunctionImpl(LessEqual, LessEqualImplScalar);
	LibSvcCompareOperatorFunctionImpl(GreaterEqual, GreaterEqualImpl);
	LibSvcCompareOperatorScalarFunctionImpl(GreaterEqual, GreaterEqualImplScalar);
	LibSvcCompareOperatorFunctionImpl(NotEqual, NotEqualImpl);
	LibSvcCompareOperatorScalarFunctionImpl(NotEqual, NotEqualImplScalar);

	void CeilImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			ceil,
			LibSvcVectorCeil<ThisType>
		);
	}

	void RoundImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			round,
			LibSvcVectorRound<ThisType>
		);
	}

	void FloorImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		using _MonoOpFnArgType = std::_Common_float_type_t<ThisType, ThisType>;
		using _MonoOpFnType = _MonoOpFnArgType(*)(_MonoOpFnArgType);
		MonoOperators<ThisType, _MonoOpFnType>(
			_Dst,
			_Src,
			CurDims,
			floor,
			LibSvcVectorFloor<ThisType>
		);
	}

	LibSvcMonoOperatorFunctionImpl(Ceil, CeilImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Ceil, CeilImpl);
	LibSvcMonoOperatorFunctionImpl(Round, RoundImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Round, RoundImpl);
	LibSvcMonoOperatorFunctionImpl(Floor, FloorImpl);
	LibSvcMonoOperatorInplaceFunctionImpl(Floor, FloorImpl);

	void SumImpl(const Tensor& _Dst, const Tensor& _Src, const SizeType CurDims)
	{
		ThisType* DataPtr1 = (ThisType*)_Dst.Data();
		const ThisType* DataPtr2 = (ThisType*)_Src.Data();

		if (CurDims > 6)
		{
			auto Steps1 = _Dst.StepsBack();
			for (auto& i : Steps1)
				i /= sizeof(ThisType);
			auto Steps2 = _Src.StepsBack();
			for (auto& i : Steps2)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Src.Shape().data();
			const SizeType* __restrict StepPtr1 = Steps1.data();
			const SizeType* __restrict StepPtr2 = Steps2.data();
			const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
			const SizeType* __restrict BeginsPtr2 = _Src.SliceBegins().data();
			const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
			const SizeType* __restrict StridesPtr2 = _Src.Strides().data();
			const auto LoopDim = CurDims - 1;
			ShapeType CurIndice(LoopDim, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				LoopDim,
				{
					SizeType Index1 = 0;
					SizeType Index2 = 0;
					for (SizeType i = 0; i < LoopDim; ++i)
					{
						Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
						Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
					}
					DataPtr1[Index1] = ThisType(0);
					for (SizeType i = 0; i < ShapePtr[LoopDim]; ++i)
					{
						const auto IndexAxisLB = Index2 + ((i * StridesPtr2[LoopDim]) + BeginsPtr2[LoopDim]) * StepPtr2[LoopDim];
						DataPtr1[Index1] += DataPtr2[IndexAxisLB];
					}
				}
			);

			return;
		}

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
		{
			auto Steps = _Dst.StepsBack();
			for (auto& i : Steps)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Dst.Shape().data();
			const SizeType* __restrict StepPtr = Steps.data();
			const SizeType* __restrict BeginsPtr = _Dst.SliceBegins().data();
			const SizeType* __restrict StridesPtr = _Dst.Strides().data();
			const auto LoopDim = CurDims - 1;
			ShapeType CurIndice(LoopDim, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				LoopDim,
				{
					SizeType Index = 0;
					for (SizeType i = 0; i < LoopDim; ++i)
						Index += ((IndicesPtr[i] * StridesPtr[i]) + BeginsPtr[i]) * StepPtr[i];
					for (SizeType ldvar = 1; ldvar < ShapePtr[LoopDim]; ++ldvar)
					{
						const auto IndexAxisCur = Index +
							((ldvar * StridesPtr[LoopDim]) + BeginsPtr[LoopDim]) * StepPtr[LoopDim];
						const auto IndexAxisLast = Index +
							(((ldvar - 1) * StridesPtr[LoopDim]) + BeginsPtr[LoopDim]) * StepPtr[LoopDim];
						DataPtr[IndexAxisCur] += DataPtr[IndexAxisLast];
					}
				}
			);

			return;
		}

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
		{
			auto Steps = _Dst.StepsBack();
			for (auto& i : Steps)
				i /= sizeof(ThisType);
			const SizeType* __restrict ShapePtr = _Dst.Shape().data();
			const SizeType* __restrict StepPtr = Steps.data();
			const SizeType* __restrict BeginsPtr = _Dst.SliceBegins().data();
			const SizeType* __restrict StridesPtr = _Dst.Strides().data();
			const auto LoopDim = CurDims - 1;
			ShapeType CurIndice(LoopDim, 0);
			SizeType* __restrict IndicesPtr = CurIndice.data();
			LibSvcCycle(
				IndicesPtr,
				ShapePtr,
				LoopDim,
				{
					SizeType Index = 0;
					for (SizeType i = 0; i < LoopDim; ++i)
						Index += ((IndicesPtr[i] * StridesPtr[i]) + BeginsPtr[i]) * StepPtr[i];
					for (SizeType ldvar = 1; ldvar < ShapePtr[LoopDim]; ++ldvar)
					{
						const auto IndexAxisCur = Index +
							((ldvar * StridesPtr[LoopDim]) + BeginsPtr[LoopDim]) * StepPtr[LoopDim];
						const auto IndexAxisLast = Index +
							(((ldvar - 1) * StridesPtr[LoopDim]) + BeginsPtr[LoopDim]) * StepPtr[LoopDim];
						DataPtr[IndexAxisCur] *= DataPtr[IndexAxisLast];
					}
				}
			);

			return;
		}

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
		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
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
		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
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
		if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE)
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

	void NearestInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{
		
	}

	void LinearInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{
		
	}

	void BilinearInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{

	}

	void TrilinearInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{

	}

	void BicubicInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{

	}

	void AreaInterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		const SizeType _InterpDim
	)
	{

	}

	void InterpolateImpl(
		const Tensor& _Dst,
		const Tensor& _Src,
		const SizeType CurDims,
		InterpolateType _Type = InterpolateType::Nearest
	)
	{
		if(_Type == InterpolateType::Nearest)
			NearestInterpolateImpl(_Dst, _Src, CurDims, 1);
		else if (_Type == InterpolateType::Linear)
			LinearInterpolateImpl(_Dst, _Src, CurDims, 1);
		else if (_Type == InterpolateType::Bilinear)
			BilinearInterpolateImpl(_Dst, _Src, CurDims, 2);
		else if (_Type == InterpolateType::Trilinear)
			TrilinearInterpolateImpl(_Dst, _Src, CurDims, 3);
		else if (_Type == InterpolateType::Bicubic)
			BicubicInterpolateImpl(_Dst, _Src, CurDims, 3);
		else if (_Type == InterpolateType::Area)
			AreaInterpolateImpl(_Dst, _Src, CurDims, 3);
	}
}

LibSvcEnd