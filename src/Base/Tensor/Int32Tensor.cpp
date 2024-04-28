
#include "Tensor/Int32Tensor.h"

LibSvcBegin

namespace Int32
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

		if (_Input.IsContinuous())
		{
			DataPtr = (ThisType*)_Input.GetPtr();
			const size_t BufferSize = VectorMul(_Input.Shape()) * sizeof(ThisType);
			LibSvcMemSet(DataPtr, &_Value, BufferSize, sizeof(ThisType));
			return;
		}

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= sizeof(ThisType);

		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();

		if (CurDims > 5)
		{
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

		auto Cont = _Input.CalcContinuous();
		Cont.resize(5);
		const SizeType* __restrict ContPtr = Cont.data();
		const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

		if (CurDims == 5)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
							{
								const auto IndexAxis4 = IndexAxis3 +
									((m * StridesPtr[Axis4]) + BeginsPtr[Axis4]) * StepPtr[Axis4];
								DataPtr[IndexAxis4] = _Value;
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
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							DataPtr[IndexAxis3] = _Value;
						}
					}
				}
			}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataPtr[IndexAxis2] = _Value;
					}
				}
			}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataPtr[IndexAxis1] = _Value;
				}
			}
		}
		else if (CurDims == 1)
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]] = _Value;
	}

	void AssignBufferImpl(
		const Tensor& _Input,
		const ThisType* __restrict Buffer,
		const ThisType* __restrict BufferEnd,
		const SizeType CurDims
	) {
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (BufferEnd < Buffer)
		{
			LogMessage("[Operator] BufferEnd* < Buffer* Is True, Make Sure BufferEnd* > Buffer*");
			return;
		}

		if (_Input.IsContinuous())
		{
			DataPtr = (ThisType*)_Input.GetPtr();
			const size_t BufferSize = (BufferEnd - Buffer) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr, Buffer, BufferSize);
			return;
		}

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= sizeof(ThisType);
		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();

		if (CurDims > 5)
		{
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

		auto Cont = _Input.CalcContinuous();
		Cont.resize(5);
		const SizeType* __restrict ContPtr = Cont.data();
		const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

		if (CurDims == 5)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
							{
								const auto IndexAxis4 = IndexAxis3 +
									((m * StridesPtr[Axis4]) + BeginsPtr[Axis4]) * StepPtr[Axis4];
								DataPtr[IndexAxis4] = *(Buffer++);
								if (Buffer == BufferEnd)
									return;
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
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							DataPtr[IndexAxis3] = *(Buffer++);
							if (Buffer == BufferEnd)
								return;
						}
					}
				}
			}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataPtr[IndexAxis2] = *(Buffer++);
						if (Buffer == BufferEnd)
							return;
					}
				}
			}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataPtr[IndexAxis1] = *(Buffer++);
					if (Buffer == BufferEnd)
						return;
				}
			}
		}
		else if (CurDims == 1)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
			{
				DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]] = *(Buffer++);
				if (Buffer == BufferEnd)
					return;
			}
		}
	}

	void AssignTensorImpl(const Tensor& _InputA, const Tensor& _InputB, const SizeType CurDims)
	{
		ThisType* DataPtr1 = (ThisType*)_InputA.Data();
		const ThisType* DataPtr2 = (ThisType*)_InputB.Data();

		if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
		{
			DataPtr1 = (ThisType*)_InputA.GetPtr();
			DataPtr2 = (ThisType*)_InputB.GetPtr();
			const size_t BufferSize = VectorMul(_InputA.Shape()) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr1, DataPtr2, BufferSize);
			return;
		}

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
					DataPtr1[Index1] = DataPtr2[Index2];
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
							for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
							{
								const auto IndexAxis4A = IndexAxis3A +
									((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
								const auto IndexAxis4B = IndexAxis3B +
									((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
								DataPtr1[IndexAxis4A] = DataPtr2[IndexAxis4B];
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
							DataPtr1[IndexAxis3A] = DataPtr2[IndexAxis3B];
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
						DataPtr1[IndexAxis2A] = DataPtr2[IndexAxis2B];
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
					DataPtr1[IndexAxis1A] = DataPtr2[IndexAxis1B];
				}
			}
		}
		else if (CurDims == 1)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
			{
				DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
					DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]];
			}
		}
	}

	void FixWithRandomImpl(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, const SizeType CurDims)
	{
		std::mt19937_64  RndDevice(_Seed);
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

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= sizeof(ThisType);
		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();

		if (CurDims > 5)
		{
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

		auto Cont = _Input.CalcContinuous();
		Cont.resize(5);
		const SizeType* __restrict ContPtr = Cont.data();
		const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

		if (CurDims == 5)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
							{
								const auto IndexAxis4 = IndexAxis3 +
									((m * StridesPtr[Axis4]) + BeginsPtr[Axis4]) * StepPtr[Axis4];
								DataPtr[IndexAxis4] = (ThisType)NormGen(RndDevice);
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
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							DataPtr[IndexAxis3] = (ThisType)NormGen(RndDevice);
						}
					}
				}
			}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataPtr[IndexAxis2] = (ThisType)NormGen(RndDevice);
					}
				}
			}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataPtr[IndexAxis1] = (ThisType)NormGen(RndDevice);
				}
			}
		}
		else if (CurDims == 1)
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]] = (ThisType)NormGen(RndDevice);
	}

	void AssignValue(const Tensor& _Input, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool)
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
						_ThreadPool->Commit(AssignImpl, SqueezedTensor.Slice(ThreadSlices), _Val, _ValType, CurDims);
						if (End == SqueezedShape[i])
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

		if (_InputA.IsBroadCasted())
			LibSvcThrow("You Can't Assign To A BroadCasted Tensor!");

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
		auto Steps = _Indices.StepsBack();
		for (auto& i : Steps)
			i /= sizeof(ThisType);

		const SizeType* __restrict ShapePtr = _Indices.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Indices.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Indices.Strides().data();

		const ThisType* IndicePtr = (ThisType*)_Indices.Data();

		auto Cont = _Indices.CalcContinuous();
		Cont.resize(5);
		const SizeType* __restrict ContPtr = Cont.data();
		const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];
		ShapeType DataIndice(CurDims, 0);

		if (CurDims == 5)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				DataIndice[Axis0] = i;
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataIndice[Axis1] = j;
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataIndice[Axis2] = k;
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							DataIndice[Axis3] = l;
							for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
							{
								const auto IndexAxis4 = IndexAxis3 +
									((m * StridesPtr[Axis4]) + BeginsPtr[Axis4]) * StepPtr[Axis4];
								DataIndice[Axis4] = m;
								_Ret[DataIndice].Assign(_Input[SizeType(IndicePtr[IndexAxis4])]);
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
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				DataIndice[Axis0] = i;
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataIndice[Axis1] = j;
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataIndice[Axis2] = k;
						for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
						{
							const auto IndexAxis3 = IndexAxis2 +
								((l * StridesPtr[Axis3]) + BeginsPtr[Axis3]) * StepPtr[Axis3];
							DataIndice[Axis3] = l;
							_Ret[DataIndice].Assign(_Input[SizeType(IndicePtr[IndexAxis3])]);
						}
					}
				}
			}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				DataIndice[Axis0] = i;
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataIndice[Axis1] = j;
					for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
					{
						const auto IndexAxis2 = IndexAxis1 +
							((k * StridesPtr[Axis2]) + BeginsPtr[Axis2]) * StepPtr[Axis2];
						DataIndice[Axis2] = k;
						_Ret[DataIndice].Assign(_Input[SizeType(IndicePtr[IndexAxis2])]);
					}
				}
			}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
			{
				const auto IndexAxis0 = ((i * StridesPtr[Axis0]) + BeginsPtr[Axis0]) * StepPtr[Axis0];
				DataIndice[Axis0] = i;
				for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
				{
					const auto IndexAxis1 = IndexAxis0 +
						((j * StridesPtr[Axis1]) + BeginsPtr[Axis1]) * StepPtr[Axis1];
					DataIndice[Axis1] = j;
					_Ret[DataIndice].Assign(_Input[SizeType(IndicePtr[IndexAxis1])]);
				}
			}
		}
		else if (CurDims == 1)
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				_Ret[i].Assign(_Input[SizeType(IndicePtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]])]);
	}

	Tensor Gather(const Tensor& _Input, const Tensor& _Indices, ThreadPool* _ThreadPool)
	{
		const auto& _InputShape = _Input.Shape();
		const auto& _IndicesShape = _Indices.Shape();
		ShapeType _NewShape(_IndicesShape.begin(), _IndicesShape.end());
		if (_InputShape.size() == 1)
			_NewShape.emplace_back(1);
		else
			_NewShape.insert(_NewShape.end(), _InputShape.begin() + 1, _InputShape.end());
		Tensor Ret(_NewShape, _Input.DType());
		const auto TotalSize = VectorMul(_NewShape);
		const auto CurDims = _Indices.DimCount();

		if (CurDims > 5)
			LibSvcThrow("Gather Operator Not Support Dim > 5!");

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
							_Input,
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
			GatherImpl(Ret, _Input, _Indices, CurDims);
		}
		else
			GatherImpl(Ret, _Input, _Indices, CurDims);

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
		using _MonoOpFnArgType = std::conditional_t<std::is_integral_v<ThisType>, int64, float64>;
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

}

LibSvcEnd