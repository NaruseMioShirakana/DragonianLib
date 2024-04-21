
#include "Tensor/Int32Tensor.h"

LibSvcBegin

namespace Int32
{

	ThisType CastFrom(TensorType _Type, void* _Val)
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

	void AssignImpl(const Tensor& _Input, void* _Val, TensorType _ValType, SizeType CurDims)
	{
		const auto _Value = CastFrom(_ValType, _Val);
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (_Input.IsContinuous())
		{
			const size_t BufferSize = VectorMul(_Input.Shape()) * sizeof(ThisType);
			LibSvcMemSet(DataPtr, &_Value, BufferSize, sizeof(ThisType));
			return;
		}

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= DType2Size(_Input.DType());
		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();
		if (CurDims == 4)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
						for (SizeType l = 0; l < ShapePtr[3]; ++l)
						{
							DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
								((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
								((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2] +
								((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3]] = _Value;
						}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
					{
						DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
							((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
							((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2]] = _Value;
					}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
				{
					DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
						((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1]] = _Value;
				}
		}
		else if (CurDims == 1)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
			{
				DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]] = _Value;
			}
		}
	}

	void AssignBufferImpl(
		const Tensor& _Input,
		const ThisType* __restrict Buffer,
		const ThisType* __restrict BufferEnd,
		SizeType CurDims
	) {
		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		if (BufferEnd < Buffer)
		{
			LogMessage("[Operator] BufferEnd* < Buffer* Is True, Make Sure BufferEnd* > Buffer*");
			return;
		}

		if (_Input.IsContinuous())
		{
			const size_t BufferSize = (BufferEnd - Buffer) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr, Buffer, BufferSize);
			return;
		}

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= DType2Size(_Input.DType());
		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();
		if (CurDims == 4)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
						for (SizeType l = 0; l < ShapePtr[3]; ++l)
						{
							DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
								((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
								((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2] +
								((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3]] = *(Buffer++);
							if (Buffer == BufferEnd)
								return;
						}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
					{
						DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
							((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
							((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2]] = *(Buffer++);
						if (Buffer == BufferEnd)
							return;
					}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
				{
					DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
						((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1]] = *(Buffer++);
					if (Buffer == BufferEnd)
						return;
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

	void AssignTensorImpl(const Tensor& _InputA, const Tensor& _InputB, SizeType CurDims)
	{
		ThisType* DataPtr1 = (ThisType*)_InputA.Data();
		const ThisType* DataPtr2 = (ThisType*)_InputB.Data();

		if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
		{
			const size_t BufferSize = VectorMul(_InputA.Shape()) * sizeof(ThisType);
			LibSvcMemCpy(DataPtr1, DataPtr2, BufferSize);
		}

		auto Steps1 = _InputA.StepsBack();
		for (auto& i : Steps1)
			i /= DType2Size(_InputA.DType());
		auto Steps2 = _InputB.StepsBack();
		for (auto& i : Steps2)
			i /= DType2Size(_InputB.DType());
		const SizeType* __restrict ShapePtr = _InputA.Shape().data();
		const SizeType* __restrict StepPtr1 = Steps1.data();
		const SizeType* __restrict StepPtr2 = Steps2.data();
		const SizeType* __restrict BeginsPtr1 = _InputA.SliceBegins().data();
		const SizeType* __restrict BeginsPtr2 = _InputB.SliceBegins().data();
		const SizeType* __restrict StridesPtr1 = _InputA.Strides().data();
		const SizeType* __restrict StridesPtr2 = _InputB.Strides().data();
		if (CurDims == 4)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
						for (SizeType l = 0; l < ShapePtr[3]; ++l)
						{
							DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0] +
								((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1] +
								((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2] +
								((l * StridesPtr1[3]) + BeginsPtr1[3]) * StepPtr1[3]] =
								DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0] +
								((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1] +
								((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2] +
								((l * StridesPtr2[3]) + BeginsPtr2[3]) * StepPtr2[3]];
						}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
					{
						DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0] +
							((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1] +
							((k * StridesPtr1[2]) + BeginsPtr1[2]) * StepPtr1[2]] =
							DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0] +
							((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1] +
							((k * StridesPtr2[2]) + BeginsPtr2[2]) * StepPtr2[2]];
					}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
				{
					DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0] +
						((j * StridesPtr1[1]) + BeginsPtr1[1]) * StepPtr1[1]] =
						DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0] +
						((j * StridesPtr2[1]) + BeginsPtr2[1]) * StepPtr2[1]];
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

	void FixWithRandomImpl(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, SizeType CurDims)
	{
		std::mt19937_64  RndDevice(_Seed);
		std::normal_distribution NormGen(_Mean, _Sigma);

		ThisType* __restrict DataPtr = (ThisType*)_Input.Data();

		auto Steps = _Input.StepsBack();
		for (auto& i : Steps)
			i /= DType2Size(_Input.DType());
		const SizeType* __restrict ShapePtr = _Input.Shape().data();
		const SizeType* __restrict StepPtr = Steps.data();
		const SizeType* __restrict BeginsPtr = _Input.SliceBegins().data();
		const SizeType* __restrict StridesPtr = _Input.Strides().data();
		if (CurDims == 4)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
						for (SizeType l = 0; l < ShapePtr[3]; ++l)
						{
							DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
								((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
								((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2] +
								((l * StridesPtr[3]) + BeginsPtr[3]) * StepPtr[3]] = (ThisType)NormGen(RndDevice);
						}
		}
		else if (CurDims == 3)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
					for (SizeType k = 0; k < ShapePtr[2]; ++k)
					{
						DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
							((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1] +
							((k * StridesPtr[2]) + BeginsPtr[2]) * StepPtr[2]] = (ThisType)NormGen(RndDevice);
					}
		}
		else if (CurDims == 2)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
				for (SizeType j = 0; j < ShapePtr[1]; ++j)
				{
					DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0] +
						((j * StridesPtr[1]) + BeginsPtr[1]) * StepPtr[1]] = (ThisType)NormGen(RndDevice);
				}
		}
		else if (CurDims == 1)
		{
			for (SizeType i = 0; i < ShapePtr[0]; ++i)
			{
				DataPtr[((i * StridesPtr[0]) + BeginsPtr[0]) * StepPtr[0]] = (ThisType)NormGen(RndDevice);
			}
		}
	}

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType, ThreadPool* _ThreadPool)
	{
		const auto CurDimSize = _Input.Shape().front();
		const auto CurDims = (SizeType)_Input.Shape().size();

		if (CurDims > 4)
			for (auto i = 0; i < CurDimSize; ++i)
				AssignValue(_Input[i], _Val, _ValType, _ThreadPool);
		else if (_ThreadPool)
			_ThreadPool->Commit(AssignImpl, _Input, _Val, _ValType, CurDims);
		else
			AssignImpl(_Input, _Val, _ValType, CurDims);
	}

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid, ThreadPool* _ThreadPool)
	{
		const byte* Buffer = (const byte*)BufferVoid;
		const byte* BufferEnd = (const byte*)BufferEndVoid;
		if ((BufferEnd - Buffer) % DType2Size(_Input.DType()))
			return;
		const auto CurDimSize = _Input.Shape().front();
		const auto CurDims = (SizeType)_Input.Shape().size();

		if (CurDims > 4)
			for (auto i = 0; i < CurDimSize; ++i)
			{
				auto InputTen = _Input[i];
				const auto BufSize = BufferEnd - Buffer;
				const auto RequiredBufSize = VectorMul(InputTen.Shape()) * DType2Size(InputTen.DType());
				if (BufSize <= 0)
					return;
				if (BufSize <= RequiredBufSize)
				{
					AssignBuffer(_Input, Buffer, BufferEnd, _ThreadPool);
					return;
				}
				AssignBuffer(_Input, Buffer, Buffer + RequiredBufSize, _ThreadPool);
				Buffer += RequiredBufSize;
			}
		else if (_ThreadPool)
			_ThreadPool->Commit(AssignBufferImpl, _Input, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims);
		else
			AssignBufferImpl(_Input, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims);
	}

	void AssignTensorBroadCasted(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		const auto CurDimSize = _InputA.Shape().front();
		const auto CurDims = (SizeType)_InputA.Shape().size();

		if (CurDims > 4)
			for (auto i = 0; i < CurDimSize; ++i)
				AssignTensorBroadCasted(_InputA[i], _InputB[i], _ThreadPool);
		else if (_ThreadPool)
			_ThreadPool->Commit(AssignTensorImpl, _InputA, _InputB, CurDims);
		else
			AssignTensorImpl(_InputA, _InputB, CurDims);
	}

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool)
	{
		if (_InputB.IsScalar())
		{
			AssignValue(_InputA, _InputB.Data(), _InputB.DType(), _ThreadPool);
			return;
		}

		const auto BroadCast = _InputA.BroadCast(_InputB);

		AssignTensorBroadCasted(BroadCast.first, BroadCast.second, _ThreadPool);
	}

	void FixWithRandom(const Tensor& _Input, uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool)
	{
		const auto CurDimSize = _Input.Shape().front();
		const auto CurDims = (SizeType)_Input.Shape().size();

		if (CurDims > 4)
			for (auto i = 0; i < CurDimSize; ++i)
				FixWithRandom(_Input[i], _Seed, _Mean, _Sigma, _ThreadPool);
		else if (_ThreadPool)
			_ThreadPool->Commit(FixWithRandomImpl, _Input, _Seed, _Mean, _Sigma, CurDims);
		else
			FixWithRandomImpl(_Input, _Seed, _Mean, _Sigma, CurDims);
	}

}

LibSvcEnd