
#include "Tensor/Float32Tensor.h"


LibSvcBegin

namespace Float32
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

	void AssignValue(const Tensor& _Input, void* _Val, TensorType _ValType)
	{
		const auto CurDimSize = _Input.Shape().front();
		const auto CurDims = (SizeType)_Input.Shape().size();

		if (CurDims > 4)
			for (auto i = 0; i < CurDimSize; ++i)
				AssignValue(_Input[i], _Val, _ValType);
		else
			AssignImpl(_Input, _Val, _ValType, CurDims);
	}

	void AssignBufferImpl(
		const Tensor& _Input,
		const ThisType* __restrict Buffer,
		const ThisType* __restrict BufferEnd,
		SizeType CurDims
	) {
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

	void AssignBuffer(const Tensor& _Input, cpvoid BufferVoid, cpvoid BufferEndVoid)
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
					AssignBuffer(_Input, Buffer, BufferEnd);
					return;
				}
				AssignBuffer(_Input, Buffer, Buffer + RequiredBufSize);
				Buffer += RequiredBufSize;
			}
		else
			AssignBufferImpl(_Input, (const ThisType*)Buffer, (const ThisType*)BufferEnd, CurDims);
	}

	void AssignTensorImpl(const Tensor& _InputA, const Tensor& _InputB)
	{
		
	}

	void AssignTensor(const Tensor& _InputA, const Tensor& _InputB)
	{
		if(_InputB.IsScalar())
		{
			AssignValue(_InputA, _InputB.Data(), _InputB.DType());
			return;
		}
		std::vector<bool> BroadCast(std::max(_InputA.Shape().size(), _InputB.Shape().size()));
	}

}

LibSvcEnd