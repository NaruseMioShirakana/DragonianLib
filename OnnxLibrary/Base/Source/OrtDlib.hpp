#pragma once
#include "../EnvManager.hpp"

template <>
struct Ort::TypeToTensorType<DragonianLib::Float16>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::BFloat16>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Complex32>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Complex64>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Float8E4M3FN>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Float8E4M3FNUZ>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Float8E5M2>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
};

template <>
struct Ort::TypeToTensorType<DragonianLib::Float8E5M2FNUZ>
{
	static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ;
};

template <typename _Type>
constexpr auto TypeToOnnxTensorType = Ort::TypeToTensorType<_Type>::type;

_D_Dragonian_Lib_Onnx_Runtime_Header

struct InputTensorsType
{
protected:
	OrtTuple OrtTensors;
	DlibTuple DlibTensors;

public:
	InputTensorsType() = default;
	InputTensorsType& Emplace(std::pair<Ort::Value, std::shared_ptr<DlibValue>>&& _InputTensor)
	{
		auto Ten = std::move(_InputTensor);
		OrtTensors.emplace_back(std::move(Ten.first));
		DlibTensors.emplace_back(std::move(Ten.second));
		return *this;
	}
	InputTensorsType& Emplace(Ort::Value&& _OrtTensor, std::shared_ptr<DlibValue>&& _DlibTensor)
	{
		OrtTensors.emplace_back(std::move(_OrtTensor));
		DlibTensors.emplace_back(std::move(_DlibTensor));
		return *this;
	}
	InputTensorsType& Emplace(Ort::Value&& _OrtTensor)
	{
		OrtTensors.emplace_back(std::move(_OrtTensor));
		DlibTensors.emplace_back(nullptr);
		return *this;
	}

	operator OrtTuple& () noexcept
	{
		return OrtTensors;
	}
	operator const OrtTuple& () const noexcept
	{
		return OrtTensors;
	}
};

template <typename _MyValueType, size_t _NRank>
auto CreateTensorViewFromOrtValue(
	Ort::Value&& _Value,
	const Dimensions<_NRank>& _Shape
)
{
	const auto BufferSize = _Shape.Multiply();
	const auto ElementCount = static_cast<Int64>(_Value.GetTensorTypeAndShapeInfo().GetElementCount());
	const auto ElementType = _Value.GetTensorTypeAndShapeInfo().GetElementType();

	if (ElementCount != BufferSize)
		_D_Dragonian_Lib_Throw_Exception("Size mismatch, expected: " + std::to_string(BufferSize) + ", got: " + std::to_string(ElementCount));

	try
	{
		if (ElementType == TypeToOnnxTensorType<_MyValueType>)
		{
			auto Data = _Value.GetTensorMutableData<_MyValueType>();
			return Functional::FromShared<_MyValueType, _NRank, Device::CPU>(
				_Shape,
				{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
				BufferSize
			);
		}

		if constexpr (TypeTraits::IsFloatingPointValue<_MyValueType>)
		{
			if (ElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
			{
				auto Data = _Value.GetTensorMutableData<Ort::Float16_t>();
				return Functional::FromShared<Float16, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
			if (ElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
			{
				auto Data = _Value.GetTensorMutableData<Ort::BFloat16_t>();
				return Functional::FromShared<BFloat16, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
		}
		else if constexpr (TypeTraits::IsIntegerValue<_MyValueType>)
		{
			if (ElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
			{
				auto Data = _Value.GetTensorMutableData<int8_t>();
				return Functional::FromShared<int8_t, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
			if (ElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
			{
				auto Data = _Value.GetTensorMutableData<int16_t>();
				return Functional::FromShared<int16_t, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
			if (ElementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
			{
				auto Data = _Value.GetTensorMutableData<int32_t>();
				return Functional::FromShared<int32_t, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
		}
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
	_D_Dragonian_Lib_Throw_Exception("Type mismatch, expected: " + std::to_string(TypeToOnnxTensorType<_MyValueType>) + ", got: " + std::to_string(_Value.GetTensorTypeAndShapeInfo().GetElementType()));
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
std::pair<Ort::Value, std::shared_ptr<DlibValue>> CreateValueFromTensor(
	const OrtMemoryInfo* _MyMemoryInfo,
	const Tensor<_TensorType, _NRank, _MyDevice>& _Tensor,
	const UInt64 _InputAxisCount,
	size_t _AxisOffset
)
{
	auto Shared = _Tensor.CreateShared();
	const auto TensorShape = Shared->Shape().Data();
	auto TensorData = Shared->Data();
	const auto ElementCount = Shared->ElementCount();
	try
	{
		return {
			Ort::Value::CreateTensor(
				_MyMemoryInfo,
				TensorData,
				static_cast<size_t>(ElementCount),
				TensorShape + _AxisOffset,
				_InputAxisCount
			),
			Shared
		};
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
auto CheckAndTryCreateValueFromTensor(
	const OrtMemoryInfo* _MyMemoryInfo,
	const Tensor<_TensorType, _NRank, _MyDevice>& _InputTensor,
	ONNXTensorElementDataType _DataType,
	const TemplateLibrary::Vector<Int64>& _InputShapes,
	const TemplateLibrary::Array<const wchar_t*, _NRank>& _AxisNames,
	const char* _TensorName,
	const DLogger& _Logger = nullptr
)
{
	auto _Tensor = _InputTensor.Continuous().Evaluate();
	const auto& TensorShape = _Tensor.Shape();
	const auto TensorAxisCount = TensorShape.Size();
	const auto InputAxisCount = _InputShapes.Size();

	if (TensorAxisCount < InputAxisCount)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid tensor axis, expected: " +
			std::to_string(InputAxisCount) +
			", got: " +
			std::to_string(TensorAxisCount) +
			", input name of the tensor is: \"" +
			_TensorName +
			"\""
		);

	const auto AxisOffset = TensorAxisCount - InputAxisCount;

	for (UInt64 i = 0; i < InputAxisCount; ++i)
	{
		if (_InputShapes[i] != -1 && _InputShapes[i] != TensorShape[i + AxisOffset])
			_D_Dragonian_Lib_Throw_Exception(
				"Invalid tensor shape at axis \"" +
				WideStringToUTF8(_AxisNames[i + AxisOffset]) +
				"\", expected: " +
				std::to_string(_InputShapes[i]) +
				", got: " +
				std::to_string(TensorShape[i + AxisOffset]) +
				", input name of the tensor is: \"" +
				_TensorName +
				"\""
			);
	}

	if (_DataType != TypeToOnnxTensorType<_TensorType>)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_TensorType>)
		{
			if ((_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
				_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) && _Logger)
				_Logger->LogWarn(
					L"Input tensor: \"" +
					UTF8ToWideString(_TensorName) +
					L"\" of this model is half precision, but the input tensor is single precision," +
					L"input will automatically converting to half precision"
				);
			if (_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
				return CreateValueFromTensor<BFloat16, _NRank, _MyDevice>(
					_MyMemoryInfo,
					_Tensor.template Cast<BFloat16>().Evaluate(),
					InputAxisCount,
					AxisOffset
				);
			if (_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
				return CreateValueFromTensor<Float16, _NRank, _MyDevice>(
					_MyMemoryInfo,
					_Tensor.template Cast<Float16>().Evaluate(),
					InputAxisCount,
					AxisOffset
				);
		}
		else if constexpr (TypeTraits::IsIntegerValue<_TensorType>)
		{
			if ((_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
				_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
				_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) && _Logger)
				_Logger->LogWarn(
					L"Input tensor: \"" +
					UTF8ToWideString(_TensorName) +
					L"\" of this model is lower bit depth, but the input tensor is higher bit depth," +
					L"input will automatically converting to lower bit depth"
				);
			if (_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
				return CreateValueFromTensor<Int8, _NRank, _MyDevice>(
					_MyMemoryInfo,
					_Tensor.template Cast<Int8>().Evaluate(),
					InputAxisCount,
					AxisOffset
				);
			if (_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
				return CreateValueFromTensor<Int16, _NRank, _MyDevice>(
					_MyMemoryInfo,
					_Tensor.template Cast<Int16>().Evaluate(),
					InputAxisCount,
					AxisOffset
				);
			if (_DataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
				return CreateValueFromTensor<Int32, _NRank, _MyDevice>(
					_MyMemoryInfo,
					_Tensor.template Cast<Int32>().Evaluate(),
					InputAxisCount,
					AxisOffset
				);
		}

		_D_Dragonian_Lib_Throw_Exception(
			"Invalid tensor type, expected: " +
			std::to_string(_DataType) +
			", got: " +
			std::to_string(TypeToOnnxTensorType<_TensorType>) +
			", input name of the tensor is: \"" +
			_TensorName +
			"\""
		);
	}

	return CreateValueFromTensor(_MyMemoryInfo, _Tensor, InputAxisCount, AxisOffset);
}

_D_Dragonian_Lib_Onnx_Runtime_End