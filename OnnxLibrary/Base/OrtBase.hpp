/**
 * @file OrtBase.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Base classes for Onnx models in DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Created <
 *  > 2025/3/19 NaruseMioShirakana Added CreateEnvironment <
 *  > 2025/3/19 NaruseMioShirakana Added OnnxModelBase <
 */

#pragma once

#include "OnnxLibrary/Base/EnvManager.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

_D_Dragonian_Lib_Onnx_Runtime_Header

/**
 * @class DragonianLib::Tensor
 * @brief Tensor type
 * @tparam ValueType Value type of the tensor
 * @tparam Rank Rank of the tensor
 * @tparam Provider Device provider of the tensor
 */
template <typename ValueType, UInt64 Rank, Device Provider>
using Tensor = DragonianLib::Tensor<ValueType, Rank, Provider>;

/**
 * @brief Tuple of Ort values
 */
using OrtTuple = std::vector<Ort::Value>;

/**
 * @brief Value of Dlib
 */
using DlibTuple = std::vector<std::shared_ptr<DlibValue>>;

/**
 * @typedef ProgressCallback
 * @brief Callback function for progress updates
 */
using ProgressCallback = std::function<void(bool, Int64)>;

/**
 * @brief Creates a DragonianLibOrtEnv
 * @param Options Environment options
 * @return Onnx environment
 */
OnnxRuntimeEnvironment CreateEnvironment(
	const OnnxEnvironmentOptions& Options
);

Ort::AllocatorWithDefaultOptions& GetDefaultOrtAllocator();

template <typename _Type>
constexpr auto TypeToOnnxTensorType = Ort::TypeToTensorType<_Type>::type;

/**
 * @class OnnxModelBase
 * @brief Base class for Onnx models in MoeVoiceStudioCore
 */
template <typename Child>
class OnnxModelBase
{
public:
	using _TMyChild = Child;
    
    OnnxModelBase(
        const OnnxRuntimeEnvironment& _Environment,
        const std::wstring& _ModelPath,
        const std::shared_ptr<Logger>& _Logger = nullptr
    ) : _MyModelExecutionProvider(_Environment->GetProvider()), _MyEnvironment(_Environment),
        _MyOnnxEnvironment(_Environment->GetEnvironment()), _MySessionOptions(_Environment->GetSessionOptions()),
		_MyMemoryInfo(_Environment->GetMemoryInfo()), _MyRunOptions(std::make_shared<Ort::RunOptions>()),
		_MyModelPath(_ModelPath)
    {
        _D_Dragonian_Lib_Rethrow_Block(
            _MyModel = _D_Dragonian_Lib_Onnx_Runtime_Space RefOnnxRuntimeModel(_ModelPath, _Environment);
            );
        if (_Logger) _MyLogger = _Logger;
        else _MyLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();

		_D_Dragonian_Lib_Rethrow_Block(GetIOInfo(););
    }

    ~OnnxModelBase() noexcept
    {
		static auto _MyStaticLogger = _D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger();
		std::wstring HexPtr;
		{
			std::wstringstream wss;
			wss << std::hex << _MyModel.get();
			wss >> HexPtr;
		}
		_MyStaticLogger->LogMessage(L"UnReference Model: Instance[PTR:" + HexPtr + L", PATH:\"" + _MyModelPath + L"\"], Current Referece Count: " + std::to_wstring(_MyModel.use_count() - 1));
    }

    /**
     * @brief Get the DragonianLibOrtEnv
     * @return Reference to DragonianLibOrtEnv
     */
	template <typename _ThisType>
    decltype(auto) GetDlEnv(this _ThisType&& _Self)
    {
		if (!_Self._MyEnvironment)
			_D_Dragonian_Lib_Throw_Exception("Environment is not initialized");
        return *(std::forward<_ThisType>(_Self)._MyEnvironment);
    }

    /**
     * @brief Get the DragonianLibOrtEnv
     * @return Pointer to DragonianLibOrtEnv
     */
    template <typename _ThisType>
    decltype(auto) GetDlEnvPtr(this _ThisType&& _Self) noexcept
    {
        return std::forward<_ThisType>(_Self)._MyEnvironment;
    }

	/**
	 * @brief Get the logger
	 * @return Reference to the logger
	 */
    template <typename _ThisType>
    decltype(auto) GetLogger(this _ThisType&& _Self) noexcept
    {
		if (!_Self._MyLogger)
			return *(_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger());
        return *(std::forward<_ThisType>(_Self)._MyLogger);
    }

	/**
	 * @brief Get the logger
	 * @return Pointer to the logger
	 */
    template <typename _ThisType>
    decltype(auto) GetLoggerPtr(this _ThisType&& _Self) noexcept
    {
        return std::forward<_ThisType>(_Self)._MyLogger;
    }

	/**
	 * @brief Get the model
	 * @return Reference to the model
	 */
	template <typename _ThisType>
	decltype(auto) GetModel(this _ThisType&& _Self) noexcept
	{
		return std::forward<_ThisType>(_Self)._MyModel;
	}

	/**
	 * @brief Get the ort environment
	 * @return Ort environment
	 */
	template <typename _ThisType>
	decltype(auto) GetOrtEnv(this _ThisType&& _Self) noexcept
	{
		return std::forward<_ThisType>(_Self)._MyOnnxEnvironment;
	}

	/**
	 * @brief Get the session options
	 * @return Session options
	 */
	template <typename _ThisType>
	decltype(auto) GetSessionOptions(this _ThisType&& _Self) noexcept
	{
		return std::forward<_ThisType>(_Self)._MySessionOptions;
	}

	/**
	 * @brief Get the memory info
	 * @return Memory info
	 */
	template <typename _ThisType>
	decltype(auto) GetMemoryInfo(this _ThisType&& _Self) noexcept
	{
		return std::forward<_ThisType>(_Self)._MyMemoryInfo;
	}
protected:
	void LogInfo(const std::wstring& _Message) const noexcept
	{
		_MyLogger->Log(_Message, Logger::LogLevel::Info);
	}

	void LogWarn(const std::wstring& _Message) const noexcept
	{
		_MyLogger->Log(_Message, Logger::LogLevel::Warn);
	}

	void LogError(const std::wstring& _Message) const noexcept
	{
		_MyLogger->Log(_Message, Logger::LogLevel::Error);
	}

	template <typename _MyValueType, size_t _NRank>
	static auto CreateTensorViewFromOrtValue(Ort::Value&& _Value, const Dimensions<_NRank>& _Shape)
	{
		const auto BufferSize = _Shape.Multiply();
		if (_Value.GetTensorTypeAndShapeInfo().GetElementCount() != BufferSize)
			_D_Dragonian_Lib_Throw_Exception("Size mismatch");
		try
		{
			if (_Value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
			{
				auto Data = _Value.GetTensorMutableData<Ort::Float16_t>();
				return Functional::FromShared<Float16, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
			if (_Value.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
			{
				auto Data = _Value.GetTensorMutableData<Ort::BFloat16_t>();
				return Functional::FromShared<BFloat16, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				).template Cast<_MyValueType>().Evaluate();
			}
			if (_Value.GetTensorTypeAndShapeInfo().GetElementType() == TypeToOnnxTensorType<_MyValueType>)
			{
				auto Data = _Value.GetTensorMutableData<_MyValueType>();
				return Functional::FromShared<_MyValueType, _NRank, Device::CPU>(
					_Shape,
					{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
					BufferSize
				);
			}
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
		_D_Dragonian_Lib_Throw_Exception("Type mismatch, expected: " + std::to_string(TypeToOnnxTensorType<_MyValueType>) + ", got: " + std::to_string(_Value.GetTensorTypeAndShapeInfo().GetElementType()));
	}

	template <typename _TensorType, size_t _NRank, Device _MyDevice>
	static std::pair<Ort::Value, std::shared_ptr<DlibValue>> CreateValueFromTensor(
		const OrtMemoryInfo* _MyMemoryInfo,
		const Tensor<_TensorType, _NRank, _MyDevice>& _Tensor,
		const auto _InputAxisCount,
		size_t _AxisOffset
	)
	{
		const auto TensorShape = _Tensor.Shape().Data();
		auto TensorData = _Tensor.Data();
		const auto ElementCount = _Tensor.ElementCount();
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
				_Tensor.CreateShared()
			};
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}
	}

	template <typename _TensorType, size_t _NRank, Device _MyDevice>
	static auto CheckAndTryCreateValueFromTensor(
		const OrtMemoryInfo* _MyMemoryInfo,
		const Tensor<_TensorType, _NRank, _MyDevice>& _Tensor,
		ONNXTensorElementDataType _DataType,
		const TemplateLibrary::Vector<Int64>& _InputShapes,
		const TemplateLibrary::Array<const wchar_t*, _NRank>& _AxisNames,
		const char* _TensorName,
		const DLogger& _Logger = nullptr
	)
	{
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

private:
	_D_Dragonian_Lib_Constexpr_Force_Inline void GetIOInfo()
	{
		_MyInputCount = _MyModel->GetInputCount();
		_MyOutputCount = _MyModel->GetOutputCount();
		for (Int64 i = 0; i < _MyInputCount; ++i)
		{
			_MyIONames.EmplaceBack(_MyModel->GetInputNameAllocated(i, GetDefaultOrtAllocator()).get());
			_MyInputNames.EmplaceBack(_MyIONames.Back().c_str());
			auto InputShape = _MyModel->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			_MyInputDims.EmplaceBack(InputShape.data(), InputShape.data() + InputShape.size());
			_MyInputTypes.EmplaceBack(_MyModel->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
		}
		for (Int64 i = 0; i < _MyOutputCount; ++i)
		{
			_MyIONames.EmplaceBack(_MyModel->GetOutputNameAllocated(i, GetDefaultOrtAllocator()).get());
			_MyOutputNames.EmplaceBack(_MyIONames.Back().c_str());
			auto OutputShape = _MyModel->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			_MyOutputDims.EmplaceBack(OutputShape.data(), OutputShape.data() + OutputShape.size());
			_MyOutputTypes.EmplaceBack(_MyModel->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
		}
	}

public:
	OnnxModelBase& operator=(OnnxModelBase&&) = default;
	OnnxModelBase& operator=(const OnnxModelBase&) = default;
	OnnxModelBase(const OnnxModelBase&) = default;
	OnnxModelBase(OnnxModelBase&&) = default;

	template <typename... _ArgumentTypes>
	decltype(auto) operator()(_ArgumentTypes&&... _Arguments)
	{
		return static_cast<_TMyChild*>(this)->Forward(std::forward<_ArgumentTypes>(_Arguments)...);
	}

	void SetTerminate() const
	{
		_MyRunOptions->SetTerminate();
	}

	void UnTerminate() const
	{
		_MyRunOptions->UnsetTerminate();
	}

protected:
	Device _MyModelExecutionProvider = Device::CPU; ///< Execution provider (device) of the model

private:
	OnnxRuntimeEnvironment _MyEnvironment = nullptr; ///< OnnxRuntimeEnviroment

protected:
    Ort::Env* _MyOnnxEnvironment = nullptr; ///< Onnx environment
    Ort::SessionOptions* _MySessionOptions = nullptr; ///< Onnx session options
    Ort::MemoryInfo* _MyMemoryInfo = nullptr; ///< Onnx memory info
    OnnxRuntimeModel _MyModel = nullptr; ///< Onnx model

private:
    std::shared_ptr<Logger> _MyLogger = nullptr; ///< Logger

protected:
	Int64 _MyInputCount = 1;
	Int64 _MyOutputCount = 1;
	TemplateLibrary::Vector<std::string> _MyIONames;
	TemplateLibrary::Vector<const char*> _MyInputNames;
	TemplateLibrary::Vector<const char*> _MyOutputNames;
	TemplateLibrary::Vector<TemplateLibrary::Vector<Int64>> _MyInputDims;
	TemplateLibrary::Vector<TemplateLibrary::Vector<Int64>> _MyOutputDims;
	TemplateLibrary::Vector<ONNXTensorElementDataType> _MyInputTypes;
	TemplateLibrary::Vector<ONNXTensorElementDataType> _MyOutputTypes;

	std::shared_ptr<Ort::RunOptions> _MyRunOptions;
	std::wstring _MyModelPath;
};

_D_Dragonian_Lib_Onnx_Runtime_End

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
