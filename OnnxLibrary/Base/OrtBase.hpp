﻿/**
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
 * @typedef OrtTuple
 * @brief Tuple of Ort values
 */
using OrtTuple = std::vector<Ort::Value>;

/**
 * @typedef ProgressCallback
 * @brief Callback function for progress updates
 */
using ProgressCallback = std::function<void(bool, Int64)>;

/**
 * @enum ExecutionProviders
 * @brief Enum for execution providers (devices)
 */
enum class ExecutionProviders
{
    CPU = 0,
    CUDA = 1,
    DML = 2
};

/**
 * @brief Creates a DragonianLibOrtEnv
 * @param _Provider Execution provider (device) of the environment
 * @param _DeviceId Device ID of the environment
 * @param _IntraOpNumThreads Number of threads for intra-op parallelism
 * @return 
 */
OnnxRuntimeEnviroment CreateEnvironment(
    ExecutionProviders _Provider = ExecutionProviders::CPU,
	UInt64 _DeviceId = 0,
	UInt64 _IntraOpNumThreads = 1
);

Ort::AllocatorWithDefaultOptions& GetDefaultOrtAllocator();

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
        const OnnxRuntimeEnviroment& _Environment,
        const std::wstring& _ModelPath,
        const std::shared_ptr<Logger>& _Logger = nullptr
    ) : _MyModelExecutionProvider(static_cast<ExecutionProviders>(_Environment->GetCurProvider())), _MyEnvironment(_Environment),
        _MyOnnxEnvironment(_Environment->GetEnv()), _MySessionOptions(_Environment->GetSessionOptions()),
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
		_MyStaticLogger->LogMessage(L"UnReference Model: Instance[PTR:" + HexPtr + L", PATH:\"" + _MyModelPath + L"\"], Current Referece Count: " + std::to_wstring(_MyModel.use_count()));
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
	static decltype(auto) CreateTensorViewFromOrtValue(Ort::Value&& _Value, const Dimensions<_NRank>& _Shape)
	{
		const auto BufferSize = _Shape.Multiply();
		if (_Value.GetTensorTypeAndShapeInfo().GetElementCount() != BufferSize)
			_D_Dragonian_Lib_Throw_Exception("Size mismatch");
		auto Data = _Value.GetTensorMutableData<_MyValueType>();
		return Functional::FromShared<_MyValueType, _NRank, Device::CPU>(
			_Shape,
			{ Data, [_Value{ std::move(_Value) }](void* _Data) mutable {} },
			BufferSize
		);
	}

private:
	void GetIOInfo()
	{
		_MyInputCount = _MyModel->GetInputCount();
		_MyOutputCount = _MyModel->GetOutputCount();
		for (Int64 i = 0; i < _MyInputCount; ++i)
		{
			_MyIONames.EmplaceBack(_MyModel->GetInputNameAllocated(i, GetDefaultOrtAllocator()).get());
			_MyInputNames.EmplaceBack(_MyIONames.Back().c_str());
			auto InputShape = _MyModel->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			_MyInputDims.EmplaceBack(InputShape.data(), InputShape.data() + InputShape.size());
		}
		for (Int64 i = 0; i < _MyOutputCount; ++i)
		{
			_MyIONames.EmplaceBack(_MyModel->GetOutputNameAllocated(i, GetDefaultOrtAllocator()).get());
			_MyOutputNames.EmplaceBack(_MyIONames.Back().c_str());
			auto OutputShape = _MyModel->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			_MyOutputDims.EmplaceBack(OutputShape.data(), OutputShape.data() + OutputShape.size());
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
	ExecutionProviders _MyModelExecutionProvider = ExecutionProviders::CPU; ///< Execution provider (device) of the model

private:
    OnnxRuntimeEnviroment _MyEnvironment = nullptr; ///< OnnxRuntimeEnviroment

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

	std::shared_ptr<Ort::RunOptions> _MyRunOptions;
	std::wstring _MyModelPath;
};

_D_Dragonian_Lib_Onnx_Runtime_End