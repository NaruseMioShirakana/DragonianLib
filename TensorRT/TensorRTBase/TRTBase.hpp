/**
 * FileName: TRTBase.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <functional>
#include "Base.h"
#include "NvInfer.h"
#include "MyTemplateLibrary/Vector.h"
#include "Util/StringPreprocess.h"

#define DragonianLibCUDAError DragonianLibThrow(cudaGetErrorString(cudaGetLastError()))

namespace TensorRTLib
{
	using ProgressCallback = std::function<void(size_t, size_t)>;

	class DLogger final : public nvinfer1::ILogger
	{
	public:
		void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
	};

	struct DynaShapeSlice
	{
		std::string Name;
		nvinfer1::Dims Min, Opt, Max;
		bool operator==(const char* _in) const
		{
			return Name == _in;
		}
		bool operator==(const std::string& _in) const
		{
			return Name == _in;
		}
	};

	struct Tensor
	{
		Tensor(
			void* data = nullptr,
			const nvinfer1::Dims& shape = nvinfer1::Dims2(0, 0),
			std::string name = "None",
			int64_t size = 0,
			nvinfer1::DataType type = nvinfer1::DataType::kFLOAT
		) :
			Data(data), Shape(shape), Name(std::move(name)), Size(size), Type(type) {}
		~Tensor()
		{
			if (IsOwner && Data)
				GetMemoryProvider(DragonianLib::Device::CPU)->Free(Data);
		}
		bool operator==(const char* _Val) const
		{
			return Name == _Val;
		}
		Tensor(Tensor&& _Val) noexcept;
		Tensor& operator=(Tensor&& _Val) noexcept;

		void DeviceData2Host();
		int64_t GetElementCount() const;

		void* Data = nullptr;
		nvinfer1::Dims Shape;
		std::string Name;
		int64_t Size = 0;
		nvinfer1::DataType Type = nvinfer1::DataType::kFLOAT;
		bool IsOwner = false;
		void* GpuBuffer = nullptr;
	private:
		Tensor(const Tensor& _Val) = delete;
		Tensor& operator=(const Tensor& _Val) = delete;
	};

	using TrtTensor = ::TensorRTLib::Tensor;

	class GPUBuffer
	{
	public:
		GPUBuffer() = default;
		GPUBuffer(const Tensor& HostTensor);
		~GPUBuffer();

		GPUBuffer& operator=(const Tensor& HostTensor);
		GPUBuffer& Resize(size_t NewSize);
		GPUBuffer(GPUBuffer&& _Val) noexcept;

		operator void* () const
		{
			return Data;
		}
	private:
		void Destory();
		void* Data = nullptr;
		int64_t Size = 0;

		GPUBuffer(const GPUBuffer& _Val) = delete;
		
		GPUBuffer& operator=(const GPUBuffer& _Val) = delete;
		GPUBuffer& operator=(GPUBuffer&& _Val) noexcept = delete;
	};

	struct InferenceDeviceBuffer;

	class TrtModel
	{
	public:
		TrtModel() = default;
		~TrtModel() = default;
		TrtModel(
			const std::wstring& _OrtPath,
			const std::wstring& _CacheFile,
			const DragonianLibSTL::Vector<DynaShapeSlice>& DynaShapeConfig,
			int DLACore = -1,
			bool Fallback = true,
			bool EnableFp16 = false,
			bool EnableBf16 = false,
			bool EnableInt8 = false,
			nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING,
			int32_t OptimizationLevel = 3
		)
		{
			LoadModel(
				_OrtPath,
				_CacheFile,
				DynaShapeConfig,
				DLACore,
				Fallback,
				EnableFp16,
				EnableBf16,
				EnableInt8,
				VerboseLevel,
				OptimizationLevel
			);
		}

		void LoadModel(
			const std::wstring& _OrtPath,
			const std::wstring& _CacheFile,
			const DragonianLibSTL::Vector<DynaShapeSlice>& DynaShapeConfig,
			int DLACore = -1,
			bool Fallback = true,
			bool EnableFp16 = false,
			bool EnableBf16 = false,
			bool EnableInt8 = false,
			nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING,
			int32_t OptimizationLevel = 3
		);

		DragonianLibSTL::Vector<Tensor> Infer(
			const DragonianLibSTL::Vector<Tensor>& Inputs,
			const InferenceDeviceBuffer& _Buffer,
			const std::vector<std::string>& _OutputNames
		) const;

		int64_t GetInputCount() const { return mInputCount; }
		int64_t GetOutputCount() const { return mOutputCount; }
		int64_t GetIOCount() const { return mEngine->getNbIOTensors(); }
		const std::vector<std::string>& GetInputNames() const { return MyInputNames; }
		const std::vector<std::string>& GetOutputNames() const { return MyOutputNames; }

	private:
		std::unique_ptr<nvinfer1::IRuntime> mRuntime = nullptr;
		std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;

		int64_t mInputCount = 0, mOutputCount = 0, mIONodeCount = 0;

		std::vector<std::string> MyInputNames, MyOutputNames;

		TrtModel(const TrtModel& _Val) = delete;
		TrtModel(TrtModel&& _Val) = delete;
		TrtModel& operator=(const TrtModel& _Val) = delete;
		TrtModel& operator=(TrtModel&& _Val) = delete;
	};

	struct InferenceDeviceBuffer
	{
		DragonianLibSTL::Vector<GPUBuffer> mGpuBuffers;
		DragonianLibSTL::Vector<void*> mDeviceBindings;

		InferenceDeviceBuffer() = default;
		InferenceDeviceBuffer(const TrtModel& _Model)
		{
			mGpuBuffers.Resize(_Model.GetIOCount());
			mDeviceBindings.Resize(_Model.GetIOCount());
		}
		InferenceDeviceBuffer& Reload(const TrtModel& _Model)
		{
			mGpuBuffers.Resize(_Model.GetIOCount());
			mDeviceBindings.Resize(_Model.GetIOCount());
			return *this;
		}
	};

	struct TrtConfig
	{
		std::wstring CacheFile;
		int DLACore = -1;
		bool Fallback = true;
		bool EnableFp16 = false;
		bool EnableBf16 = false;
		bool EnableInt8 = false;
		nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING;
		int32_t OptimizationLevel = 3;
	};
}