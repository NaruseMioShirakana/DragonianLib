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

	struct Tensor
	{
		Tensor(void* data, const nvinfer1::Dims& shape, std::string name, int64_t size, nvinfer1::DataType type) :
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

		operator void* () const
		{
			return Data;
		}
	private:
		void Destory();
		void* Data = nullptr;
		int64_t Size = 0;

		GPUBuffer(const GPUBuffer& _Val) = delete;
		GPUBuffer(GPUBuffer&& _Val) noexcept = delete;
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
			int DLACore = -1,
			bool Fallback = true,
			bool EnableFp16 = false,
			bool EnableBf16 = false,
			bool EnableInt8 = false,
			nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING
		)
		{
			LoadModel(
				_OrtPath,
				_CacheFile,
				DLACore,
				Fallback,
				EnableFp16,
				EnableBf16,
				EnableInt8,
				VerboseLevel
			);
		}

		void LoadModel(
			const std::wstring& _OrtPath,
			const std::wstring& _CacheFile,
			int DLACore = -1,
			bool Fallback = true,
			bool EnableFp16 = false,
			bool EnableBf16 = false,
			bool EnableInt8 = false,
			nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING
		);

		DragonianLibSTL::Vector<Tensor> Infer(
			const DragonianLibSTL::Vector<Tensor>& Inputs,
			const InferenceDeviceBuffer& _Buffer,
			const std::vector<std::string>& _OutputNames
		) const;

		int64_t GetInputCount() const { return mInputCount; }
		int64_t GetOutputCount() const { return mOutputCount; }
		int64_t GetIOCount() const { return mEngine->getNbIOTensors(); }

	private:
		std::shared_ptr<nvinfer1::IRuntime> mRuntime = nullptr;
		std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;
		std::shared_ptr<nvinfer1::INetworkDefinition> mNetwork = nullptr;

		int64_t mInputCount = 0, mOutputCount = 0, mIONodeCount = 0;

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
	};
}