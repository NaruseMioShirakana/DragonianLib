#include "TRTBase.hpp"
#include "Util/Logger.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

namespace TensorRTLib
{
	DLogger logger;

	static void StreamDeleter(const cudaStream_t* pStream) {
		if (pStream)
		{
			static_cast<void>(cudaStreamDestroy(*pStream));
			delete pStream;
		}
	}

	size_t NvDataType2Size(nvinfer1::DataType _Ty)
	{
		if (nvinfer1::DataType::kFLOAT == _Ty) return 4;
		if (nvinfer1::DataType::kHALF == _Ty) return 2;
		if (nvinfer1::DataType::kINT8 == _Ty) return 1;
		if (nvinfer1::DataType::kINT32 == _Ty) return 4;
		if (nvinfer1::DataType::kBOOL == _Ty) return 1;
		if (nvinfer1::DataType::kUINT8 == _Ty) return 1;
		if (nvinfer1::DataType::kFP8 == _Ty) return 1;
		if (nvinfer1::DataType::kBF16 == _Ty) return 2;
		if (nvinfer1::DataType::kINT64 == _Ty) return 8;
		if (nvinfer1::DataType::kINT4 == _Ty) DragonianLibFatalError;
		DragonianLibFatalError;
	}

	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			delete obj;
		}
	};

	static void setAllDynamicRanges(const nvinfer1::INetworkDefinition* network, float inRange = 2.0F, float outRange = 4.0F)
	{
		// Ensure that all layer inputs have a scale.
		for (int i = 0; i < network->getNbLayers(); i++)
		{
			auto layer = network->getLayer(i);
			for (int j = 0; j < layer->getNbInputs(); j++)
			{
				nvinfer1::ITensor* input{ layer->getInput(j) };
				// Optional inputs are nullptr here and are from RNN layers.
				if (input != nullptr && !input->dynamicRangeIsSet())
				{
					if (!input->setDynamicRange(-inRange, inRange)) DragonianLibFatalError;
				}
			}
		}

		// Ensure that all layer outputs have a scale.
		// Tensors that are also inputs to layers are ingored here
		// since the previous loop nest assigned scales to them.
		for (int i = 0; i < network->getNbLayers(); i++)
		{
			auto layer = network->getLayer(i);
			for (int j = 0; j < layer->getNbOutputs(); j++)
			{
				nvinfer1::ITensor* output{ layer->getOutput(j) };
				// Optional outputs are nullptr here and are from RNN layers.
				if (output != nullptr && !output->dynamicRangeIsSet())
				{
					// Pooling must have the same input and output scales.
					if (layer->getType() == nvinfer1::LayerType::kPOOLING)
					{
						if (!output->setDynamicRange(-inRange, inRange)) DragonianLibFatalError;
					}
					else
					{
						if (!output->setDynamicRange(-outRange, outRange)) DragonianLibFatalError;
					}
				}
			}
		}
	}

	void DLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept
	{
		if (severity <= Severity::kWARNING)
			DragonianLibLogMessage(msg);
	}

	Tensor::Tensor(Tensor&& _Val) noexcept
	{
		Data = _Val.Data;
		Shape = _Val.Shape;
		Name = std::move(_Val.Name);
		Size = _Val.Size;
		Type = _Val.Type;
		IsOwner = _Val.IsOwner;
		GpuBuffer = _Val.GpuBuffer;

		_Val.Data = nullptr;
		_Val.IsOwner = false;
		_Val.GpuBuffer = nullptr;
	}

	Tensor& Tensor::operator=(Tensor&& _Val) noexcept
	{
		if (IsOwner && Data)
			GetMemoryProvider(DragonianLib::Device::CPU)->Free(Data);

		Data = _Val.Data;
		Shape = _Val.Shape;
		Name = std::move(_Val.Name);
		Size = _Val.Size;
		Type = _Val.Type;
		IsOwner = _Val.IsOwner;
		GpuBuffer = _Val.GpuBuffer;

		_Val.Data = nullptr;
		_Val.IsOwner = false;
		_Val.GpuBuffer = nullptr;

		return *this;
	}

	void Tensor::DeviceData2Host()
	{
		if (!GpuBuffer)
			return;
		Data = GetMemoryProvider(DragonianLib::Device::CPU)->Allocate(Size);
		IsOwner = true;
		if (const auto Ret = cudaMemcpy(Data, GpuBuffer, Size, cudaMemcpyDeviceToHost); Ret)
			DragonianLibThrow(cudaGetErrorString(Ret));
		GpuBuffer = nullptr;
	}

	int64_t Tensor::GetElementCount() const
	{
		return Size / NvDataType2Size(Type);
	}

	GPUBuffer::GPUBuffer(const Tensor& HostTensor)
	{
		if (const auto Ret = cudaMalloc(&Data, HostTensor.Size); Ret)
			DragonianLibThrow(cudaGetErrorString(Ret));
		if (const auto Ret = cudaMemcpy(Data, HostTensor.Data, HostTensor.Size, cudaMemcpyHostToDevice); Ret)
			DragonianLibThrow(cudaGetErrorString(Ret));
		Size = HostTensor.Size;
	}

	GPUBuffer::~GPUBuffer()
	{
		Destory();
	}

	GPUBuffer::GPUBuffer(GPUBuffer&& _Val) noexcept
	{
		Size = _Val.Size;
		Data = _Val.Data;
		_Val.Size = 0;
		_Val.Data = nullptr;
	}

	GPUBuffer& GPUBuffer::operator=(const Tensor& HostTensor)
	{
		if (HostTensor.Size > Size)
		{
			Destory();
			if (const auto Ret = cudaMalloc(&Data, HostTensor.Size); Ret)
				DragonianLibThrow(cudaGetErrorString(Ret));
			Size = HostTensor.Size;
		}
		if (const auto Ret = cudaMemcpy(Data, HostTensor.Data, HostTensor.Size, cudaMemcpyHostToDevice); Ret)
			DragonianLibThrow(cudaGetErrorString(Ret));
		return *this;
	}

	GPUBuffer& GPUBuffer::Resize(size_t NewSize)
	{
		if (NewSize > Size)
		{
			Destory();
			if (const auto Ret = cudaMalloc(&Data, NewSize); Ret)
				DragonianLibThrow(cudaGetErrorString(Ret));
			Size = NewSize;
		}
		return *this;
	}

	void GPUBuffer::Destory()
	{
		if (!Data)
			return;
		if (const auto Ret = cudaFree(Data); Ret)
			DragonianLibThrow(cudaGetErrorString(Ret));
		Data = nullptr;
		Size = 0;
	}

	void TrtModel::LoadModel(
		const std::wstring& _OrtPath,
		const std::wstring& _CacheFile,
		const DragonianLibSTL::Vector<DynaShapeSlice>& DynaShapeConfig,
		int DLACore,
		bool Fallback,
		bool EnableFp16,
		bool EnableBf16,
		bool EnableInt8,
		nvinfer1::ILogger::Severity VerboseLevel,
		int32_t OptimizationLevel
	)
	{
		if(_CacheFile.empty() || !exists(std::filesystem::path(_CacheFile)))
		{
			auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(::TensorRTLib::logger));
			if (!builder)
				DragonianLibFatalError;

			auto mNetwork = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
				1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
			));
			if (!mNetwork)
				DragonianLibFatalError;

			//Parse Onnx Model
			{
				auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*mNetwork, ::TensorRTLib::logger));
				if (!parser)
					DragonianLibFatalError;
				if (!parser->parseFromFile(DragonianLib::WideStringToUTF8(_OrtPath).c_str(), static_cast<int>(VerboseLevel)))
				{
					std::string Errors;
					for (int i = 0; i < parser->getNbErrors(); ++i)
						(Errors += '\n') += parser->getError(i)->desc();
					DragonianLibThrow(Errors);
				}
			}

			auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
			if (!config)
				DragonianLibFatalError;

			//Config
			{
				auto timingCache = std::unique_ptr<nvinfer1::ITimingCache>();

				if (EnableFp16 && builder->platformHasFastFp16())
					config->setFlag(nvinfer1::BuilderFlag::kFP16);
				if (EnableBf16 && builder->platformHasFastFp16())
					config->setFlag(nvinfer1::BuilderFlag::kBF16);
				if (EnableInt8 && builder->platformHasFastInt8())
				{
					config->setFlag(nvinfer1::BuilderFlag::kINT8);
					setAllDynamicRanges(mNetwork.get(), 127.0F, 127.0F);
				}

				if (_CacheFile.size() && exists(std::filesystem::path(_CacheFile)))
				{
					DragonianLibLogMessage("Not Impl Yet!");
				}

				size_t VRAMFREE, VRAMTOTAL;
				cudaMemGetInfo(&VRAMFREE, &VRAMTOTAL);
				config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, VRAMFREE * 8 / 10);

				if (DLACore >= 0)
				{
					if (builder->getNbDLACores() == 0)
						DragonianLibThrow("Error: use DLA core on a platfrom that doesn't have any DLA cores");
					if (Fallback)
						config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
					if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
						config->setFlag(nvinfer1::BuilderFlag::kFP16);
					config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
					config->setDLACore(DLACore);
				}

				if(OptimizationLevel)
					for (int32_t i = 0; i < mNetwork->getNbInputs(); i++)
					{
						auto inp = mNetwork->getInput(i);
						auto iter = std::find(DynaShapeConfig.begin(), DynaShapeConfig.end(), inp->getName());
						if (iter == DynaShapeConfig.end())
							iter = std::find(DynaShapeConfig.begin(), DynaShapeConfig.end(), "DynaArg" + std::to_string(i));
						if (iter == DynaShapeConfig.end())
							continue;
						auto opt = builder->createOptimizationProfile();
						opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kMIN, iter->Min);
						opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kOPT, iter->Opt);
						opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kMAX, iter->Max);
						if (!opt->isValid())
							DragonianLibFatalError;
						config->addOptimizationProfile(opt);
					}

				config->setBuilderOptimizationLevel(OptimizationLevel);
			}

			std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*mNetwork, *config));
			if (!plan)
				DragonianLibFatalError;
			if(!_CacheFile.empty() && !exists(std::filesystem::path(_CacheFile)))
			{
				DragonianLib::FileGuard file(_CacheFile, L"wb");
				fwrite(plan->data(), 1, plan->size(), file);
			}

			mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(::TensorRTLib::logger));
			if (!mRuntime)
				DragonianLibFatalError;

			mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
				mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
			if (!mEngine)
				DragonianLibFatalError;
		}
		else if(exists(std::filesystem::path(_CacheFile)))
		{
			DragonianLib::FileGuard file(_CacheFile, L"rb");
			struct stat file_stat;
			stat(DragonianLib::WideStringToUTF8(_CacheFile).c_str(), &file_stat);
			DragonianLibSTL::Vector<unsigned char> Buffer(file_stat.st_size);
			fread(Buffer.Data(), 1, file_stat.st_size, file);

			mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(::TensorRTLib::logger));
			if (!mRuntime)
				DragonianLibFatalError;

			mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
				mRuntime->deserializeCudaEngine(Buffer.Data(), Buffer.Size()), InferDeleter()
			);
			if (!mEngine)
				DragonianLibFatalError;
		}

		mIONodeCount = mEngine->getNbIOTensors();
		mInputCount = 0;
		mOutputCount = 0;
		for (int32_t i = 0; i < mIONodeCount; i++)
		{
			auto const name = mEngine->getIOTensorName(i);
			if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
			{
				++mInputCount;
				MyInputNames.emplace_back(name);
			}
			if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
			{
				++mOutputCount;
				MyOutputNames.emplace_back(name);
			}
		}
	}

	DragonianLibSTL::Vector<Tensor> TrtModel::Infer(
		const DragonianLibSTL::Vector<Tensor>& Inputs,
		const InferenceDeviceBuffer& _Buffer,
		const std::vector<std::string>& _OutputNames
	) const
	{
		if (Inputs.Size() < mInputCount)
			DragonianLibThrow("Missing Inputs!");
		if (_OutputNames.size() != mOutputCount)
			DragonianLibThrow("Output Count Mismatch!");

		auto mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
		if (!mContext)
			DragonianLibFatalError;

		// Input Tensors
		for (int32_t i = 0; i < mIONodeCount; i++)
		{
			auto const name = mEngine->getIOTensorName(i);
			if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
			{
				auto _Tensor = std::find(Inputs.begin(), Inputs.end(), name);
				if (_Tensor == Inputs.End())
					DragonianLibThrow("The Input " + std::string(name) + " Is Missing, Please Fix This Input!");
				if (mEngine->getTensorDataType(name) != _Tensor->Type)
					DragonianLibThrow("Data Type Mismatch!");
				if (!mContext->setInputTensorAddress(name, _Buffer.mDeviceBindings[i] = (_Buffer.mGpuBuffers[i] = *_Tensor)))
					DragonianLibFatalError;
				if (!mContext->setInputShape(name, _Tensor->Shape))
					DragonianLibThrow("Shape Mismatch!");
			}
		}

		// Output Tensors
		DragonianLibSTL::Vector<Tensor> OutputTensors(mOutputCount);
		for (int32_t i = 0; i < mIONodeCount; i++)
		{
			auto const name = mEngine->getIOTensorName(i);
			if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
			{
				auto Iter = std::ranges::find(_OutputNames, name);
				if (Iter == _OutputNames.end())
					DragonianLibThrow("The Output Name " + std::string(name) + " Is Missing, Please Check Your Output Names!");
				auto Idx = Iter - _OutputNames.begin();

				auto& TensorRef = OutputTensors[Idx];
				TensorRef.Shape = mContext->getTensorShape(name);
				TensorRef.Type = mEngine->getTensorDataType(name);
				TensorRef.Size = NvDataType2Size(TensorRef.Type);
				for (int j = 0; j < TensorRef.Shape.nbDims; ++j)
					TensorRef.Size *= TensorRef.Shape.d[j];
				TensorRef.GpuBuffer = _Buffer.mDeviceBindings[i] = _Buffer.mGpuBuffers[i].Resize(TensorRef.Size);

				if (!mContext->setOutputTensorAddress(name, TensorRef.GpuBuffer))
					DragonianLibFatalError;
			}
		}

		bool status = mContext->executeV2(_Buffer.mDeviceBindings.Data());
		if (!status)
			DragonianLibThrow("An Error Occurred While Inference!");

		return std::move(OutputTensors);
	}
}

