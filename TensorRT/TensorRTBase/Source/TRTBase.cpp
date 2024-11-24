#include "../TRTBase.hpp"
#include "Libraries/Util/Logger.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

_D_Dragonian_TensorRT_Lib_Space_Header

DLogger logger;

void _Impl_Dragonian_Lib_Free_CPU_Memory(void* _Pointer) { GetMemoryProvider(Device::CPU)->Free(_Pointer); }
void _Impl_Dragonian_Lib_Free_CUDA_Memory(void* _Pointer) { if (cudaFree(_Pointer)) _D_Dragonian_Lib_CUDA_Error; }
struct _Impl_Dragonian_Lib_Default_Deleter { template <typename T>void operator()(T* obj) const { delete obj; } };
_Impl_Dragonian_Lib_Default_Deleter _Valdef_My_Default_Deleter;

static size_t NvDataType2Size(nvinfer1::DataType _Ty)
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
	if (nvinfer1::DataType::kINT4 == _Ty) _D_Dragonian_Lib_CUDA_Error;
	_D_Dragonian_Lib_Fatal_Error;
}

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
				if (!input->setDynamicRange(-inRange, inRange)) _D_Dragonian_Lib_Fatal_Error;
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
					if (!output->setDynamicRange(-inRange, inRange)) _D_Dragonian_Lib_Fatal_Error;
				}
				else
				{
					if (!output->setDynamicRange(-outRange, outRange)) _D_Dragonian_Lib_Fatal_Error;
				}
			}
		}
	}
}

void DLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept
{
	if (severity <= Severity::kWARNING)
		LogMessage(UTF8ToWideString(msg));
}

void TrtModel::LoadModel(
	const std::wstring& _OrtPath,
	const std::wstring& _CacheFile,
	const std::vector<DynaShapeSlice>& DynaShapeConfig,
	int DLACore,
	bool Fallback,
	bool EnableFp16,
	bool EnableBf16,
	bool EnableInt8,
	nvinfer1::ILogger::Severity VerboseLevel,
	int32_t OptimizationLevel
)
{
	if (_CacheFile.empty() || !exists(std::filesystem::path(_CacheFile)))
	{
		auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
		if (!builder)
			_D_Dragonian_Lib_Fatal_Error;

		auto mNetwork = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
			0
		));
		if (!mNetwork)
			_D_Dragonian_Lib_Fatal_Error;

		//Parse Onnx Model
		auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*mNetwork, logger));
		if (!parser)
			_D_Dragonian_Lib_Fatal_Error;
		if (!parser->parseFromFile(WideStringToUTF8(_OrtPath).c_str(), static_cast<int>(VerboseLevel)))
		{
			std::string Errors;
			for (int i = 0; i < parser->getNbErrors(); ++i)
				(Errors += '\n') += parser->getError(i)->desc();
			_D_Dragonian_Lib_Throw_Exception(Errors);
		}

		auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
		if (!config)
			_D_Dragonian_Lib_Fatal_Error;

		//Config
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
			LogMessage(L"Not Impl Yet!");
		}

		size_t VRAMFREE, VRAMTOTAL;
		cudaMemGetInfo(&VRAMFREE, &VRAMTOTAL);
		config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, VRAMFREE * 8 / 10);

		if (DLACore > 0)
		{
			if (builder->getNbDLACores() == 0)
				_D_Dragonian_Lib_Throw_Exception("Error: use DLA core on a platfrom that doesn't have any DLA cores");
			if (Fallback)
				config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
			if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
				config->setFlag(nvinfer1::BuilderFlag::kFP16);

			config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
			config->setDLACore(DLACore);
		}

		if (OptimizationLevel)
		{
			auto opt = builder->createOptimizationProfile();
			for (int32_t i = 0; i < mNetwork->getNbInputs(); i++)
			{
				auto inp = mNetwork->getInput(i);
				auto iter = std::find(DynaShapeConfig.begin(), DynaShapeConfig.end(), inp->getName());
				if (iter == DynaShapeConfig.end())
					iter = std::find(DynaShapeConfig.begin(), DynaShapeConfig.end(), "DynaArg" + std::to_string(i));
				if (iter == DynaShapeConfig.end())
					continue;
				opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kMIN, iter->Min);
				opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kOPT, iter->Opt);
				opt->setDimensions(inp->getName(), nvinfer1::OptProfileSelector::kMAX, iter->Max);
			}
			if (!opt->isValid())
				_D_Dragonian_Lib_Fatal_Error;
			config->addOptimizationProfile(opt);
		}

		config->setBuilderOptimizationLevel(OptimizationLevel);

		std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*mNetwork, *config));
		if (!plan)
			_D_Dragonian_Lib_Fatal_Error;
		if (!_CacheFile.empty() && !exists(std::filesystem::path(_CacheFile)))
		{
			FileGuard file(_CacheFile, L"wb");
			fwrite(plan->data(), 1, plan->size(), file);
		}

		mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
			nvinfer1::createInferRuntime(logger),
			_Valdef_My_Default_Deleter
		);
		if (!mRuntime)
			_D_Dragonian_Lib_Fatal_Error;

		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
			mRuntime->deserializeCudaEngine(
				plan->data(), plan->size()
			),
			_Valdef_My_Default_Deleter
		);
		if (!mEngine)
			_D_Dragonian_Lib_Fatal_Error;
	}
	else if (exists(std::filesystem::path(_CacheFile)))
	{
		FileGuard file(_CacheFile, L"rb");
		struct stat file_stat;
		stat(WideStringToUTF8(_CacheFile).c_str(), &file_stat);
		std::vector<unsigned char> Buffer(file_stat.st_size);
		fread(Buffer.data(), 1, file_stat.st_size, file);

		mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
			nvinfer1::createInferRuntime(logger),
			_Valdef_My_Default_Deleter
		);
		if (!mRuntime)
			_D_Dragonian_Lib_Fatal_Error;

		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
			mRuntime->deserializeCudaEngine(
				Buffer.data(), Buffer.size()
			),
			_Valdef_My_Default_Deleter
		);
		if (!mEngine)
			_D_Dragonian_Lib_Fatal_Error;
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
	_MyGpuBuffers.resize(mIONodeCount);
}

InferenceSession TrtModel::Construct(
	const std::vector<ITensorInfo>& Inputs,
	const std::vector<std::string>& _OutputNames
)
{
	std::lock_guard Lock(mMutex);

	InferenceSession ReInferenceSession;
	if (Inputs.size() < size_t(mInputCount))
		_D_Dragonian_Lib_Throw_Exception("Missing Inputs!");
	if (_OutputNames.size() != mOutputCount)
		_D_Dragonian_Lib_Throw_Exception("Output Count Mismatch!");

	ReInferenceSession._MyContext = std::shared_ptr<nvinfer1::IExecutionContext>(
		mEngine->createExecutionContext(),
		_Valdef_My_Default_Deleter
	);
	ReInferenceSession._MyInputInfos.resize(mInputCount);
	ReInferenceSession._MyInputGpuBuffer.resize(mInputCount);
	ReInferenceSession._MyDeviceBindings.resize(mIONodeCount);
	ReInferenceSession._MyOutputInfos.resize(mOutputCount);
	ReInferenceSession._MyOutputGpuBuffer.resize(mOutputCount);
	ReInferenceSession._MyCondition.resize(mInputCount, false);

	auto mContext = ReInferenceSession._MyContext;
	if (!mContext)
		_D_Dragonian_Lib_Fatal_Error;

	// Input Tensors
	for (int32_t i = 0; i < mIONodeCount; i++)
	{
		auto const name = mEngine->getIOTensorName(i);
		if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
		{
			auto _Tensor = std::find(Inputs.begin(), Inputs.end(), name);

			if (_Tensor == Inputs.end())
				_D_Dragonian_Lib_Throw_Exception("The Input " + std::string(name) + " Is Missing, Please Fix This Input!");
			if (mEngine->getTensorDataType(name) != _Tensor->_MyType)
				_D_Dragonian_Lib_Throw_Exception("Data Type Mismatch!");

			const auto Index = _Tensor - Inputs.begin();
			_MyGpuBuffers[i].ReAllocate(_Tensor->_MySize);
			ReInferenceSession._MyInputGpuBuffer[Index] = _MyGpuBuffers[i].GetData();
			ReInferenceSession._MyInputInfos[Index] = *_Tensor;
			ReInferenceSession._MyDeviceBindings[i] = _MyGpuBuffers[i].GetData().get();

			if (!mContext->setInputTensorAddress(name, ReInferenceSession._MyDeviceBindings[i]))
				_D_Dragonian_Lib_Fatal_Error;
			if (!mContext->setInputShape(name, _Tensor->_MyShape))
				_D_Dragonian_Lib_Throw_Exception("Shape Mismatch!");
		}
		else if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
		{
			auto Iter = std::ranges::find(_OutputNames, name);
			if (Iter == _OutputNames.end())
				_D_Dragonian_Lib_Throw_Exception("The Output Name " + std::string(name) + " Is Missing, Please Check Your Output Names!");
			auto Index = Iter - _OutputNames.begin();

			auto& TensorRef = ReInferenceSession._MyOutputInfos[Index];
			TensorRef._MyShape = mContext->getTensorShape(name);
			TensorRef._MyType = mEngine->getTensorDataType(name);
			TensorRef._MySize = NvDataType2Size(TensorRef._MyType);
			TensorRef._MyName = name;
			for (int j = 0; j < TensorRef._MyShape.nbDims; ++j)
				TensorRef._MySize *= TensorRef._MyShape.d[j];
			_MyGpuBuffers[i].ReAllocate(TensorRef._MySize);
			ReInferenceSession._MyOutputGpuBuffer[Index] = _MyGpuBuffers[i].GetData();
			ReInferenceSession._MyDeviceBindings[i] = _MyGpuBuffers[i].GetData().get();

			if (!mContext->setOutputTensorAddress(name, ReInferenceSession._MyDeviceBindings[i]))
				_D_Dragonian_Lib_Fatal_Error;
		}
	}

	return ReInferenceSession;
}

IGPUBufferImpl& IGPUBufferImpl::ReAllocate(size_t NewSize)
{
	if (NewSize > static_cast<size_t>(_MySize))
	{
		void* Data = nullptr;
		if (cudaMalloc(&Data, NewSize))
			_D_Dragonian_Lib_CUDA_Error;
		_MyData = std::shared_ptr<void>(
			Data, _Impl_Dragonian_Lib_Free_CUDA_Memory
		);
		_MySize = NewSize;
	}
	return *this;
}

std::shared_ptr<void> IGPUBufferImpl::GetData() const
{
	return _MyData;
}

IGPUBufferImpl::operator void*() const
{
	return _MyData.get();
}

ITensorInfo::ITensorInfo(
	const nvinfer1::Dims& shape,
	std::string name,
	int64_t size,
	nvinfer1::DataType type
) : _MyShape(shape), _MyName(std::move(name)),
_MySize(size), _MyType(type) {}

bool ITensorInfo::operator==(const char* _Val) const
{
	return _MyName == _Val;
}

int64_t ITensorInfo::GetElementCount() const
{
	return _MySize / NvDataType2Size(_MyType);
}

bool ITensorInfo::operator==(const ITensorInfo& _Val) const
{
	if (_MyShape != _Val._MyShape)
		return false;
	if (_MyName != _Val._MyName)
		return false;
	if (_MySize != _Val._MySize)
		return false;
	if (_MyType != _Val._MyType)
		return false;
	return true;
}

bool ITensorInfo::operator!=(const ITensorInfo& _Val) const
{
	return !(*this == _Val);
}

void InferenceSession::HostMemoryToDevice(size_t _Index, const void* _Pointer, size_t _Size)
#ifndef DRAGONIANLIB_DEBUG
const
#endif
{
#ifdef DRAGONIANLIB_DEBUG
	if (!_MyCondition[_Index])
	{
#endif
		std::lock_guard Lock(*_MyMutex);
		auto Buffer = _MyInputGpuBuffer[_Index].get();
		if (cudaMemcpy(Buffer, _Pointer, _Size, cudaMemcpyHostToDevice))
			_D_Dragonian_Lib_CUDA_Error;
#ifdef DRAGONIANLIB_DEBUG
		_MyCondition[_Index] = true;
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Input Data Already Set!");
#endif
}

void InferenceSession::Run()
#ifndef DRAGONIANLIB_DEBUG
const
#endif
{
#ifdef DRAGONIANLIB_DEBUG
	for (const auto& Condition : _MyCondition)
		if (!Condition)
			_D_Dragonian_Lib_Throw_Exception("No Input Data!");
	TidyGuard Tidy([this] { _MyCondition = { _MyCondition.size(), false, std::allocator<bool>() }; });
#endif
	std::lock_guard Lock(*_MyMutex);
	_MyContext->executeV2(_MyDeviceBindings.data());
}

void InferenceSession::DeviceMemoryToHost(size_t _Index, void* _Pointer, size_t _Size) const
{
	std::lock_guard Lock(*_MyMutex);
	auto Buffer = _MyOutputGpuBuffer[_Index].get();
	if (cudaMemcpy(_Pointer, Buffer, _Size, cudaMemcpyDeviceToHost))
		_D_Dragonian_Lib_CUDA_Error;
}

bool InferenceSession::IsReady(const std::vector<ITensorInfo>& _Check) const
{
	if (_Check.size() != _MyInputInfos.size())
		return false;
	for (size_t i = 0; i < _Check.size(); ++i)
		if (_Check[i] != _MyInputInfos[i])
			return false;
	return true;
}

DragonianLibSTL::Vector<float> InferenceSession::GetOutput(size_t _Index) const
{
	std::lock_guard Lock(*_MyMutex);
	auto Buffer = _MyOutputGpuBuffer[_Index].get();
	auto& Tensor = _MyOutputInfos[_Index];
	DragonianLibSTL::Vector<float> Result(Tensor.GetElementCount());
	if (cudaMemcpy(Result.Data(), Buffer, Tensor._MySize, cudaMemcpyDeviceToHost))
		_D_Dragonian_Lib_CUDA_Error;
	return Result;
}

bool operator==(const nvinfer1::Dims& _Left, const nvinfer1::Dims& _Right)
{
	if (_Left.nbDims != _Right.nbDims)
		return false;
	for (int i = 0; i < _Left.nbDims; ++i)
		if (_Left.d[i] != _Right.d[i])
			return false;
	return true;
}

bool operator!=(const nvinfer1::Dims& _Left, const nvinfer1::Dims& _Right)
{
	return !(_Left == _Right);
}

_D_Dragonian_TensorRT_Lib_Space_End