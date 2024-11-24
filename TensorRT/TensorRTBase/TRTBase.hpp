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
#include <mutex>
#include "Base.h"
#include "NvInfer.h"
#include "MyTemplateLibrary/Vector.h"
#include "Util/StringPreprocess.h"

#define _D_Dragonian_TensorRT_Lib_Space_Header _D_Dragonian_Lib_Space_Begin namespace TensorRTLib{
#define _D_Dragonian_TensorRT_Lib_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_TensorRT_Lib_Space_Header

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

struct ITensorInfo;
struct IGPUBufferImpl;
class InferenceSession;

class TrtModel
{
public:
	TrtModel() = default;
	~TrtModel() = default;
	TrtModel(
		const std::wstring& _OrtPath,
		const std::wstring& _CacheFile,
		const std::vector<DynaShapeSlice>& DynaShapeConfig,
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
		const std::vector<DynaShapeSlice>& DynaShapeConfig,
		int DLACore = -1,
		bool Fallback = true,
		bool EnableFp16 = false,
		bool EnableBf16 = false,
		bool EnableInt8 = false,
		nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING,
		int32_t OptimizationLevel = 3
	);

	InferenceSession Construct(
		const std::vector<ITensorInfo>& Inputs,
		const std::vector<std::string>& _OutputNames
	);

	int64_t GetInputCount() const { return mInputCount; }
	int64_t GetOutputCount() const { return mOutputCount; }
	int64_t GetIOCount() const { return mIONodeCount; }
	const std::vector<std::string>& GetInputNames() const { return MyInputNames; }
	const std::vector<std::string>& GetOutputNames() const { return MyOutputNames; }

private:
	std::shared_ptr<nvinfer1::IRuntime> mRuntime = nullptr;
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;
	std::mutex mMutex;

	int64_t mInputCount = 0, mOutputCount = 0, mIONodeCount = 0;
	std::vector<std::string> MyInputNames, MyOutputNames;
	std::vector<IGPUBufferImpl> _MyGpuBuffers;

	TrtModel(const TrtModel& _Val) = delete;
	TrtModel(TrtModel&& _Val) = delete;
	TrtModel& operator=(const TrtModel& _Val) = delete;
	TrtModel& operator=(TrtModel&& _Val) = delete;
};

struct IGPUBufferImpl
{
	friend struct ITensorInfo;
	friend class InferenceSession;
	friend class TrtModel;
	IGPUBufferImpl() = default;
	~IGPUBufferImpl() = default;
	IGPUBufferImpl& ReAllocate(size_t NewSize);
	std::shared_ptr<void> GetData() const;
	operator void* () const;

	IGPUBufferImpl(const IGPUBufferImpl& _Val) = default;
	IGPUBufferImpl(IGPUBufferImpl&& _Val) noexcept = default;
	IGPUBufferImpl& operator=(const IGPUBufferImpl& _Val) = default;
	IGPUBufferImpl& operator=(IGPUBufferImpl&& _Val) noexcept = default;
protected:
	std::shared_ptr<void> _MyData = nullptr;
	int64_t _MySize = 0;
};

struct ITensorInfo
{
	friend struct IGPUBufferImpl;
	friend class InferenceSession;
	friend class TrtModel;
	ITensorInfo(
		const nvinfer1::Dims& shape = nvinfer1::Dims2(0, 0),
		std::string name = "None",
		int64_t size = 0,
		nvinfer1::DataType type = nvinfer1::DataType::kFLOAT
	);
	~ITensorInfo() = default;
	bool operator==(const char* _Val) const;
	int64_t GetElementCount() const;

	ITensorInfo(ITensorInfo&& _Val) noexcept = default;
	ITensorInfo& operator=(ITensorInfo&& _Val) noexcept = default;
	ITensorInfo(const ITensorInfo& _Val) = default;
	ITensorInfo& operator=(const ITensorInfo& _Val) = default;

	bool operator==(const ITensorInfo& _Val) const;
	bool operator!=(const ITensorInfo& _Val) const;

	nvinfer1::Dims& GetShape() { return _MyShape; }
	const nvinfer1::Dims& GetShape() const { return _MyShape; }
	std::string& GetName() { return _MyName; }
	const std::string& GetName() const { return _MyName; }
	int64_t GetSize() const { return _MySize; }
	nvinfer1::DataType GetType() const { return _MyType; }

protected:
	nvinfer1::Dims _MyShape;
	std::string _MyName;
	int64_t _MySize = 0;
	nvinfer1::DataType _MyType = nvinfer1::DataType::kFLOAT;
};

class InferenceSession
{
public:
	friend class TrtModel;
	InferenceSession() : _MyMutex(std::make_shared<std::mutex>()) {}
	~InferenceSession() = default;
#ifndef DRAGONIANLIB_DEBUG
	void Run() const;
	void HostMemoryToDevice(size_t _Index, const void* _Pointer, size_t _Size) const;
#else
	void Run();
	void HostMemoryToDevice(size_t _Index, const void* _Pointer, size_t _Size);
#endif
	void DeviceMemoryToHost(size_t _Index, void* _Pointer, size_t _Size) const;
	bool IsReady(const std::vector<ITensorInfo>& _Check) const;

	std::vector<ITensorInfo>& GetOutputInfos() { return _MyOutputInfos; }
	const std::vector<ITensorInfo>& GetOutputInfos() const { return _MyOutputInfos; }
	std::vector<ITensorInfo>& GetInputInfos() { return _MyInputInfos; }
	const std::vector<ITensorInfo>& GetInputInfos() const { return _MyInputInfos; }

	DragonianLibSTL::Vector<float> GetOutput(size_t _Index) const;

	InferenceSession(const InferenceSession& _Val) = delete;
	InferenceSession& operator=(const InferenceSession& _Val) = delete;
	InferenceSession(InferenceSession&& _Val) noexcept = default;
	InferenceSession& operator=(InferenceSession&& _Val) noexcept = default;
protected:
	std::shared_ptr<nvinfer1::IExecutionContext> _MyContext = nullptr;
	std::vector<void*> _MyDeviceBindings;
	std::vector<std::shared_ptr<void>> _MyInputGpuBuffer;
	std::vector<ITensorInfo> _MyInputInfos;
	std::vector<std::shared_ptr<void>> _MyOutputGpuBuffer;
	std::vector<ITensorInfo> _MyOutputInfos;
private:
	std::vector<bool> _MyCondition;
	std::shared_ptr<std::mutex> _MyMutex;
};

struct TrtConfig
{
	std::unordered_map<std::wstring, std::wstring> CacheFile;
	std::vector<DynaShapeSlice> DynaSetting;
	int DLACore = -1;
	bool Fallback = true;
	bool EnableFp16 = false;
	bool EnableBf16 = false;
	bool EnableInt8 = false;
	nvinfer1::ILogger::Severity VerboseLevel = nvinfer1::ILogger::Severity::kWARNING;
	int32_t OptimizationLevel = 3;
};

bool operator==(const nvinfer1::Dims& _Left, const nvinfer1::Dims& _Right);
bool operator!=(const nvinfer1::Dims& _Left, const nvinfer1::Dims& _Right);

_D_Dragonian_TensorRT_Lib_Space_End