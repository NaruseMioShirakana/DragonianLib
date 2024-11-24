/**
 * FileName: ModelBase.hpp
 * Note: MoeVoiceStudioCore Onnx 模型基类
 *
 * Copyright (C) 2022-2023 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of MoeVoiceStudioCore library.
 * MoeVoiceStudioCore library is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * MoeVoiceStudioCore library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include <functional>
#include "Libraries/Base.h"
#include "Libraries/EnvManager.hpp"

#define _D_Dragonian_Lib_Lib_Text_To_Speech_Header namespace DragonianLib { namespace TextToSpeech { 
#define _D_Dragonian_Lib_Lib_Text_To_Speech_End } }

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

/**
 * @class LibTTSModule
 * @brief Base class for Onnx models
 */
class LibTTSModule
{
public:
	/**
	 * @typedef ProgressCallback
	 * @brief Callback function for progress updates
	 */
	using ProgressCallback = std::function<void(size_t, size_t)>;

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
	 * \brief 构造Onnx模型基类
	 * \param ExecutionProvider_ ExecutionProvider(可以理解为设备)
	 * \param DeviceID_ 设备ID
	 * \param ThreadCount_ 线程数
	 */
	LibTTSModule(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_ = 0);

	virtual ~LibTTSModule();

	/**
	 * \brief 获取采样率
	 * \return 采样率
	 */
	[[nodiscard]] long GetSamplingRate() const
	{
		return ModelSamplingRate;
	}
protected:
	/**
	 * @brief Sampling rate
	 */
	long ModelSamplingRate = 22050;

	/**
	 * @brief ONNX environment
	 */
	Ort::Env* OnnxEnv = nullptr;

	/**
	 * @brief Session options
	 */
	Ort::SessionOptions* SessionOptions = nullptr;

	/**
	 * @brief Memory info
	 */
	Ort::MemoryInfo* MemoryInfo = nullptr;

	/**
	 * @brief Execution provider
	 */
	ExecutionProviders ModelExecutionProvider = ExecutionProviders::CPU;

	/**
	 * @brief Shared pointer to DragonianLibOrtEnv
	 */
	std::shared_ptr<DragonianLibOrtEnv> OrtApiEnv;

	/**
	 * @brief Progress callback function
	 */
	ProgressCallback ProgressCallbackFunction;
public:
	LibTTSModule(const LibTTSModule&) = default;
	LibTTSModule& operator=(const LibTTSModule&) = default;
	LibTTSModule(LibTTSModule&&) noexcept = default;
	LibTTSModule& operator=(LibTTSModule&&) noexcept = default;
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End