﻿/**
 * FileName: MoeVSSamplerManager.hpp
 * Note: MoeVoiceStudioCore Diffusion 采样器管理
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
 * date: 2022-10-17 Create
*/

#pragma once
#include "BaseSampler.hpp"

LibSvcHeader

using SamplerWrp = std::shared_ptr<BaseSampler>;

using ReflowSamplerWrp = std::shared_ptr<ReflowBaseSampler>;

using GetSamplerFn = std::function<SamplerWrp(Ort::Session*, Ort::Session*, Ort::Session*, int64_t, const BaseSampler::ProgressCallback&, Ort::MemoryInfo*)>;

void RegisterSampler(const std::wstring& _name, const GetSamplerFn& _constructor_fn);

/**
 * \brief 获取采样器
 * \param _name 类名
 * \param alpha Alphas Onnx模型Session
 * \param dfn DenoiseFn Onnx模型Session
 * \param pred Predictor Onnx模型Session
 * \param Mel_Bins MelBins
 * \param _ProgressCallback 进度条回调（直接传模型的回调就可以了）
 * \param memory 模型的OrtMemoryInfo
 * \return 采样器
 */
SamplerWrp GetSampler(
	const std::wstring& _name,
	Ort::Session* alpha,
	Ort::Session* dfn,
	Ort::Session* pred,
	int64_t Mel_Bins,
	const BaseSampler::ProgressCallback& _ProgressCallback,
	Ort::MemoryInfo* memory
);

std::vector<std::wstring> GetSamplerList();

/******************************* Reflow ***********************************/

using GetReflowSamplerFn = std::function<ReflowSamplerWrp(
	Ort::Session*, 
	int64_t, 
	const ReflowBaseSampler::ProgressCallback&,
	Ort::MemoryInfo*
)>;

void RegisterReflowSampler(const std::wstring& _name, const GetReflowSamplerFn& _constructor_fn);

/**
 * \brief 获取采样器
 * \param _name 类名
 * \param velocity Velocity Onnx模型Session
 * \param Mel_Bins MelBins
 * \param _ProgressCallback 进度条回调（直接传模型的回调就可以了）
 * \param memory 模型的OrtMemoryInfo
 * \return 采样器
 */
ReflowSamplerWrp GetReflowSampler(
	const std::wstring& _name,
	Ort::Session* velocity,
	int64_t Mel_Bins,
	const BaseSampler::ProgressCallback& _ProgressCallback,
	Ort::MemoryInfo* memory
);

std::vector<std::wstring> GetReflowSamplerList();

LibSvcEnd