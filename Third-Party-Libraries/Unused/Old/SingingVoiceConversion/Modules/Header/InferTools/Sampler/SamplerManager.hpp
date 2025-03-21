/**
 * FileName: MoeVSSamplerManager.hpp
 * Note: MoeVoiceStudioCore Diffusion Sampler
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

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_Header

using SamplerWrp = std::shared_ptr<BaseSampler>;

using ReflowSamplerWrp = std::shared_ptr<ReflowBaseSampler>;

using GetSamplerFn = std::function<SamplerWrp(Ort::Session*, Ort::Session*, Ort::Session*, int64_t, const BaseSampler::ProgressCallback&, Ort::MemoryInfo*)>;

/**
 * \brief Register a sampler
 * \param _name Class name
 * \param _constructor_fn Constructor function
 */
void RegisterSampler(const std::wstring& _name, const GetSamplerFn& _constructor_fn);

/**
 * \brief Get a sampler
 * \param _name Class name
 * \param alpha Alphas Onnx model session
 * \param dfn DenoiseFn Onnx model session
 * \param pred Predictor Onnx model session
 * \param Mel_Bins MelBins
 * \param _ProgressCallback Progress bar callback (directly pass the model's callback)
 * \param memory Model's OrtMemoryInfo
 * \return Sampler
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

/**
 * \brief Get the list of samplers
 * \return List of sampler names
 */
std::vector<std::wstring> GetSamplerList();

/******************************* Reflow ***********************************/

using GetReflowSamplerFn = std::function<ReflowSamplerWrp(
	Ort::Session*,
	int64_t,
	const ReflowBaseSampler::ProgressCallback&,
	Ort::MemoryInfo*
)>;

/**
 * \brief Register a reflow sampler
 * \param _name Class name
 * \param _constructor_fn Constructor function
 */
void RegisterReflowSampler(const std::wstring& _name, const GetReflowSamplerFn& _constructor_fn);

/**
 * \brief Get a reflow sampler
 * \param _name Class name
 * \param velocity Velocity Onnx model session
 * \param Mel_Bins MelBins
 * \param _ProgressCallback Progress bar callback (directly pass the model's callback)
 * \param memory Model's OrtMemoryInfo
 * \return Reflow sampler
 */
ReflowSamplerWrp GetReflowSampler(
	const std::wstring& _name,
	Ort::Session* velocity,
	int64_t Mel_Bins,
	const BaseSampler::ProgressCallback& _ProgressCallback,
	Ort::MemoryInfo* memory
);

/**
 * \brief Get the list of reflow samplers
 * \return List of reflow sampler names
 */
std::vector<std::wstring> GetReflowSamplerList();

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Sampler_End
