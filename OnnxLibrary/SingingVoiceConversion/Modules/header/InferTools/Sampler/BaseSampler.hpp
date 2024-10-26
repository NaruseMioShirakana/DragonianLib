/**
 * FileName: MoeVSBaseSampler.hpp
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

#include "../InferTools.hpp"
#include <onnxruntime_cxx_api.h>

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

class BaseSampler
{
public:
	using ProgressCallback = std::function<void(size_t, size_t)>;

	/**
	 * \brief Constructor for the sampler
	 * \param alpha Alphas Onnx model session
	 * \param dfn DenoiseFn Onnx model session
	 * \param pred Predictor Onnx model session
	 * \param Mel_Bins MelBins
	 * \param _ProgressCallback Progress bar callback (directly pass the model's callback)
	 * \param memory Model's OrtMemoryInfo
	 */
	BaseSampler(
		Ort::Session* alpha,
		Ort::Session* dfn,
		Ort::Session* pred,
		int64_t Mel_Bins,
		const ProgressCallback& _ProgressCallback,
		Ort::MemoryInfo* memory
	);
	BaseSampler(const BaseSampler&) = delete;
	BaseSampler(BaseSampler&&) = delete;
	BaseSampler operator=(const BaseSampler&) = delete;
	BaseSampler operator=(BaseSampler&&) = delete;
	virtual ~BaseSampler() = default;

	/**
	 * \brief Sampling
	 * \param Tensors Input tensors (Tensors[0] is Condition, Tensors[1] is initial noise)
	 * \param Steps Number of sampling steps
	 * \param SpeedUp Speed up factor
	 * \param NoiseScale Noise scale
	 * \param Seed Seed
	 * \param Process Current progress
	 * \return Mel tensor
	 */
	virtual std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process);
protected:
	int64_t MelBins = 128;
	Ort::Session* Alpha = nullptr;
	Ort::Session* DenoiseFn = nullptr;
	Ort::Session* NoisePredictor = nullptr;
	ProgressCallback _callback;
	Ort::MemoryInfo* Memory = nullptr;
};

class ReflowBaseSampler
{
public:
	using ProgressCallback = std::function<void(size_t, size_t)>;

	/**
	 * \brief Get the sampler
	 * \param Velocity Velocity Onnx model session
	 * \param MelBins MelBins
	 * \param _ProgressCallback Progress bar callback (directly pass the model's callback)
	 * \param memory Model's OrtMemoryInfo
	 * \return Sampler
	 */
	ReflowBaseSampler(
		Ort::Session* Velocity,
		int64_t MelBins,
		const ProgressCallback& _ProgressCallback,
		Ort::MemoryInfo* memory
	);
	ReflowBaseSampler(const ReflowBaseSampler&) = delete;
	ReflowBaseSampler(ReflowBaseSampler&&) = delete;
	ReflowBaseSampler operator=(const ReflowBaseSampler&) = delete;
	ReflowBaseSampler operator=(ReflowBaseSampler&&) = delete;
	virtual ~ReflowBaseSampler() = default;

	/**
	 * \brief Sampling
	 * \param Tensors Input tensors
	 * \param Steps Number of sampling steps
	 * \param dt dt
	 * \param Scale Scale
	 * \param Process Current progress
	 * \return Mel tensor
	 */
	virtual std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process);
protected:
	int64_t MelBins_ = 128;
	Ort::Session* Velocity_ = nullptr;
	ProgressCallback Callback_;
	Ort::MemoryInfo* Memory_ = nullptr;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
