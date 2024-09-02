/**
 * FileName: MoeVSSamplers.hpp
 * Note: MoeVoiceStudioCore Diffusion 官方采样器
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
#include "BaseSampler.hpp"

LibSvcHeader

class PndmSampler : public BaseSampler
{
public:
	PndmSampler(Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~PndmSampler() override = default;
	PndmSampler(const PndmSampler&) = delete;
	PndmSampler(PndmSampler&&) = delete;
	PndmSampler operator=(const PndmSampler&) = delete;
	PndmSampler operator=(PndmSampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process) override;
private:
	const std::vector<const char*> denoiseInput = { "noise", "time", "condition" };
	const std::vector<const char*> predInput = { "noise", "noise_pred", "time", "time_prev" };
	const std::vector<const char*> denoiseOutput = { "noise_pred" };
	const std::vector<const char*> predOutput = { "noise_pred_o" };
};

class DDimSampler : public BaseSampler
{
public:
	DDimSampler(Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~DDimSampler() override = default;
	DDimSampler(const DDimSampler&) = delete;
	DDimSampler(DDimSampler&&) = delete;
	DDimSampler operator=(const DDimSampler&) = delete;
	DDimSampler operator=(DDimSampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process) override;
private:
	const std::vector<const char*> alphain = { "time" };
	const std::vector<const char*> alphaout = { "alphas_cumprod" };
	const std::vector<const char*> denoiseInput = { "noise", "time", "condition" };
	const std::vector<const char*> denoiseOutput = { "noise_pred" };
};

class ReflowEularSampler : public ReflowBaseSampler
{
public:
	ReflowEularSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~ReflowEularSampler() override = default;
	ReflowEularSampler(const ReflowEularSampler&) = delete;
	ReflowEularSampler(ReflowEularSampler&&) = delete;
	ReflowEularSampler operator=(const ReflowEularSampler&) = delete;
	ReflowEularSampler operator=(ReflowEularSampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;
private:
	const std::vector<const char*> velocityInput = { "x", "t", "cond" };
	const std::vector<const char*> velocityOutput = { "o" };
};

class ReflowRk4Sampler : public ReflowBaseSampler
{
public:
	ReflowRk4Sampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~ReflowRk4Sampler() override = default;
	ReflowRk4Sampler(const ReflowRk4Sampler&) = delete;
	ReflowRk4Sampler(ReflowRk4Sampler&&) = delete;
	ReflowRk4Sampler operator=(const ReflowRk4Sampler&) = delete;
	ReflowRk4Sampler operator=(ReflowRk4Sampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;
private:
	const std::vector<const char*> velocityInput = { "x", "t", "cond" };
	const std::vector<const char*> velocityOutput = { "o" };
};

class ReflowHeunSampler : public ReflowBaseSampler
{
public:
	ReflowHeunSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~ReflowHeunSampler() override = default;
	ReflowHeunSampler(const ReflowHeunSampler&) = delete;
	ReflowHeunSampler(ReflowHeunSampler&&) = delete;
	ReflowHeunSampler operator=(const ReflowHeunSampler&) = delete;
	ReflowHeunSampler operator=(ReflowHeunSampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;
private:
	const std::vector<const char*> velocityInput = { "x", "t", "cond" };
	const std::vector<const char*> velocityOutput = { "o" };
};

class ReflowPececeSampler : public ReflowBaseSampler
{
public:
	ReflowPececeSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);
	~ReflowPececeSampler() override = default;
	ReflowPececeSampler(const ReflowPececeSampler&) = delete;
	ReflowPececeSampler(ReflowPececeSampler&&) = delete;
	ReflowPececeSampler operator=(const ReflowPececeSampler&) = delete;
	ReflowPececeSampler operator=(ReflowPececeSampler&&) = delete;
	std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;
private:
	const std::vector<const char*> velocityInput = { "x", "t", "cond" };
	const std::vector<const char*> velocityOutput = { "o" };
};

LibSvcEnd