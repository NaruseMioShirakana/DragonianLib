/**
 * FileName: MoeVSSamplers.hpp
 * Note: MoeVoiceStudioCore Diffusion Samplers
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

/**
 * @class PndmSampler
 * @brief Pndm Algorithm Sampler
 */
    class PndmSampler : public BaseSampler
{
public:
    /**
     * @brief Constructor for PndmSampler
     * @param alpha Alpha session
     * @param dfn DFN session
     * @param pred Prediction session
     * @param Mel_Bins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    PndmSampler(Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for PndmSampler
     */
    ~PndmSampler() override = default;

    PndmSampler(const PndmSampler&) = delete;
    PndmSampler(PndmSampler&&) = delete;
    PndmSampler operator=(const PndmSampler&) = delete;
    PndmSampler operator=(PndmSampler&&) = delete;

    /**
     * @brief Sample function for PndmSampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param SpeedUp Speed up factor
     * @param NoiseScale Noise scale factor
     * @param Seed Random seed
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process) override;

private:
    const std::vector<const char*> denoiseInput = { "noise", "time", "condition" };
    const std::vector<const char*> predInput = { "noise", "noise_pred", "time", "time_prev" };
    const std::vector<const char*> denoiseOutput = { "noise_pred" };
    const std::vector<const char*> predOutput = { "noise_pred_o" };
};

/**
 * @class DDimSampler
 * @brief DDim Algorithm Sampler
 */
class DDimSampler : public BaseSampler
{
public:
    /**
     * @brief Constructor for DDimSampler
     * @param alpha Alpha session
     * @param dfn DFN session
     * @param pred Prediction session
     * @param Mel_Bins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    DDimSampler(Ort::Session* alpha, Ort::Session* dfn, Ort::Session* pred, int64_t Mel_Bins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for DDimSampler
     */
    ~DDimSampler() override = default;

    DDimSampler(const DDimSampler&) = delete;
    DDimSampler(DDimSampler&&) = delete;
    DDimSampler operator=(const DDimSampler&) = delete;
    DDimSampler operator=(DDimSampler&&) = delete;

    /**
     * @brief Sample function for DDimSampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param SpeedUp Speed up factor
     * @param NoiseScale Noise scale factor
     * @param Seed Random seed
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, int64_t SpeedUp, float NoiseScale, int64_t Seed, size_t& Process) override;

private:
    const std::vector<const char*> alphain = { "time" };
    const std::vector<const char*> alphaout = { "alphas_cumprod" };
    const std::vector<const char*> denoiseInput = { "noise", "time", "condition" };
    const std::vector<const char*> denoiseOutput = { "noise_pred" };
};

/**
 * @class ReflowEularSampler
 * @brief Reflow Eular Algorithm Sampler
 */
class ReflowEularSampler : public ReflowBaseSampler
{
public:
    /**
     * @brief Constructor for ReflowEularSampler
     * @param Velocity Velocity session
     * @param MelBins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    ReflowEularSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for ReflowEularSampler
     */
    ~ReflowEularSampler() override = default;

    ReflowEularSampler(const ReflowEularSampler&) = delete;
    ReflowEularSampler(ReflowEularSampler&&) = delete;
    ReflowEularSampler operator=(const ReflowEularSampler&) = delete;
    ReflowEularSampler operator=(ReflowEularSampler&&) = delete;

    /**
     * @brief Sample function for ReflowEularSampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param dt Time step
     * @param Scale Scale factor
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;

private:
    const std::vector<const char*> velocityInput = { "x", "t", "cond" };
    const std::vector<const char*> velocityOutput = { "o" };
};

/**
 * @class ReflowRk4Sampler
 * @brief Reflow RK4 Algorithm Sampler
 */
class ReflowRk4Sampler : public ReflowBaseSampler
{
public:
    /**
     * @brief Constructor for ReflowRk4Sampler
     * @param Velocity Velocity session
     * @param MelBins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    ReflowRk4Sampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for ReflowRk4Sampler
     */
    ~ReflowRk4Sampler() override = default;

    ReflowRk4Sampler(const ReflowRk4Sampler&) = delete;
    ReflowRk4Sampler(ReflowRk4Sampler&&) = delete;
    ReflowRk4Sampler operator=(const ReflowRk4Sampler&) = delete;
    ReflowRk4Sampler operator=(ReflowRk4Sampler&&) = delete;

    /**
     * @brief Sample function for ReflowRk4Sampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param dt Time step
     * @param Scale Scale factor
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;

private:
    const std::vector<const char*> velocityInput = { "x", "t", "cond" };
    const std::vector<const char*> velocityOutput = { "o" };
};

/**
 * @class ReflowHeunSampler
 * @brief Reflow Heun Algorithm Sampler
 */
class ReflowHeunSampler : public ReflowBaseSampler
{
public:
    /**
     * @brief Constructor for ReflowHeunSampler
     * @param Velocity Velocity session
     * @param MelBins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    ReflowHeunSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for ReflowHeunSampler
     */
    ~ReflowHeunSampler() override = default;

    ReflowHeunSampler(const ReflowHeunSampler&) = delete;
    ReflowHeunSampler(ReflowHeunSampler&&) = delete;
    ReflowHeunSampler operator=(const ReflowHeunSampler&) = delete;
    ReflowHeunSampler operator=(ReflowHeunSampler&&) = delete;

    /**
     * @brief Sample function for ReflowHeunSampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param dt Time step
     * @param Scale Scale factor
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;

private:
    const std::vector<const char*> velocityInput = { "x", "t", "cond" };
    const std::vector<const char*> velocityOutput = { "o" };
};

/**
 * @class ReflowPececeSampler
 * @brief Reflow PECECE Algorithm Sampler
 */
class ReflowPececeSampler : public ReflowBaseSampler
{
public:
    /**
     * @brief Constructor for ReflowPececeSampler
     * @param Velocity Velocity session
     * @param MelBins Number of Mel bins
     * @param _ProgressCallback Progress callback function
     * @param memory Memory information
     */
    ReflowPececeSampler(Ort::Session* Velocity, int64_t MelBins, const ProgressCallback& _ProgressCallback, Ort::MemoryInfo* memory);

    /**
     * @brief Destructor for ReflowPececeSampler
     */
    ~ReflowPececeSampler() override = default;

    ReflowPececeSampler(const ReflowPececeSampler&) = delete;
    ReflowPececeSampler(ReflowPececeSampler&&) = delete;
    ReflowPececeSampler operator=(const ReflowPececeSampler&) = delete;
    ReflowPececeSampler operator=(ReflowPececeSampler&&) = delete;

    /**
     * @brief Sample function for ReflowPececeSampler
     * @param Tensors Input tensors
     * @param Steps Number of steps
     * @param dt Time step
     * @param Scale Scale factor
     * @param Process Process size
     * @return Vector of output values
     */
    std::vector<Ort::Value> Sample(std::vector<Ort::Value>& Tensors, int64_t Steps, float dt, float Scale, size_t& Process) override;

private:
    const std::vector<const char*> velocityInput = { "x", "t", "cond" };
    const std::vector<const char*> velocityOutput = { "o" };
};

LibSvcEnd
