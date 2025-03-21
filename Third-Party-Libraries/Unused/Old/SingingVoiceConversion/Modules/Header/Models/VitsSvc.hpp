/**
 * FileName: VitsSvc.hpp
 * Note: MoeVoiceStudioCore Onnx Vits series SVC model definition
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
#include "SVC.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @class VitsSvc
 * @brief Derived class for Vits series singing voice conversion using Onnx models
 */
class VitsSvc : public SingingVoiceConversion
{
public:
    /**
     * @brief Constructor for VitsSvc
     * @param _Hps Hyperparameters
     * @param _ProgressCallback Progress callback function
     * @param ExecutionProvider_ Execution provider (default is CPU)
     * @param DeviceID_ Device ID (default is 0)
     * @param ThreadCount_ Number of threads (default is 0)
     */
    VitsSvc(
        const Hparams& _Hps,
        const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

    VitsSvc(
        const Hparams& _Hps,
        const ProgressCallback& _ProgressCallback,
        const std::shared_ptr<DragonianLibOrtEnv>& Env_
    );

    /**
     * @brief Destructor for VitsSvc
     */
    ~VitsSvc() override;

    /**
     * @brief Performs inference on a single slice of audio
     * @param _Slice The audio slice
     * @param _Params Inference parameters
     * @param _Process Process size
     * @return Inference result as a vector of floats
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params,
        size_t& _Process
    ) const override;

    /**
     * @brief Performs inference on PCM data
     * @param _PCMData The PCM data
     * @param _SrcSamplingRate Source sampling rate
     * @param _Params Inference parameters
     * @return Inference result as a vector of floats
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> InferPCMData(
        const DragonianLibSTL::Vector<float>& _PCMData,
        long _SrcSamplingRate,
        const InferenceParams& _Params
    ) const override;

    /**
     * @brief Mel spectrogram extractor (deleted function)
     * @param PCMAudioBegin Pointer to the beginning of PCM audio data
     * @param PCMAudioEnd Pointer to the end of PCM audio data
     * @return Mel spectrogram as a vector of Ort::Value
     */
    [[nodiscard]] std::vector<Ort::Value> MelExtractor(
        const float* PCMAudioBegin,
        const float* PCMAudioEnd
    ) const = delete;

    /**
     * @brief Gets the memory information
     * @return Pointer to the memory information
     */
    [[nodiscard]] Ort::MemoryInfo* GetMemoryInfo() const
    {
        return MemoryInfo;
    }

private:
    std::shared_ptr<Ort::Session> VitsSvcModel = nullptr;
    std::wstring VitsSvcVersion = L"SoVits4.0";

    static inline const std::vector<const char*> soVitsOutput = { "audio" };
    static inline const std::vector<const char*> soVitsInput = { "hidden_unit", "lengths", "pitch", "sid" };
    static inline const std::vector<const char*> RVCInput = { "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd" };
    static inline const std::vector<const char*> StftOutput = { "mel" };
    static inline const std::vector<const char*> StftInput = { "waveform", "aligment" };

public:
    VitsSvc(const VitsSvc&) = default;
    VitsSvc(VitsSvc&&) = default;
    VitsSvc& operator=(const VitsSvc&) = default;
    VitsSvc& operator=(VitsSvc&&) = default;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
