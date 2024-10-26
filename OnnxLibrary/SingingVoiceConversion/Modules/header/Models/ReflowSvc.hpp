/**
 * FileName: ReflowSvc.hpp
 * Note: MoeVoiceStudioCore Onnx Reflow
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
 * @class ReflowSvc
 * @brief Reflow model class, inherits from SingingVoiceConversion
 */
class ReflowSvc : public SingingVoiceConversion
{
public:
    /**
     * @brief Constructor
     * @param _Hps Hyperparameters
     * @param _ProgressCallback Progress callback function
     * @param ExecutionProvider_ Execution provider
     * @param DeviceID_ Device ID
     * @param ThreadCount_ Number of threads
     */
    ReflowSvc(
        const Hparams& _Hps,
        const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

    /**
     * @brief Destructor
     */
    ~ReflowSvc() override;

    /**
     * @brief Perform inference on a single audio slice
     * @param _Slice Audio slice
     * @param _Params Inference parameters
     * @param _Process Processing progress
     * @return Inference result
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params,
        size_t& _Process
    ) const override;

    /**
     * @brief Perform inference on PCM data
     * @param _PCMData PCM data
     * @param _SrcSamplingRate Source sampling rate
     * @param _Params Inference parameters
     * @return Inference result
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> InferPCMData(
        const DragonianLibSTL::Vector<float>& _PCMData,
        long _SrcSamplingRate,
        const InferenceParams& _Params
    ) const override;

    /**
     * @brief Perform shallow diffusion inference
     * @param _16KAudioHubert 16K audio data
     * @param _Params Inference parameters
     * @param _Mel Mel spectrogram data
     * @param _SrcF0 Source F0 data
     * @param _SrcVolume Source volume data
     * @param _SrcSpeakerMap Source speaker map data
     * @param Process Processing progress
     * @param SrcSize Source data size
     * @return Inference result
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> ShallowDiffusionInference(
        DragonianLibSTL::Vector<float>& _16KAudioHubert,
        const InferenceParams& _Params,
        std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
        const DragonianLibSTL::Vector<float>& _SrcF0,
        const DragonianLibSTL::Vector<float>& _SrcVolume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
        size_t& Process,
        int64_t SrcSize
    ) const override;

    /**
     * @brief Get the maximum step
     * @return Maximum step
     */
    [[nodiscard]] int64_t GetMaxStep() const override
    {
        return MaxStep;
    }

    /**
     * @brief Get the union svc version
     * @return Union svc version
     */
    [[nodiscard]] const std::wstring& GetUnionSvcVer() const override
    {
        return ReflowSvcVersion;
    }

    /**
     * @brief Get the number of Mel bins
     * @return Number of Mel bins
     */
    [[nodiscard]] int64_t GetMelBins() const override
    {
        return MelBins;
    }

    /**
     * @brief Normalize Mel spectrogram
     * @param MelSpec Mel spectrogram data
     */
    void NormMel(
        DragonianLibSTL::Vector<float>& MelSpec
    ) const override;

private:
    std::shared_ptr<Ort::Session> PreEncoder = nullptr; ///< Pre-encoder
    std::shared_ptr<Ort::Session> VelocityFunction = nullptr; ///< Velocity function
    std::shared_ptr<Ort::Session> PostDecoder = nullptr; ///< Post-decoder

    int64_t MelBins = 128; ///< Number of Mel bins
    int64_t MaxStep = 100; ///< Maximum step
    float SpecMin = -12; ///< Minimum spectrogram value
    float SpecMax = 2; ///< Maximum spectrogram value
    float Scale = 1000.f; ///< Scale factor
    bool VaeMode = true; ///< VAE mode

    std::wstring ReflowSvcVersion = L"ReflowSvc"; ///< Reflow svc version

    static inline const std::vector<const char*> nsfInput = { "c", "f0" }; ///< NSF input
    static inline const std::vector<const char*> nsfOutput = { "audio" }; ///< NSF output
    static inline const std::vector<const char*> afterInput = { "x" }; ///< Post-processing input
    static inline const std::vector<const char*> afterOutput = { "mel_out" }; ///< Post-processing output
    static inline const std::vector<const char*> OutputNamesEncoder = { "x", "cond", "f0_pred" }; ///< Encoder output names

public:
    ReflowSvc(const ReflowSvc&) = default; ///< Copy constructor
    ReflowSvc(ReflowSvc&&) = default; ///< Move constructor
    ReflowSvc& operator=(const ReflowSvc&) = default; ///< Copy assignment operator
    ReflowSvc& operator=(ReflowSvc&&) = default; ///< Move assignment operator
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
