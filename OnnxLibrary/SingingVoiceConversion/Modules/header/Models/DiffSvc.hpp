/**
 * FileName: DiffSvc.hpp
 * Note: MoeVoiceStudioCore Onnx DiffusionSvc
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
LibSvcHeader

/**
 * @class DiffusionSvc
 * @brief Diffusion model class, inherits from SingingVoiceConversion
 */
    class DiffusionSvc : public SingingVoiceConversion
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
    DiffusionSvc(
        const Hparams& _Hps,
        const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

    /**
     * @brief Destructor
     */
    ~DiffusionSvc() override;

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
     * @brief Check if using old version
     * @return True if using old version, otherwise false
     */
    [[nodiscard]] bool OldVersion() const
    {
        return OldDiffSvc.get();
    }

    /**
     * @brief Get the union svc version
     * @return Union svc version
     */
    [[nodiscard]] const std::wstring& GetUnionSvcVer() const override
    {
        return DiffSvcVersion;
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
    std::shared_ptr<Ort::Session> DiffusionDenoiser = nullptr; ///< Diffusion denoiser
    std::shared_ptr<Ort::Session> NoisePredictor = nullptr; ///< Noise predictor
    std::shared_ptr<Ort::Session> PostDecoder = nullptr; ///< Post-decoder
    std::shared_ptr<Ort::Session> AlphaCumprod = nullptr; ///< Alpha cumulative product
    std::shared_ptr<Ort::Session> NaiveModel = nullptr; ///< Naive model

    std::shared_ptr<Ort::Session> OldDiffSvc = nullptr; ///< Old diffusion svc

    int64_t MelBins = 128; ///< Number of Mel bins
    int64_t Pndms = 100; ///< PNDMS value
    int64_t MaxStep = 1000; ///< Maximum step
    float SpecMin = -12; ///< Minimum spectrogram value
    float SpecMax = 2; ///< Maximum spectrogram value

    std::wstring DiffSvcVersion = L"DiffSvc"; ///< Diffusion svc version

    static inline const std::vector<const char*> nsfInput = { "c", "f0" }; ///< NSF input
    static inline const std::vector<const char*> nsfOutput = { "audio" }; ///< NSF output
    static inline const std::vector<const char*> DiffInput = { "hubert", "mel2ph", "spk_embed", "f0", "initial_noise", "speedup" }; ///< Diffusion input
    static inline const std::vector<const char*> DiffOutput = { "mel_pred", "f0_pred" }; ///< Diffusion output
    static inline const std::vector<const char*> afterInput = { "x" }; ///< Post-processing input
    static inline const std::vector<const char*> afterOutput = { "mel_out" }; ///< Post-processing output
    static inline const std::vector<const char*> naiveOutput = { "mel" }; ///< Naive output

public:
    DiffusionSvc(const DiffusionSvc&) = default; ///< Copy constructor
    DiffusionSvc(DiffusionSvc&&) = default; ///< Move constructor
    DiffusionSvc& operator=(const DiffusionSvc&) = default; ///< Copy assignment operator
    DiffusionSvc& operator=(DiffusionSvc&&) = default; ///< Move assignment operator
};

LibSvcEnd
