/**
 * FileName: SVC.hpp
 * Note: MoeVoiceStudioCore OnnxSvc model base class
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
#include "ModelBase.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include "../InferTools/TensorExtractor/TensorExtractorManager.hpp"
#include "Libraries/Cluster/ClusterManager.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

using OrtTensors = std::vector<Ort::Value>;

/**
 * @class SingingVoiceConversion
 * @brief Derived class for singing voice conversion using Onnx models
 */
class SingingVoiceConversion : public LibSvcModule
{
public:
    /**
     * @brief Constructor for SingingVoiceConversion
     * @param HubertPath_ Path to the Hubert model
     * @param ExecutionProvider_ Execution provider (device)
     * @param DeviceID_ Device ID
     * @param ThreadCount_ Number of threads (default is 0)
     */
    SingingVoiceConversion(
        const std::wstring& HubertPath_,
        const ExecutionProviders& ExecutionProvider_,
        unsigned DeviceID_,
        unsigned ThreadCount_ = 0
    );

    SingingVoiceConversion(
        const std::wstring& HubertPath_,
		const std::shared_ptr<DragonianLibOrtEnv>& Env_
	);


    /**
     * @brief Performs inference with crossfade
     * @param _Audio The audio data
     * @param _SrcSamplingRate Source sampling rate
     * @param _Params Inference parameters
     * @param _Crossfade Crossfade parameters
     * @param _F0Params F0 parameters
     * @param _F0Method F0 extraction method
     * @param _F0ExtractorLoadParameter F0 extractor user parameter
     * @param _DbThreshold Threshold
     * @return Inference result as a vector of floats (_SrcSamplingRate)
     */
    DragonianLibSTL::Vector<float> InferenceWithCrossFade(
        const DragonianLibSTL::ConstantRanges<float>& _Audio,
        long _SrcSamplingRate,
        const InferenceParams& _Params,
		const CrossFadeParams& _Crossfade,
		const F0Extractor::F0ExtractorParams& _F0Params,
		const std::wstring& _F0Method,
		const F0Extractor::NetF0ExtractorSetting& _F0ExtractorLoadParameter,
        double _DbThreshold
    ) const;

    /**
     * @brief Performs inference on a single slice of audio
     * @param _Slice The audio slice
     * @param _Params Inference parameters
     * @param _Process Process size
     * @return Inference result as a vector of floats (Model.SamplingRate)
     */
    [[nodiscard]] virtual DragonianLibSTL::Vector<float> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params,
        size_t& _Process
    ) const;

    /**
     * @brief Performs inference on PCM data
     * @param _PCMData The PCM data
     * @param _SrcSamplingRate Source sampling rate
     * @param _Params Inference parameters
     * @return Inference result as a vector of floats (Model.SamplingRate)
     */
    [[nodiscard]] virtual DragonianLibSTL::Vector<float> InferPCMData(
        const DragonianLibSTL::Vector<float>& _PCMData,
        long _SrcSamplingRate,
        const InferenceParams& _Params
    ) const;

    /**
     * @brief Performs shallow diffusion inference
     * @param _16KAudioHubert 16K audio data
     * @param _Params Inference parameters
     * @param _Mel Mel spectrogram and its size
     * @param _SrcF0 Source F0 data
     * @param _SrcVolume Source volume data
     * @param _SrcSpeakerMap Source speaker map
     * @param Process Process size
     * @param SrcSize Source size
     * @return Inference result as a vector of floats (Model.SamplingRate)
     */
    [[nodiscard]] virtual DragonianLibSTL::Vector<float> ShallowDiffusionInference(
        DragonianLibSTL::Vector<float>& _16KAudioHubert,
        const InferenceParams& _Params,
        std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
        const DragonianLibSTL::Vector<float>& _SrcF0,
        const DragonianLibSTL::Vector<float>& _SrcVolume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
        size_t& Process,
        int64_t SrcSize
    ) const;

    /**
     * @brief Extracts volume from audio data
     * @param _Audio The audio data
     * @param _HopSize Hop size
     * @return Extracted volume as a vector of floats
     */
    [[nodiscard]] static DragonianLibSTL::Vector<float> ExtractVolume(
        const DragonianLibSTL::Vector<float>& _Audio,
        int _HopSize
    );

    /**
     * @brief Extracts volume from audio data
     * @param _Audio The audio data
     * @return Extracted volume as a vector of floats
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> ExtractVolume(
        const DragonianLibSTL::Vector<float>& _Audio
    ) const;

    /**
     * @brief Gets an audio slice
     * @param _InputPCM Input PCM data
     * @param _SlicePos Slice positions
	 * @param _SamplingRate Sampling rate
	 * @param _Threshold Threshold
     * @return Single audio slice
     */
    [[nodiscard]] static SingleAudio GetAudioSlice(
        const DragonianLibSTL::Vector<float>& _InputPCM,
        const DragonianLibSTL::Vector<size_t>& _SlicePos,
        long _SamplingRate,
        double _Threshold
    );

    [[nodiscard]] static SingleAudio GetAudioSlice(
        const DragonianLibSTL::ConstantRanges<float>& _InputPCM,
        const DragonianLibSTL::ConstantRanges<size_t>& _SlicePos,
        long _SamplingRate,
        double _Threshold
    );

    /**
     * @brief Pre-processes audio data
     * @param _Input Input audio data
	 * @param _Params F0 parameters
     * @param _F0Method F0 extraction method (default is "Dio")
	 * @param _F0ExtractorLoadParameter F0 extractor user parameter
     */
    static void PreProcessAudio(
		SingleAudio& _Input,
        const F0Extractor::F0ExtractorParams& _Params,
        const std::wstring& _F0Method,
        const F0Extractor::NetF0ExtractorSetting& _F0ExtractorLoadParameter
    );

    /**
     * @brief Destructor for SingingVoiceConversion
     */
    ~SingingVoiceConversion() override;

    /**
     * @brief Gets the hop size
     * @return Hop size
     */
    [[nodiscard]] int GetHopSize() const;

    /**
     * @brief Gets the hidden unit K dimensions
     * @return Hidden unit K dimensions
     */
    [[nodiscard]] int64_t GetHiddenUnitKDims() const;

    /**
     * @brief Gets the speaker count
     * @return Speaker count
     */
    [[nodiscard]] int64_t GetSpeakerCount() const;

    /**
     * @brief Checks if speaker mix is enabled
     * @return True if speaker mix is enabled, false otherwise
     */
    [[nodiscard]] bool SpeakerMixEnabled() const;

    /**
     * @brief Gets the maximum step
     * @return Maximum step
     */
    [[nodiscard]] virtual int64_t GetMaxStep() const;

    /**
     * @brief Gets the Union SVC version
     * @return Union SVC version
     */
    [[nodiscard]] virtual const std::wstring& GetUnionSvcVer() const;

    /**
     * @brief Gets the Mel bins
     * @return Mel bins
     */
    [[nodiscard]] virtual int64_t GetMelBins() const;

    /**
     * @brief Normalizes Mel spectrogram
     * @param MelSpec Mel spectrogram
     */
    virtual void NormMel(
        DragonianLibSTL::Vector<float>& MelSpec
    ) const;

    void Tick(size_t Cur, size_t Total) const
    {
        ProgressCallbackFunction(Cur, Total);
    }

protected:
    TensorExtractor::TensorExtractor Preprocessor;
    std::shared_ptr<Ort::Session> HubertModel = nullptr;

    int HopSize = 320;
    int64_t HiddenUnitKDims = 256;
    int64_t SpeakerCount = 1;
    bool EnableCharaMix = false;
    bool EnableVolume = false;

    Cluster::Cluster Cluster;
    int64_t ClusterCenterSize = 10000;
    bool EnableCluster = false;

    static inline const std::vector<const char*> hubertOutput = { "embed" };
    static inline const std::vector<const char*> hubertInput = { "source" };

public:
    SingingVoiceConversion& operator=(SingingVoiceConversion&&) = default;
    SingingVoiceConversion& operator=(const SingingVoiceConversion&) = default;
    SingingVoiceConversion(const SingingVoiceConversion&) = default;
    SingingVoiceConversion(SingingVoiceConversion&&) = default;
};

/**
 * @brief Performs vocoder inference
 * @param Mel Mel spectrogram
 * @param F0 F0 data
 * @param MelBins Mel bins
 * @param MelSize Mel size
 * @param Mem Memory info
 * @param _VocoderModel Vocoder model
 * @return Inference result as a vector of floats
 */
DragonianLibSTL::Vector<float> VocoderInfer(
    DragonianLibSTL::Vector<float>& Mel,
    DragonianLibSTL::Vector<float>& F0,
    int64_t MelBins,
    int64_t MelSize,
    const Ort::MemoryInfo* Mem,
    const std::shared_ptr<Ort::Session>& _VocoderModel
);

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
