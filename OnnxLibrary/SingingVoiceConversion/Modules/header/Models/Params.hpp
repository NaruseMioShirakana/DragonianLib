/**
 * FileName: MoeVSProject.hpp
 * Note: MoeVoiceStudioCore Parameters
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
#include "onnxruntime_cxx_api.h"
#include "../InferTools/InferTools.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

/**
 * @struct SingleSlice
 * @brief Represents a single slice of audio data
 */
struct SingleSlice
{
    /**
     * @brief Audio data
     */
    DragonianLibSTL::Vector<float> Audio;

    /**
     * @brief F0 data
     */
    DragonianLibSTL::Vector<float> F0;

    /**
     * @brief Volume data
     */
    DragonianLibSTL::Vector<float> Volume;

    /**
     * @brief Speaker data
     */
    DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> Speaker;

    /**
     * @brief Original length of the audio
     */
    size_t OrgLen = 0;

    /**
     * @brief Indicates if the slice is not mute
     */
    bool IsNotMute = false;

    /**
     * @brief Sampling rate of the audio
     */
    size_t SamplingRate = 44100;
};

/**
 * @struct SingleAudio
 * @brief Represents a single audio file
 */
struct SingleAudio
{
    /**
     * @brief Slices of the audio
     */
    DragonianLibSTL::Vector<SingleSlice> Slices;

    /**
     * @brief Path to the audio file
     */
    std::wstring Path;
};

/**
 * @struct InferenceParams
 * @brief Parameters for inference
 */
struct InferenceParams
{
    /**
     * @brief Noise scale factor (0-10)
     */
    float NoiseScale = 0.3f;

    /**
     * @brief Seed for random number generation
     */
    int64_t Seed = 52468;

    /**
     * @brief Speaker ID
     */
    int64_t SpeakerId = 0;

    /**
     * @brief Number of speakers in the model
     */
    int64_t SpkCount = 2;

    /**
     * @brief Index rate (0-1)
     */
    float IndexRate = 0.f;

    /**
     * @brief Cluster rate (0-1)
     */
    float ClusterRate = 0.f;

    /**
     * @brief DDSP noise scale factor (0-10)
     */
    float DDSPNoiseScale = 0.8f;

    /**
     * @brief Key shift (-64 to 64)
     */
    float Keys = 0.f;

    /**
     * @brief Mean filter window length (1-20)
     */
    size_t MeanWindowLength = 2;

    /**
     * @brief Diffusion acceleration factor (1-200)
     */
    size_t Pndm = 1;

    /**
     * @brief Total number of diffusion steps (1-1000)
     */
    size_t Step = 100;

    /**
     * @brief Start time for reflow
     */
    float TBegin = 0.f;

    /**
     * @brief End time for reflow
     */
    float TEnd = 1.f;

    /**
     * @brief Diffusion sampler
     */
    std::wstring Sampler = L"Pndm";

    /**
     * @brief Reflow sampler
     */
    std::wstring ReflowSampler = L"Eular";

    /**
     * @brief F0 extraction method
     */
    std::wstring F0Method = L"Dio";

    /**
     * @brief Shared pointer to the vocoder model session
     */
    std::shared_ptr<Ort::Session> VocoderModel = nullptr;

    /**
     * @brief Hop size for the vocoder
     */
    int VocoderHopSize = 512;

    /**
     * @brief Number of mel bins for the vocoder
     */
    int VocoderMelBins = 128;

    /**
     * @brief Sampling rate for the vocoder
     */
    int VocoderSamplingRate = 44100;

	/**
	 * @brief Bins of F0
	 */
	long F0Bins = 256;

    /**
	 * @brief Max F0 value
     */
    double F0Max = 1100.0;

    /**
	 * @brief Min F0 value
     */
    double F0Min = 50.0;

    /**
	 * @brief User parameter for F0 extractor
     */
    void* F0ExtractorUserParameter = nullptr;

#ifndef DRAGONIANLIB_IMPORT
    /**
     * @brief Cached 16K audio data
     */
    mutable DragonianLibSTL::Vector<float> _16KAudio;

    /**
     * @brief Cached F0 data
     */
    mutable DragonianLibSTL::Vector<float> _F0;

    /**
     * @brief Cached volume data
     */
    mutable DragonianLibSTL::Vector<float> _Volume;

    /**
     * @brief Cached speaker data
     */
    mutable DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _Speaker;

    /**
     * @brief Caches the data for inference
     * @param M16KAudio 16K audio data
     * @param MF0 F0 data
     * @param MVolume Volume data
     * @param MSpeaker Speaker data
     */
    void CacheData(
        DragonianLibSTL::Vector<float>&& M16KAudio,
        DragonianLibSTL::Vector<float>&& MF0,
        DragonianLibSTL::Vector<float>&& MVolume,
        DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>&& MSpeaker
    ) const
    {
        _16KAudio = std::move(M16KAudio);
        _F0 = std::move(MF0);
        _Volume = std::move(MVolume);
        _Speaker = std::move(MSpeaker);
    }
#endif
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
