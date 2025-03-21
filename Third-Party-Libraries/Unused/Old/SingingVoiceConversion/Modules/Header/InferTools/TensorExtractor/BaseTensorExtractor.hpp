/**
 * FileName: MoeVoiceStudioTensorExtractor.hpp
 * Note: MoeVoiceStudioCore Tensor Extractor
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
#include "onnxruntime_cxx_api.h"

#define _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_Header _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header namespace TensorExtractor {
#define _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_End } _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_Header

/**
 * @class LibSvcTensorExtractor
 * @brief Tensor Preprocessor
 */
class LibSvcTensorExtractor
{
public:
    /**
     * @struct Tensors
     * @brief Structure to hold various tensor data
     */
    struct Tensors
    {
        DragonianLibSTL::Vector<float> HiddenUnit; ///< Hidden unit tensor
        DragonianLibSTL::Vector<float> F0; ///< F0 tensor
        DragonianLibSTL::Vector<float> Volume; ///< Volume tensor
        DragonianLibSTL::Vector<float> SpkMap; ///< Speaker map tensor
        DragonianLibSTL::Vector<float> DDSPNoise; ///< DDSP noise tensor
        DragonianLibSTL::Vector<float> Noise; ///< Noise tensor
        DragonianLibSTL::Vector<int64_t> Alignment; ///< Alignment tensor
        DragonianLibSTL::Vector<float> UnVoice; ///< Unvoiced tensor
        DragonianLibSTL::Vector<int64_t> NSFF0; ///< NSFF0 tensor
        int64_t Length[1] = { 0 }; ///< Length tensor
        int64_t Speaker[1] = { 0 }; ///< Speaker tensor

        DragonianLibSTL::Vector<int64_t> HiddenUnitShape; ///< Shape of hidden unit tensor
        DragonianLibSTL::Vector<int64_t> FrameShape; ///< Shape of frame tensor
        DragonianLibSTL::Vector<int64_t> SpkShape; ///< Shape of speaker tensor
        DragonianLibSTL::Vector<int64_t> DDSPNoiseShape; ///< Shape of DDSP noise tensor
        DragonianLibSTL::Vector<int64_t> NoiseShape; ///< Shape of noise tensor
        int64_t OneShape[1] = { 1 }; ///< Shape of single element tensor
    };

    /**
     * @struct InferParams
     * @brief Structure to hold inference parameters
     */
    struct InferParams
    {
        float NoiseScale = 0.3f; ///< Noise scale factor
        float DDSPNoiseScale = 1.0f; ///< DDSP noise scale factor
        int Seed = 520468; ///< Random seed
        size_t AudioSize = 0; ///< Size of audio data
        int64_t Chara = 0; ///< Character ID
        float upKeys = 0.f; ///< Key up value
        void* Other = nullptr; ///< Other parameters
        size_t SrcSamplingRate = 32000; ///< Source sampling rate
        size_t Padding = size_t(-1); ///< Padding size
    };

    /**
     * @struct Others
     * @brief Structure to hold other parameters
     */
    struct Others
    {
        int f0_bin = 256; ///< F0 bin size
        float f0_max = 1100.0; ///< Maximum F0 value
        float f0_min = 50.0; ///< Minimum F0 value
        OrtMemoryInfo* Memory = nullptr; ///< Memory information
        void* Other = nullptr; ///< Other parameters
    };

    using Params = const InferParams&; ///< Alias for inference parameters

    /**
     * @struct Inputs
     * @brief Structure to hold input data
     */
    struct Inputs
    {
        Tensors Data; ///< Tensor data
        std::vector<Ort::Value> Tensor; ///< ONNX runtime values
        const char* const* InputNames = nullptr; ///< Input names
        const char* const* OutputNames = nullptr; ///< Output names
        size_t InputCount = 1; ///< Number of inputs
        size_t OutputCount = 1; ///< Number of outputs
    };

    /**
     * @brief Constructor for LibSvcTensorExtractor
     * @param _srcsr Source sampling rate
     * @param _sr Sampling rate
     * @param _hop Hop size
     * @param _smix Speaker mix flag
     * @param _volume Volume flag
     * @param _hidden_size Hidden size
     * @param _nspeaker Number of speakers
     * @param _other Other parameters
     */
    LibSvcTensorExtractor(
        uint64_t _srcsr,
        uint64_t _sr,
        uint64_t _hop,
        bool _smix,
        bool _volume,
        uint64_t _hidden_size,
        uint64_t _nspeaker,
        const Others& _other
    );

    /**
     * @brief Destructor for LibSvcTensorExtractor
     */
    virtual ~LibSvcTensorExtractor() = default;

    // Delete copy and move constructors and assignment operators
    LibSvcTensorExtractor(const LibSvcTensorExtractor&) = delete;
    LibSvcTensorExtractor(LibSvcTensorExtractor&&) = delete;
    LibSvcTensorExtractor operator=(const LibSvcTensorExtractor&) = delete;
    LibSvcTensorExtractor operator=(LibSvcTensorExtractor&&) = delete;

    /**
     * @brief Extract tensor inputs
     * @param HiddenUnit Hidden unit tensor
     * @param F0 F0 tensor
     * @param Volume Volume tensor
     * @param SpkMap Speaker map tensor
     * @param params Inference parameters
     * @return Extracted inputs
     */
    virtual Inputs Extract(
        const DragonianLibSTL::Vector<float>& HiddenUnit,
        const DragonianLibSTL::Vector<float>& F0,
        const DragonianLibSTL::Vector<float>& Volume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
        Params params
    );

    /**
     * @brief Get NSFF0 tensor
     * @param F0 Input tensor
     * @return NSFF0 tensor
     */
    [[nodiscard]] DragonianLibSTL::Vector<int64_t> GetNSFF0(
        const DragonianLibSTL::Vector<float>& F0
    ) const;

    /**
     * @brief Get interpolated F0 tensor
     * @param F0 Input tensor
     * @return Interpolated F0 tensor
     */
    static DragonianLibSTL::Vector<float> GetInterpedF0(
        const DragonianLibSTL::Vector<float>& F0
    );

    /**
     * @brief Interpolate UV F0 tensor
     * @param F0 Input tensor
     * @param PaddedIndex Padded index
     * @return Interpolated UV F0 tensor
     */
    static DragonianLibSTL::Vector<float> InterpUVF0(
        const DragonianLibSTL::Vector<float>& F0,
        size_t PaddedIndex = size_t(-1)
    );

    /**
     * @brief Get UV tensor
     * @param F0 Input tensor
     * @return UV tensor
     */
    static DragonianLibSTL::Vector<float> GetUV(
        const DragonianLibSTL::Vector<float>& F0
    );

    /**
     * @brief Get alignment tensor
	 * @param specLen Length of spectrogram
	 * @param hubertLen Length of Hubert
     * @return Alignment tensor
     */
    static DragonianLibSTL::Vector<int64_t> GetAligments(
        size_t specLen,
        size_t hubertLen
    );

    /**
     * @brief Perform linear combination on tensor data
     * @tparam T Data type
     * @param _data Tensor data
     * @param default_id Default ID
     * @param Value Value for combination
     */
    template <typename T>
    static void LinearCombination(
        DragonianLibSTL::Vector<DragonianLibSTL::Vector<T>>& _data,
        size_t default_id,
        T Value = T(1.0)
    )
    {
        if (_data.Empty())
            return;
        if (default_id > _data.Size())
            default_id = 0;

        for (size_t i = 0; i < _data[0].Size(); ++i)
        {
            T Sum = T(0.0);
            for (size_t j = 0; j < _data.Size(); ++j)
                Sum += _data[j][i];
            if (Sum < T(0.0001))
            {
                for (size_t j = 0; j < _data.Size(); ++j)
                    _data[j][i] = T(0);
                _data[default_id][i] = T(1);
                continue;
            }
            Sum *= T(Value);
            for (size_t j = 0; j < _data.Size(); ++j)
                _data[j][i] /= Sum;
        }
    }

    /**
     * @brief Get interpolated F0 log tensor
     * @param rF0 Input tensor
	 * @param enable_log enable log 
     * @return Interpolated F0 log tensor
     */
    [[nodiscard]] static DragonianLibSTL::Vector<float> GetInterpedF0log(
        const DragonianLibSTL::Vector<float>& rF0,
        bool enable_log
    );

    /**
     * @brief Get current speaker mix data
     * @param _input Input tensor data
     * @param dst_len Destination length
     * @param curspk Current speaker ID
     * @return Current speaker mix data
     */
    [[nodiscard]] DragonianLibSTL::Vector<float> GetCurrectSpkMixData(
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
        size_t dst_len,
        int64_t curspk
    ) const;

    /**
     * @brief Get speaker mix data
     * @param _input Input tensor data
     * @param dst_len Destination length
     * @param spk_count Speaker count
     * @return Speaker mix data
     */
    [[nodiscard]] static DragonianLibSTL::Vector<float> GetSpkMixData(
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
        size_t dst_len,
        size_t spk_count
    );

protected:
    uint64_t _NSpeaker = 1; ///< Number of speakers
    uint64_t _SamplingRate = 32000; ///< Sampling rate
    uint64_t _HopSize = 512; ///< Hop size
    bool _SpeakerMix = false; ///< Speaker mix flag
    bool _Volume = false; ///< Volume flag
    uint64_t _HiddenSize = 256; ///< Hidden size
    int f0_bin = 256; ///< F0 bin size
    float f0_max = 1100.0; ///< Maximum F0 value
    float f0_min = 50.0; ///< Minimum F0 value
    float f0_mel_min = 1127.f * log(1.f + f0_min / 700.f); ///< Minimum F0 mel value
    float f0_mel_max = 1127.f * log(1.f + f0_max / 700.f); ///< Maximum F0 mel value
    OrtMemoryInfo* Memory = nullptr; ///< Memory information
};


_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_End