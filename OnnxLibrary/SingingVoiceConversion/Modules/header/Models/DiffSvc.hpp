/**
 * FileName: DiffSvc.hpp
 * Note: MoeVoiceStudioCore Onnx Diffusion系Svc 模型定义
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
 * \brief DiffSvc模型
 */
class DiffusionSvc : public SingingVoiceConversion
{
public:

    DiffusionSvc(
        const Hparams& _Hps,
        const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

	~DiffusionSvc() override;

    [[nodiscard]] DragonianLibSTL::Vector<float> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params,
        size_t& _Process
    ) const override;

    [[nodiscard]] DragonianLibSTL::Vector<float> InferPCMData(
        const DragonianLibSTL::Vector<float>& _PCMData,
        long _SrcSamplingRate,
        const InferenceParams& _Params
    ) const override;

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

    [[nodiscard]] int64_t GetMaxStep() const override
    {
        return MaxStep;
    }

    [[nodiscard]] bool OldVersion() const
    {
        return OldDiffSvc.get();
    }

    [[nodiscard]] const std::wstring& GetUnionSvcVer() const override
    {
        return DiffSvcVersion;
    }

    [[nodiscard]] int64_t GetMelBins() const override
    {
        return MelBins;
    }

    void NormMel(
        DragonianLibSTL::Vector<float>& MelSpec
    ) const override;

private:
    std::shared_ptr<Ort::Session> PreEncoder = nullptr;
    std::shared_ptr<Ort::Session> DiffusionDenoiser = nullptr;
    std::shared_ptr<Ort::Session> NoisePredictor = nullptr;
    std::shared_ptr<Ort::Session> PostDecoder = nullptr;
    std::shared_ptr<Ort::Session> AlphaCumprod = nullptr;
    std::shared_ptr<Ort::Session> NaiveModel = nullptr;

    std::shared_ptr<Ort::Session> OldDiffSvc = nullptr;

    int64_t MelBins = 128;
    int64_t Pndms = 100;
    int64_t MaxStep = 1000;
    float SpecMin = -12;
    float SpecMax = 2;

    std::wstring DiffSvcVersion = L"DiffSvc";

    static inline const std::vector<const char*> nsfInput = { "c", "f0" };
    static inline const std::vector<const char*> nsfOutput = { "audio" };
    static inline const std::vector<const char*> DiffInput = { "hubert", "mel2ph", "spk_embed", "f0", "initial_noise", "speedup" };
    static inline const std::vector<const char*> DiffOutput = { "mel_pred", "f0_pred" };
    static inline const std::vector<const char*> afterInput = { "x" };
    static inline const std::vector<const char*> afterOutput = { "mel_out" };
    static inline const std::vector<const char*> naiveOutput = { "mel" };

public:
    DiffusionSvc(const DiffusionSvc&) = default;
    DiffusionSvc(DiffusionSvc&&) = default;
    DiffusionSvc& operator=(const DiffusionSvc&) = default;
    DiffusionSvc& operator=(DiffusionSvc&&) = default;
};

LibSvcEnd