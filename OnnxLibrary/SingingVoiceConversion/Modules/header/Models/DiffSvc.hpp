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

    DiffusionSvc(const Hparams& _Hps, const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0, unsigned ThreadCount_ = 0);
	~DiffusionSvc() override;
    DiffusionSvc(const DiffusionSvc&) = delete;
    DiffusionSvc(DiffusionSvc&&) = delete;
    DiffusionSvc& operator=(const DiffusionSvc&) = delete;
    DiffusionSvc& operator=(DiffusionSvc&&) = delete;

    void Destory();

    [[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(const SingleSlice& _Slice, const InferenceParams& _Params, size_t& _Process) const override;

    [[nodiscard]] DragonianLibSTL::Vector<int16_t> InferPCMData(const DragonianLibSTL::Vector<int16_t>& _PCMData, long _SrcSamplingRate, const InferenceParams& _Params) const override;

    [[nodiscard]] DragonianLibSTL::Vector<int16_t> ShallowDiffusionInference(
        DragonianLibSTL::Vector<float>& _16KAudioHubert,
        const InferenceParams& _Params,
        std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
        const DragonianLibSTL::Vector<float>& _SrcF0,
        const DragonianLibSTL::Vector<float>& _SrcVolume,
        const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
        size_t& Process,
        int64_t SrcSize
    ) const;

    [[nodiscard]] int64_t GetMaxStep() const
    {
        return MaxStep;
    }

    [[nodiscard]] bool OldVersion() const
    {
        return diffSvc;
    }

    [[nodiscard]] const std::wstring& GetDiffSvcVer() const
    {
        return DiffSvcVersion;
    }

    [[nodiscard]] int64_t GetMelBins() const
    {
        return melBins;
    }

    void NormMel(DragonianLibSTL::Vector<float>& MelSpec) const;

private:
    Ort::Session* encoder = nullptr;
    Ort::Session* denoise = nullptr;
    Ort::Session* pred = nullptr;
    Ort::Session* after = nullptr;
    Ort::Session* alpha = nullptr;
    Ort::Session* naive = nullptr;

    Ort::Session* diffSvc = nullptr;

    int64_t melBins = 128;
    int64_t Pndms = 100;
    int64_t MaxStep = 1000;
    float SpecMin = -12;
    float SpecMax = 2;

    std::wstring DiffSvcVersion = L"DiffSvc";

    const std::vector<const char*> nsfInput = { "c", "f0" };
    const std::vector<const char*> nsfOutput = { "audio" };
    const std::vector<const char*> DiffInput = { "hubert", "mel2ph", "spk_embed", "f0", "initial_noise", "speedup" };
    const std::vector<const char*> DiffOutput = { "mel_pred", "f0_pred" };
    const std::vector<const char*> afterInput = { "x" };
    const std::vector<const char*> afterOutput = { "mel_out" };
    const std::vector<const char*> naiveOutput = { "mel" };
};

DragonianLibSTL::Vector<int16_t> VocoderInfer(DragonianLibSTL::Vector<float>& Mel, DragonianLibSTL::Vector<float>& F0, int64_t MelBins, int64_t MelSize, const Ort::MemoryInfo* Mem, void* _VocoderModel = nullptr);

LibSvcEnd