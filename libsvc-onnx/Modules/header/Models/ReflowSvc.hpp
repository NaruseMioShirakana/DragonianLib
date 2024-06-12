/**
 * FileName: ReflowSvc.hpp
 * Note: MoeVoiceStudioCore Onnx Reflow系Svc 模型定义
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
#include <map>
#include "SVC.hpp"

LibSvcHeader

/**
 * \brief Reflow模型
 */
class ReflowSvc : public SingingVoiceConversion
{
public:
    ReflowSvc(const Hparams& _Hps, const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0, unsigned ThreadCount_ = 0);

	~ReflowSvc() override;

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

    [[nodiscard]] const std::wstring& GetReflowSvcVer() const
    {
        return ReflowSvcVersion;
    }

    [[nodiscard]] int64_t GetMelBins() const
    {
        return melBins;
    }

    void NormMel(DragonianLibSTL::Vector<float>& MelSpec) const;

private:
    Ort::Session* encoder = nullptr;
    Ort::Session* velocity = nullptr;
    Ort::Session* after = nullptr;

    int64_t melBins = 128;
    int64_t MaxStep = 100;
    float SpecMin = -12;
    float SpecMax = 2;
    float Scale = 1000.f;
    bool VaeMode = true;

    std::wstring ReflowSvcVersion = L"DiffusionSvc";

    const std::vector<const char*> nsfInput = { "c", "f0" };
    const std::vector<const char*> nsfOutput = { "audio" };
    const std::vector<const char*> afterInput = { "x" };
    const std::vector<const char*> afterOutput = { "mel_out" };
    const std::vector<const char*> OutputNamesEncoder = { "x", "cond", "f0_pred" };
};

LibSvcEnd