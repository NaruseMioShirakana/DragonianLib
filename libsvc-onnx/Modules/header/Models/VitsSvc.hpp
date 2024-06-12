/**
 * FileName: VitsSvc.hpp
 * Note: MoeVoiceStudioCore Onnx Vits系Svc 模型定义
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
#include "SVC.hpp"
#include "DiffSvc.hpp"

LibSvcHeader

class VitsSvc : public SingingVoiceConversion
{
public:

    VitsSvc(const Hparams& _Hps, const ProgressCallback& _ProgressCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0, unsigned ThreadCount_ = 0);

	~VitsSvc() override;

    void Destory();

    [[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(
        const SingleSlice& _Slice,
        const InferenceParams& _Params,
        size_t& _Process
    ) const override;

    [[nodiscard]] DragonianLibSTL::Vector<int16_t> InferPCMData(
        const DragonianLibSTL::Vector<int16_t>& _PCMData,
        long _SrcSamplingRate,
        const InferenceParams& _Params
    ) const override;

    [[nodiscard]] std::vector<Ort::Value> MelExtractor(const float* PCMAudioBegin, const float* PCMAudioEnd) const = delete;

    [[nodiscard]] Ort::MemoryInfo* GetMemoryInfo() const
    {
        return memory_info;
    }

private:
    Ort::Session* VitsSvcModel = nullptr;
    std::wstring VitsSvcVersion = L"SoVits4.0";

    const std::vector<const char*> soVitsOutput = { "audio" };
    const std::vector<const char*> soVitsInput = { "hidden_unit", "lengths", "pitch", "sid" };
    const std::vector<const char*> RVCInput = { "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd" };
    const std::vector<const char*> StftOutput = { "mel" };
    const std::vector<const char*> StftInput = { "waveform", "aligment"};
};

LibSvcEnd
