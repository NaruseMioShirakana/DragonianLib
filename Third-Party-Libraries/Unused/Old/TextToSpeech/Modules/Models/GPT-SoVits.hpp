/**
 * FileName: GPT-SoVits.hpp
 * Note: MoeVoiceStudioCore GPT-SoVits模型类
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
 * date: 2023-11-9 Create
*/

#pragma once
#include "TTS.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header
class GptSoVits : public TextToSpeech
{
public:
    GptSoVits
	(
        const ModelHParams& _Config,
        const ProgressCallback& _ProgressCallback,
        const DurationCallback& _DurationCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

    ~GptSoVits() override = default;

    GptSoVits(const GptSoVits&) = default;
    GptSoVits& operator=(const GptSoVits&) = default;
    GptSoVits(GptSoVits&&) noexcept = default;
    GptSoVits& operator=(GptSoVits&&) noexcept = default;

    DragonianLibSTL::Vector<float> Inference(TTSInputData& InputData, const TTSParams& Params) const override;

private:
    std::shared_ptr<Ort::Session> sessionVits = nullptr;
    std::shared_ptr<Ort::Session> sessionSSL = nullptr;

    std::shared_ptr<Ort::Session> sessionEncoder = nullptr;
    std::shared_ptr<Ort::Session> sessionFDecoder = nullptr;
    std::shared_ptr<Ort::Session> sessionDecoder = nullptr;

    int64_t NumLayers = 24;
    int64_t EmbeddingDim = 512;
    int64_t EOSId = 1024;

    std::vector<const char*> VitsInputNames = { "text_seq", "pred_semantic", "ref_audio" };
    static inline const std::vector<const char*> VitsOutputNames = { "audio" };

    std::vector<const char*> EncoderInputNames = { "ref_seq", "text_seq", "ref_bert", "text_bert", "ssl_content" };
    static inline const std::vector<const char*> EncoderOutputNames = { "x", "prompts" };
    std::vector<const char*> DecoderInputNames = { "iy", "ik", "iv", "iy_emb", "ix_example" };
    static inline const std::vector<const char*> DecoderOutputNames = { "y", "k", "v", "y_emb", "logits", "samples" };
    std::vector<const char*> FDecoderInputNames = { "x", "prompts" };
    static inline const std::vector<const char*> FDecoderOutputNames = { "y", "k", "v", "y_emb", "x_example" };

    std::vector<const char*> SSLInputNames = { "audio" };
    static inline const std::vector<const char*> SSLOutputNames = { "last_hidden_state" };
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End