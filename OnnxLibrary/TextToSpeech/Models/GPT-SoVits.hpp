#pragma once

/*#include "TTS.hpp"

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

_D_Dragonian_Lib_Lib_Text_To_Speech_End*/