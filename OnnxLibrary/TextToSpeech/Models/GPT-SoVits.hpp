/**
 * @file GPT-SoVits.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief GPT-SoVits
 * @changes
 *  > 2025/3/28 NaruseMioShirakana Created <
 */
#pragma once
#include "OnnxLibrary/TextToSpeech/Util/Text2Speech.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

namespace GptSoVits
{
	class T2SAR
	{
	public:
		T2SAR() = delete;
        T2SAR(
            const OnnxRuntimeEnvironment& _Environment,
			const HParams& _Config,
			const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
        );
		~T2SAR() = default;

		T2SAR& operator=(T2SAR&&) noexcept = default;
		T2SAR& operator=(const T2SAR&) = default;
		T2SAR(const T2SAR&) = default;
		T2SAR(T2SAR&&) noexcept = default;

        Tensor<Int64, 2, Device::CPU> Forward(
            const Tensor<Int64, 2, Device::CPU>& _PhonemeIds,
			const Tensor<Int64, 2, Device::CPU>& _RefPhonemeIds,
            const Tensor<Float32, 3, Device::CPU>& _BertFeature,
            Float32 _TopP = 0.6f,
			Float32 _Temperature = 0.6f,
			Float32 _RepetitionPenalty = 1.35f
        );

	private:
        OnnxModelBase<void> _MyPromptModel;
		OnnxModelBase<void> _MyDecodeModel;
		Int64 _MyEOSId = 1024;
	};

    class CfmModel : public OnnxModelBase<CfmModel>
    {
    public:
        CfmModel() = delete;
        CfmModel(
            const OnnxRuntimeEnvironment& _Environment,
            const std::wstring& _ModelPath,
            const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
        );
		~CfmModel() = default;
        CfmModel(const CfmModel&) = default;
        CfmModel& operator=(const CfmModel&) = default;
        CfmModel(CfmModel&&) noexcept = default;
        CfmModel& operator=(CfmModel&&) noexcept = default;

		Tensor<Float32, 3, Device::CPU> Forward(
			const Tensor<Float32, 3, Device::CPU>& _Feature,
            const Tensor<Float32, 3, Device::CPU>& _Mel,
			Int64 _SampleSteps = 8,
			Float32 _Temperature = 0.6f,
			Float32 _CfgRate = 0.001f
		);
    private:
        Int64 _MyInChannels = 100;
    };

    class VQModel : public OnnxModelBase<VQModel>
    {
    public:
        VQModel() = delete;
        VQModel(
			const OnnxRuntimeEnvironment& _Environment,
			const HParams& _Config,
			const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_Lib_Text_To_Speech_Space GetDefaultLogger()
		);
        VQModel(const VQModel&) = default;
        VQModel& operator=(const VQModel&) = default;
        VQModel(VQModel&&) noexcept = default;
        VQModel& operator=(VQModel&&) noexcept = default;
		~VQModel() = default;

        Tensor<Float32, 3, Device::CPU> Forward(
            const Tensor<Int64, 2, Device::CPU>& _Phonemes,
            const Tensor<Int64, 2, Device::CPU>& _PredSemantic,
            const Tensor<Float32, 2, Device::CPU>& _RefAudio,
            Int64 _RefSamplingRate,
			const std::optional<Tensor<Int64, 2, Device::CPU>>& _RefPhonemes = std::nullopt,
            const std::optional<Tensor<Int64, 2, Device::CPU>>& _RefPrompts = std::nullopt,
            Int64 _SampleSteps = 8,
            Float32 _CfgRate = 0.001f
        );

        Tensor<Int64, 3, Device::CPU> ExtractLatent(
            const Tensor<Float32, 3, Device::CPU>& _RefSSlContext
        );

    private:
        OnnxModelBase<void> _MyExtract;
		std::optional<CfmModel> _MyCfmModel = std::nullopt;
		Int64 _MySamplingRate = 32000;
		bool _IsV3 = false;
    };
}

_D_Dragonian_Lib_Lib_Text_To_Speech_End

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