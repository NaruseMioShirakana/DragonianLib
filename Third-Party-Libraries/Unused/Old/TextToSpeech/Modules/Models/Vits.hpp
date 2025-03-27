/**
 * FileName: Vits.hpp
 * Note: MoeVoiceStudioCore Vits模型类
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

class EmotionLoader
{
public:
    static constexpr long startPos = 128;
    _D_Dragonian_Lib_Force_Inline EmotionLoader() = default;
    _D_Dragonian_Lib_Force_Inline EmotionLoader(const std::wstring& EmotionFilePath)
    {
        if (EmotionFilePath.empty())
            return;
        if (!std::filesystem::exists(EmotionFilePath))
            _D_Dragonian_Lib_Throw_Exception("Emotion File Not Found");
        FileGuard File(EmotionFilePath, L"rb");
        fseek(File, 0, SEEK_END);
        const auto EmotionSize = ftell(File) / sizeof(float);
        fseek(File, startPos, SEEK_SET);
        EmotionVectors.resize(EmotionSize);
        fread(EmotionVectors.data(), sizeof(float), EmotionSize, File);
    }
    _D_Dragonian_Lib_Force_Inline DragonianLibSTL::Vector<float> operator[](size_t Index) const
    {
        if (Index < EmotionVectors.size())
            return { EmotionVectors.data() + Index * 1024, EmotionVectors.data() + (Index + 1) * 1024 };
        _D_Dragonian_Lib_Throw_Exception("Emotion Index Out Of Range");
    }
private:
    std::vector<float> EmotionVectors;
};

class Vits : public TextToSpeech
{
public:
    Vits(
        const ModelHParams& _Config,
        const ProgressCallback& _ProgressCallback,
        const DurationCallback& _DurationCallback,
        ExecutionProviders ExecutionProvider_ = ExecutionProviders::CPU,
        unsigned DeviceID_ = 0,
        unsigned ThreadCount_ = 0
    );

	~Vits() override = default;

	Vits(const Vits&) = default;
	Vits& operator=(const Vits&) = default;
	Vits(Vits&&) noexcept = default;
	Vits& operator=(Vits&&) noexcept = default;

    DragonianLibSTL::Vector<float> Inference(TTSInputData& InputData, const TTSParams& Params) const override;

protected:

    std::string VitsType;
    int64_t DefBertSize = 1024;
    int64_t VQCodeBookSize = 10;
    int64_t BertCount = 3;

    bool UseTone = false;
    bool UseBert = false;
    bool UseLength = false;
    bool UseLanguage = false;
    bool EncoderG = false;
    bool ReferenceBert = false;
    bool UseVQ = false;
    bool UseClap = false;

    std::unordered_map<std::wstring, size_t> Emotion2Id;
    bool Emotion = false;
    EmotionLoader EmotionVector;

    std::shared_ptr<Ort::Session> sessionEnc_p = nullptr;
    std::shared_ptr<Ort::Session> sessionEmb = nullptr;
    std::shared_ptr<Ort::Session> sessionSdp = nullptr;
    std::shared_ptr<Ort::Session> sessionDp = nullptr;
    std::shared_ptr<Ort::Session> sessionFlow = nullptr;
    std::shared_ptr<Ort::Session> sessionDec = nullptr;

    std::vector<const char*> EncoderInputNames = { "x" };
    static inline const std::vector<const char*> EncoderOutputNames = { "xout", "m_p", "logs_p", "x_mask" };

    std::vector<const char*> SdpInputNames = { "x", "x_mask", "zin" };
    static inline const std::vector<const char*> SdpOutputNames = { "logw" };

    std::vector<const char*> DpInputNames = { "x", "x_mask" };
    static inline const std::vector<const char*> DpOutputNames = { "logw" };

    std::vector<const char*> FlowInputNames = { "z_p", "y_mask" };
    static inline const std::vector<const char*> FlowOutputNames = { "z" };

    std::vector<const char*> DecInputNames = { "z_in" };
    static inline const std::vector<const char*> DecOutputNames = { "o" };

    static inline const std::vector<const char*> EmbiddingInputNames = { "sid" };
    static inline const std::vector<const char*> EmbiddingOutputNames = { "g" };

    static inline const std::vector<const char*> VistBertInputNames = 
    { "bert_0", "bert_1", "bert_2", "bert_3", "bert_4", "bert_5", "bert_6", "bert_7", "bert_8", "bert_9", "bert_10", "bert_11", "bert_12", "bert_13", "bert_14", "bert_15", "bert_16", "bert_17", "bert_18", "bert_19", "bert_20", "bert_21", "bert_22", "bert_23", "bert_24", "bert_25", "bert_26", "bert_27", "bert_28", "bert_29", "bert_30", "bert_31", "bert_32", "bert_33", "bert_34", "bert_35", "bert_36", "bert_37", "bert_38", "bert_39", "bert_40", "bert_41", "bert_42", "bert_43", "bert_44", "bert_45", "bert_46", "bert_47", "bert_48", "bert_49", "bert_50", "bert_51", "bert_52", "bert_53", "bert_54", "bert_55", "bert_56", "bert_57", "bert_58", "bert_59", "bert_60", "bert_61", "bert_62", "bert_63", "bert_64", "bert_65", "bert_66", "bert_67", "bert_68", "bert_69", "bert_70", "bert_71", "bert_72", "bert_73", "bert_74", "bert_75", "bert_76", "bert_77", "bert_78", "bert_79", "bert_80", "bert_81", "bert_82", "bert_83", "bert_84", "bert_85", "bert_86", "bert_87", "bert_88", "bert_89", "bert_90", "bert_91", "bert_92", "bert_93", "bert_94", "bert_95", "bert_96", "bert_97", "bert_98", "bert_99" };

private:
    DragonianLibSTL::Vector<float> GetEmotionVector(
        const DragonianLibSTL::Vector<std::wstring>& EmotionSymbol
    ) const;

    DragonianLibSTL::Vector<float> GetSpeakerEmbedding(
        const std::map<int64_t, float>& SpeakerMixIds
    ) const;
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End