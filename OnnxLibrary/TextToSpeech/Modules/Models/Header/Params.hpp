/**
 * FileName: MoeVSProject.hpp
 * Note: MoeVoiceStudioCore 项目相关的定义
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
#include "ModelBase.hpp"
#include "Libraries/MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

struct VitsConfigs
{
    std::string VitsType;
    std::wstring EncoderPath;
    std::wstring SpeakerEmbedding;
    std::wstring SDurationPredictorPath;
    std::wstring DurationPredictorPath;
    std::wstring FlowPath;
    std::wstring DecoderPath;
};

struct GptSoVitsConfigs
{
    std::wstring VitsPath;
    std::wstring SSLPath;
    std::wstring EncoderPath;
    std::wstring DecoderPath;
    std::wstring FDecoderPath;
};

struct ModelHParams
{
	VitsConfigs VitsConfig;
    GptSoVitsConfigs GptSoVitsConfig;
	long SamplingRate = 22050;
    int64_t DefBertSize = 1024;
    int64_t VQCodeBookSize = 10;
    int64_t BertCount = 3;
    int64_t NumLayers = 24;
    int64_t EmbeddingDim = 512;
    int64_t EOSId = 1024;
    bool AddBlank = true;
    bool UseTone = false;
    bool UseBert = false;
    bool UseLength = true;
    bool UseLanguageIds = false;
    bool EncoderG = false;
    bool ReferenceBert = false;
    bool UseVQ = false;
    bool UseClap = false;
    std::wstring EmotionFilePath;
    
    std::unordered_map<std::wstring, size_t> Emotion2Id;
    std::unordered_map<std::wstring, int64_t> SpeakerName2ID;
    std::unordered_map<std::wstring, int64_t> Symbols;
    std::unordered_map<std::wstring, int64_t> Language2ID;

    std::wstring PadSymbol = L"_";
};

struct TTSInputData
{
    friend LibTTSModule;
    DragonianLibSTL::Vector<float> _ReferenceAudio16KSr, _ReferenceAudioSrc;

    int64_t _BertDims = 1024;
	DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _BertVec; // If GptSoVits, Index[0] is Reference, Index[1] is Target
    DragonianLibSTL::Vector<int64_t> _Token2Phoneme;

    DragonianLibSTL::Vector<float> _ClapVec;

    DragonianLibSTL::Vector<int64_t> _PhonemesIds, _RefPhonemesIds;

    DragonianLibSTL::Vector<int64_t> _Tones;

    DragonianLibSTL::Vector<int64_t> _LanguageIds;

    DragonianLibSTL::Vector<int64_t> _Durations;

    DragonianLibSTL::Vector<float> _Emotion;

    std::map<int64_t, float> _SpeakerMixIds;

protected:
    DragonianLibSTL::Vector<std::wstring> _Phonemes;
    DragonianLibSTL::Vector<std::wstring> _RefPhonemes;
    DragonianLibSTL::Vector<std::wstring> _LanguageSymbols;
    DragonianLibSTL::Vector<std::pair<std::wstring, float>> _SpeakerMixSymbol;

public:
    _D_Dragonian_Lib_Force_Inline void SetPhonemes(DragonianLibSTL::Vector<std::wstring> Phonemes)
    {
        _Phonemes = std::move(Phonemes);
		_PhonemesIds.Clear();
    }
	_D_Dragonian_Lib_Force_Inline void SetRefPhonemes(DragonianLibSTL::Vector<std::wstring> RefPhonemes)
	{
		_RefPhonemes = std::move(RefPhonemes);
        _RefPhonemesIds.Clear();
	}
    _D_Dragonian_Lib_Force_Inline void SetLanguageSymbols(DragonianLibSTL::Vector<std::wstring> LanguageSymbols)
    {
        _LanguageSymbols = std::move(LanguageSymbols);
		_LanguageIds.Clear();
    }
    _D_Dragonian_Lib_Force_Inline void SetSpeakerMixSymbol(DragonianLibSTL::Vector<std::pair<std::wstring, float>> SpeakerMixSymbol)
    {
        _SpeakerMixSymbol = std::move(SpeakerMixSymbol);
		_SpeakerMixIds.clear();
    }
    _D_Dragonian_Lib_Force_Inline DragonianLibSTL::Vector<std::wstring>& GetPhonemes()
    {
        return _Phonemes;
    }
	_D_Dragonian_Lib_Force_Inline DragonianLibSTL::Vector<std::wstring>& GetRefPhonemes()
    {
		return _RefPhonemes;
    }
    _D_Dragonian_Lib_Force_Inline DragonianLibSTL::Vector<std::wstring>& GetLanguageSymbols()
    {
        return _LanguageSymbols;
    }
	_D_Dragonian_Lib_Force_Inline DragonianLibSTL::Vector<std::pair<std::wstring, float>>& GetSpeakerMixSymbol()
	{
		return _SpeakerMixSymbol;
	}
};

using size_type = size_t;

struct TTSParams
{
    float NoiseScale = 0.3f;                                        //噪声修正因子
    int64_t Seed = 52468;                                           //种子
    float LengthScale = 1.0f;                                       //时长修正因子
    float DurationPredictorNoiseScale = 0.8f;                       //随机时长预测器噪声修正因子
    float FactorDpSdp = 0.f;                                        //随机时长预测器与时长预测器混合比例

    float GateThreshold = 0.66666f;                                 //Tacotron2解码器EOS阈值
    int64_t MaxDecodeStep = 2000;                                   //Tacotron2/GptSoVits最大自回归步数
    int64_t VQIndex = 0;

    DragonianLibSTL::Vector<std::wstring> EmotionPrompt;            //情感标记

	int64_t LanguageID = 0;                                         //语言ID
    std::wstring LanguageSymbol;                                    //语言

	int64_t SpeakerID = 0;                                          //角色ID
    std::wstring SpeakerName;                                       //角色名
};

_D_Dragonian_Lib_Lib_Text_To_Speech_End
