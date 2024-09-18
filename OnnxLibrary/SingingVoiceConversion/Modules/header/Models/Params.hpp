/**
 * FileName: MoeVSProject.hpp
 * Note: MoeVoiceStudioCore 项目相关的定义
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
#include "../InferTools/InferTools.hpp"
#include "Util/StringPreprocess.h"

LibSvcHeader

struct SingleSlice
{
    DragonianLibSTL::Vector<int16_t> Audio;
    DragonianLibSTL::Vector<float> F0;
    DragonianLibSTL::Vector<float> Volume;
    DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> Speaker;
    int32_t OrgLen = 0;
    bool IsNotMute = false;
};

struct SingleAudio
{
    DragonianLibSTL::Vector<SingleSlice> Slices;
    std::wstring Path;
};

struct InferenceParams
{
    float NoiseScale = 0.3f;                           //噪声修正因子          0-10
    int64_t Seed = 52468;                              //种子
    int64_t SpeakerId = 0;                             //角色ID
    uint64_t SrcSamplingRate = 48000;                  //源采样率
    int64_t SpkCount = 2;                              //模型角色数
    float IndexRate = 0.f;                             //索引比               0-1
    float ClusterRate = 0.f;                           //聚类比               0-1
    float DDSPNoiseScale = 0.8f;                       //DDSP噪声修正因子      0-10
    float Keys = 0.f;                                  //升降调               -64-64
    size_t MeanWindowLength = 2;                       //均值滤波器窗口大小     1-20
    size_t Pndm = 1;                                   //Diffusion加速倍数    1-200
    size_t Step = 100;                                 //Diffusion总步数      1-1000
    float TBegin = 0.f;
    float TEnd = 1.f;
    std::wstring Sampler = L"Pndm";                    //Diffusion采样器
    std::wstring ReflowSampler = L"Eular";             //Reflow采样器
    std::wstring F0Method = L"Dio";                    //F0提取算法
    bool UseShallowDiffusion = false;                  //使用浅扩散
    void* VocoderModel = nullptr;
    void* ShallowDiffusionModel = nullptr;
    bool ShallowDiffusionUseSrcAudio = true;
    int VocoderHopSize = 512;
    int VocoderMelBins = 128;
    int VocoderSamplingRate = 44100;
    int64_t ShallowDiffuisonSpeaker = 0;
};

LibSvcEnd
