﻿/**
 * FileName: Stft.hpp
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
*/

#pragma once
#include "MyTemplateLibrary/Vector.h"
namespace DlCodecStft
{
    class STFT
    {
    public:
        STFT() = default;
        ~STFT();
        STFT(int WindowSize, int HopSize, int FFTSize = 0);
        inline static double PI = 3.14159265358979323846;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
    private:
    	int WINDOW_SIZE = 2048;
    	int HOP_SIZE = WINDOW_SIZE / 4;
    	int FFT_SIZE = WINDOW_SIZE / 2 + 1;
    };

    class Mel
    {
    public:
        Mel() = delete;
        ~Mel() = default;
        Mel(int WindowSize, int HopSize, int SamplingRate, int MelSize = 0, double FreqMin = 20., double FreqMax = 11025.);
        std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<int16_t>& audioData) const;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<double>& audioData) const;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
    private:
        STFT stft;
        int MEL_SIZE = 128;
        int FFT_SIZE = 0;
        int sr = 22050;
        DragonianLibSTL::Vector<float> MelBasis;
    };

    DragonianLibSTL::Vector<float> CQT(
        const DragonianLibSTL::Vector<float>& AudioData,
        int SamplingRate = 22050,
        int HopSize = 512,
        float FreqMin = 32.70f,
        int CQTBins = 84,
        int BinsPerOctave = 12,
        float Tuning = 0.f,
        float FilterScale = 1.f,
        float Norm = 1.f,
        float Sparsity = 0.01f,
        const char* Window = "Hann",
        bool Scale = true,
        const char* PaddingMode = "Constant",
        const char* ResourceType = "SOXR_HQ"
    );

    DragonianLibSTL::Vector<float> VQT(
        const DragonianLibSTL::Vector<float>& AudioData,
        int SamplingRate = 22050,
        int HopSize = 512,
        float FreqMin = 32.70f,
        int CQTBins = 84,
        const char* Intervals = "Equal",
        float Gamma = 0.f,
        int BinsPerOctave = 12,
        float Tuning = 0.f,
        float FilterScale = 1.f,
        float Norm = 1.f,
        float Sparsity = 0.01f,
        const char* Window = "Hann",
        bool Scale = true,
        const char* PaddingMode = "Constant",
        const char* ResourceType = "SOXR_HQ"
    );
}
