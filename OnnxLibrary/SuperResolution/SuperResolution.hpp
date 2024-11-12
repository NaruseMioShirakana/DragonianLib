/**
 * FileName: SuperResolution.hpp
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
#include "EnvManager.hpp"
#include "Image-Video/ImgVideo.hpp"
namespace DragonianLib
{
    namespace LibSuperResolution
    {
        using ProgressCallback = std::function<void(size_t, size_t)>;

        struct Hparams
        {
            std::wstring RGBModel, AlphaModel;
            long InputWidth = 64;
            long InputHeight = 64;
            long Scale = 2;
        };

        class SuperResolution
        {
        public:
            SuperResolution(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider, ProgressCallback _Callback);
            virtual ~SuperResolution() = default;
            virtual ImageVideo::Image& Infer(ImageVideo::Image& _Image, int64_t _BatchSize) const;
        protected:
            DragonianLib::DragonianLibOrtEnv Env_;
            ProgressCallback Callback_;
            std::vector<Ort::AllocatedStringPtr> Names;
            char* inputNames = nullptr;
            char* outputNames = nullptr;
        private:
            SuperResolution(const SuperResolution&) = delete;
            SuperResolution(SuperResolution&&) = delete;
            SuperResolution& operator=(const SuperResolution&) = delete;
            SuperResolution& operator=(SuperResolution&&) = delete;
        };
    }
}