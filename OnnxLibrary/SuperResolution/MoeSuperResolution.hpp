/**
 * FileName: MoeSuperResolution.hpp
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
#include "SuperResolution.hpp"

namespace DragonianLib
{
    namespace LibSuperResolution
    {
        class MoeSR : public SuperResolution
        {
        public:

            MoeSR(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
            ~MoeSR() override;

            ImageVideo::Image& Infer(ImageVideo::Image& _Image, int64_t _BatchSize) const override;
        private:
            void Destory();
            MoeSR(const MoeSR&) = delete;
            MoeSR(MoeSR&&) = delete;
            MoeSR& operator=(const MoeSR&) = delete;
            MoeSR& operator=(MoeSR&&) = delete;

            Ort::Session* Model = nullptr;
            long ScaleFactor = 2;
        };
    }
}