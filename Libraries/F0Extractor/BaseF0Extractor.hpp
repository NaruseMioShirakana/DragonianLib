/**
 * FileName: BaseF0Extractor.hpp
 * Note: DragonianLib BaseF0Extractor
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib library.
 * DragonianLib library is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
 * date: 2022-10-17 Create
*/

#pragma once
#include <cstdint>
#include "MyTemplateLibrary/Vector.h"

#define _D_Dragonian_Lib_F0_Extractor_Header namespace DragonianLib{
#define _D_Dragonian_Lib_F0_Extractor_End }

_D_Dragonian_Lib_F0_Extractor_Header

/**
 * @class BaseF0Extractor
 * @brief Base class for F0 extraction
 */
class BaseF0Extractor
{
public:

    BaseF0Extractor() = delete;
    BaseF0Extractor(int sampling_rate, int hop_size, int n_f0_bins = 256, double max_f0 = 1100.0, double min_f0 = 50.0);
    virtual ~BaseF0Extractor() = default;
    BaseF0Extractor(const BaseF0Extractor&) = delete;
    BaseF0Extractor(BaseF0Extractor&&) = delete;
    BaseF0Extractor operator=(const BaseF0Extractor&) = delete;
    BaseF0Extractor operator=(BaseF0Extractor&&) = delete;

    /**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data
	 * @param TargetLength Target length of F0
	 * @return F0
     */
    virtual DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength);

	/**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data
	 * @param TargetLength Target length of F0
	 * @return F0
	 */
    virtual DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<float>& PCMData, size_t TargetLength);

    /**
     * @brief Extract F0 from PCM data
     * @param PCMData PCM data
     * @param TargetLength Target length of F0
     * @return F0
     */
    virtual DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<int16_t>& PCMData, size_t TargetLength);
protected:
    const uint32_t fs;
    const uint32_t hop;
    const uint32_t f0_bin;
    const double f0_max;
    const double f0_min;
    double f0_mel_min;
    double f0_mel_max;
};

_D_Dragonian_Lib_F0_Extractor_End
