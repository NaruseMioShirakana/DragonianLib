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
#include "Libraries/MyTemplateLibrary/Vector.h"

#define _D_Dragonian_Lib_F0_Extractor_Header _D_Dragonian_Lib_Space_Begin namespace F0Extractor {
#define _D_Dragonian_Lib_F0_Extractor_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_F0_Extractor_Header

using namespace DragonianLibSTL;

/**
 * @struct F0ExtractorParams
 * @brief Parameters for F0 extraction
 */
struct F0ExtractorParams
{
	long SamplingRate = 16000; ///< Sampling rate
	long HopSize = 320; ///< Hop size
	long F0Bins = 256; ///< Number of bins for F0
	double F0Max = 1100.0; ///< Maximum F0
	double F0Min = 50.0; ///< Minimum F0
	void* UserParameter = nullptr; ///< User parameter
};

/**
 * @class BaseF0Extractor
 * @brief Base class for F0 extraction
 */
class BaseF0Extractor
{
public:
    BaseF0Extractor() = default;
    virtual ~BaseF0Extractor() = default;

    /**
     * @brief Extract F0 from PCM data
     * @param PCMData PCM data
	 * @param Params Parameters for F0 extraction
     * @return F0
     */
    virtual Vector<float> ExtractF0(
        const Vector<double>& PCMData,
		const F0ExtractorParams& Params
    );

    /**
     * @brief Extract F0 from PCM data
     * @param PCMData PCM data
     * @param Params Parameters for F0 extraction
     * @return F0
     */
    virtual Vector<float> ExtractF0(
        const Vector<float>& PCMData,
        const F0ExtractorParams& Params
    );

    /**
     * @brief Extract F0 from PCM data
     * @param PCMData PCM data
     * @param Params Parameters for F0 extraction
     * @return F0
     */
    virtual Vector<float> ExtractF0(
        const Vector<int16_t>& PCMData,
        const F0ExtractorParams& Params
    );

private:
    BaseF0Extractor(const BaseF0Extractor&) = delete;
    BaseF0Extractor(BaseF0Extractor&&) = delete;
    BaseF0Extractor operator=(const BaseF0Extractor&) = delete;
    BaseF0Extractor operator=(BaseF0Extractor&&) = delete;
};

_D_Dragonian_Lib_F0_Extractor_End
