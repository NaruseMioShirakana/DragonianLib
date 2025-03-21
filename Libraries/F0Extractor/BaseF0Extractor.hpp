/**
 * @file BaseF0Extractor.hpp
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
 * @brief Base class for F0 extraction
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Tensor.h"

#define _D_Dragonian_Lib_F0_Extractor_Header _D_Dragonian_Lib_Space_Begin namespace F0Extractor {
#define _D_Dragonian_Lib_F0_Extractor_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_F0_Extractor_Header

/**
 * @struct F0ExtractorParams
 * @brief Parameters for F0 extraction
 */
struct F0ExtractorParams
{
	long SamplingRate = 16000; ///< Sampling rate
	long HopSize = 320; ///< Hop size
	long F0Bins = 256; ///< Number of bins for F0
	long WindowSize = 2048; ///< Window size
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
    BaseF0Extractor() noexcept = default;
    virtual ~BaseF0Extractor() noexcept = default;
    BaseF0Extractor(const BaseF0Extractor&) noexcept = default;
    BaseF0Extractor(BaseF0Extractor&&) noexcept = default;
    BaseF0Extractor& operator=(const BaseF0Extractor&) noexcept = default;
    BaseF0Extractor& operator=(BaseF0Extractor&&) noexcept = default;

    /**
     * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
	 * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
     */
    virtual Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Float64, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
    );

    /**
     * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
     * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
     */
    virtual Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Float32, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
    );

    /**
     * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
     * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
     */
    virtual Tensor<Float32, 2, Device::CPU> ExtractF0(
        const Tensor<Int16, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
    );

    Tensor<Float32, 2, Device::CPU> operator()(
        const Tensor<Float64, 2, Device::CPU>& PCMData,
        const F0ExtractorParams& Params
        );

	Tensor<Float32, 2, Device::CPU> operator()(
		const Tensor<Float32, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
		);

	Tensor<Float32, 2, Device::CPU> operator()(
		const Tensor<Int16, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
		);

};

_D_Dragonian_Lib_F0_Extractor_End
