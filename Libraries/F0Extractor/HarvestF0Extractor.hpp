/**
 * @file HarvestF0Extractor.hpp
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
 * @brief Harvest F0 extractor
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "BaseF0Extractor.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

/**
 * @class HarvestF0Extractor
 * @brief Harvest F0 extractor
 */
class HarvestF0Extractor : public BaseF0Extractor
{
public:
	HarvestF0Extractor() noexcept = default;
	~HarvestF0Extractor() noexcept override = default;
	HarvestF0Extractor(const HarvestF0Extractor&) noexcept = default;
	HarvestF0Extractor(HarvestF0Extractor&&) noexcept = default;
	HarvestF0Extractor& operator=(const HarvestF0Extractor&) noexcept = default;
	HarvestF0Extractor& operator=(HarvestF0Extractor&&) noexcept = default;

	/**
	 * @brief Extract F0 from PCM data
	 * @param PCMData PCM data, Shape [Channel, Samples], Channel not mean the channel of audio, it means the channel of the tensor, so it should be any value except zero and negative
	 * @param Params Parameters for F0 extraction
	 * @return F0, Shape [Channel, Frames], you don't need to call the evaluate function before using it
	 */
	Tensor<Float32, 2, Device::CPU> ExtractF0(
		const Tensor<Float64, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
	) override;
private:
	
	static Tensor<Float64, 2, Device::CPU> Harvest(
		const Tensor<Float64, 2, Device::CPU>& PCMData,
		const F0ExtractorParams& Params
	);
};

_D_Dragonian_Lib_F0_Extractor_End
