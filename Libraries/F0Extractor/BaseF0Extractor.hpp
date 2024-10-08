﻿/**
 * FileName: BaseF0Extractor.hpp
 * Note: DragonianLib F0提取算法基类
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

#define DragonianLibF0ExtractorHeader namespace DragonianLib{
#define DragonianLibF0ExtractorEnd }

DragonianLibF0ExtractorHeader

class BaseF0Extractor
{
public:

	BaseF0Extractor() = delete;

	/**
	 * \brief 构造F0提取器
	 * \param sampling_rate 采样率
	 * \param hop_size HopSize
	 * \param n_f0_bins F0Bins
	 * \param max_f0 最大F0
	 * \param min_f0 最小F0
	 */
	BaseF0Extractor(int sampling_rate, int hop_size, int n_f0_bins = 256, double max_f0 = 1100.0, double min_f0 = 50.0);
	virtual ~BaseF0Extractor() = default;
	BaseF0Extractor(const BaseF0Extractor&) = delete;
	BaseF0Extractor(BaseF0Extractor&&) = delete;
	BaseF0Extractor operator=(const BaseF0Extractor&) = delete;
	BaseF0Extractor operator=(BaseF0Extractor&&) = delete;

	/**
	 * \brief 提取F0
	 * \param PCMData 音频PCM数据（SignedInt16 单声道） 
	 * \param TargetLength 目标F0长度
	 * \return F0
	 */
	virtual DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength);

	/**
	 * \brief 提取F0
	 * \param PCMData 音频PCM数据（SignedInt16 单声道）
	 * \param TargetLength 目标F0长度
	 * \return F0
	 */
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<float>& PCMData, size_t TargetLength);

	/**
	 * \brief 提取F0
	 * \param PCMData 音频PCM数据（SignedInt16 单声道）
	 * \param TargetLength 目标F0长度
	 * \return F0
	 */
	DragonianLibSTL::Vector<float> ExtractF0(const DragonianLibSTL::Vector<int16_t>& PCMData, size_t TargetLength);
protected:
	const uint32_t fs;
	const uint32_t hop;
	const uint32_t f0_bin;
	const double f0_max;
	const double f0_min;
	double f0_mel_min;
	double f0_mel_max;
};

DragonianLibF0ExtractorEnd