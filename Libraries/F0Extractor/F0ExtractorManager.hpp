/**
 * FileName: F0ExtractorManager.hpp
 * Note: DragonianLib F0提取器管理
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
#include "BaseF0Extractor.hpp"
#include <functional>
#include <memory>
#include <string>

DragonianLibF0ExtractorHeader

using F0Extractor = std::shared_ptr<BaseF0Extractor>;

using GetF0ExtractorFn = std::function<F0Extractor(uint32_t, uint32_t, uint32_t, double, double)>;

void RegisterF0Extractor(const std::wstring& _name, const GetF0ExtractorFn& _constructor_fn);

/**
 * \brief 获取F0提取器
 * \param _name 类名
 * \param fs 采样率
 * \param hop HopSize
 * \param f0_bin F0Bins
 * \param f0_max 最大F0
 * \param f0_min 最小F0
 * \return F0提取器
 */
F0Extractor GetF0Extractor(const std::wstring& _name,
                           uint32_t fs = 48000,
                           uint32_t hop = 512,
                           uint32_t f0_bin = 256,
                           double f0_max = 1100.0,
                           double f0_min = 50.0);

std::vector<std::wstring> GetF0ExtractorList();

DragonianLibF0ExtractorEnd