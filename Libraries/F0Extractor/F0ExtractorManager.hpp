﻿/**
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
#include <string>

DragonianLibF0ExtractorHeader

class F0Extractor
{
public:
	F0Extractor() = delete;
	F0Extractor(BaseF0Extractor* _ext) : _f0_ext(_ext) {}
	F0Extractor(const F0Extractor&) = delete;
	F0Extractor(F0Extractor&& _ext) noexcept
	{
		delete _f0_ext;
		_f0_ext = _ext._f0_ext;
		_ext._f0_ext = nullptr;
	}
	F0Extractor& operator=(const F0Extractor&) = delete;
	F0Extractor& operator=(F0Extractor&& _ext) noexcept
	{
		if (this == &_ext)
			return *this;
		delete _f0_ext;
		_f0_ext = _ext._f0_ext;
		_ext._f0_ext = nullptr;
		return *this;
	}
	~F0Extractor()
	{
		delete _f0_ext;
		_f0_ext = nullptr;
	}
	BaseF0Extractor* operator->() const { return _f0_ext; }

private:
	BaseF0Extractor* _f0_ext = nullptr;
};

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