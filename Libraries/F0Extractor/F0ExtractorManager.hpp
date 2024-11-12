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
#include <memory>
#include <string>
#include <vector>

_D_Dragonian_Lib_F0_Extractor_Header

using F0Extractor = std::shared_ptr<BaseF0Extractor>;

struct NetF0ExtractorSetting
{
	std::wstring ModelPath;
	void* OrtEnv;
};

/**
 * @brief Register All F0Extractors in the directory
 * @param _PluginRootDirectory Root directory of the F0Extractors
 */
void RegisterF0Extractor(const std::wstring& _PluginRootDirectory);

/**
 * @brief Get a F0Extractor
 * @param Name Name of the F0Extractor
 * @param UserParameter User parameter
 * @return F0Extractor
 */
F0Extractor GetF0Extractor(
    const std::wstring& Name,
    const void* UserParameter
);

/**
 * @brief Get a list of F0Extractor names
 * @return List of F0Extractor names
 */
const std::vector<std::wstring>& GetF0ExtractorList();

_D_Dragonian_Lib_F0_Extractor_End
