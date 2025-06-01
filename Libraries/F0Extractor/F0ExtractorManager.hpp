/**
 * @file F0ExtractorManager.hpp
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
 * @brief F0 Extractor Manager
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/F0Extractor/BaseF0Extractor.hpp"
#include <memory>
#include <string>
#include <vector>

_D_Dragonian_Lib_F0_Extractor_Header

using F0Extractor = ModulePointer<BaseF0Extractor>;
using Constructor = std::function<F0Extractor(const void*)>;

struct PEModelHParams
{
    const wchar_t* ModelPath; // Path to the PE Model (const wchar_t*)
    const void* Enviroment; // Pointer to OnnxRuntime::OnnxRuntimeEnvironment (const OnnxRuntime::OnnxRuntimeEnvironment*)
    const void* Logger; // Pointer to DragonianLib::DLogger (const DragonianLib::DLogger*)
    Int64 SamplingRate; // Sampling Rate of the Audio (Int64)
};

/**
 * @brief Register All F0Extractors in the directory, the name of the F0Extractor is the name of the file
 * @param _PluginRootDirectory Root directory of the F0Extractors
 */
void RegisterF0Extractors(
	const std::wstring& _PluginRootDirectory
);

/**
 * @brief Register a F0Extractor
 * @param _PluginName Name of the F0Extractor
 * @param _Constructor Constructor of the F0Extractor
 */
void RegisterF0Extractor(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
);

/**
 * @brief Create a new F0Extractor instance
 * @param Name Name of the F0Extractor
 * @param UserParameter User parameter
 * @return F0Extractor
 */
F0Extractor New(
    const std::wstring& Name,
    const void* UserParameter
);

/**
 * @brief Get a list of F0Extractor names
 * @return List of F0Extractor names
 */
std::vector<std::wstring>& GetList();

_D_Dragonian_Lib_F0_Extractor_End
