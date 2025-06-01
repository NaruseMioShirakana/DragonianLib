/**
 * @file G2PModule.hpp
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
 * @brief Module loader for Grapheme-to-Phoneme models in DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <functional>
#include "Libraries/G2P/G2PPlugin.hpp"

_D_Dragonian_Lib_G2P_Header

using G2PModule = ModulePointer<G2PBase>;
using Constructor = std::function<G2PModule(const void*)>;

/**
 * @brief Get a G2PModule
 * @param Name Name of the G2PModule
 * @param Parameter User parameter
 * @return G2PModule
 */
G2PModule New(
	const std::wstring& Name,
	const void* Parameter = nullptr
);

/**
 * @brief Register All G2PModules in the directory
 * @param _PluginRootDirectory Root directory of the G2PModules
 */
void RegisterG2PModules(
	const std::wstring& _PluginRootDirectory
);

/**
 * @brief Register a G2PModule
 * @param _PluginName Name of the G2PModule
 * @param _Constructor Constructor of the G2PModule
 */
void RegisterG2PModule(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
);

/**
 * @brief Get a list of G2PModule names
 * @return List of G2PModule names
 */
std::vector<std::wstring>& GetList();

_D_Dragonian_Lib_G2P_End