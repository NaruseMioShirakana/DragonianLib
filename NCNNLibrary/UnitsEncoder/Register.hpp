/**
 * @file Register.hpp
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
 * @brief Register of UnitsEncoder
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 */

#pragma once
#include "NCNNLibrary/UnitsEncoder/Hubert.hpp"

_D_Dragonian_Lib_NCNN_UnitsEncoder_Header

using UnitsEncoder = std::shared_ptr<UnitsEncoderBase>;

using Constructor = std::function< UnitsEncoder(
	const std::wstring& _Path,
	const NCNNOptions& Options,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
)>;

void RegisterUnitsEncoder(
	const std::wstring& _PluginName,
	const Constructor& _Constructor
);

UnitsEncoder New(
	const std::wstring& Name,
	const std::wstring& _Path,
	const NCNNOptions& Options,
	Int64 _SamplingRate = 16000,
	Int64 _UnitsDims = 768,
	bool _AddCache = false,
	const std::shared_ptr<Logger>& _Logger = _D_Dragonian_Lib_NCNN_UnitsEncoder_Space GetDefaultLogger()
);

std::vector<std::wstring>& GetList();

_D_Dragonian_Lib_NCNN_UnitsEncoder_End