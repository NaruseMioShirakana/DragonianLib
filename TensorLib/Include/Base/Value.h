﻿/**
 * @file Value.h
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
 * @brief Object base class for Dlib
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Base.h"
#include "Libraries/MyTemplateLibrary/Array.h"
#include "Libraries/MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Space_Begin

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
// Define WeightHeader struct
struct WeightHeader
{
    Int64 Shape[8] = { 0,0,0,0,0,0,0,0 };
    char LayerName[DRAGONIANLIB_NAME_MAX_SIZE];
    char Type[16];
};
#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

// Define WeightData struct
struct WeightData
{
    WeightHeader Header_;
    std::vector<Byte> Data_;
    std::vector<Int64> Shape_;
    std::string Type_, LayerName_;
};

// Type alias for dictionary
using DictType = std::unordered_map<std::string, WeightData>;

class DlibValue
{
public:
    DlibValue() = default;
    virtual ~DlibValue() = default;
    DlibValue(const DlibValue& _Left) = default;
    DlibValue& operator=(const DlibValue& _Left) = default;
    DlibValue(DlibValue&& _Right) noexcept = default;
    DlibValue& operator=(DlibValue&& _Right) noexcept = default;

protected:
    std::wstring RegName_;

public:
	virtual DlibValue& Load(const std::wstring& _Path, bool _Strict = false);
    virtual DlibValue& Save(const std::wstring& _Path);
    virtual void LoadData(const DictType& _WeightDict, bool _Strict = false);
    virtual void SaveData(FileGuard& _File);
};

_D_Dragonian_Lib_Space_End
