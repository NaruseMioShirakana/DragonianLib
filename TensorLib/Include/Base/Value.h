/**
 * FileName: Value.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "Libraries/Base.h"

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

class Value
{
public:
    Value() = default;
    virtual ~Value() = default;
    Value(const Value& _Left) = default;
    Value& operator=(const Value& _Left) = default;
    Value(Value&& _Right) noexcept = default;
    Value& operator=(Value&& _Right) noexcept = default;

protected:
    std::wstring RegName_;

public:
	virtual Value& Load(const std::wstring& _Path, bool _Strict = false);
    virtual Value& Save(const std::wstring& _Path);
    virtual void LoadData(const DictType& _WeightDict, bool _Strict = false);
    virtual void SaveData(FileGuard& _File);
};

_D_Dragonian_Lib_Space_End
