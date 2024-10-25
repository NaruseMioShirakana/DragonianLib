/**
 * FileName: TensorBase.h
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

#include <complex>
#include <set>
#include <unordered_map>
#include "Value.h"
#include "MyTemplateLibrary/Vector.h"

DragonianLibSpaceBegin

using namespace DragonianLibSTL;

using SizeType = int64;
template <typename _Ty>
using ContainerSet = std::set<_Ty>;
template <typename _TyA, typename _TyB>
using UnorderedMap = std::unordered_map<_TyA, _TyB>;
template <typename _TyA, typename _TyB>
using ContainerMap = std::unordered_map<_TyA, _TyB>;

DragonianLibSpaceEnd