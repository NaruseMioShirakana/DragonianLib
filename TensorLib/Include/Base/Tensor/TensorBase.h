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
#include "../Value.h"
#include "Libraries/MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Space_Begin

using namespace DragonianLibSTL;

using SizeType = Int64;
template <typename _Ty>
using ContainerSet = std::set<_Ty>;
template <typename _TyA, typename _TyB>
using UnorderedMap = std::unordered_map<_TyA, _TyB>;
template <typename _TyA, typename _TyB>
using ContainerMap = std::unordered_map<_TyA, _TyB>;

enum class DType
{
	Bool,
	Int8,
	Int16,
	Int32,
	Int64,
	UInt8,
	UInt16,
	UInt32,
	UInt64,
	Float16,
	Float32,
	Float64,
	Complex32,
	Complex64,
	BFloat16,
	Unknown
};

template<typename _Type>
struct _Impl_Dragonian_Lib_Decldtype { static constexpr auto _DType = DType::Unknown; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<bool> { static constexpr auto _DType = DType::Bool; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int8> { static constexpr auto _DType = DType::Int8; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int16> { static constexpr auto _DType = DType::Int16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int32> { static constexpr auto _DType = DType::Int32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int64> { static constexpr auto _DType = DType::Int64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt8> { static constexpr auto _DType = DType::UInt8; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt16> { static constexpr auto _DType = DType::UInt16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt32> { static constexpr auto _DType = DType::UInt32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt64> { static constexpr auto _DType = DType::UInt64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float16> { static constexpr auto _DType = DType::Float16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float32> { static constexpr auto _DType = DType::Float32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float64> { static constexpr auto _DType = DType::Float64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Complex32> { static constexpr auto _DType = DType::Complex32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Complex64> { static constexpr auto _DType = DType::Complex64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<BFloat16> { static constexpr auto _DType = DType::BFloat16; };
template<typename _Type>
constexpr auto _Impl_Dragonian_Lib_Decldtype_v = _Impl_Dragonian_Lib_Decldtype<_Type>::_DType;

_D_Dragonian_Lib_Space_End