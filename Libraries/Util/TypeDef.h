/**
 * @file TypeDef.h
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
 * @brief Type definitions for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include <cstdint>
#include <complex>

#include "Libraries/Util/Util.h"

_D_Dragonian_Lib_Space_Begin

using Boolean = bool; ///< Boolean
using Int8 = int8_t; ///< 8-bit integer
using Int16 = int16_t; ///< 16-bit integer
using Int32 = int32_t; ///< 32-bit integer
using Int64 = int64_t; ///< 64-bit integer
using Float32 = float; ///< 32-bit floating point
using Float = float; ///< Floating point
using Float64 = double; ///< 64-bit floating point
using Double = double; ///< Double
using Byte = unsigned char; ///< Byte
using LPVoid = void*; ///< Pointer to void
using LPCVoid = const void*; ///< Pointer to Constant void
using LCPVoid = const LPVoid; ///< Constant pointer to void
using UInt8 = uint8_t; ///< 8-bit unsigned integer
using UInt16 = uint16_t; ///< 16-bit unsigned integer
using UInt32 = uint32_t; ///< 32-bit unsigned integer
using UInt64 = uint64_t; ///< 64-bit unsigned integer
using Complex32 = std::complex<float>; ///< 32-bit complex
using Complex64 = std::complex<double>; ///< 64-bit complex
using Int = int; ///< Int
using UInt = unsigned int; ///< Unsigned int
using Long = long; ///< Long
using ULong = unsigned long; ///< Unsigned long
using LongLong = long long; ///< Int64
using ULongLong = unsigned long long; ///< Unsigned Int64
using Short = short; ///< Short
using UShort = unsigned short; ///< Unsigned short
using PtrDiff = ptrdiff_t; ///< Pointer difference type
using StdSize = std::size_t; ///< Size type
using IPointer = Int64; ///< Integer pointer type
using UPointer = UInt64; ///< Unsigned integer pointer type

namespace TypeDef
{
	union F8Base
	{
		uint8_t U8;
		int8_t I8;
	};
	union F16Base
	{
		uint16_t U16;
		int16_t I16;
	};

	constexpr UInt16 kSignMask = 0x8000U;
	constexpr UInt16 kBiasedExponentMask = 0x7F80U;
	constexpr UInt16 kPositiveInfinityBits = 0x7F80U;
	constexpr UInt16 kNegativeInfinityBits = 0xFF80U;
	constexpr UInt16 kPositiveQNaNBits = 0x7FC1U;
	constexpr UInt16 kNegativeQNaNBits = 0xFFC1U;
	constexpr UInt16 kSignaling_NaNBits = 0x7F80U;
	constexpr UInt16 kEpsilonBits = 0x0080U;
	constexpr UInt16 kMinValueBits = 0xFF7FU;
	constexpr UInt16 kMaxValueBits = 0x7F7FU;
	constexpr UInt16 kRoundToNearest = 0x7FFFU;
	constexpr UInt16 kOneBits = 0x3F80U;
	constexpr UInt16 kMinusOneBits = 0xBF80U;

	enum OperatorType : UInt8
	{
		UnaryOperatorType,
		BinaryOperatorType,
		ConstantOperatorType,
		ReversedConstantOperatorType
	};

	template <typename _Type, size_t _Rank>
	struct _Impl_NDInitilizerListType
	{
		using Type = std::initializer_list<typename _Impl_NDInitilizerListType<_Type, _Rank - 1>::Type>;
	};
	template <typename _Type>
	struct _Impl_NDInitilizerListType<_Type, 0>
	{
		using Type = _Type;
	};

	template <typename _Type, size_t _Rank, size_t _Size, size_t ..._RSize>
	struct _Impl_NDArray
	{
		static_assert(_Size > 0, "Size must be greater than 0");
		static_assert(sizeof...(_RSize) == _Rank - 1, "Rank must be equal to the number of sizes");
		static_assert(_Rank > 0, "Rank must be greater than 0");
		using Type = _Impl_NDArray<_Type, _Rank - 1, _RSize...>[_Size];
	};
	template <typename _Type, size_t _Size, size_t ..._RSize>
	struct _Impl_NDArray<_Type, 1, _Size, _RSize...>
	{
		static_assert(_Size > 0, "Size must be greater than 0");
		static_assert(sizeof...(_RSize) == 0, "Rank must be equal to the number of sizes");
		using Type = _Type[_Size];
	};

	enum class BuiltInTypes : UInt8
	{
		None,
		Int8,
		Int16,
		Int32,
		Int64,
		UInt8,
		UInt16,
		UInt32,
		UInt64,
		Float8,
		BFloat16,
		Float16,
		Float32,
		Float64,
		Complex32,
		Complex64,
		Byte,
		LPVoid,
		CPVoid
	};

	/**
	 * @struct Float16_t
	 * @brief Half precision floating point struct
	 */
	struct Float16_t {
		Float16_t() noexcept : Val(FromFloat32(0.f)) {}

		Float16_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		Float16_t(Float64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(Int64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(Int32 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(Int16 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(Int8 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(UInt64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(UInt32 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(UInt16 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(UInt8 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		Float16_t(Boolean _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}

		Float16_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		Float16_t& operator=(Float64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(Int64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(Int32 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(Int16 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(Int8 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(UInt64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(UInt32 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(UInt16 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(UInt8 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		Float16_t& operator=(Boolean _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }

		operator Float32 () const noexcept { return Cast2Float32(Val); }
		operator Float64 () const noexcept { return static_cast<Float64>(Cast2Float32(Val)); }
		operator Int64 () const noexcept { return static_cast<Int64>(Cast2Float32(Val)); }
		operator Int32 () const noexcept { return static_cast<Int32>(Cast2Float32(Val)); }
		operator Int16 () const noexcept { return static_cast<Int16>(Cast2Float32(Val)); }
		operator Int8 () const noexcept { return static_cast<Int8>(Cast2Float32(Val)); }
		operator UInt64 () const noexcept { return static_cast<UInt64>(Cast2Float32(Val)); }
		operator UInt32 () const noexcept { return static_cast<UInt32>(Cast2Float32(Val)); }
		operator UInt16 () const noexcept { return static_cast<UInt16>(Cast2Float32(Val)); }
		operator UInt8 () const noexcept { return static_cast<UInt8>(Cast2Float32(Val)); }
		operator Boolean () const noexcept { return Cast2Float32(Val) != 0.f; }
	private:
		F16Base Val;
		static F16Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F16Base f16) noexcept;
	};

	/**
	 * @struct BFloat16_t
	 * @brief bfloat16 struct
	 */
	struct BFloat16_t
	{
		BFloat16_t() noexcept : Val(FromFloat32(0.f)) {}

		BFloat16_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		BFloat16_t(Float64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(Int64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(Int32 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(Int16 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(Int8 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(UInt64 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(UInt32 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(UInt16 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(UInt8 _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(Boolean _Val) noexcept : Val(FromFloat32(static_cast<Float32>(_Val))) {}
		BFloat16_t(const Float16_t& _Val) noexcept : Val(FromFloat32(_Val)) {}

		BFloat16_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		BFloat16_t& operator=(Float64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(Int64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(Int32 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(Int16 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(Int8 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(UInt64 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(UInt32 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(UInt16 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(UInt8 _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(Boolean _Val) noexcept { Val = FromFloat32(static_cast<Float32>(_Val)); return *this; }
		BFloat16_t& operator=(const Float16_t& _Val) noexcept { Val = FromFloat32(_Val); return *this; }

		operator Float32 () const noexcept { return Cast2Float32(Val); }
		operator Float64 () const noexcept { return static_cast<Float64>(Cast2Float32(Val)); }
		operator Int64 () const noexcept { return static_cast<Int64>(Cast2Float32(Val)); }
		operator Int32 () const noexcept { return static_cast<Int32>(Cast2Float32(Val)); }
		operator Int16 () const noexcept { return static_cast<Int16>(Cast2Float32(Val)); }
		operator Int8 () const noexcept { return static_cast<Int8>(Cast2Float32(Val)); }
		operator UInt64 () const noexcept { return static_cast<UInt64>(Cast2Float32(Val)); }
		operator UInt32 () const noexcept { return static_cast<UInt32>(Cast2Float32(Val)); }
		operator UInt16 () const noexcept { return static_cast<UInt16>(Cast2Float32(Val)); }
		operator UInt8 () const noexcept { return static_cast<UInt8>(Cast2Float32(Val)); }
		operator Boolean () const noexcept { return Cast2Float32(Val) != 0.f; }
		operator Float16_t () const noexcept { return { Cast2Float32(Val) }; }
	private:
		F16Base Val;
		static F16Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F16Base f16) noexcept;
	};

	/**
	 * @struct Float8E4M3FN_t
	 * @brief 8-bit floating point struct with exponent 4, mantissa 3, fraction 1
	 */
	struct Float8E4M3FN_t
	{
		Float8E4M3FN_t() noexcept : Val(FromFloat32(0.f)) {}
		Float8E4M3FN_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		Float8E4M3FN_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		operator Float32 () const noexcept { return Cast2Float32(Val); }
	private:
		F8Base Val;
		static F8Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F8Base f16) noexcept;
	};

	/**
	 * @struct Float8E4M3FNUZ_t
	 * @brief 8-bit floating point struct with exponent 4, mantissa 3, fraction 1, no zero
	 */
	struct Float8E4M3FNUZ_t
	{
		Float8E4M3FNUZ_t() noexcept : Val(FromFloat32(0.f)) {}
		Float8E4M3FNUZ_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		Float8E4M3FNUZ_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		operator Float32 () const noexcept { return Cast2Float32(Val); }
	private:
		F8Base Val;
		static F8Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F8Base f16) noexcept;
	};

	/**
	 * @struct Float8E5M2_t
	 * @brief 8-bit floating point struct with exponent 5, mantissa 2, fraction 1
	 */
	struct Float8E5M2_t
	{
		Float8E5M2_t() noexcept : Val(FromFloat32(0.f)) {}
		Float8E5M2_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		Float8E5M2_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		operator Float32 () const noexcept { return Cast2Float32(Val); }
	private:
		F8Base Val;
		static F8Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F8Base f16) noexcept;
	};

	/**
	 * @struct Float8E5M2FNUZ_t
	 * @brief 8-bit floating point struct with exponent 5, mantissa 2, fraction 1, no zero
	 */
	struct Float8E5M2FNUZ_t
	{
		Float8E5M2FNUZ_t() noexcept : Val(FromFloat32(0.f)) {}
		Float8E5M2FNUZ_t(Float32 _Val) noexcept : Val(FromFloat32(_Val)) {}
		Float8E5M2FNUZ_t& operator=(Float32 _Val) noexcept { Val = FromFloat32(_Val); return *this; }
		operator Float32 () const noexcept { return Cast2Float32(Val); }
	private:
		F8Base Val;
		static F8Base FromFloat32(Float32 f32) noexcept;
		static Float32 Cast2Float32(F8Base f16) noexcept;
	};
}

using Float8 = TypeDef::Float8E4M3FN_t; ///< 8-bit floating point with exponent 4, mantissa 3, fraction 1
using BFloat16 = TypeDef::BFloat16_t; ///< bfloat16
using Float16 = TypeDef::Float16_t; ///< Half precision floating point
using Float8E4M3FN = TypeDef::Float8E4M3FN_t; ///< 8-bit floating point with exponent 4, mantissa 3, fraction 1
using Float8E4M3FNUZ = TypeDef::Float8E4M3FNUZ_t; ///< 8-bit floating point with exponent 4, mantissa 3, fraction 1, no zero
using Float8E5M2 = TypeDef::Float8E5M2_t; ///< 8-bit floating point with exponent 5, mantissa 2, fraction 1
using Float8E5M2FNUZ = TypeDef::Float8E5M2FNUZ_t; ///< 8-bit floating point with exponent 5, mantissa 2, fraction 1, no zero

template <typename _Type, size_t _Rank>
using NDInitilizerList = typename ::DragonianLib::TypeDef::_Impl_NDInitilizerListType<_Type, _Rank>::Type;

template <typename _Type, bool _Cond>
struct OptionalType {};
template <typename _Type>
struct OptionalType<_Type, true>
{
	_Type _MyValue;
};

template <typename _Type, _Type... _Values>
struct BuildTimeList
{
	using value_type = _Type;
	static constexpr size_t _MySize = sizeof...(_Values);
	static constexpr size_t size() { return _MySize; }
};

template<typename ..._ArgTypes>
struct GeneralizedList
{
	static constexpr int64_t _Size = sizeof...(_ArgTypes);
	static constexpr size_t size() { return _Size; }
};

template <typename _IntegerType, size_t _Size, _IntegerType _Index, _IntegerType... _Indices>
	requires(std::is_integral_v<_IntegerType>)
struct __ImplMakeIntegerSequence
{
	using _MyType = typename ::DragonianLib::__ImplMakeIntegerSequence<_IntegerType, _Size - 1, _Index + 1, _Indices..., _Index>::_MyType;
};

template <typename _IntegerType, _IntegerType _Index, _IntegerType... _Indices>
	requires(std::is_integral_v<_IntegerType>)
struct __ImplMakeIntegerSequence<_IntegerType, 0, _Index, _Indices...>
{
	using _MyType = ::DragonianLib::BuildTimeList<_IntegerType, _Indices...>;
};

template <typename _IntegerType, size_t _Size>
	requires(std::is_integral_v<_IntegerType>)
using MakeIntegerSequence = typename ::DragonianLib::__ImplMakeIntegerSequence<_IntegerType, _Size, 0>::_MyType;

template <size_t _Size>
using MakeIndexSequence = ::DragonianLib::MakeIntegerSequence<size_t, _Size>;

template <size_t ..._Indices>
using IndexSequence = ::DragonianLib::BuildTimeList<size_t, _Indices...>;

_D_Dragonian_Lib_Space_End