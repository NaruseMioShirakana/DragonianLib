#pragma once
#include <cstdint>
#include <complex>
#include "Util.h"

_D_Dragonian_Lib_Space_Begin

using f16base_t = uint16_t;

/**
 * @struct float16_t
 * @brief Half precision floating point struct
 */
struct float16_t {
	float16_t(float _Val) noexcept;
	float16_t& operator=(float _Val) noexcept;
	operator float() const noexcept;
private:
	uint16_t Val;
	static float16_t float32_to_float16(uint32_t f32) noexcept;
	static uint32_t float16_to_float32(float16_t f16) noexcept;
};

/**
 * @struct float8_t
 * @brief 8-bit floating point struct
 */
struct float8_t
{
	float8_t(float _Val) noexcept;
	float8_t& operator=(float _Val) noexcept;
	operator float() const noexcept;
private:
	uint8_t Val;
	static float8_t float32_to_float8(uint32_t f32) noexcept;
	static uint32_t float8_to_float32(float8_t f8) noexcept;
};

/**
 * @struct bfloat16_t
 * @brief bfloat16 struct
 */
struct bfloat16_t
{
	bfloat16_t(float _Val) noexcept;
	bfloat16_t& operator=(float _Val) noexcept;
	operator float() const noexcept;
private:
	uint16_t Val;
	static bfloat16_t float32_to_bfloat16(uint32_t f32) noexcept;
	static uint32_t bfloat16_to_float32(bfloat16_t bf16) noexcept;
};

using Boolean = bool; ///< Boolean
using Int8 = int8_t; ///< 8-bit integer
using Int16 = int16_t; ///< 16-bit integer
using Int32 = int32_t; ///< 32-bit integer
using Int64 = int64_t; ///< 64-bit integer
using Float8 = float8_t; ///< 8-bit floating point
using BFloat16 = bfloat16_t; ///< bfloat16
using Float16 = float16_t; ///< Half precision floating point
using Float32 = float; ///< 32-bit floating point
using Float64 = double; ///< 64-bit floating point
using Byte = unsigned char; ///< Byte
using LPVoid = void*; ///< Pointer to void
using CPVoid = const void*; ///< Constant pointer to void
using UInt8 = uint8_t; ///< 8-bit unsigned integer
using UInt16 = uint16_t; ///< 16-bit unsigned integer
using UInt32 = uint32_t; ///< 32-bit unsigned integer
using UInt64 = uint64_t; ///< 64-bit unsigned integer
using Complex32 = std::complex<float>; ///< 32-bit complex
using Complex64 = std::complex<double>; ///< 64-bit complex

namespace TypeDef
{
	enum OperatorType
	{
		UnaryOperatorType,
		BinaryOperatorType,
		ConstantOperatorType
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

	enum class BuiltInTypes
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
}

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