#pragma once
#include <cstdint>
#include <complex>

namespace DragonianLib
{

	/**
	 * @struct float16_t
	 * @brief Half precision floating point struct
	 */
	struct float16_t {
		float16_t(float _Val);
		float16_t& operator=(float _Val);
		operator float() const;
	private:
		uint16_t Val;
		static float16_t float32_to_float16(uint32_t f32);
		static uint32_t float16_to_float32(float16_t f16);
	};

	/**
	 * @struct float8_t
	 * @brief 8-bit floating point struct
	 */
	struct float8_t
	{
		float8_t(float _Val);
		float8_t& operator=(float _Val);
		operator float() const;
	private:
		uint8_t Val;
		static float8_t float32_to_float8(uint32_t f32);
		static uint32_t float8_to_float32(float8_t f8);
	};

	/**
	 * @struct bfloat16_t
	 * @brief bfloat16 struct
	 */
	struct bfloat16_t
	{
		bfloat16_t(float _Val);
		bfloat16_t& operator=(float _Val);
		operator float() const;
	private:
		uint16_t Val;
		static bfloat16_t float32_to_bfloat16(uint32_t f32);
		static uint32_t bfloat16_to_float32(bfloat16_t bf16);
	};

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
	struct NoneType {}; ///< None type
	static constexpr NoneType None; ///< None constant
	template <typename _Type>
	constexpr bool operator==(const _Type&, const NoneType&)
	{
		if constexpr (std::is_same_v<_Type, NoneType>)
			return true;
		else
			return false;
	}
}