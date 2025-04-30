#include "Libraries/Util/TypeDef.h"

_D_Dragonian_Lib_Space_Begin

enum class endian {
#if defined(_WIN32)
	little = 0,
	big = 1,
	native = little,
#elif defined(__GNUC__) || defined(__clang__)
	little = __ORDER_LITTLE_ENDIAN__,
	big = __ORDER_BIG_ENDIAN__,
	native = __BYTE_ORDER__,
#else
#error onnxruntime_float16::detail::endian is not implemented in this environment.
#endif
};

namespace TypeDef
{
	union float32_bits
	{
		uint32_t u;
		float f;
	};

	F16Base Float16_t::FromFloat32(Float32 f32) noexcept
	{
		float32_bits f;
		f.f = f32;

		constexpr float32_bits f32infty = { 255 << 23 };
		constexpr float32_bits f16max = { (127 + 16) << 23 };
		constexpr float32_bits denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
		constexpr unsigned int sign_mask = 0x80000000u;
		uint16_t val;

		unsigned int sign = f.u & sign_mask;
		f.u ^= sign;

		// NOTE all the integer compares in this function can be safely
		// compiled into signed compares since all operands are below
		// 0x80000000. Important if you want fast straight SSE2 code
		// (since there's no unsigned PCMPGTD).

		if (f.u >= f16max.u) {                         // result is Inf or NaN (all exponent bits set)
			val = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
		}
		else {                                       // (De)normalized number or zero
			if (f.u < (113 << 23)) {                     // resulting FP16 is subnormal or zero
				// use a magic value to align our 10 mantissa bits at the bottom of
				// the float. as long as FP addition is round-to-nearest-even this
				// just works.
				f.f += denorm_magic.f;

				// and one integer subtract of the bias later, we have our final float!
				val = static_cast<uint16_t>(f.u - denorm_magic.u);
			}
			else {
				unsigned int mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

				// update exponent, rounding bias part 1
				// Equivalent to `f.u += ((unsigned int)(15 - 127) << 23) + 0xfff`, but
				// without arithmetic overflow.
				f.u += 0xc8000fffU;
				// rounding bias part 2
				f.u += mant_odd;
				// take the bits!
				val = static_cast<uint16_t>(f.u >> 13);
			}
		}

		val |= static_cast<uint16_t>(sign >> 16);
		return { val };
	}

	Float32 Float16_t::Cast2Float32(F16Base f16) noexcept
	{
		constexpr float32_bits magic = { 113 << 23 };
		constexpr unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
		float32_bits o;

		o.u = (f16.U16 & 0x7fff) << 13;            // exponent/mantissa bits
		unsigned int exp = shifted_exp & o.u;  // just the exponent
		o.u += (127 - 15) << 23;               // exponent adjust

		// handle exponent special cases
		if (exp == shifted_exp) {   // Inf/NaN?
			o.u += (128 - 16) << 23;  // extra exp adjust
		}
		else if (exp == 0) {      // Zero/Denormal?
			o.u += 1 << 23;           // extra exp adjust
			o.f -= magic.f;           // re-normalize
		}

#if (defined _MSC_VER) && (defined _M_ARM || defined _M_ARM64 || defined _M_ARM64EC)
		if (static_cast<int16_t>(f16) < 0) {
			return -o.f;
		}
#else
  // original code:
		o.u |= (f16.U16 & 0x8000U) << 16U;  // sign bit
#endif
		return o.f;
	}

	F16Base BFloat16_t::FromFloat32(Float32 f32) noexcept
	{
		uint16_t result;
		if (std::isnan(f32))
			result = kPositiveQNaNBits;
		else 
		{
			auto get_msb_half = [](float fl)
				{
					uint16_t result;
					if constexpr (endian::native == endian::little)
						std::memcpy(&result, reinterpret_cast<char*>(&fl) + sizeof(uint16_t), sizeof(uint16_t));
					else
						std::memcpy(&result, &fl, sizeof(uint16_t));
					return result;
				};

			uint16_t upper_bits = get_msb_half(f32);
			union {
				uint32_t U32;
				float F32;
			};
			F32 = f32;
			U32 += (upper_bits & 1) + kRoundToNearest;
			result = get_msb_half(F32);
		}
		return { result };
	}

	Float32 BFloat16_t::Cast2Float32(F16Base f16) noexcept
	{
		if (static_cast<uint16_t>(f16.U16 & ~kSignMask) > kPositiveInfinityBits)
			return std::numeric_limits<float>::quiet_NaN();
		float result;
		char* const first = reinterpret_cast<char*>(&result);
		char* const second = first + sizeof(uint16_t);
		if constexpr (endian::native == endian::little)
		{
			std::memset(first, 0, sizeof(uint16_t));
			std::memcpy(second, &f16, sizeof(uint16_t));
		}
		else
		{
			std::memcpy(first, &f16, sizeof(uint16_t));
			std::memset(second, 0, sizeof(uint16_t));
		}
		return result;
	}

	template <typename T>
	struct Float8Traits;

	template <>
	struct Float8Traits<Float8E4M3FN_t> {
		static constexpr uint8_t exp_bits = 4;
		static constexpr uint8_t mant_bits = 3;
		static constexpr bool has_zero = true;
	};

	template <>
	struct Float8Traits<Float8E4M3FNUZ_t> {
		static constexpr uint8_t exp_bits = 4;
		static constexpr uint8_t mant_bits = 3;
		static constexpr bool has_zero = false;
	};

	template <>
	struct Float8Traits<Float8E5M2_t> {
		static constexpr uint8_t exp_bits = 5;
		static constexpr uint8_t mant_bits = 2;
		static constexpr bool has_zero = true;
	};

	template <>
	struct Float8Traits<Float8E5M2FNUZ_t> {
		static constexpr uint8_t exp_bits = 5;
		static constexpr uint8_t mant_bits = 2;
		static constexpr bool has_zero = false;
	};

	template <typename T>
	static F8Base FromFloat32Impl(Float32 f32) noexcept {
		constexpr uint8_t exp_bits = Float8Traits<T>::exp_bits;
		constexpr uint8_t mant_bits = Float8Traits<T>::mant_bits;
		constexpr bool has_zero = Float8Traits<T>::has_zero;

		union {
			Float32 f;
			uint32_t u;
		} v = { f32 };

		uint32_t sign = (v.u >> 31) & 0x1;
		int32_t exp = ((v.u >> 23) & 0xFF) - 127 + ((1 << (exp_bits - 1)) - 1);
		uint32_t mant = (v.u >> (23 - mant_bits)) & ((1 << mant_bits) - 1);

		if (exp <= 0) {
			if (has_zero) {
				exp = 0;
				mant = 0;
			}
			else {
				exp = 1;
				mant >>= 1;
			}
		}
		else if (exp >= (1 << exp_bits) - 1) {
			exp = (1 << exp_bits) - 1;
			mant = 0;
		}

		F8Base result;
		result.U8 = (sign << 7) | (exp << mant_bits) | mant;
		return result;
	}

	template <typename T>
	static Float32 Cast2Float32Impl(F8Base f8) noexcept {
		constexpr uint8_t exp_bits = Float8Traits<T>::exp_bits;
		constexpr uint8_t mant_bits = Float8Traits<T>::mant_bits;

		uint32_t sign = (f8.U8 >> 7) & 0x1;
		int32_t exp = ((f8.U8 >> mant_bits) & ((1 << exp_bits) - 1)) - ((1 << (exp_bits - 1)) - 1) + 127;
		uint32_t mant = f8.U8 & ((1 << mant_bits) - 1);

		if (exp <= 0) {
			exp = 0;
			mant = 0;
		}
		else if (exp >= 255) {
			exp = 255;
			mant = 0;
		}

		union {
			Float32 f;
			uint32_t u;
		} v;
		v.u = (sign << 31) | (exp << 23) | (mant << (23 - mant_bits));
		return v.f;
	}

	F8Base Float8E4M3FN_t::FromFloat32(Float32 f32) noexcept
	{
		return FromFloat32Impl<Float8E4M3FN_t>(f32);
	}

	Float32 Float8E4M3FN_t::Cast2Float32(F8Base f16) noexcept
	{
		return Cast2Float32Impl<Float8E4M3FN_t>(f16);
	}

	F8Base Float8E4M3FNUZ_t::FromFloat32(Float32 f32) noexcept
	{
		return FromFloat32Impl<Float8E4M3FNUZ_t>(f32);
	}

	Float32 Float8E4M3FNUZ_t::Cast2Float32(F8Base f16) noexcept
	{
		return Cast2Float32Impl<Float8E4M3FNUZ_t>(f16);
	}

	F8Base Float8E5M2_t::FromFloat32(Float32 f32) noexcept
	{
		return FromFloat32Impl<Float8E5M2_t>(f32);
	}

	Float32 Float8E5M2_t::Cast2Float32(F8Base f16) noexcept
	{
		return Cast2Float32Impl<Float8E5M2_t>(f16);
	}

	F8Base Float8E5M2FNUZ_t::FromFloat32(Float32 f32) noexcept
	{
		return FromFloat32Impl<Float8E5M2FNUZ_t>(f32);
	}

	Float32 Float8E5M2FNUZ_t::Cast2Float32(F8Base f16) noexcept
	{
		return Cast2Float32Impl<Float8E5M2FNUZ_t>(f16);
	}

}

_D_Dragonian_Lib_Space_End
