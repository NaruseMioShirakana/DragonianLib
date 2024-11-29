/**
 * FileName: Simd.h
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
#include "../OperatorBase.h"
#include <immintrin.h>
#define _D_Dragonian_Lib_Simd_Not_Implemented_Error _D_Dragonian_Lib_Throw_With_Inline_Function("SIMD Not Implemented Error!")
#define _D_Dragonian_Lib_Simd_Not_Mask(_Type) (::DragonianLib::Operators::Vectorized<_Type>((_Type)(0)))
#define _D_Dragonian_Lib_Simd_Complement_Mask(_Type) (::DragonianLib::Operators::Vectorized<_Type>((_Type)(-1)))
_D_Dragonian_Lib_Operator_Space_Begin
//*****************//

/**
 * @class Vectorized
 * @brief Warpper for SIMD
 * @tparam Type Type of the data
 * @tparam T TypeCheak
 */
template<typename Type, typename T = std::enable_if_t<!std::is_pointer_v<Type>, Type>>
class Vectorized
{
public:
	static_assert(std::is_same_v<Type, T>);

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~Vectorized() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(const Vectorized&) = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(Vectorized&&) = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized& operator=(const Vectorized&) = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized& operator=(Vectorized&&) = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(const Type* _ValPtr)
	{
		if constexpr (std::is_same_v<Type, float>)
			_YMMX = _mm256_castps_si256(_mm256_load_ps(_ValPtr));
		else if constexpr (std::is_same_v<Type, double>)
			_YMMX = _mm256_castpd_si256(_mm256_load_pd(_ValPtr));
		else
			_YMMX = _mm256_load_si256((const __m256i*)_ValPtr);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(__m256i _Val) : _YMMX(_Val) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(__m256 _Val) : _YMMX(_mm256_castps_si256(_Val)) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(__m256d _Val) : _YMMX(_mm256_castpd_si256(_Val)) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized(Type _Val)
	{
		if constexpr (std::is_same_v<Type, float>)
			_YMMX = _mm256_castps_si256(_mm256_set1_ps(_Val));
		else if constexpr (std::is_same_v<Type, double>)
			_YMMX = _mm256_castpd_si256(_mm256_set1_pd(_Val));
		else if constexpr (std::is_same_v<Type, int8_t>)
			_YMMX = _mm256_set1_epi8(_Val);
		else if constexpr (std::is_same_v<Type, int16_t>)
			_YMMX = _mm256_set1_epi16(_Val);
		else if constexpr (std::is_same_v<Type, int32_t>)
			_YMMX = _mm256_set1_epi32(_Val);
		else if constexpr (std::is_same_v<Type, int64_t>)
			_YMMX = _mm256_set1_epi64x(_Val);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Store(Type* _Dest) const
	{
		if constexpr (std::is_same_v<Type, float>)
			_mm256_store_ps(_Dest, *this);
		else if constexpr (std::is_same_v<Type, double>)
			_mm256_store_pd(_Dest, *this);
		else
			_mm256_store_si256((const __m256i*)_Dest, _YMMX);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Store(Type* _Dest, __mmask32 _Mask) const
	{
		if constexpr (std::is_same_v<Type, float>)
			_mm256_mask_storeu_ps(_Dest, _Mask >> 24, *this);
		else if constexpr (std::is_same_v<Type, double>)
			_mm256_mask_storeu_pd(_Dest, _Mask >> 24, *this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_mm256_mask_storeu_epi8(_Dest, _Mask, _YMMX);
		else if constexpr (std::is_same_v<Type, int16_t>)
			_mm256_mask_storeu_epi16(_Dest, _Mask >> 16, _YMMX);
		else if constexpr (std::is_same_v<Type, int32_t>)
			_mm256_mask_storeu_epi32(_Dest, _Mask >> 24, _YMMX);
		else if constexpr (std::is_same_v<Type, int64_t>)
			_mm256_mask_storeu_epi64(_Dest, _Mask >> 24, _YMMX);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void MaskedStore(void* _Dest, const Vectorized& _Mask) const
	{
		_mm256_store_si256((__m256i*)_Dest, _mm256_and_si256(*this, _Mask));
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void StoreBool(void* _Dest) const
	{
		if constexpr (std::is_same_v<Type, float>)
		{
			const auto Mask = _mm256_movemask_ps(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 8; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (std::is_same_v<Type, double>)
		{
			const auto Mask = _mm256_movemask_pd(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 4; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (std::is_same_v<Type, int8_t>)
		{
			const auto Mask = _mm256_movemask_epi8(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 32; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (std::is_same_v<Type, int16_t>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi16_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 16; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (std::is_same_v<Type, int32_t>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi32_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 8; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (std::is_same_v<Type, int64_t>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi64_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 4; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __m256i() const
	{
		static_assert(std::is_same_v<Type, int8_t> || std::is_same_v<Type, int16_t> || std::is_same_v<Type, int32_t> || std::is_same_v<Type, int64_t>);
		return _YMMX;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __m256() const
	{
		static_assert(std::is_same_v<Type, float>);
		return _mm256_castsi256_ps(_YMMX);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __m256d() const
	{
		static_assert(std::is_same_v<Type, double>);
		return _mm256_castsi256_pd(_YMMX);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __mmask32() const
	{
		return _mm256_movemask_epi8(_YMMX);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __mmask16() const
	{
		return _mm256_movemask_epi8(_YMMX) >> 16;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator __mmask8() const
	{
		return _mm256_movemask_epi8(_YMMX) >> 24;
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator<(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Right, _CMP_LT_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Right, _CMP_LT_OQ);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_cmpgt_epi8(_Right, *this);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_cmpgt_epi16(_Right, *this);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cmpgt_epi32(_Right, *this);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cmpgt_epi64(_Right, *this);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator<=(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Right, _CMP_LE_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Right, _CMP_LE_OQ);
		else
			return *this < _Right || *this == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator>(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Right, _CMP_GT_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Right, _CMP_GT_OQ);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_cmpgt_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_cmpgt_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cmpgt_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cmpgt_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator>=(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Right, _CMP_GE_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Right, _CMP_GE_OQ);
		else
			return *this > _Right || *this == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator==(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
		{
			const auto Diff = _mm256_sub_ps(*this, _Right);
			const auto Abs = _mm256_and_ps(Diff, _mm256_set1_ps(-0.0f));
			return _mm256_cmp_ps(Abs, _mm256_set1_ps(FLT_EPSILON), _CMP_LE_OQ);
		}
		else if constexpr (std::is_same_v<Type, double>)
		{
			const auto Diff = _mm256_sub_pd(*this, _Right);
			const auto Abs = _mm256_and_pd(Diff, _mm256_set1_pd(-0.0));
			return _mm256_cmp_pd(Abs, _mm256_set1_pd(DBL_EPSILON), _CMP_LE_OQ);
		}
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_cmpeq_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_cmpeq_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cmpeq_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cmpeq_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator!=(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
		{
			const auto Diff = _mm256_sub_ps(*this, _Right);
			const auto Abs = _mm256_and_ps(Diff, _mm256_set1_ps(-0.0f));
			return _mm256_cmp_ps(Abs, _mm256_set1_ps(FLT_EPSILON), _CMP_GT_OQ);
		}
		else if constexpr (std::is_same_v<Type, double>)
		{
			const auto Diff = _mm256_sub_pd(*this, _Right);
			const auto Abs = _mm256_and_pd(Diff, _mm256_set1_pd(-0.0));
			return _mm256_cmp_pd(Abs, _mm256_set1_pd(DBL_EPSILON), _CMP_GT_OQ);
		}
		else
			return _mm256_xor_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized NotEqual(const Vectorized& _Right, const Vectorized& _Mask) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Right, _CMP_NEQ_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Right, _CMP_NEQ_OQ);
		else
			return (*this == _Right).Not(_Mask);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator+(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_add_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_add_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_add_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_add_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_add_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_add_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator+=(const Vectorized& _Right)
	{
		return *this = *this + _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator-(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_sub_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_sub_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_sub_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_sub_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_sub_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_sub_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator-=(const Vectorized& _Right)
	{
		return *this = *this - _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator*(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_mul_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_mul_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t> || std::is_same_v<Type, int16_t>)
			return _mm256_mullo_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_mullo_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_mullo_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error; 
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator*=(const Vectorized& _Right)
	{
		return *this = *this * _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator/(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_div_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_div_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_div_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_div_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_div_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_div_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator/=(const Vectorized& _Right)
	{
		return *this = *this / _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator&(const Vectorized& _Right) const
	{
		return _mm256_and_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator|(const Vectorized& _Right) const
	{
		return _mm256_or_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator^(const Vectorized& _Right) const
	{
		return _mm256_xor_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Complement(const Vectorized& _Mask) const
	{
		return _mm256_xor_si256(*this, _Mask);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator<<(const int _Count) const
	{

		if constexpr (std::is_same_v<Type, float> || std::is_same_v<Type, int32_t>)
			return _mm256_slli_epi32(*this, _Count);
		else if constexpr (std::is_same_v<Type, double> || std::is_same_v<Type, int64_t>)
			return _mm256_slli_epi64(*this, _Count);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_slli_epi16(*this, _Count);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator>>(const int _Count) const
	{
		if constexpr (std::is_same_v<Type, float> || std::is_same_v<Type, int32_t>)
			return _mm256_srli_epi32(*this, _Count);
		else if constexpr (std::is_same_v<Type, double> || std::is_same_v<Type, int64_t>)
			return _mm256_srli_epi64(*this, _Count);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_srli_epi16(*this, _Count);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator&&(const Vectorized& _Right) const
	{
		return _mm256_and_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized operator||(const Vectorized& _Right) const
	{
		return _mm256_or_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Not(const Vectorized& _Mask) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cmp_ps(*this, _Mask, _CMP_EQ_OQ);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cmp_pd(*this, _Mask, _CMP_EQ_OQ);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_cmpeq_epi8(*this, _Mask);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_cmpeq_epi16(*this, _Mask);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cmpeq_epi32(*this, _Mask);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cmpeq_epi64(*this, _Mask);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Pow(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_pow_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_pow_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_pow_ps(_mm256_cvtepi32_ps(*this), _mm256_cvtepi32_ps(_Right)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_pow_pd(_mm256_cvtepi64_pd(*this), _mm256_cvtepi64_pd(_Right)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Sqrt() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_sqrt_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_sqrt_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_sqrt_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized RSqrt() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_rsqrt_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_rsqrt_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(*this)))));
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Reciprocal() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_rcp_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_rcp_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(*this)))));
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Abs(const Vectorized& _Mask) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_and_ps(*this, _Mask);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_and_pd(*this, _Mask);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_abs_epi8(*this);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_abs_epi16(*this);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_abs_epi32(*this);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_abs_epi64(*this);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;

	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Sin() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_sin_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_sin_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Cos() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cos_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cos_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Tan() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_tan_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_tan_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ASin() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_asin_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_asin_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ACos() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_acos_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_acos_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ATan() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_atan_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_atan_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ATan2(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_atan2_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_atan2_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int64_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Sinh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_sinh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_sinh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_sinh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_sinh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Cosh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_cosh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_cosh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_cosh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_cosh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Tanh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_tanh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_tanh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_tanh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_tanh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ASinh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_asinh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_asinh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_asinh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_asinh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ACosh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_acosh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_acosh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_acosh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_acosh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized ATanh() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_atanh_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_atanh_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_atanh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_atanh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Log() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_log_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_log_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_log_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_log_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Log2() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_log2_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_log2_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_log2_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_log2_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Log10() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_log10_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_log10_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_log10_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_log10_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Exp() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_exp_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_exp_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_exp_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_exp_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Exp2() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_exp2_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_exp2_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_exp2_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_exp2_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Exp10() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_exp10_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_exp10_pd(*this);
		else if constexpr (std::is_same_v<Type, int8_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int16_t>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_cvtps_epi32(_mm256_exp10_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_cvtpd_epi64(_mm256_exp10_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Ceil() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_ceil_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_ceil_pd(*this);
		else
			return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Floor() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_floor_ps(*this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_floor_pd(*this);
		else
			return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Round() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_round_ps(*this, _MM_FROUND_TO_NEAREST_INT);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_round_pd(*this, _MM_FROUND_TO_NEAREST_INT);
		else
			return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Trunc() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_round_ps(*this, _MM_FROUND_TO_ZERO);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_round_pd(*this, _MM_FROUND_TO_ZERO);
		else
			return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Frac() const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_sub_ps(*this, _mm256_floor_ps(*this));
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_sub_pd(*this, _mm256_floor_pd(*this));
		else
			return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Min(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_min_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_min_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_min_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_min_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_min_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_min_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Max(const Vectorized& _Right) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_max_ps(*this, _Right);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_max_pd(*this, _Right);
		else if constexpr (std::is_same_v<Type, int8_t>)
			return _mm256_max_epi8(*this, _Right);
		else if constexpr (std::is_same_v<Type, int16_t>)
			return _mm256_max_epi16(*this, _Right);
		else if constexpr (std::is_same_v<Type, int32_t>)
			return _mm256_max_epi32(*this, _Right);
		else if constexpr (std::is_same_v<Type, int64_t>)
			return _mm256_max_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Lerp(const Vectorized& _Right, const Vectorized& _Alpha) const
	{
		if constexpr (std::is_same_v<Type, float>)
			return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_Right, *this), _Alpha), *this);
		else if constexpr (std::is_same_v<Type, double>)
			return _mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(_Right, *this), _Alpha), *this);
		else
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vectorized Clamp(const Vectorized& _Min, const Vectorized& _Max) const
	{
		return Max(_Min).Min(_Max);
	}

private:
	__m256i _YMMX;


public:
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void DragonianLibMemcpy256_8(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
		const __m256i m0 = _mm256_load_si256(_Src + 0);
		const __m256i m1 = _mm256_load_si256(_Src + 1);
		const __m256i m2 = _mm256_load_si256(_Src + 2);
		const __m256i m3 = _mm256_load_si256(_Src + 3);
		const __m256i m4 = _mm256_load_si256(_Src + 4);
		const __m256i m5 = _mm256_load_si256(_Src + 5);
		const __m256i m6 = _mm256_load_si256(_Src + 6);
		const __m256i m7 = _mm256_load_si256(_Src + 7);
		_mm256_store_si256(_Dst + 0, m0);
		_mm256_store_si256(_Dst + 1, m1);
		_mm256_store_si256(_Dst + 2, m2);
		_mm256_store_si256(_Dst + 3, m3);
		_mm256_store_si256(_Dst + 4, m4);
		_mm256_store_si256(_Dst + 5, m5);
		_mm256_store_si256(_Dst + 6, m6);
		_mm256_store_si256(_Dst + 7, m7);
	}
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void DragonianLibMemcpy256_4(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
		const __m256i m0 = _mm256_load_si256(_Src + 0);
		const __m256i m1 = _mm256_load_si256(_Src + 1);
		const __m256i m2 = _mm256_load_si256(_Src + 2);
		const __m256i m3 = _mm256_load_si256(_Src + 3);
		_mm256_store_si256(_Dst + 0, m0);
		_mm256_store_si256(_Dst + 1, m1);
		_mm256_store_si256(_Dst + 2, m2);
		_mm256_store_si256(_Dst + 3, m3);
	}
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void DragonianLibMemcpy256_2(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
		const __m256i m0 = _mm256_load_si256(_Src + 0);
		const __m256i m1 = _mm256_load_si256(_Src + 1);
		_mm256_store_si256(_Dst + 0, m0);
		_mm256_store_si256(_Dst + 1, m1);
	}
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void DragonianLibMemcpy256_1(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
		const __m256i m0 = _mm256_load_si256(_Src + 0);
		_mm256_store_si256(_Dst + 0, m0);
	}

	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void DragonianLibMemCpy(void* const __restrict _Dst, const void* const __restrict _Src, size_t _Size)
	{
		__m256i* __restrict _Dst_Ptr = (__m256i*)_Dst;
		const __m256i* __restrict _Src_Ptr = (const __m256i*)_Src;
		while (true)
		{
			if (!(_Size >> 8))
				break;
			DragonianLibMemcpy256_8(_Dst_Ptr, _Src_Ptr);
			_Dst_Ptr += 8;
			_Src_Ptr += 8;
			_Size -= alignof(__m256i) * 8;
		}
		if (_Size >> 7)
		{
			DragonianLibMemcpy256_4(_Dst_Ptr, _Src_Ptr);
			_Dst_Ptr += 4;
			_Src_Ptr += 4;
			_Size -= alignof(__m256i) * 4;
		}
		if (_Size >> 6)
		{
			DragonianLibMemcpy256_2(_Dst_Ptr, _Src_Ptr);
			_Dst_Ptr += 2;
			_Src_Ptr += 2;
			_Size -= alignof(__m256i) * 2;
		}
		if (_Size >> 5)
		{
			DragonianLibMemcpy256_1(_Dst_Ptr, _Src_Ptr);
			++_Dst_Ptr;
			++_Src_Ptr;
			_Size -= alignof(__m256i);
		}
		if (_Size)
			memcpy(_Dst_Ptr, _Src_Ptr, _Size);
	}
};

template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Is_Vectorized_v = false;
template <typename Type>
constexpr bool _Impl_Dragonian_Lib_Is_Vectorized_v<Vectorized<Type>> = true;

_D_Dragonian_Lib_Operator_Space_End