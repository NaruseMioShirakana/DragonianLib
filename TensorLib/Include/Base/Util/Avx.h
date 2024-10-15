/**
 * FileName: Avx.h
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

#ifndef INLINE
#ifdef __GNUC__
#if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1))
#define INLINE         __inline__ __attribute__((always_inline))
#else
#define INLINE         __inline__
#endif
#elif defined(_MSC_VER)
#define INLINE __forceinline
#elif (defined(__BORLANDC__) || defined(__WATCOMC__))
#define INLINE __inline
#else
#define INLINE 
#endif
#endif

#ifndef ALIGN_ALLOC
#ifdef _MSC_VER
#define ALIGN_ALLOC(_Size, _Alig) _aligned_malloc(_Size, _Alig)
#else
#error
#endif
#endif
#ifndef ALIGN_FREE
#ifdef _MSC_VER
#define ALIGN_FREE(_Ptr) _aligned_free(_Ptr)
#else
#error
#endif
#endif

static INLINE void DragonianLibMemcpy256(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
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

static INLINE void DragonianLibMemcpy128(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	const __m256i m1 = _mm256_load_si256(_Src + 1);
	const __m256i m2 = _mm256_load_si256(_Src + 2);
	const __m256i m3 = _mm256_load_si256(_Src + 3);
	_mm256_store_si256(_Dst + 0, m0);
	_mm256_store_si256(_Dst + 1, m1);
	_mm256_store_si256(_Dst + 2, m2);
	_mm256_store_si256(_Dst + 3, m3);
}

static INLINE void DragonianLibMemcpy64(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	const __m256i m1 = _mm256_load_si256(_Src + 1);
	_mm256_store_si256(_Dst + 0, m0);
	_mm256_store_si256(_Dst + 1, m1);
}

static INLINE void DragonianLibMemcpy32(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	_mm256_store_si256(_Dst + 0, m0);
}

static INLINE void DragonianLibMemCpy(void* const __restrict _Dst, const void* const __restrict _Src, size_t _Size)
{
	unsigned char* __restrict _Dst_Ptr = (unsigned char*)_Dst;
	const unsigned char* __restrict _Src_Ptr = (const unsigned char*)_Src;
	while (true)
	{
		if (!(_Size >> 8))
			break;
		DragonianLibMemcpy256((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 256;
		_Src_Ptr += 256;
		_Size -= 256;
	}
	if (_Size >> 7)
	{
		DragonianLibMemcpy128((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 128;
		_Src_Ptr += 128;
		_Size -= 128;
	}
	if (_Size >> 6)
	{
		DragonianLibMemcpy64((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 64;
		_Src_Ptr += 64;
		_Size -= 64;
	}
	if (_Size >> 5)
	{
		DragonianLibMemcpy32((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 32;
		_Src_Ptr += 32;
		_Size -= 32;
	}
	if (_Size)
		memcpy(_Dst_Ptr, _Src_Ptr, _Size);
}

static INLINE void DragonianLibMemSet(void* const __restrict _Dst, const void* const __restrict _Src, size_t _BufferSize, size_t _AlignSize)
{
	unsigned char* __restrict _Dst_Ptr = (unsigned char*)_Dst;

	if (256 % _AlignSize != 0)
		return;
	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(256, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 256;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, _Src, _AlignSize);
		_Src_Ptr += _AlignSize;
	}

	while (true)
	{
		if (!(_BufferSize >> 8))
			break;
		DragonianLibMemcpy256((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 256;
		_BufferSize -= 256;
	}
	if (_BufferSize >> 7)
	{
		DragonianLibMemcpy128((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 128;
		_BufferSize -= 128;
	}
	if (_BufferSize >> 6)
	{
		DragonianLibMemcpy64((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 64;
		_BufferSize -= 64;
	}
	if (_BufferSize >> 5)
	{
		DragonianLibMemcpy32((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 32;
		_BufferSize -= 32;
	}
	if (_BufferSize)
		memcpy(_Dst_Ptr, _Src_Base, _BufferSize);
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
_Ty DragonianLibAddFn(_Ty a, _Ty b)
{
	return a + b;
}

template <typename _Ty>
_Ty DragonianLibSubFn(_Ty a, _Ty b)
{
	return a - b;
}

template <typename _Ty>
_Ty DragonianLibMulFn(_Ty a, _Ty b)
{
	return a * b;
}

template <typename _Ty>
_Ty DragonianLibDivFn(_Ty a, _Ty b)
{
	return a / b;
}

template <typename _Ty>
_Ty DragonianLibModFn(_Ty a, _Ty b)
{
	return a % b;
}

template <typename _Ty>
_Ty DragonianLibExp10(_Ty inp)
{
	return (_Ty)pow((_Ty)10, inp);
}

template <typename _Ty>
void DragonianLibVectorAdd(_Ty* _Dst, const _Ty* _SrcA, const _Ty* _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty) * 2;
	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2_1 = _mm256_load_si256((const __m256i*)_SrcA);
		const __m256i b_avx2_1 = _mm256_load_si256((const __m256i*)_SrcB);
		const __m256i a_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcA) + 1);
		const __m256i b_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcB) + 1);
		__m256i result_avx2_1, result_avx2_2;
		if (std::is_same_v<_Ty, float>)
		{
			auto res = _mm256_add_ps(*(const __m256*)(&a_avx2_1), *(const __m256*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_add_ps(*(const __m256*)(&a_avx2_2), *(const __m256*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			auto res = _mm256_add_pd(*(const __m256d*)(&a_avx2_1), *(const __m256d*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_add_pd(*(const __m256d*)(&a_avx2_2), *(const __m256d*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
		{
			result_avx2_1 = _mm256_add_epi8(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_add_epi8(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int16_t>)
		{
			result_avx2_1 = _mm256_add_epi16(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_add_epi16(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int32_t>)
		{
			result_avx2_1 = _mm256_add_epi32(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_add_epi32(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int64_t>)
		{
			result_avx2_1 = _mm256_add_epi64(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_add_epi64(a_avx2_2, b_avx2_2);
		}
		_mm256_store_si256((__m256i*)_Dst, result_avx2_1);
		_mm256_store_si256(((__m256i*)_Dst) + 1, result_avx2_2);
		_Dst += Stride;
		_SrcA += Stride;
		_SrcB += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) + *(_SrcB++);
}

template <typename _Ty>
void DragonianLibVectorSub(_Ty* _Dst, const _Ty* _SrcA, const _Ty* _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty) * 2;
	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2_1 = _mm256_load_si256((const __m256i*)_SrcA);
		const __m256i b_avx2_1 = _mm256_load_si256((const __m256i*)_SrcB);
		const __m256i a_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcA) + 1);
		const __m256i b_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcB) + 1);
		__m256i result_avx2_1, result_avx2_2;
		if (std::is_same_v<_Ty, float>)
		{
			auto res = _mm256_sub_ps(*(const __m256*)(&a_avx2_1), *(const __m256*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_sub_ps(*(const __m256*)(&a_avx2_2), *(const __m256*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			auto res = _mm256_sub_pd(*(const __m256d*)(&a_avx2_1), *(const __m256d*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_sub_pd(*(const __m256d*)(&a_avx2_2), *(const __m256d*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
		{
			result_avx2_1 = _mm256_sub_epi8(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_sub_epi8(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int16_t>)
		{
			result_avx2_1 = _mm256_sub_epi16(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_sub_epi16(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int32_t>)
		{
			result_avx2_1 = _mm256_sub_epi32(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_sub_epi32(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int64_t>)
		{
			result_avx2_1 = _mm256_sub_epi64(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_sub_epi64(a_avx2_2, b_avx2_2);
		}
		_mm256_store_si256((__m256i*)_Dst, result_avx2_1);
		_mm256_store_si256(((__m256i*)_Dst) + 1, result_avx2_2);
		_Dst += Stride;
		_SrcA += Stride;
		_SrcB += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) - *(_SrcB++);
}

template <typename _Ty>
void DragonianLibVectorMul(_Ty* _Dst, const _Ty* _SrcA, const _Ty* _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty) * 2;
	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2_1 = _mm256_load_si256((const __m256i*)_SrcA);
		const __m256i b_avx2_1 = _mm256_load_si256((const __m256i*)_SrcB);
		const __m256i a_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcA) + 1);
		const __m256i b_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcB) + 1);
		__m256i result_avx2_1, result_avx2_2;
		if (std::is_same_v<_Ty, float>)
		{
			auto res = _mm256_mul_ps(*(const __m256*)(&a_avx2_1), *(const __m256*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_mul_ps(*(const __m256*)(&a_avx2_2), *(const __m256*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			auto res = _mm256_mul_pd(*(const __m256d*)(&a_avx2_1), *(const __m256d*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_mul_pd(*(const __m256d*)(&a_avx2_2), *(const __m256d*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
		{
			break;
		}
		else if (std::is_same_v<_Ty, int16_t>)
		{
			result_avx2_1 = _mm256_mullo_epi16(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_mullo_epi16(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int32_t>)
		{
			result_avx2_1 = _mm256_mullo_epi32(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_mullo_epi32(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int64_t>)
		{
			result_avx2_1 = _mm256_mullo_epi64(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_mullo_epi64(a_avx2_2, b_avx2_2);
		}
		_mm256_store_si256((__m256i*)_Dst, result_avx2_1);
		_mm256_store_si256(((__m256i*)_Dst) + 1, result_avx2_2);
		_Dst += Stride;
		_SrcA += Stride;
		_SrcB += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) * *(_SrcB++);
}

template <typename _Ty>
void DragonianLibVectorDiv(_Ty* _Dst, const _Ty* _SrcA, const _Ty* _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty) * 2;
	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2_1 = _mm256_load_si256((const __m256i*)_SrcA);
		const __m256i b_avx2_1 = _mm256_load_si256((const __m256i*)_SrcB);
		const __m256i a_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcA) + 1);
		const __m256i b_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcB) + 1);
		__m256i result_avx2_1, result_avx2_2;
		if (std::is_same_v<_Ty, float>)
		{
			auto res = _mm256_div_ps(*(const __m256*)(&a_avx2_1), *(const __m256*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_div_ps(*(const __m256*)(&a_avx2_2), *(const __m256*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			auto res = _mm256_div_pd(*(const __m256d*)(&a_avx2_1), *(const __m256d*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_div_pd(*(const __m256d*)(&a_avx2_2), *(const __m256d*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
		{
			result_avx2_1 = _mm256_div_epi8(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_div_epi8(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int16_t>)
		{
			result_avx2_1 = _mm256_div_epi16(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_div_epi16(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int32_t>)
		{
			result_avx2_1 = _mm256_div_epi32(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_div_epi32(a_avx2_2, b_avx2_2);
		}
		else if (std::is_same_v<_Ty, int64_t>)
		{
			result_avx2_1 = _mm256_div_epi64(a_avx2_1, b_avx2_1);
			result_avx2_2 = _mm256_div_epi64(a_avx2_2, b_avx2_2);
		}
		_mm256_store_si256((__m256i*)_Dst, result_avx2_1);
		_mm256_store_si256(((__m256i*)_Dst) + 1, result_avx2_2);
		_Dst += Stride;
		_SrcA += Stride;
		_SrcB += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) / *(_SrcB++);
}

template <typename _Ty>
void DragonianLibVectorAddScalar(_Ty* _Dst, const _Ty* _SrcA, const _Ty _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(32, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 32;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, &_SrcB, sizeof(_Ty));
		_Src_Ptr += sizeof(_Ty);
	}
	const __m256i b_avx2 = _mm256_load_si256((const __m256i*)_Src_Base);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2 = _mm256_load_si256((const __m256i*)_SrcA);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_add_ps(*(const __m256*)(&a_avx2), *(const __m256*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_add_pd(*(const __m256d*)(&a_avx2), *(const __m256d*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			result_avx2 = _mm256_add_epi8(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int16_t>)
			result_avx2 = _mm256_add_epi16(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int32_t>)
			result_avx2 = _mm256_add_epi32(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int64_t>)
			result_avx2 = _mm256_add_epi64(a_avx2, b_avx2);
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_SrcA += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) + _SrcB;
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
void DragonianLibVectorSubScalar(_Ty* _Dst, const _Ty* _SrcA, const _Ty _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(32, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 32;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, &_SrcB, sizeof(_Ty));
		_Src_Ptr += sizeof(_Ty);
	}
	const __m256i b_avx2 = _mm256_load_si256((const __m256i*)_Src_Base);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2 = _mm256_load_si256((const __m256i*)_SrcA);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_sub_ps(*(const __m256*)(&a_avx2), *(const __m256*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_sub_pd(*(const __m256d*)(&a_avx2), *(const __m256d*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			result_avx2 = _mm256_sub_epi8(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int16_t>)
			result_avx2 = _mm256_sub_epi16(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int32_t>)
			result_avx2 = _mm256_sub_epi32(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int64_t>)
			result_avx2 = _mm256_sub_epi64(a_avx2, b_avx2);
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_SrcA += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) - _SrcB;
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
void DragonianLibVectorMulScalar(_Ty* _Dst, const _Ty* _SrcA, const _Ty _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(32, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 32;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, &_SrcB, sizeof(_Ty));
		_Src_Ptr += sizeof(_Ty);
	}
	const __m256i b_avx2 = _mm256_load_si256((const __m256i*)_Src_Base);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2 = _mm256_load_si256((const __m256i*)_SrcA);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_mul_ps(*(const __m256*)(&a_avx2), *(const __m256*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_mul_pd(*(const __m256d*)(&a_avx2), *(const __m256d*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			result_avx2 = _mm256_mullo_epi16(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int32_t>)
			result_avx2 = _mm256_mullo_epi32(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int64_t>)
			result_avx2 = _mm256_mullo_epi64(a_avx2, b_avx2);
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_SrcA += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) * _SrcB;
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
void DragonianLibVectorDivScalar(_Ty* _Dst, const _Ty* _SrcA, const _Ty _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(32, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 32;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, &_SrcB, sizeof(_Ty));
		_Src_Ptr += sizeof(_Ty);
	}
	const __m256i b_avx2 = _mm256_load_si256((const __m256i*)_Src_Base);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2 = _mm256_load_si256((const __m256i*)_SrcA);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_div_ps(*(const __m256*)(&a_avx2), *(const __m256*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_div_pd(*(const __m256d*)(&a_avx2), *(const __m256d*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			result_avx2 = _mm256_div_epi8(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int16_t>)
			result_avx2 = _mm256_div_epi16(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int32_t>)
			result_avx2 = _mm256_div_epi32(a_avx2, b_avx2);
		else if (std::is_same_v<_Ty, int64_t>)
			result_avx2 = _mm256_div_epi64(a_avx2, b_avx2);
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_SrcA += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = *(_SrcA++) / _SrcB;
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
void DragonianLibVectorAbs(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
			break;
		else if (std::is_same_v<_Ty, double>)
			break;
		else if (std::is_same_v<_Ty, int8_t>)
			result_avx2 = _mm256_abs_epi8(input_avx2);
		else if (std::is_same_v<_Ty, int16_t>)
			result_avx2 = _mm256_abs_epi16(input_avx2);
		else if (std::is_same_v<_Ty, int32_t>)
			result_avx2 = _mm256_abs_epi32(input_avx2);
		else if (std::is_same_v<_Ty, int64_t>)
			result_avx2 = _mm256_abs_epi64(input_avx2);
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)abs(*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorSin(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_sin_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_sin_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)sin((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorSinh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_sinh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_sinh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)sinh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorCos(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_cos_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_cos_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)cos((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorCosh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_cosh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_cosh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)cosh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorTan(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_tan_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_tan_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)tan((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorTanh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_tanh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_tanh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)tanh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorASin(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_asin_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_asin_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)asin((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorACos(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_acos_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_acos_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)acos((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorATan(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_atan_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_atan_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)atan((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorASinh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_asinh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_asinh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)asinh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorACosh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_acosh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_acosh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)acosh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorATanh(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_atanh_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_atanh_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)atanh((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorExp(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_exp_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_exp_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)exp((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorExp10(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_exp10_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_exp10_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)DragonianLibExp10((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorExp2(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_exp2_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_exp2_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)exp2((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorLog(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_log_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_log_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)log((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorLog10(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_log10_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_log10_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)log10((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorLog2(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_log2_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_log2_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)log2((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorPow(_Ty* _Dst, const _Ty* _SrcA, const _Ty* _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty) * 2;
	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2_1 = _mm256_load_si256((const __m256i*)_SrcA);
		const __m256i b_avx2_1 = _mm256_load_si256((const __m256i*)_SrcB);
		const __m256i a_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcA) + 1);
		const __m256i b_avx2_2 = _mm256_load_si256(((const __m256i*)_SrcB) + 1);
		__m256i result_avx2_1, result_avx2_2;
		if (std::is_same_v<_Ty, float>)
		{
			auto res = _mm256_pow_ps(*(const __m256*)(&a_avx2_1), *(const __m256*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_pow_ps(*(const __m256*)(&a_avx2_2), *(const __m256*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			auto res = _mm256_pow_pd(*(const __m256d*)(&a_avx2_1), *(const __m256d*)(&b_avx2_1));
			result_avx2_1 = *(const __m256i*)(&res);
			res = _mm256_pow_pd(*(const __m256d*)(&a_avx2_2), *(const __m256d*)(&b_avx2_2));
			result_avx2_2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
		{
			break;
		}
		else if (std::is_same_v<_Ty, int16_t>)
		{
			break;
		}
		else if (std::is_same_v<_Ty, int32_t>)
		{
			break;
		}
		else if (std::is_same_v<_Ty, int64_t>)
		{
			break;
		}
		_mm256_store_si256((__m256i*)_Dst, result_avx2_1);
		_mm256_store_si256(((__m256i*)_Dst) + 1, result_avx2_2);
		_Dst += Stride;
		_SrcA += Stride;
		_SrcB += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)pow(*(_SrcA++), *(_SrcB++));
}

template <typename _Ty>
void DragonianLibVectorPowScalar(_Ty* _Dst, const _Ty* _SrcA, const _Ty _SrcB, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	unsigned char* _Src_Base = (unsigned char*)ALIGN_ALLOC(32, 32);
	unsigned char* __restrict _Src_Ptr = _Src_Base;
	const unsigned char* const _Src_Ptr_End = _Src_Base + 32;
	while (_Src_Ptr != _Src_Ptr_End)
	{
		memcpy(_Src_Ptr, &_SrcB, sizeof(_Ty));
		_Src_Ptr += sizeof(_Ty);
	}
	const __m256i b_avx2 = _mm256_load_si256((const __m256i*)_Src_Base);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i a_avx2 = _mm256_load_si256((const __m256i*)_SrcA);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_pow_ps(*(const __m256*)(&a_avx2), *(const __m256*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_pow_pd(*(const __m256d*)(&a_avx2), *(const __m256d*)(&b_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_SrcA += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)pow(*(_SrcA++), _SrcB);
	ALIGN_FREE(_Src_Base);
}

template <typename _Ty>
void DragonianLibVectorCeil(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_ceil_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_ceil_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)ceil((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorFloor(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_floor_ps(*(const __m256*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_floor_pd(*(const __m256d*)(&input_avx2));
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)floor((double)*(_Src++));
}

template <typename _Ty>
void DragonianLibVectorRound(_Ty* _Dst, const _Ty* _Src, size_t _DataSize)
{
	constexpr size_t Stride = alignof(__m256) / sizeof(_Ty);

	while (true)
	{
		if (_DataSize < Stride)
			break;
		const __m256i input_avx2 = _mm256_load_si256((const __m256i*)_Src);
		__m256i result_avx2;
		if (std::is_same_v<_Ty, float>)
		{
			const auto res = _mm256_round_ps(*(const __m256*)(&input_avx2), _MM_ROUND_MODE_DEFAULT);
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, double>)
		{
			const auto res = _mm256_round_pd(*(const __m256d*)(&input_avx2), _MM_ROUND_MODE_DEFAULT);
			result_avx2 = *(const __m256i*)(&res);
		}
		else if (std::is_same_v<_Ty, int8_t>)
			break;
		else if (std::is_same_v<_Ty, int16_t>)
			break;
		else if (std::is_same_v<_Ty, int32_t>)
			break;
		else if (std::is_same_v<_Ty, int64_t>)
			break;
		_mm256_store_si256((__m256i*)_Dst, result_avx2);
		_Dst += Stride;
		_Src += Stride;
		_DataSize -= Stride;
	}
	for (size_t i = 0; i < _DataSize; ++i)
		*(_Dst++) = (_Ty)round((double)*(_Src++));
}