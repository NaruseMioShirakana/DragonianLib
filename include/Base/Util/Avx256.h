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

static INLINE void LibSvcMemcpy256(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_loadu_si256(_Src + 0);
	const __m256i m1 = _mm256_loadu_si256(_Src + 1);
	const __m256i m2 = _mm256_loadu_si256(_Src + 2);
	const __m256i m3 = _mm256_loadu_si256(_Src + 3);
	const __m256i m4 = _mm256_loadu_si256(_Src + 4);
	const __m256i m5 = _mm256_loadu_si256(_Src + 5);
	const __m256i m6 = _mm256_loadu_si256(_Src + 6);
	const __m256i m7 = _mm256_loadu_si256(_Src + 7);
	_mm256_storeu_si256(_Dst + 0, m0);
	_mm256_storeu_si256(_Dst + 1, m1);
	_mm256_storeu_si256(_Dst + 2, m2);
	_mm256_storeu_si256(_Dst + 3, m3);
	_mm256_storeu_si256(_Dst + 4, m4);
	_mm256_storeu_si256(_Dst + 5, m5);
	_mm256_storeu_si256(_Dst + 6, m6);
	_mm256_storeu_si256(_Dst + 7, m7);
}

static INLINE void LibSvcMemcpy128(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_loadu_si256(_Src + 0);
	const __m256i m1 = _mm256_loadu_si256(_Src + 1);
	const __m256i m2 = _mm256_loadu_si256(_Src + 2);
	const __m256i m3 = _mm256_loadu_si256(_Src + 3);
	_mm256_storeu_si256(_Dst + 0, m0);
	_mm256_storeu_si256(_Dst + 1, m1);
	_mm256_storeu_si256(_Dst + 2, m2);
	_mm256_storeu_si256(_Dst + 3, m3);
}

static INLINE void LibSvcMemcpy64(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_loadu_si256(_Src + 0);
	const __m256i m1 = _mm256_loadu_si256(_Src + 1);
	_mm256_storeu_si256(_Dst + 0, m0);
	_mm256_storeu_si256(_Dst + 1, m1);
}

static INLINE void LibSvcMemcpy32(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_loadu_si256(_Src + 0);
	_mm256_storeu_si256(_Dst + 0, m0);
}

static INLINE void LibSvcMemCpy(void* const __restrict _Dst, const void* const __restrict _Src, size_t _Size)
{
	unsigned char* __restrict _Dst_Ptr = (unsigned char*)_Dst;
	const unsigned char* __restrict _Src_Ptr = (const unsigned char*)_Src;
	while (true)
	{
		if (!(_Size >> 8))
			break;
		LibSvcMemcpy256((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 256;
		_Src_Ptr += 256;
		_Size -= 256;
	}
	if (_Size >> 7)
	{
		LibSvcMemcpy128((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 128;
		_Src_Ptr += 128;
		_Size -= 128;
	}
	if (_Size >> 6)
	{
		LibSvcMemcpy64((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 64;
		_Src_Ptr += 64;
		_Size -= 64;
	}
	if (_Size >> 5)
	{
		LibSvcMemcpy32((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Ptr);
		_Dst_Ptr += 32;
		_Src_Ptr += 32;
		_Size -= 32;
	}
	if(_Size)
		memcpy(_Dst_Ptr, _Src_Ptr, _Size);
}

static INLINE void LibSvcMemSet(void* const __restrict _Dst, const void* const __restrict _Src, size_t _BufferSize, size_t _AlignSize)
{
	unsigned char* __restrict _Dst_Ptr = (unsigned char*)_Dst;

	if (256 % _AlignSize != 0)
		return;
	unsigned char _Src_Base[256];
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
		LibSvcMemcpy256((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 256;
		_BufferSize -= 256;
	}
	if (_BufferSize >> 7)
	{
		LibSvcMemcpy128((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 128;
		_BufferSize -= 128;
	}
	if (_BufferSize >> 6)
	{
		LibSvcMemcpy64((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 64;
		_BufferSize -= 64;
	}
	if (_BufferSize >> 5)
	{
		LibSvcMemcpy32((__m256i*)_Dst_Ptr, (const __m256i*)_Src_Base);
		_Dst_Ptr += 32;
		_BufferSize -= 32;
	}
	if (_BufferSize)
		memcpy(_Dst_Ptr, _Src_Base, _BufferSize);
}