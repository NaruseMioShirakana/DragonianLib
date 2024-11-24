#include "../../header/InferTools/inferTools.hpp"
#include "string"
#ifdef LIBSVC_FLOAT_TENSOR_AVX_WRP
#include <immintrin.h>
#endif

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

#ifdef LIBSVC_FLOAT_TENSOR_AVX_WRP
FloatTensorWrapper& FloatTensorWrapper::operator+=(const FloatTensorWrapper& _right)
{
    if (_data_size != _right._data_size)
        LibDLVoiceCodecThrow("Vector Size MisMatch");
    const size_t num_avx2_elements = _data_size / 8;
    for (size_t i = 0; i < num_avx2_elements; i++) {
        const __m256 a_avx2 = _mm256_load_ps(&_data_ptr[i * 8]);
        const __m256 b_avx2 = _mm256_load_ps(&_right[i * 8]);
        const __m256 result_avx2 = _mm256_add_ps(a_avx2, b_avx2);
        _mm256_store_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] += _right[i];
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator-=(const FloatTensorWrapper& _right)
{
    if (_data_size != _right._data_size)
        LibDLVoiceCodecThrow("Vector Size MisMatch");
    const size_t num_avx2_elements = _data_size / 8;
    for (size_t i = 0; i < num_avx2_elements; i++) {
        const __m256 a_avx2 = _mm256_load_ps(&_data_ptr[i * 8]);
        const __m256 b_avx2 = _mm256_load_ps(&_right[i * 8]);
        const __m256 result_avx2 = _mm256_sub_ps(a_avx2, b_avx2);
        _mm256_store_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] -= _right[i];
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator*=(const FloatTensorWrapper& _right)
{
    if (_data_size != _right._data_size)
        LibDLVoiceCodecThrow("Vector Size MisMatch");
    const size_t num_avx2_elements = _data_size / 8;
    for (size_t i = 0; i < num_avx2_elements; i++) {
        const __m256 a_avx2 = _mm256_load_ps(&_data_ptr[i * 8]);
        const __m256 b_avx2 = _mm256_load_ps(&_right[i * 8]);
        const __m256 result_avx2 = _mm256_mul_ps(a_avx2, b_avx2);
        _mm256_store_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] *= _right[i];
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator/=(const FloatTensorWrapper& _right)
{
    if (_data_size != _right._data_size)
        LibDLVoiceCodecThrow("Vector Size MisMatch");
    const size_t num_avx2_elements = _data_size / 8;
    for (size_t i = 0; i < num_avx2_elements; i++) {
        const __m256 a_avx2 = _mm256_load_ps(&_data_ptr[i * 8]);
        const __m256 b_avx2 = _mm256_load_ps(&_right[i * 8]);
        const __m256 result_avx2 = _mm256_div_ps(a_avx2, b_avx2);
        _mm256_store_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] /= _right[i];
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator+=(float _right)
{
    const size_t num_avx2_elements = _data_size / 8;
    const __m256 value_avx2 = _mm256_set1_ps(_right);
    for (size_t i = 0; i < num_avx2_elements; ++i) {
        const __m256 vec_avx2 = _mm256_loadu_ps(&_data_ptr[i * 8]);
        const __m256 result_avx2 = _mm256_add_ps(vec_avx2, value_avx2);
        _mm256_storeu_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] += _right;
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator-=(float _right)
{
    const size_t num_avx2_elements = _data_size / 8;
    const __m256 value_avx2 = _mm256_set1_ps(_right);
    for (size_t i = 0; i < num_avx2_elements; ++i) {
        const __m256 vec_avx2 = _mm256_loadu_ps(&_data_ptr[i * 8]);
        const __m256 result_avx2 = _mm256_sub_ps(vec_avx2, value_avx2);
        _mm256_storeu_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] -= _right;
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator*=(float _right)
{
    const size_t num_avx2_elements = _data_size / 8;
    const __m256 value_avx2 = _mm256_set1_ps(_right);
    for (size_t i = 0; i < num_avx2_elements; ++i) {
        const __m256 vec_avx2 = _mm256_loadu_ps(&_data_ptr[i * 8]);
        const __m256 result_avx2 = _mm256_mul_ps(vec_avx2, value_avx2);
        _mm256_storeu_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] *= _right;
    return *this;
}

FloatTensorWrapper& FloatTensorWrapper::operator/=(float _right)
{
    const size_t num_avx2_elements = _data_size / 8;
    const __m256 value_avx2 = _mm256_set1_ps(_right);
    for (size_t i = 0; i < num_avx2_elements; ++i) {
        const __m256 vec_avx2 = _mm256_loadu_ps(&_data_ptr[i * 8]);
        const __m256 result_avx2 = _mm256_div_ps(vec_avx2, value_avx2);
        _mm256_storeu_ps(&_data_ptr[i * 8], result_avx2);
    }
    for (size_t i = num_avx2_elements * 8; i < _data_size; ++i)
        _data_ptr[i] /= _right;
    return *this;
}
#endif

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End