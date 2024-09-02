#include "../../header/InferTools/inferTools.hpp"
#include "string"
#ifdef LIBSVC_FLOAT_TENSOR_AVX_WRP
#include <immintrin.h>
#endif

LibSvcHeader

DragonianLibSTL::Vector<size_t> SliceAudio(
    const DragonianLibSTL::Vector<int16_t>& _PcmData,
    const SlicerSettings& _SlicerSettings
)
{
    if (_PcmData.Size() < size_t(_SlicerSettings.MinLength) * _SlicerSettings.SamplingRate)
        return { 0, _PcmData.Size() };

    DragonianLibSTL::Vector<unsigned long long> slice_point;
    bool slice_tag = true;
    slice_point.EmplaceBack(0);

    unsigned long CurLength = 0;
    for (size_t i = 0; i + _SlicerSettings.WindowLength < _PcmData.Size(); i += _SlicerSettings.HopSize)
    {

        if (slice_tag)
        {
            const auto vol = abs(DragonianLibSTL::Average(_PcmData.Data() + i, _PcmData.Data() + i + _SlicerSettings.WindowLength));
            if (vol < _SlicerSettings.Threshold)
            {
                slice_tag = false;
                if (CurLength > _SlicerSettings.MinLength * _SlicerSettings.SamplingRate)
                {
                    CurLength = 0;
                    slice_point.EmplaceBack(i + (_SlicerSettings.WindowLength / 2));
                }
            }
            else
                slice_tag = true;
        }
        else
        {
            const auto vol = abs(DragonianLibSTL::Average(_PcmData.Data() + i, _PcmData.Data() + i + _SlicerSettings.WindowLength));
            if (vol < _SlicerSettings.Threshold)
                slice_tag = false;
            else
            {
                slice_tag = true;
                if (CurLength > _SlicerSettings.MinLength * _SlicerSettings.SamplingRate)
                {
                    CurLength = 0;
                    slice_point.EmplaceBack(i + (_SlicerSettings.WindowLength / 2));
                }
            }
        }
        CurLength += _SlicerSettings.HopSize;
    }
    slice_point.EmplaceBack(_PcmData.Size());
    return slice_point;
}

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

LibSvcEnd