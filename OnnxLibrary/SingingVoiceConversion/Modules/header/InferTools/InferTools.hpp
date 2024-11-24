/**
 * FileName: InferTools.hpp
 * Note: MoeVoiceStudioCore InferTools
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
 * date: 2022-10-17 Create
*/

#pragma once
#include "MyTemplateLibrary/Vector.h"

#define _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header namespace DragonianLib { namespace SingingVoiceConversion { 
#define _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End } }
#define _D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Space ::DragonianLib::SingingVoiceConversion::
#define _D_Dragonian_Lib_Lib_Av_Codec_Space ::DragonianLib::AvCodec::

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header
#ifdef LIBSVC_FLOAT_TENSOR_AVX_WRP
class FloatTensorWrapper
{
public:
	FloatTensorWrapper() = delete;
	~FloatTensorWrapper() { _data_ptr = nullptr; }
	FloatTensorWrapper(float* const data_p, size_t _size) : _data_ptr(data_p), _data_size(_size) {}
	FloatTensorWrapper(const FloatTensorWrapper& _copy) = delete;
	FloatTensorWrapper& operator=(const FloatTensorWrapper&) = delete;
	FloatTensorWrapper(FloatTensorWrapper&& _move) noexcept :_data_ptr(_move._data_ptr), _data_size(_move._data_size) {}
	FloatTensorWrapper& operator=(FloatTensorWrapper&& _move) noexcept
	{
		_data_ptr = _move._data_ptr;
		_data_size = _move._data_size;
		return *this;
	}
	template<typename T>
	static const T& Min(const T& a, const T& b) { return (a > b) ? b : a; }
	float& operator[](size_t index) const { return *(_data_ptr + Min(index, _data_size)); }
	FloatTensorWrapper& operator+=(const FloatTensorWrapper& _right);
	FloatTensorWrapper& operator-=(const FloatTensorWrapper& _right);
	FloatTensorWrapper& operator*=(const FloatTensorWrapper& _right);
	FloatTensorWrapper& operator/=(const FloatTensorWrapper& _right);
	FloatTensorWrapper& operator+=(float _right);
	FloatTensorWrapper& operator-=(float _right);
	FloatTensorWrapper& operator*=(float _right);
	FloatTensorWrapper& operator/=(float _right);
private:
	float* _data_ptr = nullptr;
	size_t _data_size = 0;
};
#endif
_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End