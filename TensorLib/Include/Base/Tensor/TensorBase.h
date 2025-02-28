/**
 * FileName: TensorBase.h
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

#include <complex>
#include <set>
#include <unordered_map>
#include "TensorLib/Include/Base/Value.h"


_D_Dragonian_Lib_Space_Begin

template <typename _ValueType, size_t _Size>
using IDLArray = TemplateLibrary::Array<_ValueType, _Size>;

//using namespace DragonianLibSTL;
using SizeType = Int64;
template <typename _Ty>
using ContainerSet = std::set<_Ty>;
template <typename _TyA, typename _TyB>
using UnorderedMap = std::unordered_map<_TyA, _TyB>;
template <typename _TyA, typename _TyB>
using ContainerMap = std::unordered_map<_TyA, _TyB>;

enum class DType
{
	Bool,
	Int8,
	Int16,
	Int32,
	Int64,
	UInt8,
	UInt16,
	UInt32,
	UInt64,
	Float16,
	Float32,
	Float64,
	Complex32,
	Complex64,
	BFloat16,
	Unknown
};

template<typename _Type>
struct _Impl_Dragonian_Lib_Decldtype { static constexpr auto _DType = DType::Unknown; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<bool> { static constexpr auto _DType = DType::Bool; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int8> { static constexpr auto _DType = DType::Int8; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int16> { static constexpr auto _DType = DType::Int16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int32> { static constexpr auto _DType = DType::Int32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Int64> { static constexpr auto _DType = DType::Int64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt8> { static constexpr auto _DType = DType::UInt8; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt16> { static constexpr auto _DType = DType::UInt16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt32> { static constexpr auto _DType = DType::UInt32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<UInt64> { static constexpr auto _DType = DType::UInt64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float16> { static constexpr auto _DType = DType::Float16; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float32> { static constexpr auto _DType = DType::Float32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Float64> { static constexpr auto _DType = DType::Float64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Complex32> { static constexpr auto _DType = DType::Complex32; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<Complex64> { static constexpr auto _DType = DType::Complex64; };
template<>
struct _Impl_Dragonian_Lib_Decldtype<BFloat16> { static constexpr auto _DType = DType::BFloat16; };
template<typename _Type>
constexpr auto _Impl_Dragonian_Lib_Decldtype_v = _Impl_Dragonian_Lib_Decldtype<_Type>::_DType;

//using Dimensions = Vector<SizeType>; ///< Alias for vector of size types
//using ShapeIterator = Dimensions::Iterator; ///< Alias for iterator of shape type

template <typename _Type, size_t _Rank>
class TensorIterator
{
public:
	static_assert(_Rank > 0, "Rank must be greater than 0.");
	using value_type = TypeTraits::ConditionalType<_Rank == 1, _Type, TensorIterator<_Type, _Rank>>;
	using difference_type = ptrdiff_t;
	using pointer = TypeTraits::ConditionalType<_Rank == 1, _Type*, TensorIterator<_Type, _Rank - 1>>;
	using reference = TypeTraits::ConditionalType<_Rank == 1, _Type&, TensorIterator<_Type, _Rank>>;
	using iterator_category = std::random_access_iterator_tag;

	TensorIterator() = default;

	TensorIterator(
		_Type* _Data, const int64_t* _Shape, const int64_t* _Stride)
		: _MyData(_Data), _MyShape(_Shape), _MyStride(_Stride), _MyStepStride(*_Stride) {}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator& operator++()
	{
		_MyData += _MyStepStride;
		return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator operator++(int)
	{
		TensorIterator _Tmp = *this;
		_MyData += _MyStepStride;
		return _Tmp;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator& operator--()
	{
		_MyData -= _MyStepStride;
		return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator operator--(int)
	{
		TensorIterator _Tmp = *this;
		_MyData -= _MyStepStride;
		return _Tmp;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator& operator+=(int64_t _Off)
	{
		_MyData += _Off * _MyStepStride;
		return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator& operator-=(int64_t _Off)
	{
		_MyData -= _Off * _MyStepStride;
		return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator operator+(int64_t _Off) const
	{
		TensorIterator _Tmp = *this;
		_Tmp += _Off;
		return _Tmp;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline TensorIterator operator-(int64_t _Off) const
	{
		TensorIterator _Tmp = *this;
		_Tmp -= _Off;
		return _Tmp;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline int64_t operator-(const TensorIterator& _Right) const
	{
		return (_MyData - _Right._MyData) / _MyStepStride;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator==(const TensorIterator& _Right) const
	{
		return _MyData == _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator!=(const TensorIterator& _Right) const
	{
		return _MyData != _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator<(const TensorIterator& _Right) const
	{
		if (_MyStepStride < 0)
			return _MyData > _Right._MyData;
		return _MyData < _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator>(const TensorIterator& _Right) const
	{
		if (_MyStepStride < 0)
			return _MyData < _Right._MyData;
		return _MyData > _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator<=(const TensorIterator& _Right) const
	{
		if (_MyStepStride < 0)
			return _MyData >= _Right._MyData;
		return _MyData <= _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator>=(const TensorIterator& _Right) const
	{
		if (_MyStepStride < 0)
			return _MyData <= _Right._MyData;
		return _MyData >= _Right._MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline reference operator*() const
	{
		if constexpr (_Rank == 1)
			return *_MyData;
		else
			return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline reference operator[](int64_t _Off) const
	{
		return *(*this + _Off);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline int64_t Shape() const
	{
		return *_MyShape;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline int64_t Stride() const
	{
		return *_MyStride;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline int64_t StepStride() const
	{
		return _MyStepStride;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline const int64_t* ShapePtr() const
	{
		return _MyShape;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline const int64_t* StridePtr() const
	{
		return _MyStride;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline const _Type* DataPtr() const
	{
		return _MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline _Type* DataPtr()
	{
		return _MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline pointer Begin() const
	{
		if constexpr (_Rank == 1)
			return _MyData;
		else
			return TensorIterator<_Type, _Rank - 1>(_MyData, _MyShape + 1, _MyStride + 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline pointer End() const
	{
		if constexpr (_Rank == 1)
			return _MyData + 1;
		else
			return Begin() + *(_MyShape + 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline pointer begin() const
	{
		return Begin();
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline pointer end() const
	{
		return End();
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContinuous() const
	{
		return _MyStepStride == 1;
	}

protected:
	_Type* _MyData = nullptr;
	const int64_t* _MyShape = nullptr;
	const int64_t* _MyStride = nullptr;
	int64_t _MyStepStride = 1;
};

_D_Dragonian_Lib_Space_End