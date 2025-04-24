﻿/**
 * @file Array.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Array type for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Util.h"
#include "Libraries/Util/TypeTraits.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

template <class _ValueType, size_t _Rank>
class Array
{
public:
	static constexpr size_t _MyRank = _Rank;
	static_assert(_MyRank > 0, "The rank of the array must be greater than 0.");
	using ArrayType = _ValueType[_Rank];

	_D_Dragonian_Lib_Constexpr_Force_Inline Array& Assign(const _ValueType* _Right, size_t _Begin = 0, size_t _End = _MyRank)
	{
		for (size_t i = _Begin; i < _End; ++i)
			_MyData[i] = *_Right++;
		return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static Array ConstantOf(const _ValueType& _Value)
	{
		Array _Tmp;
		for (size_t i = 0; i < _Rank; ++i)
			_Tmp._MyData[i] = _Value;
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void AssignConstant(const _ValueType& _Value, size_t _Begin = 0, size_t _End = _MyRank)
	{
		for (size_t i = _Begin; i < _End; ++i)
			_MyData[i] = _Value;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto Sum() const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		_ValueType _Sum = 0;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum += _MyData[i];
		return _Sum;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto InnerProduct(const Array<_Type, _Rank>& _Right) const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		_ValueType _Sum = 0;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum += _MyData[i] * _Right._MyData[i];
		return _Sum;
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto Multiply() const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		_ValueType _Sum = 1;
		for (size_t i = 0; i < _Rank; ++i)
			_Sum *= _MyData[i];
		return _Sum;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Size()
	{
		return _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* Data()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* Data() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* Begin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* End() const
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* Begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* End()
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& operator[](size_t _Index)
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& operator[](size_t _Index) const
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& At(size_t _Index)
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& At(size_t _Index) const
	{
		return _MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& Front()
	{
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& Front() const
	{
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& Back()
	{
		return _MyData[_Rank - 1];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& Back() const
	{
		return _MyData[_Rank - 1];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* end()
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* begin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* end() const
	{
		return _MyData + _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedBegin()
	{
		return _MyData + _Rank - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedEnd()
	{
		return &_MyData[0] - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedBegin() const
	{
		return _MyData + _Rank - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedEnd() const
	{
		return &_MyData[0] - 1;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static bool Empty()
	{
		return _Rank == 0;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Rank()
	{
		return _Rank;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array<_ValueType, _Rank + 1> Insert(
		const _ValueType& _Value, size_t _Index
	) const
	{
		Array<_ValueType, _Rank + 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		_Tmp._MyData[_Index] = _Value;
		for (size_t i = _Index; i < _Rank; ++i)
			_Tmp._MyData[i + 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array<_ValueType, _Rank - 1> Erase(size_t _Index) const
	{
		Array<_ValueType, _Rank - 1> _Tmp;
		for (size_t i = 0; i < _Index; ++i)
			_Tmp._MyData[i] = _MyData[i];
		for (size_t i = _Index + 1; i < _Rank; ++i)
			_Tmp._MyData[i - 1] = _MyData[i];
		return _Tmp;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline std::string ToString() const
	{
		std::string _Str = "[";
		for (size_t i = 0; i < _Rank; ++i)
		{
			_Str += std::to_string(_MyData[i]);
			if (i != _Rank - 1)
				_Str += ", ";
		}
		_Str += "]";
		return _Str;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline std::wstring ToWString() const
	{
		std::wstring _Str = L"[";
		for (size_t i = 0; i < _Rank; ++i)
		{
			_Str += std::to_wstring(_MyData[i]);
			if (i != _Rank - 1)
				_Str += L", ";
		}
		_Str += L"]";
		return _Str;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline ArrayType& RawArray()
	{
		return _MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline const ArrayType& RawArray() const
	{
		return _MyData;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator==(const Array& _Right) const
	{
		for (size_t i = 0; i < _Rank; ++i)
			if (_MyData[i] != _Right._MyData[i])
				return false;
		return true;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator!=(const Array& _Right) const
	{
		return !(*this == _Right);
	}

	_ValueType _MyData[_Rank]; ///< Data of the dimensions
};

template<typename _Tp, typename... _Up>
Array(_Tp, _Up...)
-> ::DragonianLib::TemplateLibrary::Array<std::enable_if_t<(TypeTraits::IsSameTypeValue<_Tp, _Up> && ...), _Tp>,
         1 + sizeof...(_Up)>;

template <typename _ValueType, size_t _Size>
using MArray = Array<_ValueType, _Size>;

template <typename _Type, size_t _Rank>
struct _Impl_Static_Array_Type
{
	_Impl_Static_Array_Type() = delete;
	template<size_t _TRank>
		_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
			const _Type& _Value,
			const _Impl_Static_Array_Type<_Type, _TRank>& _Array
		) requires ((_Rank > 1) && _TRank == _Rank - 1)
	{
		for (size_t i = 0; i < _Array.Rank; ++i)
			Data[i + 1] = _Array.Data[i];
		Data[0] = _Value;
	}
	template<size_t _TRank>
		_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
			const _Impl_Static_Array_Type<_Type, _TRank>& _Array,
			const _Type& _Value
		) requires ((_Rank > 1) && _TRank == _Rank - 1)
	{
		for (size_t i = 0; i < _Array.Rank; ++i)
			Data[i] = _Array.Data[i];
		Data[_Array.Rank] = _Value;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Type& _Value,
		const _Impl_Static_Array_Type<_Type, 0>& _Array
	) requires (_Rank == 1)
	{
		UNUSED(_Array);
		Data[0] = _Value;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Impl_Static_Array_Type<_Type, 0>& _Array,
		const _Type& _Value
	)
	{
		UNUSED(_Array);
		Data[0] = _Value;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline _Impl_Static_Array_Type(
		const _Type& _Value
	) requires (_Rank == 1)
	{
		Data[0] = _Value;
	}

	static constexpr size_t Rank = _Rank;
	Array<_Type, _Rank> Data;
};
template <typename _Type>
struct _Impl_Static_Array_Type<_Type, 0> {};

template <typename _Type>
struct ExtractAllShapesOfArrayLikeType;
template <typename _Type, size_t _Size>
struct ExtractAllShapesOfArrayLikeType<_Type[_Size]>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_Type> +1;
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape() requires (Rank > 0)
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_Size) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_Size),
				ExtractAllShapesOfArrayLikeType<_Type>::GetShape()
			);
			return Shape;
		}
	}
};
template <template <typename, size_t> typename _ObjType, typename _ValueType, size_t _ValueSize>
struct ExtractAllShapesOfArrayLikeType<_ObjType<_ValueType, _ValueSize>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape() requires (Rank > 0)
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_ValueSize) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_ValueSize),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};
template <template <size_t, typename> typename _ObjType, size_t _ValueSize, typename _ValueType>
struct ExtractAllShapesOfArrayLikeType<_ObjType<_ValueSize, _ValueType>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
		static constexpr const _Impl_Static_Array_Type<int64_t, Rank>& GetShape() requires (Rank > 0)
	{
		if constexpr (Rank == 1)
		{
			static _Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_ValueSize) };
			return Shape;
		}
		else
		{
			static _Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_ValueSize),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};
template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArrayLikeValue<_Type>>>
const auto& GetAllShapesOfArrayLikeType = _D_Dragonian_Lib_TL_Namespace ExtractAllShapesOfArrayLikeType<_Type>::GetShape().Data;

template <typename _ValueType>
struct ExtractAllShapesOfInitializerList;
template <typename _ValueType>
struct ExtractAllShapesOfInitializerList<std::initializer_list<_ValueType>>
{
	static constexpr size_t Rank = TypeTraits::ExtractRankValue<_ValueType> +1;
		static constexpr _Impl_Static_Array_Type<int64_t, Rank> GetShape(const std::initializer_list<_ValueType>& _Val) requires (Rank > 0)
	{
		if constexpr (Rank == 1)
		{
			_Impl_Static_Array_Type<int64_t, 1> Shape{ static_cast<int64_t>(_Val.size()) };
			return Shape;
		}
		else
		{
			_Impl_Static_Array_Type<int64_t, Rank> Shape(
				static_cast<int64_t>(_Val.size()),
				ExtractAllShapesOfArrayLikeType<_ValueType>::GetShape()
			);
			return Shape;
		}
	}
};

_D_Dragonian_Lib_Template_Library_Space_End