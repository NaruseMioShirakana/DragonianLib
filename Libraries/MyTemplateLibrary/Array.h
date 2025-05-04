/**
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
#include "Libraries/MyTemplateLibrary/Util.h"
#include "Libraries/Util/TypeTraits.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

template <typename _ValueType, size_t _Rank>
class Array
{
public:
	static constexpr size_t _MyRank = _Rank;
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
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& operator[](
		this const Array& _Self,
		size_t _Index
		)
	{
		return _Self._MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& operator[](
		this Array& _Self,
		size_t _Index
		)
	{
		return _Self._MyData[_Index];
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

template <typename _ValueType>
class Array<_ValueType, 0>
{
	static constexpr size_t _MyRank = 0;
	using ArrayType = _ValueType[1];

	_D_Dragonian_Lib_Constexpr_Force_Inline Array& Assign(const _ValueType*, size_t = 0, size_t = _MyRank)
	{
		return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static Array ConstantOf(const _ValueType&)
	{
		return {};
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static void AssignConstant(const _ValueType&, size_t = 0, size_t = _MyRank)
	{
		
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto Sum() const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		return _ValueType(0);
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto InnerProduct(const Array<_Type, _MyRank>&) const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		return _ValueType(0);
	}
	template <typename _Type = _ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline auto Multiply() const
		requires (TypeTraits::IsArithmeticValue<_Type>&& TypeTraits::IsSameTypeValue<_Type, _ValueType>)
	{
		return _ValueType(0);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Size()
	{
		return 0;
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
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* Begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* End()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& operator[](
		this const Array& _Self,
		size_t _Index
		)
	{
		return _Self._MyData[_Index];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType& operator[](
		this Array& _Self,
		size_t _Index
		)
	{
		return _Self._MyData[_Index];
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
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType& Back() const
	{
		return _MyData[0];
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* begin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* end()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* begin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* end() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedBegin()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline _ValueType* ReversedEnd()
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedBegin() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline const _ValueType* ReversedEnd() const
	{
		return _MyData;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static bool Empty()
	{
		return true;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static size_t Rank()
	{
		return 0;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array<_ValueType, 1> Insert(
		const _ValueType& _Value, size_t
	) const
	{
		return Array<_ValueType, 1>{ _Value };
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Array Erase(size_t _Index) const
	{
		return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::string ToString()
	{
		return "[]";
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static std::wstring ToWString()
	{
		return L"[]";
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
		return true;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool operator!=(const Array& _Right) const
	{
		return true;
	}

	_ValueType _MyData[1]{ };
};

template<typename _Tp, typename... _Up>
Array(_Tp&&, _Up&&...)
-> ::DragonianLib::TemplateLibrary::Array<std::enable_if_t<(TypeTraits::CouldBeConvertedFromValue<TypeTraits::RemoveReferenceType<_Tp>, _Up> && ...), TypeTraits::RemoveReferenceType<_Tp>>,
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

template <typename _Fs, typename _Sec>
constexpr bool IsAllCmbAbleImpl()
{
	return (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> &&
		TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec>) ||
		(TypeTraits::IsArrayLikeValue<_Fs> && TypeTraits::IsArrayLikeValue<_Sec> &&
			TypeTraits::IsSameTypeValue<TypeTraits::ArrayType<_Fs>, TypeTraits::ArrayType<_Sec>>);
}
template <typename>
constexpr bool IsAllCmbAble()
{
	return true;
}
template <typename _Fs, typename _Sec, typename... _Rest>
constexpr bool IsAllCmbAble()
{
	return IsAllCmbAbleImpl<_Fs, _Sec>() && IsAllCmbAble<_Sec, _Rest...>();
}

template <typename _Fs, typename _Sec>
constexpr bool IsAllMovCmbAbleImpl()
{
	return (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> &&
		TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec> && 
		std::is_move_assignable_v<TypeTraits::ArrayType<_Fs>> &&
		std::is_move_assignable_v<TypeTraits::ArrayType<_Sec>>) ||
		(TypeTraits::IsArrayLikeValue<_Fs> && TypeTraits::IsArrayLikeValue<_Sec> &&
			TypeTraits::IsSameTypeValue<TypeTraits::ArrayType<_Fs>, TypeTraits::ArrayType<_Sec>>);
}
template <typename>
constexpr bool IsAllMovCmbAble()
{
	return true;
}
template <typename _Fs, typename _Sec, typename... _Rest>
constexpr bool IsAllMovCmbAble()
{
	return IsAllMovCmbAbleImpl<_Fs, _Sec>() && IsAllMovCmbAble<_Sec, _Rest...>();
}

template <typename _ArrayLikeType>
constexpr auto ImplCombineArray(
	_ArrayLikeType&& _Left
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeType>)
{
	return std::forward<_ArrayLikeType>(_Left);
}

template <typename _ArrayLikeType>
constexpr auto ImplMoveCombineArray(
	_ArrayLikeType&& _Left
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeType>)
{
	return std::forward<_ArrayLikeType>(_Left);
}

template <typename _ArrayLikeTypeA, typename _ArrayLikeTypeB>
constexpr auto ImplCombineArray(
	_ArrayLikeTypeA&& _ArrayLikeA,
	_ArrayLikeTypeB&& _ArrayLikeB
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeTypeA>&& TypeTraits::IsArrayLikeValue<_ArrayLikeTypeB>&& TypeTraits::IsSameTypeValue<TypeTraits::ArrayType<_ArrayLikeTypeA>, TypeTraits::ArrayType<_ArrayLikeTypeB>>)
{
	using ArrayType = TypeTraits::ArrayType<_ArrayLikeTypeA>;
	constexpr size_t _RankA = TypeTraits::ArraySize<_ArrayLikeTypeA>;
	constexpr size_t _RankB = TypeTraits::ArraySize<_ArrayLikeTypeB>;
	auto ABegin = Begin(std::forward<_ArrayLikeTypeA>(_ArrayLikeA));
	auto BBegin = Begin(std::forward<_ArrayLikeTypeB>(_ArrayLikeB));
	Array<ArrayType, _RankA + _RankB> _Ret;
	for (size_t i = 0; i < _RankA; ++i)
		_Ret[i] = ABegin[i];
	for (size_t i = 0; i < _RankB; ++i)
		_Ret[i + _RankA] = BBegin[i];
	return _Ret;
}

template <typename _ArrayLikeTypeA, typename _ArrayLikeTypeB>
constexpr auto ImplMoveCombineArray(
	_ArrayLikeTypeA&& _ArrayLikeA,
	_ArrayLikeTypeB&& _ArrayLikeB
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeTypeA>&& TypeTraits::IsArrayLikeValue<_ArrayLikeTypeB>&& TypeTraits::IsSameTypeValue<TypeTraits::ArrayType<_ArrayLikeTypeA>, TypeTraits::ArrayType<_ArrayLikeTypeB>>&& std::is_move_assignable_v<TypeTraits::ArrayType<_ArrayLikeTypeA>>&& std::is_move_assignable_v<TypeTraits::ArrayType<_ArrayLikeTypeB>>)
{
	using ArrayType = TypeTraits::ArrayType<_ArrayLikeTypeA>;
	constexpr size_t _RankA = TypeTraits::ArraySize<_ArrayLikeTypeA>;
	constexpr size_t _RankB = TypeTraits::ArraySize<_ArrayLikeTypeB>;
	auto ABegin = Begin(std::forward<_ArrayLikeTypeA>(_ArrayLikeA));
	auto BBegin = Begin(std::forward<_ArrayLikeTypeB>(_ArrayLikeB));
	Array<ArrayType, _RankA + _RankB> _Ret;
	for (size_t i = 0; i < _RankA; ++i)
		_Ret[i] = std::move(ABegin[i]);
	for (size_t i = 0; i < _RankB; ++i)
		_Ret[i + _RankA] = std::move(BBegin[i]);
	return _Ret;
}

template <typename _ArrayLikeType, typename _ValueTy>
constexpr auto ImplCombineArrayValue(
	_ArrayLikeType&& _Left,
	_ValueTy&& _Right
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeType> && !TypeTraits::IsArrayLikeValue<_ValueTy>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_ArrayLikeType>, _ValueTy>)
{
	using ArrayType = TypeTraits::ArrayType<_ArrayLikeType>;
	constexpr size_t _Rank = TypeTraits::ArraySize<_ArrayLikeType>;
	auto ABegin = Begin(std::forward<_ArrayLikeType>(_Left));
	Array<ArrayType, _Rank + 1> _Ret;
	for (size_t i = 0; i < _Rank; ++i)
		_Ret[i] = ABegin[i];
	_Ret[_Rank] = std::forward<_ValueTy>(_Right);
	return _Ret;
}

template <typename _ArrayLikeType, typename _ValueTy>
constexpr auto ImplMoveCombineArrayValue(
	_ArrayLikeType&& _Left,
	_ValueTy&& _Right
)
	requires (TypeTraits::IsArrayLikeValue<_ArrayLikeType> && !TypeTraits::IsArrayLikeValue<_ValueTy>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_ArrayLikeType>, _ValueTy>)
{
	using ArrayType = TypeTraits::ArrayType<_ArrayLikeType>;
	constexpr size_t _Rank = TypeTraits::ArraySize<_ArrayLikeType>;
	auto ABegin = Begin(std::forward<_ArrayLikeType>(_Left));
	Array<ArrayType, _Rank + 1> _Ret;
	for (size_t i = 0; i < _Rank; ++i)
		_Ret[i] = std::move(ABegin[i]);
	_Ret[_Rank] = std::forward<_ValueTy>(_Right);
	return _Ret;
}

template <typename _Fs, typename _Sec, typename... _Rest>
constexpr auto CombineArrayImpl(_Fs&& _First, _Sec&& _Second, _Rest&&... _RestArr)
{
	if constexpr (sizeof...(_Rest))
	{
		if constexpr (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> && TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec>)
			return CombineArrayImpl(
				ImplCombineArrayValue(
					std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
				),
				std::forward<_Rest>(_RestArr)...
			);
		else
			return CombineArrayImpl(
				ImplCombineArray(
					std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
				),
				std::forward<_Rest>(_RestArr)...
			);
	}
	else
	{
		if constexpr (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> && TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec>)
			return ImplCombineArrayValue(
				std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
			);
		else
			return ImplCombineArray(
				std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
			);
	}
}

template <typename _Fs>
constexpr auto CombineArrayImpl(_Fs&& _First)
{
	return std::forward<_Fs>(_First);
}

template <typename _Fs, typename _Sec, typename... _Rest>
constexpr auto MoveCombineArrayImpl(_Fs&& _First, _Sec&& _Second, _Rest&&... _RestArr)
{
	if constexpr (sizeof...(_Rest))
	{
		if constexpr (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> && TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec>)
			return MoveCombineArrayImpl(
				ImplMoveCombineArrayValue(
					std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
				),
				std::forward<_Rest>(_RestArr)...
			);
		else
			return MoveCombineArrayImpl(
				ImplMoveCombineArray(
					std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
				),
				std::forward<_Rest>(_RestArr)...
			);
	}
	else
	{
		if constexpr (TypeTraits::IsArrayLikeValue<_Fs> && !TypeTraits::IsArrayLikeValue<_Sec> && TypeTraits::CouldBeConvertedFromValue<TypeTraits::ArrayType<_Fs>, _Sec>)
			return ImplMoveCombineArrayValue(
				std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
			);
		else
			return ImplMoveCombineArray(
				std::forward<_Fs>(_First), std::forward<_Sec>(_Second)
			);
	}
}

template <typename _Fs>
constexpr auto MoveCombineArrayImpl(_Fs&& _First)
{
	return std::forward<_Fs>(_First);
}

template <typename... _Args>
constexpr auto CombineArray(_Args&&... _ArgsArr)
	requires (IsAllCmbAble<_Args...>())
{
	return CombineArrayImpl(std::forward<_Args>(_ArgsArr)...);
}

template <typename... _Args>
constexpr auto MoveCombineArray(_Args&&... _ArgsArr)
	requires (IsAllMovCmbAble<_Args...>())
{
	return MoveCombineArrayImpl(std::forward<_Args>(_ArgsArr)...);
}

constexpr decltype(auto) ConstexprSum()
{
	return 0;
}
template <typename _Type>
constexpr decltype(auto) ConstexprSum(_Type&& _First)
{
	return std::forward<_Type>(_First);
}
template <typename _Type, typename... _Rest>
constexpr decltype(auto) ConstexprSum(_Type&& _Arg, _Rest&&... _Args)
{
	return std::forward<_Type>(_Arg) + ConstexprSum(std::forward<_Rest>(_Args)...);
}
template <typename... _Type>
constexpr decltype(auto) ExpandSum(_Type&&... _Args)
{
	return ConstexprSum(std::forward<_Type>(_Args)...);
}

constexpr decltype(auto) ConstexprMul()
{
	return 0;
}
template <typename _Type>
constexpr decltype(auto) ConstexprMul(_Type&& _First)
{
	return std::forward<_Type>(_First);
}
template <typename _Type, typename... _Rest>
constexpr decltype(auto) ConstexprMul(_Type&& _Arg, _Rest&&... _Args)
{
	return std::forward<_Type>(_Arg) * ConstexprMul(std::forward<_Rest>(_Args)...);
}
template <typename... _Type>
constexpr decltype(auto) ExpandMul(_Type&&... _Args)
{
	return ConstexprMul(std::forward<_Type>(_Args)...);
}

_D_Dragonian_Lib_Template_Library_Space_End