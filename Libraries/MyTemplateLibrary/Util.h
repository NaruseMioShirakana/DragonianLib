/**
 * @file Util.h
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
 * @brief Utility functions for DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/Base.h"

#define _D_Dragonian_Lib_Template_Library_Space_Begin _D_Dragonian_Lib_Space_Begin namespace TemplateLibrary {
#define _D_Dragonian_Lib_Template_Library_Space_End } _D_Dragonian_Lib_Space_End
#define _D_Dragonian_Lib_TL_Namespace _D_Dragonian_Lib_Namespace TemplateLibrary::
#define _D_Dragonian_Lib_Stl_Throw(message) _D_Dragonian_Lib_Throw_Exception(message)

_D_Dragonian_Lib_Template_Library_Space_Begin

constexpr size_t _Valdef_Dragonian_Lib_StlUnfoldCount = 8;

template <typename ValueType, typename ...ArgTypes>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplConstructAt(
	ValueType& _Where,
	ArgTypes&&... _Args
) requires (std::is_constructible_v<ValueType, ArgTypes...>)
{
	return *new (&_Where) ValueType(std::forward<ArgTypes>(_Args)...);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline void ImplDestroyRange(
	IteratorType1 _First,
	IteratorType2 _Last
) requires (TypeTraits::IsSameIterator<IteratorType1, IteratorType2> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType1>;
	if constexpr (!std::is_trivially_destructible_v<ValueType> && std::is_destructible_v<ValueType>)
	{
		if (_First >= _Last)
			return;
		const auto Size = static_cast<size_t>(_Last - _First);
		size_t i = 0;
		if (Size >= _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (; i <= Size - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
				for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
					(_First + i + j)->~ValueType();
		for (; i < Size; ++i)
			(_First + i)->~ValueType();
	}
}

template <typename IteratorType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorDefaultConstruct(
	IteratorType _Ptr,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType>&& std::is_default_constructible_v<TypeTraits::IteratorValueType<IteratorType>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Ptr[i + j]);
	for (; i < _Count; ++i)
		ImplConstructAt(_Ptr[i]);
}

template <typename DestIterator, typename SrcIterator>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorCast(
	DestIterator _Dest,
	SrcIterator _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<DestIterator>&& TypeTraits::IsIterator<SrcIterator>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<DestIterator>, TypeTraits::IteratorValueType<SrcIterator>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<DestIterator>>)
{
	using DestType = TypeTraits::IteratorValueType<DestIterator>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[i + j] = DestType(_Src[i + j]);
	for (; i < _Count; ++i)
		_Dest[i] = DestType(_Src[i]);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorCopy(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType1>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[i + j] = ValueType(_Src[i + j]);
	for (; i < _Count; ++i)
		_Dest[i] = ValueType(_Src[i]);
}

template <typename IteratorType, typename ValueType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorCopyOne(
	IteratorType _Dest,
	size_t _Count,
	const ValueType& _Src
) requires (TypeTraits::IsIterator<IteratorType> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType>, const ValueType&>)
{
	using DestType = TypeTraits::IteratorValueType<IteratorType>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[i + j] = DestType(_Src);
	for (; i < _Count; ++i)
		_Dest[i] = DestType(_Src);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorMove(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType2>>)
{
	using DestType = TypeTraits::IteratorValueType<IteratorType1>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[i + j] = DestType(std::move(_Src[i + j]));
	for (; i < _Count; ++i)
		_Dest[i] = DestType(std::move(_Src[i]));
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplReversedIteratorCopy(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType1>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[_Count - (1 + i + j)] = ValueType(_Src[_Count - (1 + i + j)]);
	for (; i < _Count; ++i)
		_Dest[_Count - (1 + i)] = ValueType(_Src[_Count - (1 + i)]);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplReversedIteratorMove(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType2>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType1>;
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				_Dest[_Count - (1 + i + j)] = ValueType(std::move(_Src[_Count - (1 + i + j)]));
	for (; i < _Count; ++i)
		_Dest[_Count - (i + 1)] = ValueType(std::move(_Src[_Count - (i + 1)]));
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplReversedIteratorCopyConstruct(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
)
requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&&
	TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>,
	TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Dest[_Count - (1 + i + j)], _Src[_Count - (1 + i + j)]);
	for (; i < _Count; ++i)
		ImplConstructAt(_Dest[_Count - (i + 1)], _Src[_Count - (i + 1)]);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplReversedIteratorMoveConstruct(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType2>>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Dest[_Count - (1 + i + j)], std::move(_Src[_Count - (1 + i + j)]));
	for (; i < _Count; ++i)
		ImplConstructAt(_Dest[_Count - (i + 1)], std::move(_Src[_Count - (i + 1)]));
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorCopyConstruct(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&&
	TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>,
	TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Dest[i + j], _Src[i + j]);
	for (; i < _Count; ++i)
		ImplConstructAt(_Dest[i], _Src[i]);
}

template <typename IteratorType, typename ValueType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorCopyConstructOne(
	IteratorType _Dest,
	size_t _Count,
	const ValueType& _Src
) requires (TypeTraits::IsIterator<IteratorType> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType>, const ValueType&>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Dest[i + j], _Src);
	for (; i < _Count; ++i)
		ImplConstructAt(_Dest[i], _Src);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorMoveConstruct(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	size_t i = 0;
	if (_Count >= _Valdef_Dragonian_Lib_StlUnfoldCount)
		for (; i <= _Count - _Valdef_Dragonian_Lib_StlUnfoldCount; i += _Valdef_Dragonian_Lib_StlUnfoldCount)
			for (size_t j = 0; j < _Valdef_Dragonian_Lib_StlUnfoldCount; ++j)
				ImplConstructAt(_Dest[i + j], std::move(_Src[i + j]));
	for (; i < _Count; ++i)
		ImplConstructAt(_Dest[i], std::move(_Src[i]));
}

template <typename IteratorType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorOffset(
	IteratorType _Src,
	size_t _Count,
	int64_t Offset
) requires (TypeTraits::IsIterator<IteratorType> && (std::is_copy_assignable_v<TypeTraits::IteratorValueType<IteratorType>> || std::is_move_assignable_v<TypeTraits::IteratorValueType<IteratorType>>) && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType>;
	if (Offset == 0 || _Count == 0)
		return;
	auto _Dest = _Src + Offset;
	if (Offset < 0)
	{
		if constexpr (std::is_move_assignable_v<ValueType>)
			ImplIteratorMove(_Dest, _Src, _Count);
		else
			ImplIteratorCopy(_Dest, _Src, _Count);
	}
	else
	{
		if constexpr (std::is_move_assignable_v<ValueType>)
			ImplReversedIteratorMove(_Dest, _Src, _Count);
		else
			ImplReversedIteratorCopy(_Dest, _Src, _Count);
	}
}

template <typename IteratorType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorOffsetConstruct(
	IteratorType _Src,
	size_t _Count,
	int64_t Offset
) requires (TypeTraits::IsIterator<IteratorType> && (std::is_copy_assignable_v<TypeTraits::IteratorValueType<IteratorType>> || std::is_move_assignable_v<TypeTraits::IteratorValueType<IteratorType>>) && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>)
{
	using ValueType = TypeTraits::IteratorValueType<IteratorType>;
	if (Offset == 0 || _Count == 0)
		return;
	auto _Dest = _Src + Offset;
	if (Offset < 0)
	{
		if constexpr (std::is_move_constructible_v<ValueType>)
			ImplIteratorMoveConstruct(_Dest, _Src, _Count);
		else
			ImplIteratorCopyConstruct(_Dest, _Src, _Count);
	}
	else
	{
		if constexpr (std::is_move_constructible_v<ValueType>)
			ImplReversedIteratorMoveConstruct(_Dest, _Src, _Count);
		else
			ImplReversedIteratorCopyConstruct(_Dest, _Src, _Count);
	}
}

template <typename IteratorType>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorOffsetCopy(
	IteratorType _Src,
	size_t _Count,
	int64_t Offset
) requires (TypeTraits::IsIterator<IteratorType>&& std::is_copy_assignable_v<TypeTraits::IteratorValueType<IteratorType>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType>>)
{
	if (Offset == 0)
		return;
	auto _Dest = _Src + Offset;
	if (Offset < 0)
		ImplIteratorCopy(_Dest, _Src, _Count);
	else
		ImplReversedIteratorCopy(_Dest, _Src, _Count);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorTryMoveConstruct(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&& TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	if constexpr (TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>>)
		ImplIteratorMoveConstruct(_Dest, _Src, _Count);
	else
		ImplIteratorCopyConstruct(_Dest, _Src, _Count);
}

template <typename IteratorType1, typename IteratorType2>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ImplIteratorTryMoveAssign(
	IteratorType1 _Dest,
	IteratorType2 _Src,
	size_t _Count
) requires (TypeTraits::IsIterator<IteratorType1>&& TypeTraits::IsIterator<IteratorType2>&&
	TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>,
	TypeTraits::IteratorValueType<IteratorType2>> && !TypeTraits::IsConstValue<TypeTraits::IteratorValueType<IteratorType1>>)
{
	if constexpr (TypeTraits::CouldBeConvertedFromValue<TypeTraits::IteratorValueType<IteratorType1>, TypeTraits::RReferenceType<TypeTraits::IteratorValueType<IteratorType2>>>)
		ImplIteratorMove(_Dest, _Src, _Count);
	else
		ImplIteratorCopy(_Dest, _Src, _Count);
}

template <typename _Type>
constexpr decltype(auto) CBegin(const _Type& _Container)
	requires (TypeTraits::HasCRange<_Type>)
{
	if constexpr (TypeTraits::HasCLRange<_Type>)
		return _Container.cbegin();
	else if constexpr (TypeTraits::HasCHRange<_Type>)
		return _Container.CBegin();
}
template <typename _Type>
constexpr decltype(auto) CEnd(const _Type& _Container)
	requires (TypeTraits::HasCRange<_Type>)
{
	if constexpr (TypeTraits::HasCLRange<_Type>)
		return _Container.cend();
	else if constexpr (TypeTraits::HasCHRange<_Type>)
		return _Container.CEnd();
}
template <typename _Type>
constexpr decltype(auto) Begin(_Type&& _Container)
	requires (TypeTraits::HasRange<_Type>)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).begin();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).Begin();
	else
		return CBegin(std::forward<_Type>(_Container));
}
template <typename _Type>
constexpr decltype(auto) End(_Type&& _Container)
	requires (TypeTraits::HasRange<_Type>)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).end();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).End();
	else
		return CEnd(std::forward<_Type>(_Container));
}
template <typename _Type, size_t _Rank>
constexpr decltype(auto) Begin(_Type (& _Arr)[_Rank])
{
	return _Arr + 0;
}
template <typename _Type, size_t _Rank>
constexpr decltype(auto) End(_Type (& _Arr)[_Rank])
{
	return _Arr + _Rank;
}
template <typename _Type, size_t _Rank>
constexpr decltype(auto) Begin(const _Type(&_Arr)[_Rank])
{
	return _Arr + 0;
}
template <typename _Type, size_t _Rank>
constexpr decltype(auto) End(const _Type(&_Arr)[_Rank])
{
	return _Arr + _Rank;
}

template <typename>
struct NEpsilonImpl
{
	using _EpsilonTy = std::nullopt_t;
	static constexpr _EpsilonTy Epsilon = std::nullopt;
};

template <>
struct NEpsilonImpl<float>
{
	using _EpsilonTy = float;
	static constexpr _EpsilonTy Epsilon = std::numeric_limits<float>::epsilon();
};

template <>
struct NEpsilonImpl<double>
{
	using _EpsilonTy = double;
	static constexpr _EpsilonTy Epsilon = std::numeric_limits<double>::epsilon();
};

template <>
struct NEpsilonImpl<long double>
{
	using _EpsilonTy = long double;
	static constexpr _EpsilonTy Epsilon = std::numeric_limits<long double>::epsilon();
};

template <typename _Type>
using EpsilonImpl = NEpsilonImpl<TypeTraits::RemoveConstType<TypeTraits::RemoveVolatileType<TypeTraits::RemoveReferenceType<_Type>>>>;

class CCmpEqual
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return std::abs(_Left - _Right) < _Epsilon;
		else
			return std::equal(_Left, _Right);
	}
};
constexpr CCmpEqual CmpEqual{};

class CCmpNotEqual
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return std::abs(_Left - _Right) > _Epsilon;
		else
			return !std::equal(_Left, _Right);
	}
};
constexpr CCmpNotEqual CmpNotEqual{};

class CCmpLess
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return (_Left < _Right) && CmpNotEqual(_Left, _Right);
		else
			return std::less<_Type>()(_Left, _Right);
	}
};
constexpr CCmpLess CmpLess{};

class CCmpLessEqual
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return (_Left < _Right) || CmpEqual(_Left, _Right);
		else
			return std::less_equal<_Type>()(_Left, _Right);
	}
};
constexpr CCmpLessEqual CmpLessEqual{};

class CCmpGreater
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return (_Left > _Right) && CmpNotEqual(_Left, _Right);
		else
			return std::greater<_Type>()(_Left, _Right);
	}
};
constexpr CCmpGreater CmpGreater{};

class CCmpGreaterEqual
{
public:
	template <typename _Type, typename _EpsilonTy = typename EpsilonImpl<_Type>::_EpsilonTy, _EpsilonTy _Epsilon = EpsilonImpl<_Type>::Epsilon>
	constexpr static bool operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (TypeTraits::IsFloatingPointValue<_Type>)
			return (_Left > _Right) || CmpEqual(_Left, _Right);
		else
			return std::greater_equal<_Type>()(_Left, _Right);
	}
};
constexpr CCmpGreaterEqual CmpGreaterEqual{};

template <typename _IteratorTypeBeg, typename _IteratorTypeEnd>
size_t GetSize(_IteratorTypeBeg _MyBegin, _IteratorTypeEnd _MyEnd)
	requires(TypeTraits::IsSameIterator<_IteratorTypeBeg, _IteratorTypeEnd>)
{
	using MyIterTypeBeg = _IteratorTypeBeg;
	using MyIterTypeEnd = _IteratorTypeEnd;

	if constexpr (TypeTraits::HasIntegerSubtraction<MyIterTypeBeg>)
		return static_cast<UInt64>(_MyEnd - _MyBegin);
	else if constexpr (TypeTraits::HasFrontIncrement<MyIterTypeBeg>)
	{
		auto _Tmp = _MyBegin;
		UInt64 _Size = 0;
		if constexpr (TypeTraits::_HasLessOperator<MyIterTypeBeg, MyIterTypeEnd>)
			while (_Tmp < _MyEnd)
			{
				++_Tmp;
				++_Size;
			}
		else if constexpr (TypeTraits::_HasEqualOperator<MyIterTypeBeg, MyIterTypeEnd>)
			while (_Tmp != _MyEnd)
			{
				++_Tmp;
				++_Size;
			}
		else
			_D_Dragonian_Lib_Throw_Exception("Could not get size!");
		return _Size;
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Could not get size!");
}

template <typename _IteratorTypeBeg, typename _IteratorTypeEnd, typename = std::enable_if_t<
	TypeTraits::IsSameIterator<_IteratorTypeBeg, _IteratorTypeEnd>>>
	class RangesWrp
{
public:
	using MyIterTypeBeg = _IteratorTypeBeg;
	using MyIterTypeEnd = _IteratorTypeEnd;
	using MyRefType = decltype(*TypeTraits::InstanceOf<MyIterTypeBeg>());
	using MyValueType = TypeTraits::RemoveReferenceType<MyRefType>;

	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline ~RangesWrp() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp(const RangesWrp&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp(RangesWrp&&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp(const MyIterTypeBeg& _Begin, const MyIterTypeEnd& _End)
		: _MyBegin(_Begin), _MyEnd(_End)
	{
		if constexpr (TypeTraits::HasLessOperator<MyIterTypeBeg> && TypeTraits::HasLessOperator<MyIterTypeEnd>)
			if (_MyEnd < _MyBegin)
				_D_Dragonian_Lib_Throw_Exception("End could not be less than Begin!");
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data() const noexcept
	{
		return _MyBegin;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) data() const noexcept
	{
		return _MyBegin;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const noexcept
	{
		return _MyBegin;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const noexcept
	{
		return _MyEnd;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin() const noexcept
	{
		return _MyBegin;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end() const noexcept
	{
		return _MyEnd;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline UInt64 Size() const noexcept
	{
		return GetSize(_MyBegin, _MyEnd);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](size_t _Index) const
	{
		if constexpr (TypeTraits::CouldIndex<MyIterTypeBeg>)
			return _MyBegin[_Index];
		else if constexpr (TypeTraits::HasIntegerAddition<MyIterTypeBeg>)
			return *(_MyBegin + _Index);
		else if constexpr (TypeTraits::HasFrontIncrement<MyIterTypeBeg>)
		{
			auto _Tmp = _MyBegin;
			for (size_t i = 0; i < _Index; ++i)
				++_Tmp;
			return *_Tmp;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Could not index!");
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(RangesWrp&& _RRight) noexcept = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(const RangesWrp& _Right) noexcept = default;

	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& Assign(const RangesWrp& _Right)
	{
		if (this == &_Right)
			return *this;
		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			operator[](i) = _Right[i];
		return *this;
	}

	template <typename _Type2>
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(const _Type2& _Right)
		requires (TypeTraits::CouldBeConvertedFromValue<MyValueType, _Type2>)
	{
		for (size_t i = 0; i < Size(); ++i)
			operator[](i) = static_cast<MyValueType>(_Right);
		return *this;
	}

	RangesWrp<const MyValueType*, const MyValueType*> RawConst() const
	{
		return RangesWrp<const TypeTraits::RemoveReferenceType<decltype(*_MyBegin)>*, const TypeTraits::RemoveReferenceType<decltype(*_MyEnd)>*>(&*_MyBegin, &*_MyEnd);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Raw() const noexcept
	{
		return RangesWrp<TypeTraits::RemoveReferenceType<decltype(*_MyBegin)>*, TypeTraits::RemoveReferenceType<decltype(*_MyEnd)>*>{ &*_MyBegin, &* _MyEnd };
	}

	decltype(auto) Byte() const noexcept
	{
		auto RawRange = Raw();
		if constexpr (TypeTraits::IsConstValue<MyValueType>)
			return RangesWrp<const DragonianLib::Byte*, const DragonianLib::Byte*>(
				reinterpret_cast<const DragonianLib::Byte*>(RawRange._MyBegin),
				reinterpret_cast<const DragonianLib::Byte*>(RawRange._MyEnd)
			);
		else
			return RangesWrp<DragonianLib::Byte*, DragonianLib::Byte*>(
				reinterpret_cast<DragonianLib::Byte*>(RawRange._MyBegin),
				reinterpret_cast<DragonianLib::Byte*>(RawRange._MyEnd)
			);
	}

	bool Null() const noexcept
	{
		return (_MyBegin == nullptr) || (_MyEnd == nullptr);
	}

	bool Contains(const MyIterTypeBeg& _Pointer, bool _End = false) const
	{
		return (_Pointer >= _MyBegin) && (_End ? (_Pointer <= _MyEnd) : (_Pointer < _MyEnd));
	}

	bool IsSubRangeOf(const RangesWrp& _Right) const
	{
		return (_MyBegin >= _Right._MyBegin) && (_MyEnd <= _Right._MyEnd);
	}

	bool IsSubRangeOf(const MyIterTypeBeg& _Begin, const MyIterTypeEnd& _End) const
	{
		return (_MyBegin >= _Begin) && (_MyEnd <= _End);
	}

	operator RangesWrp<const TypeTraits::RemoveARPCV<MyValueType>*, const TypeTraits::RemoveARPCV<MyValueType>*>()
	{
		return { &*_MyBegin, &*_MyEnd };
	}

protected:
	MyIterTypeBeg _MyBegin = nullptr;
	MyIterTypeEnd _MyEnd = nullptr;
};

template <typename _ValueType>
using ConstantRanges = RangesWrp<const _ValueType*, const _ValueType*>;
template <typename _ValueType>
using MutableRanges = RangesWrp<_ValueType*, _ValueType*>;

template <typename _Type>
	requires (TypeTraits::IsArithmeticValue<_Type>)
class NumberRangesIterator
{
public:
	NumberRangesIterator() = delete;
	NumberRangesIterator(_Type _Value, _Type _Step) : _MyValue(_Value), _MyStep(_Step) {}
	const _Type& operator*() const { return _MyValue; }
	const _Type* operator->() const { return &_MyValue; }
	NumberRangesIterator& operator++() { _MyValue += _MyStep; return *this; }
	NumberRangesIterator operator++(int) { auto _Tmp = *this; _MyValue += _MyStep; return _Tmp; }
	bool operator==(const NumberRangesIterator& _Right) const { return _MyValue == _Right._MyValue; }
	bool operator!=(const NumberRangesIterator& _Right) const { return _MyValue != _Right._MyValue; }
	bool operator<(const NumberRangesIterator& _Right) const { return _MyStep > 0 ? (_MyValue < _Right._MyValue) : (_MyValue > _Right._MyValue); }

private:
	_Type _MyValue;
	_Type _MyStep;
};

template <typename _Type>
	requires (TypeTraits::IsArithmeticValue<_Type>)
class NumberRanges
{
public:
	NumberRanges() = delete;
	NumberRanges(_Type _Begin, _Type _End, _Type _Step) : _MyBegin(_Begin), _MyStep(_Step), _MyEnd(_End) {}
	NumberRangesIterator<_Type> begin() const { return NumberRangesIterator<_Type>(_MyBegin, _MyStep); }
	NumberRangesIterator<_Type> end() const { return NumberRangesIterator<_Type>(_MyEnd, _MyStep); }
	NumberRangesIterator<_Type> Begin() const { return NumberRangesIterator<_Type>(_MyBegin, _MyStep); }
	NumberRangesIterator<_Type> End() const { return NumberRangesIterator<_Type>(_MyEnd, _MyStep); }
	NumberRangesIterator<_Type> rbegin() const { return NumberRangesIterator<_Type>(_MyEnd, -_MyStep); }
	NumberRangesIterator<_Type> rend() const { return NumberRangesIterator<_Type>(_MyBegin, -_MyStep); }
private:
	_Type _MyBegin;
	_Type _MyStep;
	_Type _MyEnd;
};

template <typename _IteratorType, typename _IntegerType = Int64>
	requires (TypeTraits::IsIntegerValue<_IntegerType>)
class EnumratedRangesIterator
{
public:
	using _MyReferenceType = decltype(*TypeTraits::InstanceOf<_IteratorType>());

	EnumratedRangesIterator() = delete;
	EnumratedRangesIterator(_IteratorType _Iterator, _IntegerType _Index) : _MyIterator(_Iterator), _MyIndex(_Index) {}
	EnumratedRangesIterator& operator++() { ++_MyIterator; ++_MyIndex; return *this; }
	EnumratedRangesIterator operator++(int) { auto _Tmp = *this; ++_MyIterator; ++_MyIndex; return _Tmp; }
	bool operator==(const EnumratedRangesIterator& _Right) const { return _MyIterator == _Right._MyIterator; }
	bool operator!=(const EnumratedRangesIterator& _Right) const { return _MyIterator != _Right._MyIterator; }
	bool operator<(const EnumratedRangesIterator& _Right) const { return _MyIterator < _Right._MyIterator; }
	decltype(auto) operator*() const { return std::forward_as_tuple(_MyIndex, *_MyIterator); }
	decltype(auto) operator->() const { return _MyIterator; }

private:
	_IteratorType _MyIterator;
	_IntegerType _MyIndex;
};

template <typename _Type, typename _IntegerType = Int64>
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
class MutableEnumrate
{
public:
	MutableEnumrate() = delete;
	MutableEnumrate(_Type& _Value) : _MyValue(&_Value) {}
	decltype(auto) begin() const { return EnumratedRangesIterator(_MyValue->begin(), _IntegerType(0)); }
	decltype(auto) end() const { return EnumratedRangesIterator(_MyValue->end(), _IntegerType(0)); }
private:
	_Type* _MyValue;
};

template <typename _Type, typename _IntegerType = Int64>
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
class ConstEnumrate
{
public:
	ConstEnumrate() = delete;
	ConstEnumrate(const _Type& _Value) : _MyValue(&_Value) {}
	decltype(auto) begin() const { return EnumratedRangesIterator(_MyValue->begin(), _IntegerType(0)); }
	decltype(auto) end() const { return EnumratedRangesIterator(_MyValue->end(), _IntegerType(0)); }
private:
	const _Type* _MyValue;
};

template <typename _Type, typename _IntegerType = Int64>
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
class ObjectiveEnumrate
{
public:
	ObjectiveEnumrate() = delete;
	ObjectiveEnumrate(_Type&& _Value) : _MyValue(std::move(_Value)) {}
	decltype(auto) begin() const { return EnumratedRangesIterator(_MyValue->begin(), _IntegerType(0)); }
	decltype(auto) end() const { return EnumratedRangesIterator(_MyValue->end(), _IntegerType(0)); }
private:
	_Type _MyValue;
};

template <typename _IntegerType = Int64, typename _Type>
decltype(auto) Enumrate(_Type& _Value)
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
{
	return MutableEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _IntegerType = Int64, typename _Type>
decltype(auto) Enumrate(const _Type& _Value)
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
{
	return ConstEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _IntegerType = Int64, typename _Type>
decltype(auto) Enumrate(_Type&& _Value)
	requires (TypeTraits::HasRange<_Type>&& TypeTraits::IsIntegerValue<_IntegerType>)
{
	return ObjectiveEnumrate<_Type, _IntegerType>(std::forward<_Type>(_Value));
}

template <typename _IteratorTypeBeg, typename _IteratorTypeEnd>
decltype(auto) Ranges(_IteratorTypeBeg _Begin, _IteratorTypeEnd _End)
{
	return RangesWrp(_Begin, _End);
}

template <typename _Type>
decltype(auto) Ranges(_Type&& _Rng)
	requires (TypeTraits::HasRange<_Type>)
{
	return RangesWrp(Begin(std::forward<_Type>(_Rng)), End(std::forward<_Type>(_Rng)));
}

template <typename _Type>
decltype(auto) Ranges(_Type _Begin, _Type _End, _Type _Step)
	requires (TypeTraits::IsArithmeticValue<_Type>)
{
	if (_Step == 0)
		_D_Dragonian_Lib_Throw_Exception("Step cannot be 0.");
	if (_Begin < _End && _Step < 0)
		_D_Dragonian_Lib_Throw_Exception("Step must be positive.");
	if (_Begin > _End && _Step > 0)
		_D_Dragonian_Lib_Throw_Exception("Step must be negative.");
	return NumberRanges<_Type>(_Begin, _End, _Step);
}

template <typename _Type>
decltype(auto) Ranges(_Type _Begin, _Type _End)
	requires (TypeTraits::IsArithmeticValue<_Type>)
{
	return NumberRanges<_Type>(_Begin, _End, _Begin < _End ? _Type(1) : _Type(-1));
}

template <typename _Type>
decltype(auto) Ranges(_Type _End)
	requires (TypeTraits::IsArithmeticValue<_Type>)
{
	return NumberRanges<_Type>(_Type(0), _End, _End > _Type(0) ? _Type(1) : _Type(-1));
}

template <typename _CouldBeIndexedType, size_t... _Index>
_D_Dragonian_Lib_Constexpr_Force_Inline auto MakeTuple(_CouldBeIndexedType&& _Object, std::index_sequence<_Index...>)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return std::make_tuple(std::forward<_CouldBeIndexedType>(_Object)[_Index]...);
}

template <typename _CouldBeIndexedType, size_t... _Index>
_D_Dragonian_Lib_Constexpr_Force_Inline auto MoveAsTuple(_CouldBeIndexedType&& _Object, std::index_sequence<_Index...>)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return std::make_tuple(std::move(std::forward<_CouldBeIndexedType>(_Object)[_Index])...);
}

template <typename _CouldBeIndexedType, size_t... _Index>
_D_Dragonian_Lib_Constexpr_Force_Inline auto ReferenceAsTuple(_CouldBeIndexedType&& _Object, std::index_sequence<_Index...>)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return std::forward_as_tuple(std::forward<_CouldBeIndexedType>(_Object)[_Index]...);
}

template <size_t _Count, typename _CouldBeIndexedType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto MakeTuple(_CouldBeIndexedType&& _Object)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return MakeTuple(std::forward<_CouldBeIndexedType>(_Object), std::make_index_sequence<_Count>{});
}

template <size_t _Count, typename _CouldBeIndexedType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto MoveAsTuple(_CouldBeIndexedType&& _Object)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return MoveAsTuple(std::forward<_CouldBeIndexedType>(_Object), std::make_index_sequence<_Count>{});
}

template <size_t _Count, typename _CouldBeIndexedType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto ReferenceAsTuple(_CouldBeIndexedType&& _Object)
	requires(TypeTraits::CouldIndex<_CouldBeIndexedType>)
{
	return ReferenceAsTuple(std::forward<_CouldBeIndexedType>(_Object), std::make_index_sequence<_Count>{});
}

template <typename _TupleType, size_t... _Index>
_D_Dragonian_Lib_Constexpr_Force_Inline auto SubTupleView(_TupleType&& _Tuple, IndexSequence<_Index...>)
{
	return std::forward_as_tuple(std::get<_Index>(std::forward<_TupleType>(_Tuple))...);
}

template <size_t _Begin = 0, size_t _End = size_t(-1), typename _TupleType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto SubTupleView(_TupleType&& _Tuple)
	requires (_End >= _Begin && _Begin <= std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
{
	constexpr auto MyEnd = std::min(_End, std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>);
	if constexpr (MyEnd == _Begin || _Begin == std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
		return std::tuple();
	else
		return SubTupleView(std::forward<_TupleType>(_Tuple), MakeIndexRange<_Begin, MyEnd>{});
}

template <typename _TupleType, size_t... _Index>
_D_Dragonian_Lib_Constexpr_Force_Inline auto SubTuple(_TupleType&& _Tuple, IndexSequence<_Index...>)
{
	return std::make_tuple(std::get<_Index>(std::forward<_TupleType>(_Tuple))...);
}

template <size_t _Begin = 0, size_t _End = size_t(-1), typename _TupleType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto SubTuple(_TupleType&& _Tuple)
	requires (_End >= _Begin && _Begin <= std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
{
	constexpr auto MyEnd = std::min(_End, std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>);
	if constexpr (MyEnd == _Begin || _Begin == std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
		return std::tuple();
	else
		return SubTuple(std::forward<_TupleType>(_Tuple), MakeIndexRange<_Begin, MyEnd>{});
}

template <size_t _Index, typename _TupleType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto DropElementView(_TupleType&& _Tuple)
	requires (_Index < std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
{
	return std::tuple_cat(
		SubTupleView<0, _Index>(std::forward<_TupleType>(_Tuple)),
		SubTupleView<_Index + 1, std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>>(std::forward<_TupleType>(_Tuple))
	);
}

template <size_t _Index, typename _TupleType>
_D_Dragonian_Lib_Constexpr_Force_Inline auto DropElement(_TupleType&& _Tuple)
	requires (_Index < std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>)
{
	return std::make_tuple(
		std::get<_Index>(std::forward<_TupleType>(_Tuple)),
		std::tuple_cat(
			SubTuple<0, _Index>(std::forward<_TupleType>(_Tuple)),
			SubTuple<_Index + 1, std::tuple_size_v<TypeTraits::RemoveARPCVType<_TupleType>>>(std::forward<_TupleType>(_Tuple))
		)
	);
}

_D_Dragonian_Lib_Template_Library_Space_End

_D_Dragonian_Lib_Space_Begin

enum class Device : UInt8
{
	CPU = 0,
	CUDA,
	HIP,
	DIRECTX,
	CUSTOM
};

static inline size_t NopID = size_t(-1);

namespace Dtl
{
	using namespace _D_Dragonian_Lib_Namespace TemplateLibrary;
}

namespace DragonianLibSTL
{
	using namespace _D_Dragonian_Lib_Namespace TemplateLibrary;
}

_D_Dragonian_Lib_Space_End