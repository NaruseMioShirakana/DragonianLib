/**
 * FileName: Alloc.h
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
#define _D_Dragonian_Lib_Template_Library_Space_Begin namespace DragonianLibSTL {
#define _D_Dragonian_Lib_Template_Library_Space_End }
//Force Inline
#ifdef _MSC_VER
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr __attribute__((always_inline)) inline
#else
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr inline
#endif
#include <memory>

_D_Dragonian_Lib_Template_Library_Space_Begin

class BaseAllocator;
using Allocator = BaseAllocator*;

template <typename ValueType>
class LinearIterator
{
public:
	using value_type = ValueType;
	using difference_type = ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;
	using iterator_category = std::random_access_iterator_tag;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator ValueType* () const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType& operator*() const
	{
		return *_MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType* operator->() const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator++()
	{
		++_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator operator++(int)
	{
		LinearIterator _Tmp = *this;
		++_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator--()
	{
		--_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator operator--(int)
	{
		LinearIterator _Tmp = *this;
		--_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator+=(ptrdiff_t _Off)
	{
		_MyIter += _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator-=(ptrdiff_t _Off)
	{
		_MyIter -= _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator operator+(ptrdiff_t _Off) const
	{
		return LinearIterator(_MyIter + _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator operator-(ptrdiff_t _Off) const
	{
		return LinearIterator(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const LinearIterator& _Right) const
	{
		return _MyIter == _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const LinearIterator& _Right) const
	{
		return _MyIter != _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const LinearIterator& _Right) const
	{
		return _MyIter < _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const LinearIterator& _Right) const
	{
		return _MyIter > _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const LinearIterator& _Right) const
	{
		return _MyIter <= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const LinearIterator& _Right) const
	{
		return _MyIter >= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ValueType* _Right) const
	{
		return _MyIter == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ValueType* _Right) const
	{
		return _MyIter != _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ValueType* _Right) const
	{
		return _MyIter < _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ValueType* _Right) const
	{
		return _MyIter > _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ValueType* _Right) const
	{
		return _MyIter <= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ValueType* _Right) const
	{
		return _MyIter >= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType& operator[](size_t _Off)
	{
		return _MyIter[_Off];
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator=(const LinearIterator& _Right)
	{
		if (this != &_Right)
			_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator=(LinearIterator&& _Right) noexcept
	{
		_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator& operator=(ValueType* _Right)
	{
		_MyIter = _Right;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator(const LinearIterator& _Right)
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator(LinearIterator&& _Right) noexcept
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator(ValueType* _Iter)
		: _MyIter(_Iter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~LinearIterator() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline LinearIterator() = default;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType* Get() const
	{
		return _MyIter;
	}
private:
	ValueType* _MyIter{};
};

template <typename ValueType>
class ConstLinearIterator
{
public:
	using value_type = ValueType;
	using difference_type = ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;
	using iterator_category = std::random_access_iterator_tag;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator const ValueType* () const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType& operator*() const
	{
		return *_MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType* operator->() const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator++()
	{
		++_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator operator++(int)
	{
		ConstLinearIterator _Tmp = *this;
		++_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator--()
	{
		--_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator operator--(int)
	{
		ConstLinearIterator _Tmp = *this;
		--_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator+=(ptrdiff_t _Off)
	{
		_MyIter += _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator-=(ptrdiff_t _Off)
	{
		_MyIter -= _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator operator+(ptrdiff_t _Off) const
	{
		return ConstLinearIterator(_MyIter + _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator operator-(ptrdiff_t _Off) const
	{
		return ConstLinearIterator(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ConstLinearIterator& _Right) const
	{
		return _MyIter == _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ConstLinearIterator& _Right) const
	{
		return _MyIter != _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ConstLinearIterator& _Right) const
	{
		return _MyIter < _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ConstLinearIterator& _Right) const
	{
		return _MyIter > _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ConstLinearIterator& _Right) const
	{
		return _MyIter <= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ConstLinearIterator& _Right) const
	{
		return _MyIter >= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ValueType* _Right) const
	{
		return _MyIter == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ValueType* _Right) const
	{
		return _MyIter != _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ValueType* _Right) const
	{
		return _MyIter < _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ValueType* _Right) const
	{
		return _MyIter > _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ValueType* _Right) const
	{
		return _MyIter <= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ValueType* _Right) const
	{
		return _MyIter >= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType& operator[](size_t _Off) const
	{
		return _MyIter[_Off];
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator=(const ConstLinearIterator& _Right)
	{
		if (this != &_Right)
			_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator=(ConstLinearIterator&& _Right) noexcept
	{
		_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator& operator=(const ValueType* _Right)
	{
		_MyIter = _Right;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator(const ConstLinearIterator& _Right)
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator(ConstLinearIterator&& _Right) noexcept
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator(const ValueType* _Iter)
		: _MyIter(_Iter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator(const LinearIterator<ValueType>& _Right)
		: _MyIter(_Right.Get()) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~ConstLinearIterator() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstLinearIterator() = default;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType* Get() const
	{
		return _MyIter;
	}
private:
	const ValueType* _MyIter{};
};

template <typename ValueType>
class ReversedLinearIterator
{
public:
	using value_type = ValueType;
	using difference_type = ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;
	using iterator_category = std::random_access_iterator_tag;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator ValueType* () const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType& operator*() const
	{
		return *_MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType* operator->() const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator++()
	{
		--_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator operator++(int)
	{
		ReversedLinearIterator _Tmp = *this;
		--_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator--()
	{
		++_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator operator--(int)
	{
		ReversedLinearIterator _Tmp = *this;
		++_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator+=(ptrdiff_t _Off)
	{
		_MyIter -= _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator-=(ptrdiff_t _Off)
	{
		_MyIter += _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator operator+(ptrdiff_t _Off) const
	{
		return ReversedLinearIterator(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator operator-(ptrdiff_t _Off) const
	{
		return ReversedLinearIterator(_MyIter + _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ReversedLinearIterator& _Right) const
	{
		return _MyIter == _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ReversedLinearIterator& _Right) const
	{
		return _MyIter != _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ReversedLinearIterator& _Right) const
	{
		return _MyIter > _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ReversedLinearIterator& _Right) const
	{
		return _MyIter < _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ReversedLinearIterator& _Right) const
	{
		return _MyIter >= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ReversedLinearIterator& _Right) const
	{
		return _MyIter <= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ValueType* _Right) const
	{
		return _MyIter == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ValueType* _Right) const
	{
		return _MyIter != _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ValueType* _Right) const
	{
		return _MyIter > _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ValueType* _Right) const
	{
		return _MyIter < _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ValueType* _Right) const
	{
		return _MyIter >= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ValueType* _Right) const
	{
		return _MyIter <= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType& operator[](size_t _Off)
	{
		return *(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator=(const ReversedLinearIterator& _Right)
	{
		if (this != &_Right)
			_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator=(ReversedLinearIterator&& _Right) noexcept
	{
		_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator& operator=(ValueType* _Right)
	{
		_MyIter = _Right;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator(const ReversedLinearIterator& _Right)
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator(ReversedLinearIterator&& _Right) noexcept
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator(ValueType* _Iter)
		: _MyIter(_Iter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~ReversedLinearIterator() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ReversedLinearIterator() = default;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType* Get() const
	{
		return _MyIter;
	}
private:
	ValueType* _MyIter{};
};

template <typename ValueType>
class ConstReversedLinearIterator
{
public:
	using value_type = ValueType;
	using difference_type = ptrdiff_t;
	using pointer = ValueType*;
	using reference = ValueType&;
	using iterator_category = std::random_access_iterator_tag;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline operator const ValueType* () const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType& operator*() const
	{
		return *_MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType* operator->() const
	{
		return _MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator++()
	{
		--_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator operator++(int)
	{
		ConstReversedLinearIterator _Tmp = *this;
		--_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator--()
	{
		++_MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator operator--(int)
	{
		ConstReversedLinearIterator _Tmp = *this;
		++_MyIter;
		return _Tmp;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator+=(ptrdiff_t _Off)
	{
		_MyIter -= _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator-=(ptrdiff_t _Off)
	{
		_MyIter += _Off;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator operator+(ptrdiff_t _Off) const
	{
		return ConstReversedLinearIterator(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator operator-(ptrdiff_t _Off) const
	{
		return ConstReversedLinearIterator(_MyIter + _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter == _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter != _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter > _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter < _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter >= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ConstReversedLinearIterator& _Right) const
	{
		return _MyIter <= _Right._MyIter;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator==(const ValueType* _Right) const
	{
		return _MyIter == _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator!=(const ValueType* _Right) const
	{
		return _MyIter != _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<(const ValueType* _Right) const
	{
		return _MyIter > _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>(const ValueType* _Right) const
	{
		return _MyIter < _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator<=(const ValueType* _Right) const
	{
		return _MyIter >= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool operator>=(const ValueType* _Right) const
	{
		return _MyIter <= _Right;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType& operator[](size_t _Off) const
	{
		return *(_MyIter - _Off);
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator=(const ConstReversedLinearIterator& _Right)
	{
		if (this != &_Right)
			_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator=(ConstReversedLinearIterator&& _Right) noexcept
	{
		_MyIter = _Right._MyIter;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator& operator=(const ValueType* _Right)
	{
		_MyIter = _Right;
		return *this;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator(const ConstReversedLinearIterator& _Right)
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator(ConstReversedLinearIterator&& _Right) noexcept
		: _MyIter(_Right._MyIter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator(const ValueType* _Iter)
		: _MyIter(_Iter) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator(const ReversedLinearIterator<ValueType>& _Right)
		: _MyIter(_Right.Get()) {}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~ConstReversedLinearIterator() = default;
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReversedLinearIterator() = default;

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType* Get() const
	{
		return _MyIter;
	}
private:
	const ValueType* _MyIter{};
};

template <typename ValueType>
LinearIterator<ValueType> operator+(ptrdiff_t _Off, const LinearIterator<ValueType>& _Iter)
{
	return _Iter + _Off;
}

template <typename ValueType>
ConstLinearIterator<ValueType> operator+(ptrdiff_t _Off, const ConstLinearIterator<ValueType>& _Iter)
{
	return _Iter + _Off;
}

template <typename ValueType>
ReversedLinearIterator<ValueType> operator+(ptrdiff_t _Off, const ReversedLinearIterator<ValueType>& _Iter)
{
	return _Iter + _Off;
}

template <typename ValueType>
ConstReversedLinearIterator<ValueType> operator+(ptrdiff_t _Off, const ConstReversedLinearIterator<ValueType>& _Iter)
{
	return _Iter + _Off;
}

_D_Dragonian_Lib_Template_Library_Space_End

namespace DragonianLib
{
	static inline size_t NopID = size_t(-1);
	using DragonianLibSTL::Allocator;

	enum class Device
	{
		CPU = 0,
		CUDA,
		HIP,
		DIRECTX,
	};

	Allocator GetMemoryProvider(Device _Device);
}

_D_Dragonian_Lib_Template_Library_Space_Begin
using DragonianLib::Device;
using DragonianLib::GetMemoryProvider;
using DragonianLib::NopID;

class BaseAllocator
{
public:
	friend class MemoryProvider;
	virtual ~BaseAllocator() {}
	BaseAllocator(const BaseAllocator&) = default;
	BaseAllocator(BaseAllocator&&) = default;
	BaseAllocator& operator=(const BaseAllocator&) = default;
	BaseAllocator& operator=(BaseAllocator&&) = default;
	virtual unsigned char* Allocate(size_t _Size);
	virtual void Free(void* _Block);
	Device GetDevice() const;
	BaseAllocator(Device _Type) : Type_(_Type) {}
	Device Type_;
};

class CPUAllocator : public BaseAllocator
{
public:
	friend class MemoryProvider;
	~CPUAllocator() override {}
	CPUAllocator(const CPUAllocator&) = default;
	CPUAllocator(CPUAllocator&&) = default;
	CPUAllocator& operator=(const CPUAllocator&) = default;
	CPUAllocator& operator=(CPUAllocator&&) = default;
	unsigned char* Allocate(size_t _Size) override;
	void Free(void* _Block) override;
	CPUAllocator() : BaseAllocator(Device::CPU) {}
};

_D_Dragonian_Lib_Template_Library_Space_End