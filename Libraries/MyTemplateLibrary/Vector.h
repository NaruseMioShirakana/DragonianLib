/**
 * FileName: Vector.h
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
#include "Alloc.h"
#include <initializer_list>
#include <algorithm>
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#define _D_Dragonian_Lib_Stl_Throw(message) _Impl_Dragonian_Lib_Template_Library_Throw_Exception(message, __FILE__, __FUNCSIG__, __LINE__)

_D_Dragonian_Lib_Template_Library_Space_Begin

constexpr size_t _D_Dragonian_Lib_Stl_Unfold_Count = 8;

[[noreturn]] void _Impl_Dragonian_Lib_Template_Library_Throw_Exception(const char* Message, const char* FILE, const char* FUN, int LINE);

template <typename ValueType, typename ...ArgTypes>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_constructible_v<ValueType, ArgTypes...>,
    ValueType&
> _Impl_Dragonian_Lib_Construct_At(ValueType& _Where, ArgTypes&&... _Args)
{
    return *new (std::addressof(_Where)) ValueType(std::forward<ArgTypes>(_Args)...);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void
_Impl_Dragonian_Lib_Destroy_Range(ValueType* _First, ValueType* _Last)
{
    if (_First >= _Last)
        return;

    if constexpr (!std::is_trivially_destructible_v<ValueType>)
    {
        const auto Size = static_cast<size_t>(_Last - _First);
        size_t i = 0;
        if (Size >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= Size - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    (_First + i + j)->~ValueType();
        for (; i < Size; ++i)
            (_First + i)->~ValueType();
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_default_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Default_Construct(ValueType* _Ptr, size_t _Count)
{
    if constexpr (std::is_trivially_copyable_v<ValueType>)
        return;
    else
    {
        size_t i = 0;
        if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    _Impl_Dragonian_Lib_Construct_At(_Ptr[i + j]);
        for (; i < _Count; ++i)
            _Impl_Dragonian_Lib_Construct_At(_Ptr[i]);
    }
}

template <typename DestType, typename SrcType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_assignable_v<DestType, SrcType>
> _Impl_Dragonian_Lib_Iterator_Cast(DestType* _Dest, const SrcType* _Src, size_t _Count)
{
    size_t i = 0;
    if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
        for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
            for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                _Dest[i + j] = _Src[i + j];
    for (; i < _Count; ++i)
        _Dest[i] = _Src[i];
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Copy(ValueType* _Dest, const ValueType* _Src, size_t _Count)
{
    if constexpr (std::is_trivially_copyable_v<ValueType>)
        memcpy(_Dest, _Src, sizeof(ValueType) * _Count);
    else
    {
        size_t i = 0;
        if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    _Dest[i + j] = _Src[i + j];
        for (; i < _Count; ++i)
            _Dest[i] = _Src[i];
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Copy_One(ValueType* _Dest, size_t _Count, const ValueType& _Src)
{
    size_t i = 0;
    if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
        for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
            for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                _Dest[i + j] = _Src;
    for (; i < _Count; ++i)
        _Dest[i] = _Src;
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_move_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Move(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
    if constexpr (std::is_trivially_copyable_v<ValueType>)
        memcpy(_Dest, _Src, sizeof(ValueType) * _Count);
    else
    {
        size_t i = 0;
        if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    _Dest[i + j] = std::move(_Src[i + j]);
        for (; i < _Count; ++i)
            _Dest[i] = std::move(_Src[i]);
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Reversed_Iterator_Copy(ValueType* _Dest, const ValueType* _Src, size_t _Count)
{
    size_t i = 0;
    if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
        for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
            for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                _Dest[_Count - (1 + i + j)] = _Src[_Count - (1 + i + j)];
    for (; i < _Count; ++i)
        _Dest[_Count - (i + 1)] = _Src[_Count - (i + 1)];
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_move_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Reversed_Iterator_Move(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
    size_t i = 0;
    if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
        for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
            for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                _Dest[_Count - (1 + i + j)] = std::move(_Src[_Count - (1 + i + j)]);
    for (; i < _Count; ++i)
        _Dest[_Count - (i + 1)] = std::move(_Src[_Count - (i + 1)]);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Reversed_Iterator_Copy_Construct(ValueType* _Dest, const ValueType* _Src, size_t _Count)
{
	size_t i = 0;
	if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
		for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
			for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
				_Impl_Dragonian_Lib_Construct_At(_Dest[_Count - (1 + i + j)], _Src[_Count - (1 + i + j)]);
	for (; i < _Count; ++i)
		_Impl_Dragonian_Lib_Construct_At(_Dest[_Count - (i + 1)], _Src[_Count - (i + 1)]);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t <
    std::is_move_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Reversed_Iterator_Move_Construct(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
	size_t i = 0;
	if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
		for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
			for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
				_Impl_Dragonian_Lib_Construct_At(_Dest[_Count - (1 + i + j)], std::move(_Src[_Count - (1 + i + j)]));
	for (; i < _Count; ++i)
		_Impl_Dragonian_Lib_Construct_At(_Dest[_Count - (i + 1)], std::move(_Src[_Count - (i + 1)]));
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Copy_Construct(ValueType* _Dest, const ValueType* _Src, size_t _Count)
{
    if constexpr (std::is_trivially_copyable_v<ValueType>)
        memcpy(_Dest, _Src, sizeof(ValueType) * _Count);
    else
    {
        size_t i = 0;
        if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    _Impl_Dragonian_Lib_Construct_At(_Dest[i + j], _Src[i + j]);
        for (; i < _Count; ++i)
            _Impl_Dragonian_Lib_Construct_At(_Dest[i], _Src[i]);
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(ValueType* _Dest, size_t _Count, const ValueType& _Src)
{
    size_t i = 0;
    if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
        for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
            for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                _Impl_Dragonian_Lib_Construct_At(_Dest[i + j], _Src);
    for (; i < _Count; ++i)
        _Impl_Dragonian_Lib_Construct_At(_Dest[i], _Src);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_move_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Move_Construct(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
    if constexpr (std::is_trivially_copyable_v<ValueType>)
        memcpy(_Dest, _Src, sizeof(ValueType) * _Count);
    else
    {
        size_t i = 0;
        if (_Count >= _D_Dragonian_Lib_Stl_Unfold_Count)
            for (; i <= _Count - _D_Dragonian_Lib_Stl_Unfold_Count; i += _D_Dragonian_Lib_Stl_Unfold_Count)
                for (size_t j = 0; j < _D_Dragonian_Lib_Stl_Unfold_Count; ++j)
                    _Impl_Dragonian_Lib_Construct_At(_Dest[i + j], std::move(_Src[i + j]));
        for (; i < _Count; ++i)
            _Impl_Dragonian_Lib_Construct_At(_Dest[i], std::move(_Src[i]));
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Offset(ValueType* _Src, size_t _Count, int64_t Offset)
{
    if (Offset == 0 || _Count == 0)
        return;
    auto _Dest = _Src + Offset;
    if (Offset < 0)
    {
        if constexpr (std::is_move_assignable_v<ValueType>)
            _Impl_Dragonian_Lib_Iterator_Move(_Dest, _Src, _Count);
        else
            _Impl_Dragonian_Lib_Iterator_Copy(_Dest, _Src, _Count);
    }
    else
    {
        if constexpr (std::is_move_assignable_v<ValueType>)
            _Impl_Dragonian_Lib_Reversed_Iterator_Move(_Dest, _Src, _Count);
        else
            _Impl_Dragonian_Lib_Reversed_Iterator_Copy(_Dest, _Src, _Count);
    }
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_constructible_v<ValueType> || std::is_move_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Offset_Construct(ValueType* _Src, size_t _Count, int64_t Offset)
{
	if (Offset == 0 || _Count == 0)
		return;
	auto _Dest = _Src + Offset;
	if (Offset < 0)
	{
		if constexpr (std::is_move_constructible_v<ValueType>)
			_Impl_Dragonian_Lib_Iterator_Move_Construct(_Dest, _Src, _Count);
		else
			_Impl_Dragonian_Lib_Iterator_Copy_Construct(_Dest, _Src, _Count);
	}
	else
	{
		if constexpr (std::is_move_constructible_v<ValueType>)
			_Impl_Dragonian_Lib_Reversed_Iterator_Move_Construct(_Dest, _Src, _Count);
		else
			_Impl_Dragonian_Lib_Reversed_Iterator_Copy_Construct(_Dest, _Src, _Count);
	}
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
    std::is_copy_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Offset_Copy(ValueType* _Src, size_t _Count, int64_t Offset)
{
    if (Offset == 0)
        return;
    auto _Dest = _Src + Offset;
    if (Offset < 0)
        _Impl_Dragonian_Lib_Iterator_Copy(_Dest, _Src, _Count);
    else
        _Impl_Dragonian_Lib_Reversed_Iterator_Copy(_Dest, _Src, _Count);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
	std::is_copy_constructible_v<ValueType> || std::is_move_constructible_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
	if constexpr (std::is_move_constructible_v<ValueType>)
		_Impl_Dragonian_Lib_Iterator_Move_Construct(_Dest, _Src, _Count);
	else
		_Impl_Dragonian_Lib_Iterator_Copy_Construct(_Dest, _Src, _Count);
}

template <typename ValueType>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t <
	std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>
> _Impl_Dragonian_Lib_Iterator_Try_Move_Assign(ValueType* _Dest, ValueType* _Src, size_t _Count)
{
	if constexpr (std::is_move_assignable_v<ValueType>)
		_Impl_Dragonian_Lib_Iterator_Move(_Dest, _Src, _Count);
	else
		_Impl_Dragonian_Lib_Iterator_Copy(_Dest, _Src, _Count);
}

//using Type_ = float;
template <typename Type_>
class Vector
{
public:
    friend std::_Tidy_guard<Vector>;
    using TidyGuard = std::_Tidy_guard<Vector>;

    using ValueType = Type_;
    using Reference = ValueType&;
    using ConstReference = const ValueType&;
    using Pointer = ValueType*;
    using ConstPointer = const ValueType*;
    using Iterator = LinearIterator<ValueType>;
    using ConstIterator = ConstLinearIterator<ValueType>;
	using ReversedIterator = ReversedLinearIterator<ValueType>;
	using ConstReversedIterator = ConstReversedLinearIterator<ValueType>;
    using SizeType = size_t;
    using IndexType = long long;

    static_assert(std::is_copy_assignable_v<ValueType>, "ValueType Must Be Copy Assignable!");
	static_assert(std::is_copy_constructible_v<ValueType>, "ValueType Must Be Copy Constructible!");

protected:
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void _Tidy() noexcept
    {
        _Impl_Dragonian_Lib_Destroy_Range(_MyFirst, _MyLast);
        if (_MyFirst && _MyOwner)
            _MyAllocator->Free(_MyFirst);
        _MyOwner = true;
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
    }

private:
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Destory() noexcept
    {
        _Tidy();
        _MyAllocator = nullptr;
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void AllocateMemory(SizeType _Size)
	{
        if (_Size == 0)
        {
            _MyFirst = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        _MyFirst = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _Size * 2);
        if (!_MyFirst) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyFirst + _Size * 2;
	}

protected:
    Pointer _MyFirst, _MyLast, _MyEnd;
    Allocator _MyAllocator;
    bool _MyOwner = true;

public:
	~Vector() noexcept
    {
        Destory();
    }

	Vector()
    {
        _MyAllocator = GetMemoryProvider(Device::CPU);
        _MyFirst = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
        _MyLast = _MyFirst;
        _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
        return;
    }

	Vector(SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
		if constexpr (std::is_default_constructible_v<ValueType>)
	    {
		    if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
	    	_MyAllocator = _Alloc;
	    	AllocateMemory(_Size);
	    	_Impl_Dragonian_Lib_Iterator_Default_Construct(_MyFirst, _Size);
	    }
        else
			_D_Dragonian_Lib_Stl_Throw("Default Construct Of ValueType Not Allowed!");
    }

	Vector(SizeType _Size, ConstReference _Value, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(_MyFirst, _Size, _Value);
    }

	Vector(Pointer* _Block, SizeType _Size, Allocator _Alloc, bool _Owner = true)
    {
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        _MyOwner = _Owner;
        _MyFirst = *_Block;
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyLast;
        *_Block = nullptr;
    }

	Vector(ConstPointer _Begin, ConstPointer _End, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        auto _Size = _End - _Begin;
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Begin, _Size);
    }

	Vector(const ConstIterator& _Begin, const ConstIterator& _End, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        auto _Size = _End - _Begin;
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
		_Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Begin.Get(), _Size);
    }

	Vector(ConstPointer _Buffer, SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Buffer, _Size);
    }

	Vector(const std::initializer_list<ValueType>& _List, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        auto _Size = _List.size();
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _List.begin(), _Size);
    }

	Vector(const Vector& _Left)
    {
		if constexpr (!std::is_copy_constructible_v<ValueType>)
			_D_Dragonian_Lib_Stl_Throw("Copy Assign Of ValueType Not Allowed!");
        else
        {
	        _MyAllocator = _Left._MyAllocator;
        	auto _Size = _Left.Size();
        	AllocateMemory(_Size);
        	_Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Left._MyFirst, _Size);
        }
    }

	Vector(Vector&& _Right) noexcept
    {
        _MyFirst = _Right._MyFirst;
        _MyLast = _Right._MyLast;
        _MyEnd = _Right._MyEnd;
        _MyAllocator = _Right._MyAllocator;
        _MyOwner = _Right._MyOwner;

        _Right._MyAllocator = nullptr;
        _Right._MyFirst = nullptr;
        _Right._MyLast = nullptr;
        _Right._MyEnd = nullptr;
    }

	Vector& operator=(const Vector& _Left)
    {
        if (&_Left == this)
            return *this;
        if constexpr (!std::is_copy_constructible_v<ValueType>)
            _D_Dragonian_Lib_Stl_Throw("Copy Assign Of ValueType Not Allowed!");
        else
        {
            Destory();
            _MyAllocator = _Left._MyAllocator;
            auto _Size = _Left.Size();
            AllocateMemory(_Size);
            _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Left._MyFirst, _Size);
        }
        return *this;
    }

	Vector& operator=(Vector&& _Right) noexcept
    {
        if (&_Right != this)
            Destory();

        _MyFirst = _Right._MyFirst;
        _MyLast = _Right._MyLast;
        _MyEnd = _Right._MyEnd;
        _MyAllocator = _Right._MyAllocator;
        _MyOwner = _Right._MyOwner;

        _Right._MyAllocator = nullptr;
        _Right._MyFirst = nullptr;
        _Right._MyLast = nullptr;
        _Right._MyEnd = nullptr;

        return *this;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReference operator[](SizeType _Index) const
    {
#ifdef DRAGONIANLIB_DEBUG
        if (size_t(_Index) >= Size())
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        return _MyFirst[_Index];
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference operator[](SizeType _Index)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ConstReference At(SizeType _Index) const
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference At(SizeType _Index)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

public:
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) Begin()
    {
		return Iterator(_MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) End()
    {
		return Iterator(_MyLast);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedBegin()
    {
		return ReversedIterator(_MyLast - 1);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedEnd()
    {
		return ReversedIterator(_MyFirst - 1);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) begin()
    {
        return Iterator(_MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) end()
    {
		return Iterator(_MyLast);
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) rbegin()
	{
		return ReversedIterator(_MyLast - 1);
	}

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) rend()
    {
		return ReversedIterator(_MyFirst - 1);
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) Begin() const
	{
		return ConstIterator(_MyFirst);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) End() const
	{
		return ConstIterator(_MyLast);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedBegin() const
	{
		return ConstReversedIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedEnd() const
	{
		return ConstReversedIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) begin() const
	{
		return ConstIterator(_MyFirst);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) end() const
	{
		return ConstIterator(_MyLast);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) rbegin() const
	{
		return ConstReversedIterator(_MyLast - 1);
	}

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) rend() const
    {
		return ConstReversedIterator(_MyFirst - 1);
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) FrontReversedBegin()
	{
		return LinearIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) FrontReversedEnd()
	{
		return LinearIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) FrontReversedBegin() const
	{
		return ConstLinearIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) FrontReversedEnd() const
	{
		return ConstLinearIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Size() const
    {
        return static_cast<SizeType>(_MyLast - _MyFirst);
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Capacity() const
    {
        return static_cast<SizeType>(_MyEnd - _MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::pair<Pointer, SizeType> Release()
    {
        auto Ptr = _MyFirst;
        auto _Size = Size();
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
        _MyAllocator = nullptr;
        return { Ptr, _Size };
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType* Data()
    {
        return std::_Unfancy_maybe_null(_MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const ValueType* Data() const
    {
        return std::_Unfancy_maybe_null(_MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Allocator GetAllocator() const
    {
        return _MyAllocator;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference Back() const
    {
        return *(_MyLast - 1);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference Front() const
    {
        return *(_MyFirst);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool Empty() const
    {
        return _MyFirst == _MyLast;
    }

private:
    template<typename... _ArgsTy>
    static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>,
        Reference> EmplaceImpl(Reference _Obj, _ArgsTy &&... _Args)
    {
		return _Impl_Dragonian_Lib_Construct_At(_Obj, std::forward<_ArgsTy>(_Args)...);
    }

    template<typename... _ArgsTy>
	std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>
    > EmplaceImpl(const ConstIterator& _Where, _ArgsTy&&... _Args)
    {
        constexpr auto _Size = 1ll;
        const auto _MySize = _MyLast - _MyFirst;
        const auto _NewSize = _MySize + _Size;
        const auto _FrontCount = _Where - _MyFirst;
        const auto _Remainder = _MySize - _FrontCount;
        if (_NewSize > static_cast<int64_t>(Capacity()))
        {
            auto _NewBuffer = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _NewSize * 2);
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer, _MyFirst, _FrontCount
            );
            _Impl_Dragonian_Lib_Construct_At(
				*(_NewBuffer + _FrontCount), std::forward<_ArgsTy>(_Args)...
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer + _FrontCount + _Size, _MyFirst + _FrontCount, _Remainder
            );
            _Tidy();
            _MyFirst = _NewBuffer;
            _MyLast = _NewBuffer + _NewSize;
            _MyEnd = _NewBuffer + _NewSize * 2;
        }
        else
        {
            const auto NeedConstruct = _Size - _Remainder;
            const auto SrcConstructCountSub = std::min(0ll, NeedConstruct);
            const auto SrcNoConstructCount = abs(SrcConstructCountSub);
            _Impl_Dragonian_Lib_Iterator_Offset_Construct(
                _MyFirst + _FrontCount + SrcNoConstructCount,
                _Remainder + SrcConstructCountSub,
                _Size
            );
            _Impl_Dragonian_Lib_Iterator_Offset(
                _MyFirst + _FrontCount,
                SrcNoConstructCount,
                _Size
            );
            const auto DstConstructCount = std::max(0ll, NeedConstruct);
            const auto DstNoConstructCount = _Size - DstConstructCount;
            _Impl_Dragonian_Lib_Destroy_Range(
                _MyFirst + _FrontCount,
                _MyFirst + _FrontCount + DstNoConstructCount
            );
            _Impl_Dragonian_Lib_Construct_At(
                *(_MyFirst + _FrontCount), std::forward<_ArgsTy>(_Args)...
            );
            _MyLast = _MyFirst + _NewSize;
        }
    }

	void InsertImpl(const ConstIterator& _Where, const ConstIterator& _Begin, const ConstIterator& _End)
    {
        const auto _Size = _End - _Begin;
        const auto _MySize = _MyLast - _MyFirst;
        const auto _NewSize = _MySize + _Size;
        const auto _FrontCount = _Where - _MyFirst;
        const auto _Remainder = _MySize - _FrontCount;
        if (_NewSize > static_cast<int64_t>(Capacity()))
        {
            auto _NewBuffer = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _NewSize * 2);
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer, _MyFirst, _FrontCount
            );
            _Impl_Dragonian_Lib_Iterator_Copy_Construct(
                _NewBuffer + _FrontCount, _Begin.Get(), _Size
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer + _FrontCount + _Size, _MyFirst + _FrontCount, _Remainder
            );
            _Tidy();
            _MyFirst = _NewBuffer;
            _MyLast = _NewBuffer + _NewSize;
            _MyEnd = _NewBuffer + _NewSize * 2;
        }
        else
        {
            const auto NeedConstruct = _Size - _Remainder;
            const auto SrcConstructCountSub = std::min(0ll, NeedConstruct);
            const auto SrcNoConstructCount = abs(SrcConstructCountSub);
            _Impl_Dragonian_Lib_Iterator_Offset_Construct(
                _MyFirst + _FrontCount + SrcNoConstructCount,
                _Remainder + SrcConstructCountSub,
                _Size
            );
            _Impl_Dragonian_Lib_Iterator_Offset(
                _MyFirst + _FrontCount,
                SrcNoConstructCount,
                _Size
            );
            const auto DstConstructCount = std::max(0ll, NeedConstruct);
            const auto DstNoConstructCount = _Size - DstConstructCount;
            _Impl_Dragonian_Lib_Iterator_Copy_Construct(
                _MyFirst + _FrontCount + DstNoConstructCount,
                _End.Get() - DstConstructCount,
                DstConstructCount
            );
            _Impl_Dragonian_Lib_Iterator_Copy(
                _MyFirst + _FrontCount,
                _Begin.Get(),
                DstNoConstructCount
            );
            _MyLast = _MyFirst + _NewSize;
        }
    }

    void InsertImpl(const ConstIterator& _Where, SizeType _Count, const ValueType& _Value)
    {
        const auto _Size = static_cast<int64_t>(_Count);
        const auto _MySize = _MyLast - _MyFirst;
        const auto _NewSize = _MySize + _Size;
        const auto _FrontCount = _Where - _MyFirst;
        const auto _Remainder = _MySize - _FrontCount;
        if (_NewSize > static_cast<int64_t>(Capacity()))
        {
            auto _NewBuffer = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _NewSize * 2);
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer, _MyFirst, _FrontCount
            );
            _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(
                _NewBuffer + _FrontCount, _Size, _Value
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer + _FrontCount + _Size, _MyFirst + _FrontCount, _Remainder
            );
            _Tidy();
            _MyFirst = _NewBuffer;
            _MyLast = _NewBuffer + _NewSize;
            _MyEnd = _NewBuffer + _NewSize * 2;
        }
        else
        {
            const auto NeedConstruct = _Size - _Remainder;
            const auto SrcConstructCountSub = std::min(0ll, NeedConstruct);
            const auto SrcNoConstructCount = abs(SrcConstructCountSub);
            _Impl_Dragonian_Lib_Iterator_Offset_Construct(
                _MyFirst + _FrontCount + SrcNoConstructCount,
                _Remainder + SrcConstructCountSub,
                _Size
            );
            _Impl_Dragonian_Lib_Iterator_Offset(
                _MyFirst + _FrontCount,
                SrcNoConstructCount,
                _Size
            );
            const auto DstConstructCount = std::max(0ll, NeedConstruct);
            const auto DstNoConstructCount = _Size - DstConstructCount;
            _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(
                _MyFirst + _FrontCount + DstNoConstructCount,
                DstConstructCount,
                _Value
            );
            _Impl_Dragonian_Lib_Iterator_Copy_One(
                _MyFirst + _FrontCount,
                DstNoConstructCount,
                _Value
            );
            _MyLast = _MyFirst + _NewSize;
        }
    }

	void MoveInsertImpl(const ConstIterator& _Where, const Iterator& _Begin, const Iterator& _End)
    {
        const auto _Size = _End - _Begin;
        const auto _MySize = _MyLast - _MyFirst;
        const auto _NewSize = _MySize + _Size;
        const auto _FrontCount = _Where - _MyFirst;
        const auto _Remainder = _MySize - _FrontCount;
        if (_NewSize > Capacity())
        {
            auto _NewBuffer = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _NewSize * 2);
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer, _MyFirst, _FrontCount
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer + _FrontCount, _Begin.Get(), _Size
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _NewBuffer + _FrontCount + _Size, _MyFirst + _FrontCount, _Remainder
            );
            _Tidy();
            _MyFirst = _NewBuffer;
            _MyLast = _NewBuffer + _NewSize;
            _MyEnd = _NewBuffer + _NewSize * 2;
        }
        else
        {
            const auto NeedConstruct = _Size - _Remainder;
            const auto SrcConstructCountSub = std::min(0ll, NeedConstruct);
            const auto SrcNoConstructCount = abs(SrcConstructCountSub);
            _Impl_Dragonian_Lib_Iterator_Offset_Construct(
                _MyFirst + _FrontCount + SrcNoConstructCount,
                _Remainder + SrcConstructCountSub,
                _Size
            );
            _Impl_Dragonian_Lib_Iterator_Offset(
                _MyFirst + _FrontCount,
                SrcNoConstructCount,
                _Size
            );
            const auto DstConstructCount = std::max(0ll, NeedConstruct);
            const auto DstNoConstructCount = _Size - DstConstructCount;
            _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
                _MyFirst + _FrontCount + DstNoConstructCount,
                _End.Get() - DstConstructCount,
                DstConstructCount
            );
            _Impl_Dragonian_Lib_Iterator_Try_Move_Assign(
                _MyFirst + _FrontCount,
                _Begin.Get(),
                DstNoConstructCount
            );
            _MyLast = _MyFirst + _NewSize;
        }
    }

public:
	void Reserve(SizeType _NewCapacity)
    {
        if (_NewCapacity == Capacity())
            return;

        if (_NewCapacity == 0)
        {
            _Tidy();
            _MyFirst = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        auto _Data = (Pointer)_MyAllocator->Allocate(sizeof(ValueType) * _NewCapacity);
        auto _Size = std::min(_NewCapacity, Size());

        _Impl_Dragonian_Lib_Iterator_Try_Move_Construct(
            _Data, _MyFirst, _Size
        );

        _Tidy();
        _MyFirst = _Data;
        _MyLast = _Data + _Size;
        _MyEnd = _Data + _NewCapacity;
    }

	template <typename TmpTy = ValueType>
	std::enable_if_t<
		std::is_default_constructible_v<TmpTy>&& std::is_same_v<TmpTy, ValueType>
	> Resize(SizeType _NewSize)
    {
        if (_NewSize == Size())
            return;
        if (_NewSize < Size())
        {
            _Impl_Dragonian_Lib_Destroy_Range(
                _MyFirst + _NewSize,
                _MyLast
            );
			_MyLast = _MyFirst + _NewSize;
            return;
        }

        if (_NewSize > Capacity())
            Reserve(_NewSize * 2);

		_Impl_Dragonian_Lib_Iterator_Default_Construct(
			_MyLast, _NewSize - Size()
		);
        _MyLast = _MyFirst + _NewSize;
    }

	void Resize(SizeType _NewSize, const ValueType& _Value)
    {
        if (_NewSize == Size())
            return;
        if (_NewSize < Size())
        {
            _Impl_Dragonian_Lib_Destroy_Range(
                _MyFirst + _NewSize,
                _MyLast
            );
            _MyLast = _MyFirst + _NewSize;
            return;
        }

        if (_NewSize >= Capacity())
            Reserve(_NewSize * 2);

        _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(
			_MyLast, _NewSize - Size(), _Value
        );
        _MyLast = _MyFirst + _NewSize;
    }

    template<typename... _ArgsTy>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>,
        Reference> Emplace(const ConstIterator& _Where, _ArgsTy &&... _Args)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
        EmplaceImpl(_Where, std::forward<_ArgsTy>(_Args)...);
		return *(_MyFirst + Idx);
    }

    template<typename... _ArgsTy>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>,
        Reference> EmplaceBack(_ArgsTy &&... _Args)
    {
		if (_MyLast == _MyEnd)
			Reserve(Capacity() * 2);
		return EmplaceImpl(*_MyLast++, std::forward<_ArgsTy>(_Args)...);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference Insert(const ConstIterator& _Where, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return Emplace(_Where, _Value);
    }

    template <typename TmpTy = ValueType>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
        std::is_move_assignable_v<TmpTy>&& std::is_move_constructible_v<TmpTy>&& std::is_same_v<TmpTy, ValueType>,
        Reference> Insert(const ConstIterator& _Where, ValueType&& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        return Emplace(_Where, std::move(_Value));
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Insert(const ConstIterator& _Where, SizeType _Count, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        InsertImpl(_Where, _Count, _Value);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Insert(const ConstIterator& _Where, const ConstIterator& _First, const ConstIterator& _Last)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
		if (_First > _Last)
			_D_Dragonian_Lib_Stl_Throw("Invalid Range!");
#endif
		InsertImpl(_Where, _First, _Last);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void MoveTo(const ConstIterator& _Where, const Iterator& _First, const Iterator& _Last)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
		if (_First > _Last)
			_D_Dragonian_Lib_Stl_Throw("Invalid Range!");
#endif
		MoveInsertImpl(_Where, _First, _Last);
    }

	ValueType Erase(const ConstIterator& _Where)
    {
#ifdef DRAGONIANLIB_DEBUG
		if (_Where >= _MyLast || _Where < _MyFirst)
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		const auto _Idx = _Where - _MyFirst;
        auto _Value = std::move(*(_MyFirst + _Idx));
        _Impl_Dragonian_Lib_Iterator_Offset(
            _MyFirst + _Idx + 1,
            1,
            -1
        );
		--_MyLast;
        if constexpr (!std::is_trivially_copyable_v<ValueType>)
            _MyLast->~ValueType();
		return _Value;
	}

	void Erase(const ConstIterator& _First, const ConstIterator& _Last)
	{
#ifdef DRAGONIANLIB_DEBUG
        if (_Last >= _MyLast || _First < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
		if (_First >= _Last)
			_D_Dragonian_Lib_Stl_Throw("Invalid Range!");
#endif
        const auto _Idx = _First - _MyFirst;
		const auto _Count = _Last - _First;
		const auto _MySize = _MyLast - _MyFirst;
		_Impl_Dragonian_Lib_Iterator_Offset(
			_MyFirst + _Idx + _Count,
			_MySize - _Idx - _Count,
			-(_Count)
		);
        _Impl_Dragonian_Lib_Destroy_Range(
			_MyFirst + _MySize - _Count,
			_MyLast
        );
		_MyLast -= _Count;
    }

	void Clear()
    {
        _Impl_Dragonian_Lib_Destroy_Range(
			_MyFirst,
			_MyLast
		);
        _MyLast = _MyFirst;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void PopBack()
    {
        --_MyLast;
        if constexpr (!std::is_trivially_copyable_v<ValueType>)
            _MyLast->~ValueType();
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector operator+(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) += ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector operator-(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) -= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector operator*(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) *= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector operator/(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) /= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector operator^(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) = (ValueType)pow(*(Iter++), _Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator+=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) += ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator-=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) -= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator*=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) *= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator/=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) /= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator^=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) = (ValueType)pow(*(Iter++), _Val);
        return *this;
    }
};

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator+(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
    if (_ValA.Size() != _ValB.Size())
        _D_Dragonian_Lib_Stl_Throw("Size MisMatch!");
    auto Temp = _ValA;
    auto Iter = Temp.Data();
    auto ValIter = _ValB.Data();
    while (Iter != Temp.End())
        *(Iter++) += _TypeA(*(ValIter++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator-(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
    if (_ValA.Size() != _ValB.Size())
        _D_Dragonian_Lib_Stl_Throw("Size MisMatch!");
    auto Temp = _ValA;
    auto Iter = Temp.Data();
    auto ValIter = _ValB.Data();
    while (Iter != Temp.End())
        *(Iter++) -= _TypeA(*(ValIter++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator*(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
    if (_ValA.Size() != _ValB.Size())
        _D_Dragonian_Lib_Stl_Throw("Size MisMatch!");
    auto Temp = _ValA;
    auto Iter = Temp.Data();
    auto ValIter = _ValB.Data();
    while (Iter != Temp.End())
        *(Iter++) *= _TypeA(*(ValIter++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator/(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
    if (_ValA.Size() != _ValB.Size())
        _D_Dragonian_Lib_Stl_Throw("Size MisMatch!");
    auto Temp = _ValA;
    auto Iter = Temp.Data();
    auto ValIter = _ValB.Data();
    while (Iter != Temp.End())
        *(Iter++) /= _TypeA(*(ValIter++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator^(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
    if (_ValA.Size() != _ValB.Size())
        _D_Dragonian_Lib_Stl_Throw("Size MisMatch!");
    auto Temp = _ValA;
    auto Iter = Temp.Data();
    auto ValIter = _ValB.Data();
    while (Iter != Temp.End())
        *(Iter++) = (_TypeA)pow(*(Iter++), *(ValIter++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator+(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
    Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
    auto IterA = Temp.Data();
    auto IterB = _ValB.Data();
    while (IterA != Temp.End())
        *(IterA++) = _ValA + (_TypeA)(*(IterB++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator-(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
    Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
    auto IterA = Temp.Data();
    auto IterB = _ValB.Data();
    while (IterA != Temp.End())
        *(IterA++) = _ValA + (_TypeA)(*(IterB++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator*(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
    Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
    auto IterA = Temp.Data();
    auto IterB = _ValB.Data();
    while (IterA != Temp.End())
        *(IterA++) = _ValA + (_TypeA)(*(IterB++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator/(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
    Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
    auto IterA = Temp.Data();
    auto IterB = _ValB.Data();
    while (IterA != Temp.End())
        *(IterA++) = _ValA + (_TypeA)(*(IterB++));
    return Temp;
}

template <typename _TypeA, typename _TypeB>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_TypeA> operator^(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
    Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
    auto IterA = Temp.Data();
    auto IterB = _ValB.Data();
    while (IterA != Temp.End())
        *(IterA++) = (_TypeA)pow(_ValA, *(IterB++));
    return Temp;
}

/**
 * @brief Generate an arithmetic progression within the range [Start, End) and step size Step.
 * @tparam Type The type of the elements in the vector.
 * @param Start Start value.
 * @param End End value.
 * @param Step Step size.
 * @param NDiv Division factor.
 * @return
 */
template <typename Type>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<Type> Arange(Type Start, Type End, Type Step = Type(1.), Type NDiv = Type(1.))
{
    Vector<Type> OutPut(size_t((End - Start) / Step));
    auto OutPutPtr = OutPut.Begin();
    const auto OutPutPtrEnd = OutPut.End();
    while (OutPutPtr != OutPutPtrEnd)
    {
        *(OutPutPtr++) = Start / NDiv;
        Start += Step;
    }
    return OutPut;
}

/**
 * @brief Mean filter for 1D signal.
 * @tparam _Type The type of the elements in the vector.
 * @param _Signal Input signal.
 * @param _WindowSize Window size.
 * @return Signal after mean filter.
 */
template <typename _Type>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<_Type> MeanFliter(const Vector<_Type>& _Signal, size_t _WindowSize)
{
    Vector<_Type> Result(_Signal.Size());

    if (_WindowSize > _Signal.Size() || _WindowSize < 2)
        return _Signal;

    auto WndSz = (_Type)(_WindowSize % 2 ? _WindowSize : _WindowSize + 1);

    const size_t half = _WindowSize / 2;
    auto Ptr = Result.Data();

    for (size_t i = 0; i < half; ++i)
        *(Ptr++) = _Signal[i];

    for (size_t i = half; i < _Signal.Size() - half; i++) {
        _Type sum = 0.0f;
        for (size_t j = i - half; j <= i + half; j++)
            sum += _Signal[j];
        *(Ptr++) = (sum / WndSz);
    }

    for (size_t i = _Signal.Size() - half; i < _Signal.Size(); ++i)
        *(Ptr++) = _Signal[i];

    return Result;
}

/**
 * @brief Calculate the average value of the elements in the range [Start, End].
 * @tparam T The type of the elements in the range.
 * @param Start The start of the range.
 * @param End The end of the range.
 * @return The average value of the elements in the range.
 */
template<typename T>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline double Average(const T* Start, const T* End)
{
    const auto Size = End - Start + 1;
    double Avg = (double)(*Start);
    for (auto i = 1; i < Size; i++)
        Avg = Avg + (abs((double)Start[i]) - Avg) / (double)(i + 1ull);
    return Avg;
}

/**
 * @brief Calculate the size of the resampled signal.
 * @param SrcSize Source signal size.
 * @param SrcSamplingRate Source signal sampling rate.
 * @param DstSamplingRate Destination signal sampling rate.
 * @return Size of the resampled signal.
 */
inline size_t CalculateResampledSize(size_t SrcSize, double SrcSamplingRate, double DstSamplingRate) {
    return static_cast<size_t>(ceil(double(SrcSize) * DstSamplingRate / SrcSamplingRate));
}

/**
 * @brief Resample the signal.
 * @tparam TypeInput The type of the source signal.
 * @tparam TypeOutput The type of the destination signal.
 * @param SrcBuffer Source signal buffer.
 * @param SrcSize Source signal size.
 * @param DstBuffer Destination signal buffer.
 * @param DstSize Destination signal size.
 * @param Div Division factor.
 */
template<typename TypeInput, typename TypeOutput>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Resample(
    const TypeInput* SrcBuffer,
    size_t SrcSize,
    TypeOutput* DstBuffer,
    size_t DstSize,
    TypeOutput Div
) {
    if (SrcSize == 0 || DstSize == 0) _D_Dragonian_Lib_Stl_Throw("Invalid size!");
    if (SrcSize == DstSize) {
        for (size_t i = 0; i < SrcSize; ++i)
            DstBuffer[i] = static_cast<TypeOutput>(SrcBuffer[i]) / Div;
        return;
    }
    const double scale = static_cast<double>(SrcSize - 1) / static_cast<double>(DstSize - 1);
    for (size_t i = 0; i < DstSize; ++i) {
        const double srcPos = static_cast<double>(i) * scale;
        const size_t srcIndex = static_cast<size_t>(srcPos);
        const double frac = srcPos - static_cast<double>(srcIndex);

        if (srcIndex >= SrcSize - 1)
            DstBuffer[i] = static_cast<TypeOutput>(SrcBuffer[SrcSize - 1] / Div);
        else
        {
            TypeInput val1 = SrcBuffer[srcIndex];
            TypeInput val2 = SrcBuffer[srcIndex + 1];
            if (isnan(val1) && isnan(val2))
                DstBuffer[i] = std::numeric_limits<TypeOutput>::quiet_NaN();
            else if (isnan(val1))
                DstBuffer[i] = static_cast<TypeOutput>(val2 / Div);
            else if (isnan(val2))
                DstBuffer[i] = static_cast<TypeOutput>(val1 / Div);
            else
                DstBuffer[i] = static_cast<TypeOutput>((val1 * (1.0 - frac) + val2 * frac) / Div);
        }
    }
}

/**
 * @brief Resample the signal.
 * @tparam TypeInput The type of the source signal.
 * @tparam TypeOutput The type of the destination signal.
 * @param SrcBuffer Source signal buffer.
 * @param SrcSize Source signal size.
 * @param DstBuffer Destination signal buffer.
 * @param DstSize Destination signal size.
 */
template<typename TypeInput, typename TypeOutput>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Resample(
    const TypeInput* SrcBuffer,
    size_t SrcSize,
    TypeOutput* DstBuffer,
    size_t DstSize
) {
    if (SrcSize == 0 || DstSize == 0) _D_Dragonian_Lib_Stl_Throw("Invalid size!");
    if (SrcSize == DstSize) {
        for (size_t i = 0; i < SrcSize; ++i)
            DstBuffer[i] = static_cast<TypeOutput>(SrcBuffer[i]);
        return;
    }
    const double scale = static_cast<double>(SrcSize - 1) / static_cast<double>(DstSize - 1);
    for (size_t i = 0; i < DstSize; ++i) {
        const double srcPos = static_cast<double>(i) * scale;
        const size_t srcIndex = static_cast<size_t>(srcPos);
        const double frac = srcPos - double(srcIndex);

        if (srcIndex >= SrcSize - 1)
            DstBuffer[i] = static_cast<TypeOutput>(SrcBuffer[SrcSize - 1]);
        else
        {
            TypeInput val1 = SrcBuffer[srcIndex];
            TypeInput val2 = SrcBuffer[srcIndex + 1];
            if (isnan(val1) && isnan(val2))
                DstBuffer[i] = std::numeric_limits<TypeOutput>::quiet_NaN();
            else if (isnan(val1))
                DstBuffer[i] = static_cast<TypeOutput>(val2);
            else if (isnan(val2))
                DstBuffer[i] = static_cast<TypeOutput>(val1);
            else
                DstBuffer[i] = static_cast<TypeOutput>((val1 * (1.0 - frac) + val2 * frac));
        }
    }
}

/**
 * @brief Resample the signal.
 * @tparam TypeOutput The type of the destination signal.
 * @tparam TypeInput The type of the source signal.
 * @param Data Source signal.
 * @param SrcSamplingRate Source signal sampling rate.
 * @param DstSamplingRate Destination signal sampling rate.
 * @param Div Division factor.
 * @return Resampled signal.
 */
template<typename TypeOutput, typename TypeInput>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<TypeOutput> InterpResample(
    const Vector<TypeInput>& Data,
    long SrcSamplingRate,
    long DstSamplingRate,
    TypeOutput Div
)
{
    Vector<TypeOutput> Output(CalculateResampledSize(Data.Size(), (double)SrcSamplingRate, (double)DstSamplingRate));
    Resample(Data.Data(), Data.Size(), Output.Data(), Output.Size(), Div);
    return Output;
}

/**
 * @brief Resample the signal.
 * @tparam TypeOutput The type of the destination signal.
 * @tparam TypeInput The type of the source signal.
 * @param Data Source signal.
 * @param SrcSamplingRate Source signal sampling rate.
 * @param DstSamplingRate Destination signal sampling rate.
 * @return Resampled signal.
 */
template<typename TypeOutput, typename TypeInput>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<TypeOutput> InterpResample(
    const Vector<TypeInput>& Data,
    long SrcSamplingRate,
    long DstSamplingRate
)
{
    Vector<TypeOutput> Output(CalculateResampledSize(Data.Size(), (double)SrcSamplingRate, (double)DstSamplingRate));
    Resample(Data.Data(), Data.Size(), Output.Data(), Output.Size());
    return Output;
}

/**
 * @brief Cast the elements in the vector.
 * @tparam TypeOutput The type of the destination signal.
 * @tparam TypeInput The type of the source signal.
 * @param Data Source signal.
 * @return Signal.
 */
template<typename TypeOutput, typename TypeInput>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<TypeOutput> SignalCast(
    const Vector<TypeInput>& Data
)
{
	Vector<TypeOutput> Output(Data.Size());
	for (size_t i = 0; i < Data.Size(); ++i)
		Output[i] = static_cast<TypeOutput>(Data[i]);
	return Output;
}

/**
 * @brief Resample the signal.
 * @tparam T The type of the signal.
 * @param _Data Source signal.
 * @param _SrcSamplingRate Source signal sampling rate.
 * @param _DstSamplingRate Destination signal sampling rate.
 * @return Resampled signal.
 */
template<typename T>
_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector<T> InterpFunc(
    const Vector<T>& _Data,
    long _SrcSamplingRate,
    long _DstSamplingRate
)
{
    if constexpr (!std::is_floating_point_v<T>)
        return InterpResample<T, T>(_Data, _SrcSamplingRate, _DstSamplingRate);

    Vector<T> Temp(_Data.Size());
    for (size_t i = 0; i < _Data.Size(); ++i)
        Temp[i] = _Data[i] < 0.001 ? NAN : _Data[i];
    auto Output = InterpResample<T, T>(Temp, _SrcSamplingRate, _DstSamplingRate);
    for (auto& f0 : Output) if (isnan(f0)) f0 = 0;
    return Output;
}

_D_Dragonian_Lib_Template_Library_Space_End
