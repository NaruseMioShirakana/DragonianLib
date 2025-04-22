﻿/**
 * @file Vector.h
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
 * @brief Vector type of DragonianLib
 * @changes
 *  > 2025/3/19 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Alloc.h"
#include "Iterator.h"
#include <initializer_list>
#include <algorithm>

_D_Dragonian_Lib_Template_Library_Space_Begin

constexpr struct VectorViewPlaceholder {} TypeVectorView;

//using Type_ = float;
template <typename Type_, Device Device_ = Device::CPU>
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
    using Allocator = GetAllocatorType<Device_>;
	static constexpr auto _MyDevice = Device_;

protected:
    _D_Dragonian_Lib_Constexpr_Force_Inline void _Tidy() noexcept
    {
        _Impl_Dragonian_Lib_Destroy_Range(_MyFirst, _MyLast);
        if (_MyFirst && _MyOwner)
            _MyAllocator.deallocate(_MyFirst);
        _MyOwner = true;
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
    }

private:
    _D_Dragonian_Lib_Constexpr_Force_Inline void Destory() noexcept
    {
        _Tidy();
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline void AllocateMemory(SizeType _Size)
	{
        if (_Size == 0)
        {
            _MyFirst = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        _MyFirst = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _Size * 2);
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

    Vector(Allocator _Allocator = Allocator())
	{
        _MyAllocator = _Allocator;
        _MyFirst = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
        _MyLast = _MyFirst;
        _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
	}

	Vector(SizeType _Size, Allocator _Alloc = Allocator())
    {
		if constexpr (std::is_default_constructible_v<ValueType>)
	    {
	    	_MyAllocator = _Alloc;
	    	AllocateMemory(_Size);
	    	_Impl_Dragonian_Lib_Iterator_Default_Construct(_MyFirst, _Size);
	    }
        else
			_D_Dragonian_Lib_Stl_Throw("Default Construct Of ValueType Not Allowed!");
    }

	Vector(SizeType _Size, ConstReference _Value, Allocator _Alloc = Allocator())
    {
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct_One(_MyFirst, _Size, _Value);
    }

	Vector(Pointer* _Block, SizeType _Size, Allocator _Alloc, bool _Owner = true)
    {
        _MyAllocator = _Alloc;
        _MyOwner = _Owner;
        _MyFirst = *_Block;
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyLast;
        *_Block = nullptr;
    }

    Vector(VectorViewPlaceholder, Pointer _Block, SizeType _Size)
    {
        _MyOwner = false;
        _MyFirst = _Block;
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyLast;
    }

    static Vector CreateView(Pointer _Block, SizeType _Size, Allocator _Alloc)
	{
        Vector _Result;
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _Result._MyAllocator = _Alloc;
        _Result._MyOwner = false;
        _Result._MyFirst = _Block;
        _Result._MyLast = _Result._MyFirst + _Size;
        _Result._MyEnd = _Result._MyLast;
        return _Result;
	}

	Vector(ConstPointer _Begin, ConstPointer _End, Allocator _Alloc = Allocator())
    {
        auto _Size = _End - _Begin;
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Begin, _Size);
    }

	Vector(const ConstIterator& _Begin, const ConstIterator& _End, Allocator _Alloc = Allocator())
    {
        auto _Size = _End - _Begin;
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
		_Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Begin.Get(), _Size);
    }

	Vector(ConstPointer _Buffer, SizeType _Size, Allocator _Alloc = Allocator())
    {
        if (!_Alloc || _Size < 0) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyAllocator = _Alloc;
        AllocateMemory(_Size);
        _Impl_Dragonian_Lib_Iterator_Copy_Construct(_MyFirst, _Buffer, _Size);
    }
    
	Vector(const std::initializer_list<ValueType>& _List, Allocator _Alloc = Allocator())
    {
        auto _Size = _List.size();
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

        _Right._MyFirst = nullptr;
        _Right._MyLast = nullptr;
        _Right._MyEnd = nullptr;

        return *this;
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline ConstReference operator[](SizeType _Index) const
    {
#ifdef DRAGONIANLIB_DEBUG
        if (size_t(_Index) >= Size())
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        return _MyFirst[_Index];
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline Reference operator[](SizeType _Index)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline ConstReference At(SizeType _Index) const
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline Reference At(SizeType _Index)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

public:
    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin()
    {
		return Iterator(_MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End()
    {
		return Iterator(_MyLast);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ReversedBegin()
    {
		return ReversedIterator(_MyLast - 1);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ReversedEnd()
    {
		return ReversedIterator(_MyFirst - 1);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin()
    {
        return Iterator(_MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end()
    {
		return Iterator(_MyLast);
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) rbegin()
	{
		return ReversedIterator(_MyLast - 1);
	}

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) rend()
    {
		return ReversedIterator(_MyFirst - 1);
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const
	{
		return ConstIterator(_MyFirst);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const
	{
		return ConstIterator(_MyLast);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ReversedBegin() const
	{
		return ConstReversedIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ReversedEnd() const
	{
		return ConstReversedIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin() const
	{
		return ConstIterator(_MyFirst);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end() const
	{
		return ConstIterator(_MyLast);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) rbegin() const
	{
		return ConstReversedIterator(_MyLast - 1);
	}

    _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) rend() const
    {
		return ConstReversedIterator(_MyFirst - 1);
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) FrontReversedBegin()
	{
		return LinearIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) FrontReversedEnd()
	{
		return LinearIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) FrontReversedBegin() const
	{
		return ConstLinearIterator(_MyLast - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) FrontReversedEnd() const
	{
		return ConstLinearIterator(_MyFirst - 1);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Size() const
    {
        return static_cast<SizeType>(_MyLast - _MyFirst);
    }

	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Capacity() const
    {
        return static_cast<SizeType>(_MyEnd - _MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline std::pair<Pointer, SizeType> Release()
    {
        auto Ptr = _MyFirst;
        auto _Size = Size();
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
        return { Ptr, _Size };
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline ValueType* Data()
    {
        return std::_Unfancy_maybe_null(_MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline const ValueType* Data() const
    {
        return std::_Unfancy_maybe_null(_MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline Allocator GetAllocator() const
    {
        return _MyAllocator;
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline Reference Back() const
    {
        return *(_MyLast - 1);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline Reference Front() const
    {
        return *(_MyFirst);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline bool Empty() const
    {
        return _MyFirst == _MyLast;
    }

private:
    template<typename... _ArgsTy>
    static _D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>,
        Reference> EmplaceImpl(Reference _Obj, _ArgsTy &&... _Args)
    {
		return _Impl_Dragonian_Lib_Construct_At(_Obj, std::forward<_ArgsTy>(_Args)...);
    }

    template<typename... _ArgsTy, typename = std::enable_if_t<std::is_constructible_v<ValueType, _ArgsTy...>>>
	decltype(auto) EmplaceImpl(const ConstIterator& _Where, _ArgsTy&&... _Args)
    {
        constexpr auto _Size = 1ll;
        const auto _MySize = _MyLast - _MyFirst;
        const auto _NewSize = _MySize + _Size;
        const auto _FrontCount = _Where - _MyFirst;
        const auto _Remainder = _MySize - _FrontCount;
        if (_NewSize > static_cast<int64_t>(Capacity()))
        {
            auto _NewBuffer = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _NewSize * 2);
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
            auto _NewBuffer = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _NewSize * 2);
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
            auto _NewBuffer = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _NewSize * 2);
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
            auto _NewBuffer = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _NewSize * 2);
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
            _MyFirst = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        auto _Data = (Pointer)_MyAllocator.allocate(sizeof(ValueType) * _NewCapacity);
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
    _D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
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
    _D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
        std::is_constructible_v<ValueType, _ArgsTy...>,
        Reference> EmplaceBack(_ArgsTy &&... _Args)
    {
		if (_MyLast == _MyEnd)
			Reserve(Capacity() * 2);
		return EmplaceImpl(*_MyLast++, std::forward<_ArgsTy>(_Args)...);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline Reference Insert(const ConstIterator& _Where, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		return Emplace(_Where, _Value);
    }

    template <typename TmpTy = ValueType>
    _D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
        std::is_move_assignable_v<TmpTy>&& std::is_move_constructible_v<TmpTy>&& std::is_same_v<TmpTy, ValueType>,
        Reference> Insert(const ConstIterator& _Where, ValueType&& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        return Emplace(_Where, std::move(_Value));
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline void Insert(const ConstIterator& _Where, SizeType _Count, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        InsertImpl(_Where, _Count, _Value);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline void Insert(const ConstIterator& _Where, const ConstIterator& _First, const ConstIterator& _Last)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
		if (_First > _Last)
			_D_Dragonian_Lib_Stl_Throw("Invalid Range!");
#endif
		InsertImpl(_Where, _First, _Last);
    }

    _D_Dragonian_Lib_Constexpr_Force_Inline void MoveTo(const ConstIterator& _Where, const Iterator& _First, const Iterator& _Last)
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

    _D_Dragonian_Lib_Constexpr_Force_Inline void PopBack()
    {
        --_MyLast;
        if constexpr (!std::is_trivially_copyable_v<ValueType>)
            _MyLast->~ValueType();
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector operator+(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) += ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector operator-(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) -= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector operator*(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) *= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector operator/(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) /= ValueType(_Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector operator^(const _T& _Val) const
    {
        auto Temp = *this;
        auto Iter = Temp._MyFirst;
        while (Iter != Temp._MyLast)
            *(Iter++) = (ValueType)pow(*(Iter++), _Val);
        return Temp;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector& operator+=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) += ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector& operator-=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) -= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector& operator*=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) *= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector& operator/=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) /= ValueType(_Val);
        return *this;
    }

    template<typename _T>
    _D_Dragonian_Lib_Constexpr_Force_Inline Vector& operator^=(const _T& _Val)
    {
        auto Iter = _MyFirst;
        while (Iter != _MyLast)
            *(Iter++) = (ValueType)pow(*(Iter++), _Val);
        return *this;
    }

    template <typename... _ArgTypes>
	static decltype(auto) MakeVector(_ArgTypes&&... _Args)
	{
        constexpr auto _Size = sizeof...(_Args);
        Vector Result;
        if constexpr (_Size)
        {
            if constexpr (DRAGONIANLIB_EMPTY_CAPACITY < _Size)
                Result.Reserve(_Size);
            Result._MyLast = Result._MyFirst + _Size;
            MakeVector(Result._MyFirst, std::forward<_ArgTypes>(_Args)...);
        }
        return Result;
	}

private:
    template <typename _ArgType, typename ... _RestTypes>
    _D_Dragonian_Lib_Constexpr_Force_Inline static decltype(auto) MakeVector(ValueType* _Result, _ArgType&& _Arg, _RestTypes&&... _Args)
    {
        _Impl_Dragonian_Lib_Construct_At(*_Result, std::forward<_ArgType>(_Arg));
        if constexpr (sizeof...(_Args))
            MakeVector(_Result + 1, std::forward<_RestTypes>(_Args)...);
    }
};

template <typename _Type, Device _Device = Device::CPU, typename ... _ArgTypes>
decltype(auto) MakeVector(_Type&& _First, _ArgTypes&&... _Args)
{
	return Vector<_Type, _Device>::MakeVector(std::forward<_Type>(_First), std::forward<_ArgTypes>(_Args)...);
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<Type> Arange(Type Start, Type End, Type Step = Type(1.), Type NDiv = Type(1.))
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<_Type> MeanFliter(const Vector<_Type>& _Signal, size_t _WindowSize)
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
_D_Dragonian_Lib_Constexpr_Force_Inline double Average(const T* Start, const T* End)
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
_D_Dragonian_Lib_Constexpr_Force_Inline void Resample(
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
_D_Dragonian_Lib_Constexpr_Force_Inline void Resample(
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<TypeOutput> InterpResample(
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

template<typename TypeOutput, typename TypeInput>
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<TypeOutput> InterpResample(
    const ConstantRanges<TypeInput>& Data,
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<TypeOutput> InterpResample(
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<TypeOutput> SignalCast(
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
_D_Dragonian_Lib_Constexpr_Force_Inline Vector<T> InterpFunc(
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
