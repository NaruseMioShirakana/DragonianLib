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

//Force Inline
#ifdef _MSC_VER
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr __attribute__((always_inline)) inline
#else
#define _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline constexpr inline
#endif
#define _D_Dragonian_Lib_Stl_Throw(message) _Impl_Dragonian_Lib_Template_Library_Throw_Exception(message, __FILE__, __FUNCSIG__, __LINE__)

_D_Dragonian_Lib_Template_Library_Space_Begin

[[noreturn]] void _Impl_Dragonian_Lib_Template_Library_Throw_Exception(const char* Message, const char* FILE, const char* FUN, int LINE);

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
    using Iterator = ValueType*;
    using ConstIterator = const ValueType*;
    using SizeType = size_t;
    using IndexType = long long;

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ~Vector() noexcept
    {
        Destory();
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector()
    {
        Allocator_ = GetMemoryProvider(Device::CPU);
        _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
        _MyLast = _MyFirst;
        _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
        return;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        Allocator_ = _Alloc;

        if (_Size == 0)
        {
            _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
        if (!_MyFirst) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyFirst + _Size * 2;

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                new (Iter++) ValueType;
        }
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(SizeType _Size, ConstReference _Value, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        Allocator_ = _Alloc;

        if (_Size == 0)
        {
            _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
        if (!_MyFirst) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyFirst + _Size * 2;

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                new (Iter++) ValueType(_Value);
        }
        else
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                *(Iter++) = _Value;
        }
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(Pointer* _Block, SizeType _Size, Allocator _Alloc, bool _Owner = true)
    {
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        Allocator_ = _Alloc;
        _MyOwner = _Owner;
        _MyFirst = *_Block;
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyLast;
        *_Block = nullptr;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        ConstuctWithIteratorImpl(_Begin, _End, _Alloc);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(ConstPointer _Buffer, SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        ConstuctWithIteratorImpl(_Buffer, _Buffer + _Size, _Alloc);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(const std::initializer_list<ValueType>& _List, Allocator _Alloc = GetMemoryProvider(Device::CPU))
    {
        ConstuctWithIteratorImpl(_List.begin(), _List.end(), _Alloc);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(const Vector& _Left)
    {
        ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector(Vector&& _Right) noexcept
    {
        _MyFirst = _Right._MyFirst;
        _MyLast = _Right._MyLast;
        _MyEnd = _Right._MyEnd;
        Allocator_ = _Right.Allocator_;
        _MyOwner = _Right._MyOwner;

        _Right.Allocator_ = nullptr;
        _Right._MyFirst = nullptr;
        _Right._MyLast = nullptr;
        _Right._MyEnd = nullptr;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator=(const Vector& _Left)
    {
        if (&_Left == this)
            return *this;
        Destory();
        ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
        return *this;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Vector& operator=(Vector&& _Right) noexcept
    {
        if (&_Right != this)
            Destory();

        _MyFirst = _Right._MyFirst;
        _MyLast = _Right._MyLast;
        _MyEnd = _Right._MyEnd;
        Allocator_ = _Right.Allocator_;
        _MyOwner = _Right._MyOwner;

        _Right.Allocator_ = nullptr;
        _Right._MyFirst = nullptr;
        _Right._MyLast = nullptr;
        _Right._MyEnd = nullptr;
        return *this;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference operator[](SizeType _Index) const
    {
#ifdef DRAGONIANLIB_DEBUG
        if (size_t(_Index) >= Size())
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        return _MyFirst[_Index];
    }

protected:
    Pointer _MyFirst, _MyLast, _MyEnd;
    Allocator Allocator_;
    bool _MyOwner = true;

public:
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) Begin() const
    {
        return _MyFirst;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) End() const
    {
        return _MyLast;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedBegin() const
    {
        return _MyLast - 1;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) ReversedEnd() const
    {
        return _MyFirst - 1;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) begin() const
    {
        return _MyFirst;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) end() const
    {
        return _MyLast;
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Size() const
    {
        return _MyLast - _MyFirst;
    }

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Capacity() const
    {
        return _MyEnd - _MyFirst;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::pair<Pointer, SizeType> Release()
    {
        auto Ptr = _MyFirst;
        auto _Size = Size();
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
        Allocator_ = nullptr;
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
        return Allocator_;
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
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void _Tidy()
    {
        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                (Iter++)->~ValueType();
        }
        if (_MyFirst && _MyOwner)
            Allocator_->Free(_MyFirst);
        _MyOwner = true;
        _MyFirst = nullptr;
        _MyLast = nullptr;
        _MyEnd = nullptr;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Destory()
    {
        _Tidy();
        Allocator_ = nullptr;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void ConstuctWithIteratorImpl(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc)
    {
        if (!_Alloc) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        Allocator_ = _Alloc;

        const auto _Size = _End - _Begin;

        if (_Size <= 0)
        {
            _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
        if (!_MyFirst) _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");
        _MyLast = _MyFirst + _Size;
        _MyEnd = _MyFirst + _Size * 2;

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                new (Iter++) ValueType(*(_Begin++));
        }
        else
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                *(Iter++) = *(_Begin++);
        }
    }

    template<typename... _ArgsTy>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) EmplaceImpl(Reference _Obj, _ArgsTy &&... _Args)
    {
        ::new (static_cast<void*>(std::addressof(_Obj))) ValueType(std::forward<_ArgsTy>(_Args)...);
        return _Obj;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void ReserveImpl(SizeType _NewCapacity, IndexType _Front, IndexType _Tail)
    {
        auto _Size = Size() + _Tail - _Front;
        auto _TailSize = (_MyLast - _MyFirst) - _Front;

        if (_NewCapacity <= _Size || _Front > _Tail)
            _D_Dragonian_Lib_Stl_Throw("Bad Alloc!");

        auto _Data = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _NewCapacity);

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            for (IndexType i = 0; i < _Front; ++i)
                new (_Data + i) ValueType(std::move(_MyFirst[i]));
            for (IndexType i = 0; i < _TailSize; ++i)
                new (_Data + _Tail + i) ValueType(std::move(_MyFirst[_Front + i]));
        }
        else
        {
            memcpy(_Data, _MyFirst, sizeof(ValueType) * _Front);
            memcpy(_Data + _Tail, _MyFirst + _Front, sizeof(ValueType) * _TailSize);
        }

        _Tidy();
        _MyFirst = _Data;
        _MyLast = _Data + _Size;
        _MyEnd = _Data + _NewCapacity;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void CopyImpl(IndexType _Front, IndexType _Tail)
    {
        auto _Size = Size() + _Tail - _Front;
        auto _TailSize = (_MyLast - _MyFirst) - _Front;

        if (_Front > _Tail || Capacity() < _Size)
            _D_Dragonian_Lib_Stl_Throw("Index Out Of Range!");

        if constexpr (!std::is_arithmetic_v<ValueType>)
            for (IndexType i = _TailSize - 1; i >= 0; --i)
                new (_MyFirst + _Tail + i) ValueType(std::move(_MyFirst[_Front + i]));
        else
            for (IndexType i = _TailSize - 1; i >= 0; --i)
                *(_MyFirst + _Tail + i) = _MyFirst[_Front + i];

        _MyLast = _MyFirst + _Size;
    }
public:
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Reserve(SizeType _NewCapacity)
    {
        if (_NewCapacity == Capacity())
            return;

        if (_NewCapacity == 0)
        {
            _Tidy();
            _MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
            _MyLast = _MyFirst;
            _MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
            return;
        }

        auto _Data = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _NewCapacity);
        auto _Size = std::min(_NewCapacity, Size());

        if constexpr (!std::is_arithmetic_v<ValueType>)
            for (SizeType i = 0; i < _Size; ++i)
                new (_Data + i) ValueType(std::move(_MyFirst[i]));
        else
            memcpy(_Data, _MyFirst, sizeof(ValueType) * _Size);

        _Tidy();
        _MyFirst = _Data;
        _MyLast = _Data + _Size;
        _MyEnd = _Data + _NewCapacity;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Resize(SizeType _NewSize)
    {
        if (_NewSize == Size())
            return;

        if (_NewSize >= Capacity())
            Reserve(_NewSize * 2);

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            for (auto i = _NewSize; i < Size(); ++i)
                (_MyFirst + i)->~ValueType();
            for (auto i = Size(); i < _NewSize; ++i)
                new (_MyFirst + i) ValueType();
        }
        else
            for (auto i = Size(); i < _NewSize; ++i)
                _MyFirst[i] = ValueType(0);
        _MyLast = _MyFirst + _NewSize;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Resize(SizeType _NewSize, ConstReference _Val)
    {
        if (_NewSize == Size())
            return;

        if (_NewSize >= Capacity())
            Reserve(_NewSize * 2);

        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            for (auto i = _NewSize; i < Size(); ++i)
                (_MyFirst + i)->~ValueType();
            for (auto i = Size(); i < _NewSize; ++i)
                new (_MyFirst + i) ValueType(_Val);
        }
        else
            for (auto i = Size(); i < _NewSize; ++i)
                _MyFirst[i] = _Val;
        _MyLast = _MyFirst + _NewSize;
    }

    template<typename... _ArgsTy>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) Emplace(ConstIterator _Where, _ArgsTy &&... _Args)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        auto Idx = _Where - _MyFirst;
        if (_MyLast + 1 > _MyEnd)
            ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
        else
            CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);
        return EmplaceImpl(*(_MyFirst + Idx), std::forward<_ArgsTy>(_Args)...);
    }

    template<typename... _ArgsTy>
    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline decltype(auto) EmplaceBack(_ArgsTy &&... _Args)
    {
        return Emplace(_MyLast, std::forward<_ArgsTy>(_Args)...);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference Insert(ConstIterator _Where, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        auto Idx = _Where - _MyFirst;
        if (_MyLast + 1 > _MyEnd)
            ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
        else
            CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);

        if constexpr (!std::is_arithmetic_v<ValueType>)
            new (_MyFirst + Idx) ValueType(_Value);
        else
            *(_MyFirst + Idx) = _Value;
        return *(_MyFirst + Idx);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline Reference Insert(ConstIterator _Where, ValueType&& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        auto Idx = _Where - _MyFirst;
        if (_MyLast + 1 > _MyEnd)
            ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
        else
            CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);

        if constexpr (!std::is_arithmetic_v<ValueType>)
            new (_MyFirst + Idx) ValueType(std::move(_Value));
        else
            *(_MyFirst + Idx) = _Value;
        return *(_MyFirst + Idx);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Insert(ConstIterator _Where, SizeType _Count, const ValueType& _Value)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        auto Idx = _Where - _MyFirst;
        if (_MyLast + _Count > _MyEnd)
            ReserveImpl((_Count + Size()) * 2, _Where - _MyFirst, _Where - _MyFirst + IndexType(_Count));
        else
            CopyImpl(_Where - _MyFirst, _Where - _MyFirst + IndexType(_Count));

        if constexpr (!std::is_arithmetic_v<ValueType>)
            for (SizeType i = 0; i < _Count; ++i)
                new (_MyFirst + Idx + i) ValueType(_Value);
        else
            for (SizeType i = 0; i < _Count; ++i)
                *(_MyFirst + Idx + i) = _Value;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Insert(ConstIterator _Where, ConstIterator _First, ConstIterator _Last)
    {
#ifdef DRAGONIANLIB_DEBUG
        if (_Where > _MyLast || _Where < _MyFirst)
            _D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
        auto Idx = _Where - _MyFirst;
        SizeType _Count = _Last - _First;
        if (_Last < _First)
            _D_Dragonian_Lib_Stl_Throw("Range Error!");

        if (_MyLast + _Count > _MyEnd)
            ReserveImpl((_Count + Size()) * 2, _Where - _MyFirst, _Where - _MyFirst + IndexType(_Count));
        else
            CopyImpl(_Where - _MyFirst, _Where - _MyFirst + IndexType(_Count));

        if constexpr (!std::is_arithmetic_v<ValueType>)
            for (SizeType i = 0; i < _Count; ++i)
                new (_MyFirst + Idx + i) ValueType(*(_First++));
        else
            for (SizeType i = 0; i < _Count; ++i)
                *(_MyFirst + Idx + i) = *(_First++);
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline ValueType Erase(ConstIterator _Where)
    {
#ifdef DRAGONIANLIB_DEBUG
		if (_Where >= _MyLast || _Where < _MyFirst)
			_D_Dragonian_Lib_Stl_Throw("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		ValueType _Value = std::move(*_Where);
		for (auto i = Idx + 1; i < Size(); ++i)
			*(_MyFirst + i - 1) = std::move(*(_MyFirst + i));
		--_MyLast;
		return _Value;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void Clear()
    {
        if constexpr (!std::is_arithmetic_v<ValueType>)
        {
            auto Iter = _MyFirst;
            while (Iter != _MyLast)
                (Iter++)->~ValueType();
        }
        _MyLast = _MyFirst;
    }

    _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void PopBack()
    {
        --_MyLast;
        if constexpr (!std::is_arithmetic_v<ValueType>)
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
