#pragma once
#include "Alloc.h"
#include <exception>
#include <initializer_list>
#include <algorithm>
#include "matlabfunctions.h"

#define DRAGONIANLIBCONSTEXPR inline
#define DragonianLibStlThrow(message) ThrowException(message, __FILE__, __FUNCSIG__, __LINE__)

DRAGONIANLIBSTLBEGIN

[[noreturn]] void ThrowException(const char* Message, const char* FILE, const char* FUN, int LINE);

//using Type_ = float;
template <typename Type_>
class Vector
{
public:
	friend std::_Tidy_guard<Vector>;

	using ValueType = Type_;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;
	using Pointer = ValueType*;
	using ConstPointer = const ValueType*;
	using Iterator = ValueType*;
	using ConstIterator = const ValueType*;
	using SizeType = size_t;
	using IndexType = long long;

	DRAGONIANLIBCONSTEXPR ~Vector() noexcept
	{
		Destory();
	}

	DRAGONIANLIBCONSTEXPR Vector()
	{
		Allocator_ = GetMemoryProvider(Device::CPU);
		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
		_MyLast = _MyFirst;
		_MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
		return;
	}

	DRAGONIANLIBCONSTEXPR Vector(SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		if (!_Alloc) DragonianLibStlThrow("Bad Alloc!");
		Allocator_ = _Alloc;

		if(_Size == 0)
		{
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
			return;
		}

		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
		if (!_MyFirst) DragonianLibStlThrow("Bad Alloc!");
		_MyLast = _MyFirst + _Size;
		_MyEnd = _MyFirst + _Size * 2;

		if constexpr (!std::is_arithmetic_v<ValueType>)
		{
			auto Iter = _MyFirst;
			while (Iter != _MyLast)
				new (Iter++) ValueType;
		}
	}

	DRAGONIANLIBCONSTEXPR Vector(SizeType _Size, ConstReference _Value, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		if (!_Alloc) DragonianLibStlThrow("Bad Alloc!");
		Allocator_ = _Alloc;

		if (_Size == 0)
		{
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * DRAGONIANLIB_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + DRAGONIANLIB_EMPTY_CAPACITY;
			return;
		}

		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
		if (!_MyFirst) DragonianLibStlThrow("Bad Alloc!");
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

	DRAGONIANLIBCONSTEXPR Vector(Pointer* _Block, SizeType _Size, Allocator _Alloc, bool _Owner = true)
	{
		if (!_Alloc) DragonianLibStlThrow("Bad Alloc!");
		Allocator_ = _Alloc;
		_MyOwner = _Owner;
		_MyFirst = *_Block;
		_MyLast = _MyFirst + _Size;
		_MyEnd = _MyLast;
		*_Block = nullptr;
	}

	DRAGONIANLIBCONSTEXPR Vector(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_Begin, _End, _Alloc);
	}

	DRAGONIANLIBCONSTEXPR Vector(ConstPointer _Buffer, SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_Buffer, _Buffer + _Size, _Alloc);
	}

	DRAGONIANLIBCONSTEXPR Vector(const std::initializer_list<ValueType>& _List, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_List.begin(), _List.end(), _Alloc);
	}

	DRAGONIANLIBCONSTEXPR Vector(const Vector& _Left)
	{
		ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
	}

	DRAGONIANLIBCONSTEXPR Vector(Vector&& _Right) noexcept
	{
		_MyFirst = _Right._MyFirst;
		_MyLast = _Right._MyLast;
		_MyEnd = _Right._MyEnd;
		Allocator_ = _Right.Allocator_;

		_Right.Allocator_ = nullptr;
		_Right._MyFirst = nullptr;
		_Right._MyLast = nullptr;
		_Right._MyEnd = nullptr;
	}

	DRAGONIANLIBCONSTEXPR Vector& operator=(const Vector& _Left)
	{
		if (&_Left == this)
			return *this;
		Destory();
		ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
		return *this;
	}

	DRAGONIANLIBCONSTEXPR Vector& operator=(Vector&& _Right) noexcept
	{
		if (&_Right != this)
			Destory();

		_MyFirst = _Right._MyFirst;
		_MyLast = _Right._MyLast;
		_MyEnd = _Right._MyEnd;
		Allocator_ = _Right.Allocator_;

		_Right.Allocator_ = nullptr;
		_Right._MyFirst = nullptr;
		_Right._MyLast = nullptr;
		_Right._MyEnd = nullptr;
		return *this;
	}

	DRAGONIANLIBCONSTEXPR Reference operator[](SizeType _Index) const
	{
#ifdef DRAGONIANLIB_DEBUG
		if (size_t(_Index) >= Size())
			DragonianLibStlThrow("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

protected:
	Pointer _MyFirst, _MyLast, _MyEnd;
	Allocator Allocator_;
	bool _MyOwner = true;

public:
	DRAGONIANLIBCONSTEXPR Iterator Begin()
	{
		return _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR Iterator End()
	{
		return _MyLast;
	}

	DRAGONIANLIBCONSTEXPR ConstIterator Begin() const
	{
		return _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR ConstIterator End() const
	{
		return _MyLast;
	}

	DRAGONIANLIBCONSTEXPR Iterator begin()
	{
		return _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR Iterator end()
	{
		return _MyLast;
	}

	DRAGONIANLIBCONSTEXPR ConstIterator begin() const
	{
		return _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR ConstIterator end() const
	{
		return _MyLast;
	}

	DRAGONIANLIBCONSTEXPR SizeType Size() const
	{
		return _MyLast - _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR SizeType Capacity() const
	{
		return _MyEnd - _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR std::pair<Pointer, SizeType> Release()
	{
		auto Ptr = _MyFirst;
		auto _Size = Size();
		_MyFirst = nullptr;
		_MyLast = nullptr;
		_MyEnd = nullptr;
		Allocator_ = nullptr;
		return { Ptr, _Size };
	}

	DRAGONIANLIBCONSTEXPR ValueType* Data()
	{
		return std::_Unfancy_maybe_null(_MyFirst);
	}

	DRAGONIANLIBCONSTEXPR const ValueType* Data() const
	{
		return std::_Unfancy_maybe_null(_MyFirst);
	}

	DRAGONIANLIBCONSTEXPR Allocator GetAllocator() const
	{
		return Allocator_;
	}

	DRAGONIANLIBCONSTEXPR Reference Back() const
	{
		return *(_MyLast - 1);
	}

	DRAGONIANLIBCONSTEXPR Reference Front() const
	{
		return *(_MyFirst);
	}

	DRAGONIANLIBCONSTEXPR bool Empty() const
	{
		return _MyFirst == _MyLast;
	}

private:
	DRAGONIANLIBCONSTEXPR void _Tidy()
	{
		if constexpr (!std::is_arithmetic_v<ValueType>)
		{
			auto Iter = _MyFirst;
			while (Iter != _MyLast)
				(Iter++)->~ValueType();
		}
		if (_MyFirst && _MyOwner)
			Allocator_->Free(_MyFirst);
		_MyFirst = nullptr;
		_MyLast = nullptr;
		_MyEnd = nullptr;
	}

	DRAGONIANLIBCONSTEXPR void Destory()
	{
		_Tidy();
		Allocator_ = nullptr;
	}

	DRAGONIANLIBCONSTEXPR void ConstuctWithIteratorImpl(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc)
	{
		if (!_Alloc) DragonianLibStlThrow("Bad Alloc!");
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
		if (!_MyFirst) DragonianLibStlThrow("Bad Alloc!");
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
	DRAGONIANLIBCONSTEXPR void EmplaceImpl(Iterator _Where, _ArgsTy &&... _Args)
	{
		new (_Where) ValueType(std::forward<_ArgsTy>(_Args)...);
	}

	DRAGONIANLIBCONSTEXPR void ReserveImpl(SizeType _NewCapacity, IndexType _Front, IndexType _Tail)
	{
		auto _Size = Size() + _Tail - _Front;
		auto _TailSize = (_MyLast - _MyFirst) - _Front;

		if (_NewCapacity <= _Size || _Front > _Tail)
			DragonianLibStlThrow("Bad Alloc!");

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

	DRAGONIANLIBCONSTEXPR void CopyImpl(IndexType _Front, IndexType _Tail)
	{
		auto _Size = Size() + _Tail - _Front;
		auto _TailSize = (_MyLast - _MyFirst) - _Front;

		if (_Front > _Tail || Capacity() < _Size)
			DragonianLibStlThrow("Index Out Of Range!");

		if constexpr (!std::is_arithmetic_v<ValueType>)
			for (IndexType i = _TailSize - 1; i >= 0; --i)
				new (_MyFirst + _Tail + i) ValueType(std::move(_MyFirst[_Front + i]));
		else
			for (IndexType i = _TailSize - 1; i >= 0; --i)
				*(_MyFirst + _Tail + i) = _MyFirst[_Front + i];

		_MyLast = _MyFirst + _Size;
	}
public:
	DRAGONIANLIBCONSTEXPR void Reserve(SizeType _NewCapacity)
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
			for (IndexType i = 0; i < _Size; ++i)
				new (_Data + i) ValueType(std::move(_MyFirst[i]));
		else
			memcpy(_Data, _MyFirst, sizeof(ValueType) * _Size);

		_Tidy();
		_MyFirst = _Data;
		_MyLast = _Data + _Size;
		_MyEnd = _Data + _NewCapacity;
	}

	DRAGONIANLIBCONSTEXPR void Resize(SizeType _NewSize, ConstReference _Val = ValueType(0))
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
	DRAGONIANLIBCONSTEXPR void Emplace(ConstIterator _Where, _ArgsTy &&... _Args)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			DragonianLibStlThrow("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		if (_MyLast + 1 > _MyEnd)
			ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
		else
			CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);
		EmplaceImpl(_MyFirst + Idx, std::forward<_ArgsTy>(_Args)...);
	}

	template<typename... _ArgsTy>
	DRAGONIANLIBCONSTEXPR void EmplaceBack(_ArgsTy &&... _Args)
	{
		Emplace(_MyLast, std::forward<_ArgsTy>(_Args)...);
	}

	DRAGONIANLIBCONSTEXPR void Insert(ConstIterator _Where, const ValueType& _Value)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			DragonianLibStlThrow("Out Of Range!");
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
	}

	DRAGONIANLIBCONSTEXPR void Insert(ConstIterator _Where, ValueType&& _Value)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			DragonianLibStlThrow("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		if (_MyLast + 1 > _MyEnd)
			ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
		else
			CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);

		if constexpr (!std::is_arithmetic_v<ValueType>)
			new (_MyFirst + Idx) ValueType(std::move(_Value));
		else
			*(_MyFirst + Idx) = std::move(_Value);
	}

	DRAGONIANLIBCONSTEXPR void Insert(ConstIterator _Where, SizeType _Count = 1, const ValueType& _Value = ValueType(0))
	{
#ifdef DRAGONIANLIB_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			DragonianLibStlThrow("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		if (_MyLast + _Count > _MyEnd)
			ReserveImpl((_Count + Size()) * 2, _Where - _MyFirst, _Where - _MyFirst + _Count);
		else
			CopyImpl(_Where - _MyFirst, _Where - _MyFirst + _Count);

		if constexpr (!std::is_arithmetic_v<ValueType>)
			for (SizeType i = 0; i < _Count; ++i)
				new (_MyFirst + Idx + i) ValueType(_Value);
		else
			for (SizeType i = 0; i < _Count; ++i)
				*(_MyFirst + Idx + i) = _Value;
	}

	DRAGONIANLIBCONSTEXPR void Insert(ConstIterator _Where, ConstIterator _First, ConstIterator _Last)
	{
#ifdef DRAGONIANLIB_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			DragonianLibStlThrow("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		SizeType _Count = _Last - _First;
		if (_Count < 0)
			DragonianLibStlThrow("Range Error!");

		if (_MyLast + _Count > _MyEnd)
			ReserveImpl((_Count + Size()) * 2, _Where - _MyFirst, _Where - _MyFirst + _Count);
		else
			CopyImpl(_Where - _MyFirst, _Where - _MyFirst + _Count);

		if constexpr (!std::is_arithmetic_v<ValueType>)
			for (SizeType i = 0; i < _Count; ++i)
				new (_MyFirst + Idx + i) ValueType(*(_First++));
		else
			for (SizeType i = 0; i < _Count; ++i)
				*(_MyFirst + Idx + i) = *(_First++);
	}

	DRAGONIANLIBCONSTEXPR void Clear()
	{
		if constexpr (!std::is_arithmetic_v<ValueType>)
		{
			auto Iter = _MyFirst;
			while (Iter != _MyLast)
				(Iter++)->~ValueType();
		}
		_MyLast = _MyFirst;
	}

	DRAGONIANLIBCONSTEXPR void PopBack()
	{
		--_MyLast;
		if constexpr (!std::is_arithmetic_v<ValueType>)
			_MyLast->~ValueType();
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector operator+(const _T& _Val) const
	{
		auto Temp = *this;
		auto Iter = Temp._MyFirst;
		while (Iter != Temp._MyLast)
			*(Iter++) += ValueType(_Val);
		return Temp;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector operator-(const _T& _Val) const
	{
		auto Temp = *this;
		auto Iter = Temp._MyFirst;
		while (Iter != Temp._MyLast)
			*(Iter++) -= ValueType(_Val);
		return Temp;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector operator*(const _T& _Val) const
	{
		auto Temp = *this;
		auto Iter = Temp._MyFirst;
		while (Iter != Temp._MyLast)
			*(Iter++) *= ValueType(_Val);
		return Temp;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector operator/(const _T& _Val) const
	{
		auto Temp = *this;
		auto Iter = Temp._MyFirst;
		while (Iter != Temp._MyLast)
			*(Iter++) /= ValueType(_Val);
		return Temp;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector operator^(const _T& _Val) const
	{
		auto Temp = *this;
		auto Iter = Temp._MyFirst;
		while (Iter != Temp._MyLast)
			*(Iter++) = (ValueType)pow(*(Iter++), _Val);
		return Temp;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector& operator+=(const _T& _Val)
	{
		auto Iter = _MyFirst;
		while (Iter != _MyLast)
			*(Iter++) += ValueType(_Val);
		return *this;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector& operator-=(const _T& _Val)
	{
		auto Iter = _MyFirst;
		while (Iter != _MyLast)
			*(Iter++) -= ValueType(_Val);
		return *this;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector& operator*=(const _T& _Val)
	{
		auto Iter = _MyFirst;
		while (Iter != _MyLast)
			*(Iter++) *= ValueType(_Val);
		return *this;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector& operator/=(const _T& _Val)
	{
		auto Iter = _MyFirst;
		while (Iter != _MyLast)
			*(Iter++) /= ValueType(_Val);
		return *this;
	}

	template<typename _T>
	DRAGONIANLIBCONSTEXPR Vector& operator^=(const _T& _Val)
	{
		auto Iter = _MyFirst;
		while (Iter != _MyLast)
			*(Iter++) = (ValueType)pow(*(Iter++), _Val);
		return *this;
	}
};

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator+(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
	if (_ValA.Size() != _ValB.Size())
		DragonianLibStlThrow("Size MisMatch!");
	auto Temp = _ValA;
	auto Iter = Temp.Data();
	auto ValIter = _ValB.Data();
	while (Iter != Temp.End())
		*(Iter++) += _TypeA(*(ValIter++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator-(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
	if (_ValA.Size() != _ValB.Size())
		DragonianLibStlThrow("Size MisMatch!");
	auto Temp = _ValA;
	auto Iter = Temp.Data();
	auto ValIter = _ValB.Data();
	while (Iter != Temp.End())
		*(Iter++) -= _TypeA(*(ValIter++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator*(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
	if (_ValA.Size() != _ValB.Size())
		DragonianLibStlThrow("Size MisMatch!");
	auto Temp = _ValA;
	auto Iter = Temp.Data();
	auto ValIter = _ValB.Data();
	while (Iter != Temp.End())
		*(Iter++) *= _TypeA(*(ValIter++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator/(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
	if (_ValA.Size() != _ValB.Size())
		DragonianLibStlThrow("Size MisMatch!");
	auto Temp = _ValA;
	auto Iter = Temp.Data();
	auto ValIter = _ValB.Data();
	while (Iter != Temp.End())
		*(Iter++) /= _TypeA(*(ValIter++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator^(const Vector<_TypeA>& _ValA, const Vector<_TypeB>& _ValB)
{
	if (_ValA.Size() != _ValB.Size())
		DragonianLibStlThrow("Size MisMatch!");
	auto Temp = _ValA;
	auto Iter = Temp.Data();
	auto ValIter = _ValB.Data();
	while (Iter != Temp.End())
		*(Iter++) = (_TypeA)pow(*(Iter++), *(ValIter++));
	return Temp;
}

//using _TypeA = float;
//using _TypeB = double;
template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator+(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
	Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
	auto IterA = Temp.Data();
	auto IterB = _ValB.Data();
	while (IterA != Temp.End())
		*(IterA++) = _ValA + (_TypeA)(*(IterB++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator-(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
	Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
	auto IterA = Temp.Data();
	auto IterB = _ValB.Data();
	while (IterA != Temp.End())
		*(IterA++) = _ValA + (_TypeA)(*(IterB++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator*(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
	Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
	auto IterA = Temp.Data();
	auto IterB = _ValB.Data();
	while (IterA != Temp.End())
		*(IterA++) = _ValA + (_TypeA)(*(IterB++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator/(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
	Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
	auto IterA = Temp.Data();
	auto IterB = _ValB.Data();
	while (IterA != Temp.End())
		*(IterA++) = _ValA + (_TypeA)(*(IterB++));
	return Temp;
}

template <typename _TypeA, typename _TypeB>
DRAGONIANLIBCONSTEXPR Vector<_TypeA> operator^(const _TypeA& _ValA, const Vector<_TypeB>& _ValB)
{
	Vector<_TypeA> Temp{ _ValB.Size(), _ValB.GetAllocator() };
	auto IterA = Temp.Data();
	auto IterB = _ValB.Data();
	while (IterA != Temp.End())
		*(IterA++) = (_TypeA)pow(_ValA, *(IterB++));
	return Temp;
}

template <typename _Type>
DRAGONIANLIBCONSTEXPR Vector<_Type> Arange(_Type _Start, _Type _End, _Type _Step = _Type(1.), _Type _NDiv = _Type(1.))
{
	Vector<_Type> OutPut(size_t((_End - _Start) / _Step));
	auto OutPutPtr = OutPut.Begin();
	const auto OutPutPtrEnd = OutPut.End();
	while (OutPutPtr != OutPutPtrEnd)
	{
		*(OutPutPtr++) = _Start / _NDiv;
		_Start += _Step;
	}
	return OutPut;
}

template <typename _Type>
DRAGONIANLIBCONSTEXPR Vector<_Type> MeanFliter(const Vector<_Type>& _Signal, size_t _WindowSize)
{
	Vector<_Type> Result(_Signal.Size());

	if (_WindowSize > _Signal.Size() || _WindowSize < 2)
		return _Signal;

	auto WndSz = (_Type)(_WindowSize % 2 ? _WindowSize : _WindowSize + 1);

	const size_t half = _WindowSize / 2; // 窗口半径，向下取整
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

template<typename T>
DRAGONIANLIBCONSTEXPR double Average(const T* start, const T* end)
{
	const auto size = end - start + 1;
	auto avg = (double)(*start);
	for (auto i = 1; i < size; i++)
		avg = avg + (abs((double)start[i]) - avg) / (double)(i + 1ull);
	return avg;
}

/**
 * \brief 重采样（插值）
 * \tparam TOut 输出类型
 * \tparam TIn 输入类型
 * \param _Data 输入数据
 * \param _SrcSamplingRate 输入采样率
 * \param _DstSamplingRate 输出采样率
 * \param n_Div 给输出的数据统一除以这个数
 * \return 输出数据
 */
template<typename TOut, typename TIn>
DRAGONIANLIBCONSTEXPR Vector<TOut> InterpResample(
	const Vector<TIn>& _Data,
	long _SrcSamplingRate,
	long _DstSamplingRate,
	TOut n_Div = TOut(1)
)
{
	if (_SrcSamplingRate != _DstSamplingRate)
	{
		const double intstep = double(_SrcSamplingRate) / double(_DstSamplingRate);
		const auto xi = Arange(0., double(_Data.Size()), intstep);
		auto x0 = Arange(0., double(_Data.Size()));
		while (x0.Size() < _Data.Size())
			x0.EmplaceBack(x0[x0.Size() - 1] + 1.0);
		while (x0.Size() > _Data.Size())
			x0.PopBack();

		Vector<double> y0(_Data.Size());
		for (size_t i = 0; i < _Data.Size(); ++i)
			y0[i] = double(_Data[i]) / double(n_Div);

		Vector<double> yi(xi.Size());
		interp1(x0.Data(), y0.Data(), long(x0.Size()), xi.Data(), long(xi.Size()), yi.Data());

		Vector<TOut> out(xi.Size());
		for (size_t i = 0; i < yi.Size(); ++i)
			out[i] = TOut(yi[i]);
		return out;
	}
	Vector<TOut> out(_Data.Size());
	for (size_t i = 0; i < _Data.Size(); ++i)
		out[i] = TOut(_Data[i]) / n_Div;
	return out;
}

/**
 * \brief 重采样（插值）
 * \tparam T 数据类型
 * \param _Data 输入数据
 * \param _SrcSamplingRate 输入采样率
 * \param _DstSamplingRate 输出采样率
 * \return 输出数据
 */
template<typename T>
DRAGONIANLIBCONSTEXPR Vector<T> InterpFunc(
	const Vector<T>& _Data,
	long _SrcSamplingRate,
	long _DstSamplingRate
)
{
	if (_SrcSamplingRate != _DstSamplingRate)
	{
		const double intstep = double(_SrcSamplingRate) / double(_DstSamplingRate);
		auto xi = Arange(0., double(_Data.Size()), intstep);
		while (xi.Size() < size_t(_DstSamplingRate))
			xi.EmplaceBack(xi[xi.Size() - 1] + 1.0);
		while (xi.Size() > size_t(_DstSamplingRate))
			xi.PopBack();
		auto x0 = Arange(0., double(_Data.Size()));
		while (x0.Size() < _Data.Size())
			x0.EmplaceBack(x0[x0.Size() - 1] + 1.0);
		while (x0.Size() > _Data.Size())
			x0.PopBack();
		Vector<double> y0(_Data.Size());
		for (size_t i = 0; i < _Data.Size(); ++i)
			y0[i] = _Data[i] <= T(0.0001) ? NAN : double(_Data[i]);
		Vector<double> yi(xi.Size());
		interp1(x0.Data(), y0.Data(), long(x0.Size()), xi.Data(), long(xi.Size()), yi.Data());
		Vector<T> out(xi.Size());
		for (size_t i = 0; i < yi.Size(); ++i)
			out[i] = isnan(yi[i]) ? T(0.0) : T(yi[i]);
		return out;
	}
	return _Data;
}

DRAGONIANLIBSTLEND