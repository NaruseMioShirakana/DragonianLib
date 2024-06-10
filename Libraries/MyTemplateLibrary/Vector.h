#pragma once
#include "Alloc.h"
#include <exception>
#include <initializer_list>
#include <algorithm>

LIBSVCSTLBEGIN

//using Type_ = float;
template <typename Type_>
class Vector
{
public:
	using ValueType = Type_;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;
	using Pointer = ValueType*;
	using ConstPointer = const ValueType*;
	using Iterator = ValueType*;
	using ConstIterator = const ValueType*;
	using SizeType = size_t;
	using IndexType = long long;

	~Vector()
	{
		Destory();
	}

	Vector(SizeType _Size = 0, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		if (!_Alloc) throw std::bad_alloc();
		Allocator_ = _Alloc;

		if(_Size == 0)
		{
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * LIBSVC_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + LIBSVC_EMPTY_CAPACITY;
			return;
		}

		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
		if (!_MyFirst) throw std::bad_alloc();
		_MyLast = _MyFirst + _Size;
		_MyEnd = _MyFirst + _Size * 2;

		if constexpr (!std::is_arithmetic_v<ValueType>)
		{
			auto Iter = _MyFirst;
			while (Iter != _MyLast)
				new (Iter++) ValueType;
		}
	}

	Vector(SizeType _Size, ConstReference _Value, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		if (!_Alloc) throw std::bad_alloc();
		Allocator_ = _Alloc;

		if (_Size == 0)
		{
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * LIBSVC_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + LIBSVC_EMPTY_CAPACITY;
			return;
		}

		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
		if (!_MyFirst) throw std::bad_alloc();
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

	Vector(Pointer* _Block, SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		if (!_Alloc) throw std::bad_alloc();
		Allocator_ = _Alloc;

		_MyFirst = *_Block;
		_MyLast = _MyFirst + _Size;
		_MyEnd = _MyLast;
		*_Block = nullptr;
	}

	Vector(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_Begin, _End, _Alloc);
	}

	Vector(ConstPointer _Buffer, SizeType _Size, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_Buffer, _Buffer + _Size, _Alloc);
	}

	Vector(const std::initializer_list<ValueType>& _List, Allocator _Alloc = GetMemoryProvider(Device::CPU))
	{
		ConstuctWithIteratorImpl(_List.begin(), _List.end(), _Alloc);
	}

	Vector(const Vector& _Left)
	{
		ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
	}

	Vector(Vector&& _Right) noexcept
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

	Vector& operator=(const Vector& _Left)
	{
		if (&_Left == this)
			return *this;
		Destory();
		ConstuctWithIteratorImpl(_Left._MyFirst, _Left._MyLast, _Left.Allocator_);
		return *this;
	}

	Vector& operator=(Vector&& _Right) noexcept
	{
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

	Reference operator[](IndexType _Index) const
	{
#ifdef LIBSVC_DEBUG
		if (size_t(_Index) >= Size())
			throw std::exception("Out Of Range!");
#endif
		return _MyFirst[_Index];
	}

protected:
	Pointer _MyFirst, _MyLast, _MyEnd;
	Allocator Allocator_;

public:
	Iterator Begin() const
	{
		return _MyFirst;
	}

	Iterator End() const
	{
		return _MyLast;
	}

	SizeType Size() const
	{
		return _MyLast - _MyFirst;
	}

	SizeType Capacity() const
	{
		return _MyEnd - _MyFirst;
	}

	Pointer Release()
	{
		auto Ptr = _MyFirst;
		_MyFirst = nullptr;
		_MyLast = nullptr;
		_MyEnd = nullptr;
		Allocator_ = nullptr;
		return Ptr;
	}

	Pointer Data()
	{
		return _MyFirst;
	}

	const ValueType* Data() const
	{
		return _MyFirst;
	}

	Allocator GetAllocator() const
	{
		return Allocator_;
	}

	Reference Back() const
	{
		return *(_MyLast - 1);
	}

	Reference Front() const
	{
		return *(_MyFirst);
	}

	bool Empty() const
	{
		return _MyFirst == _MyLast;
	}

private:
	void DestoryData()
	{
		if constexpr (!std::is_arithmetic_v<ValueType>)
		{
			auto Iter = _MyFirst;
			while (Iter != _MyLast)
				(Iter++)->~ValueType();
		}
		if (_MyFirst)
			Allocator_->Free(_MyFirst);
		_MyFirst = nullptr;
		_MyLast = nullptr;
		_MyEnd = nullptr;
	}

	void Destory()
	{
		DestoryData();
		Allocator_ = nullptr;
	}

	void ConstuctWithIteratorImpl(ConstIterator _Begin, ConstIterator _End, Allocator _Alloc)
	{
		if (!_Alloc) throw std::bad_alloc();
		Allocator_ = _Alloc;

		const auto _Size = _End - _Begin;

		if (_Size <= 0)
		{
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * LIBSVC_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + LIBSVC_EMPTY_CAPACITY;
			return;
		}

		_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _Size * 2);
		if (!_MyFirst) throw std::bad_alloc();
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
	void EmplaceImpl(Iterator _Where, _ArgsTy &&... _Args)
	{
		new (_Where) ValueType(std::forward<_ArgsTy>(_Args)...);
	}

	void ReserveImpl(SizeType _NewCapacity, IndexType _Front, IndexType _Tail)
	{
		auto _Size = Size() + _Tail - _Front;
		auto _TailSize = (_MyLast - _MyFirst) - _Front;

		if (_NewCapacity <= _Size || _Front > _Tail)
			throw std::bad_alloc();

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

		DestoryData();
		_MyFirst = _Data;
		_MyLast = _Data + _Size;
		_MyEnd = _Data + _NewCapacity;
	}

	void CopyImpl(IndexType _Front, IndexType _Tail)
	{
		auto _Size = Size() + _Tail - _Front;
		auto _TailSize = (_MyLast - _MyFirst) - _Front;

		if (_Front > _Tail || Capacity() < _Size)
			throw std::exception("Index Out Of Range!");

		if constexpr (!std::is_arithmetic_v<ValueType>)
			for (IndexType i = _TailSize - 1; i >= 0; --i)
				new (_MyFirst + _Tail + i) ValueType(std::move(_MyFirst[_Front + i]));
		else
			for (IndexType i = _TailSize - 1; i >= 0; --i)
				*(_MyFirst + _Tail + i) = _MyFirst[_Front + i];

		_MyLast = _MyFirst + _Size;
	}
public:
	void Reserve(SizeType _NewCapacity)
	{
		if (_NewCapacity == Capacity())
			return;

		if (_NewCapacity == 0)
		{
			DestoryData();
			_MyFirst = (Pointer)Allocator_->Allocate(sizeof(ValueType) * LIBSVC_EMPTY_CAPACITY);
			_MyLast = _MyFirst;
			_MyEnd = _MyFirst + LIBSVC_EMPTY_CAPACITY;
			return;
		}

		auto _Data = (Pointer)Allocator_->Allocate(sizeof(ValueType) * _NewCapacity);
		auto _Size = std::min(_NewCapacity, Size());

		if constexpr (!std::is_arithmetic_v<ValueType>)
			for (IndexType i = 0; i < _Size; ++i)
				new (_Data + i) ValueType(std::move(_MyFirst[i]));
		else
			memcpy(_Data, _MyFirst, sizeof(ValueType) * _Size);

		DestoryData();
		_MyFirst = _Data;
		_MyLast = _Data + _Size;
		_MyEnd = _Data + _NewCapacity;
	}

	void Resize(SizeType _NewSize, ConstReference _Val = ValueType(0))
	{
		if (_NewSize == Size())
			return;

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
	void Emplace(ConstIterator _Where, _ArgsTy &&... _Args)
	{
#ifdef LIBSVC_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			throw std::exception("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		if (_MyLast + 1 > _MyEnd)
			ReserveImpl(Capacity() * 2, _Where - _MyFirst, _Where - _MyFirst + 1);
		else
			CopyImpl(_Where - _MyFirst, _Where - _MyFirst + 1);
		EmplaceImpl(_MyFirst + Idx, std::forward<_ArgsTy>(_Args)...);
	}

	template<typename... _ArgsTy>
	void EmplaceBack(_ArgsTy &&... _Args)
	{
		Emplace(_MyLast, std::forward<_ArgsTy>(_Args)...);
	}

	void Insert(ConstIterator _Where, const ValueType& _Value)
	{
#ifdef LIBSVC_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			throw std::exception("Out Of Range!");
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

	void Insert(ConstIterator _Where, ValueType&& _Value)
	{
#ifdef LIBSVC_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			throw std::exception("Out Of Range!");
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

	void Insert(ConstIterator _Where, SizeType _Count = 1, const ValueType& _Value = ValueType(0))
	{
#ifdef LIBSVC_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			throw std::exception("Out Of Range!");
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

	void Insert(ConstIterator _Where, ConstIterator _First, ConstIterator _Last)
	{
#ifdef LIBSVC_DEBUG
		if (_Where > _MyLast || _Where < _MyFirst)
			throw std::exception("Out Of Range!");
#endif
		auto Idx = _Where - _MyFirst;
		SizeType _Count = _Last - _First;
		if (_Count < 0)
			throw std::exception("Range Error!");

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

	void Clear()
	{
		auto Iter = _MyFirst;
		if constexpr (!std::is_arithmetic_v<ValueType>)
			while (Iter != _MyLast)
				Iter->~ValueType();
		_MyLast = _MyFirst;
	}

	void PopBack()
	{
		--_MyLast;
		if constexpr (!std::is_arithmetic_v<ValueType>)
			_MyLast->~ValueType();
	}
};

LIBSVCSTLEND