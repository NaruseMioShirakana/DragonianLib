#pragma once
#include "Libraries/Base.h"

#define _D_Dragonian_Lib_Template_Library_Space_Begin _D_Dragonian_Lib_Space_Begin namespace TemplateLibrary {
#define _D_Dragonian_Lib_Template_Library_Space_End } _D_Dragonian_Lib_Space_End
#define _D_Dragonian_Lib_TL_Namespace _D_Dragonian_Lib_Namespace TemplateLibrary::

_D_Dragonian_Lib_Template_Library_Space_Begin

template <typename _Type>
class _MyLess
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A < _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type & _Left, const _Type & _Right)
	{
		return _Left < _Right;
	}

};

template <typename _Type>
class _MyGreater
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A > _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type& _Left, const _Type& _Right)
	{
		return _Left > _Right;
	}
};

template <typename _Type>
class _MyEqual
{
	template <typename Ty>
	static constexpr auto Check(const Ty& _A, const Ty& _B) -> decltype(_A == _B, std::true_type()) { return {}; }
	static constexpr std::false_type Check(...) { return {}; }

public:
	static constexpr bool HasOperator = decltype(Check(TypeTraits::InstanceOf<_Type>(), TypeTraits::InstanceOf<_Type>()))::value;

	_D_Dragonian_Lib_Force_Inline std::enable_if_t<HasOperator, bool> operator()(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (std::is_floating_point_v<_Type>)
			return std::abs(_Left - _Right) < std::numeric_limits<_Type>::epsilon();
		else
			return _Left == _Right;
	}
};

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Begin(_Type&& _Container)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).begin();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).Begin();
}
template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) End(_Type&& _Container)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).end();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).End();
}

template <typename _IteratorType, typename = std::enable_if_t<TypeTraits::IsIterator<_IteratorType>>>
class IteratorRanges
{
public:
	using _ValueType = decltype(*TypeTraits::InstanceOf<_IteratorType>());

	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges() = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges(const _IteratorType& _Begin, const _IteratorType& _End) : _MyBegin(_Begin), _MyEnd(_End) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline ~IteratorRanges() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges(const IteratorRanges&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges(IteratorRanges&&) = default;

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data() const { return &*_MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) data() const { return &*_MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const { return _MyEnd; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end() const { return _MyEnd; }

	_D_Dragonian_Lib_Constexpr_Force_Inline UInt64 Size() const { return _MyEnd - _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](size_t _Index) const
	{
		if constexpr (TypeTraits::CouldIndex<_IteratorType>)
			return _MyBegin[_Index];
		else
			return *(_MyBegin + _Index);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges& operator=(IteratorRanges&& _Right) = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges& operator=(const IteratorRanges& _Right) = delete;

	template <typename _IteratorType2, typename = std::enable_if_t<
		TypeTraits::IsIterator<_IteratorType2>&&
		TypeTraits::CouldBeConvertedFromValue<_ValueType, decltype(*TypeTraits::InstanceOf<_IteratorType2>())>>>
		_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges& Assign(const IteratorRanges<_IteratorType2>& _Right)
	{
		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right[i];
		return *this;
	}

	template <typename _Type2, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_ValueType, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline IteratorRanges& operator=(const _Type2& _Right)
	{
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right;
		return *this;
	}

protected:
	_IteratorType _MyBegin;
	_IteratorType _MyEnd;
};

template <typename _Type>
class ConstantRanges
{
public:
	_D_Dragonian_Lib_Constexpr_Force_Inline ConstantRanges() = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline ConstantRanges(const _Type* _Begin, const _Type* _End) : _MyBegin(_Begin), _MyEnd(_End) {}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) data() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const { return _MyEnd; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end() const { return _MyEnd; }

	_D_Dragonian_Lib_Constexpr_Force_Inline UInt64 Size() const { return _MyEnd - _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](size_t _Index) const { return _MyBegin[_Index]; }

protected:
	const _Type* _MyBegin = nullptr;
	const _Type* _MyEnd = nullptr;
};

template <typename _Type>
class MutableRanges
{
public:
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges() = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline ~MutableRanges() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges(const MutableRanges&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges(MutableRanges&&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges(_Type* _Begin, _Type* _End) : _MyBegin(_Begin), _MyEnd(_End) {}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) data() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) End() const { return _MyEnd; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) begin() const { return _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) end() const { return _MyEnd; }

	_D_Dragonian_Lib_Constexpr_Force_Inline UInt64 Size() const { return _MyEnd - _MyBegin; }
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](size_t _Index) const { return _MyBegin[_Index]; }

	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges& operator=(MutableRanges&& _Right) = delete;

	_D_Dragonian_Lib_Constexpr_Force_Inline operator ConstantRanges<_Type>() const { return ConstantRanges<_Type>(_MyBegin, _MyEnd); }
	_D_Dragonian_Lib_Constexpr_Force_Inline ConstantRanges<_Type> C() const { return ConstantRanges<_Type>(_MyBegin, _MyEnd); }

	template <typename _Type2, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_Type, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges& operator=(const MutableRanges<_Type2>& _Right)
	{
		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right[i];
		return *this;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges& operator=(const MutableRanges& _Right)
	{
		if (this == &_Right)
			return *this;

		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right[i];
		return *this;
	}

	template <typename _Type2, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_Type, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges& operator=(const ConstantRanges<_Type2>& _Right)
	{
		if (Size() != _Right.Size())
			_D_Dragonian_Lib_Throw_Exception("Size not match!");
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right[i];
		return *this;
	}

	template <typename _Type2, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_Type, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline MutableRanges& operator=(const _Type2& _Right)
	{
		for (size_t i = 0; i < Size(); ++i)
			_MyBegin[i] = _Right;
		return *this;
	}

protected:
	_Type* _MyBegin = nullptr;
	_Type* _MyEnd = nullptr;
};

template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArithmeticValue<_Type>>>
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

template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArithmeticValue<_Type>>>
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

template <typename _IteratorType, typename _IntegerType = Int64, typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>>
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
	std::pair<_IntegerType, _MyReferenceType> operator*() const { return { _MyIndex, *_MyIterator }; }
	decltype(auto) operator->() const { return _MyIterator; }

private:
	_IteratorType _MyIterator;
	_IntegerType _MyIndex;
};

template <typename _Type, typename _IntegerType = Int64,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<std::ranges::range<_Type>>>
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

template <typename _Type, typename _IntegerType = Int64,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<std::ranges::range<_Type>>>
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

template <typename _IntegerType = Int64, typename _Type,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<std::ranges::range<_Type>>>
decltype(auto) Enumrate(_Type& _Value)
{
	return MutableEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _IntegerType = Int64, typename _Type,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<std::ranges::range<_Type>>>
decltype(auto) Enumrate(const _Type& _Value)
{
	return ConstEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _Type>
decltype(auto) Ranges(const _Type* _Begin, const _Type* _End)
{
	return ConstantRanges<_Type>(_Begin, _End);
}

template <typename _Type>
decltype(auto) Ranges(_Type* _Begin, _Type* _End)
{
	return MutableRanges<_Type>(_Begin, _End);
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Ranges(_Type&& _Container)
{
	return Ranges(&*Begin(std::forward<_Type>(_Container)), &*End(std::forward<_Type>(_Container)));
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) CRanges(_Type&& _Container)
{
	return ConstantRanges(&*Begin(std::forward<_Type>(_Container)), &*End(std::forward<_Type>(_Container)));
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) CBRanges(_Type&& _Container)
{
	return ConstantRanges((const Byte*)&*Begin(std::forward<_Type>(_Container)), (const Byte*)&*End(std::forward<_Type>(_Container)));
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>&& TypeTraits::IsPointerLike<decltype(Begin(TypeTraits::InstanceOf<_Type>()))>>>
decltype(auto) ByteRanges(const _Type& _Container)
{
	using ValueType = decltype(*Begin(_Container));
	const auto _Begin = reinterpret_cast<const Byte*>(&*Begin(_Container));
	const auto _Size = End(_Container) - Begin(_Container);
	const auto _End = _Begin + sizeof(ValueType) * _Size;
	return ConstantRanges<Byte>(_Begin, _End);
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArithmeticValue<_Type>>>
decltype(auto) Ranges(_Type _Begin, _Type _End, _Type _Step)
{
	if (_Step == 0)
		_D_Dragonian_Lib_Throw_Exception("Step cannot be 0.");
	if (_Begin < _End && _Step < 0)
		_D_Dragonian_Lib_Throw_Exception("Step must be positive.");
	if (_Begin > _End && _Step > 0)
		_D_Dragonian_Lib_Throw_Exception("Step must be negative.");
	return NumberRanges<_Type>(_Begin, _End, _Step);
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArithmeticValue<_Type>>>
decltype(auto) Ranges(_Type _Begin, _Type _End)
{
	return NumberRanges<_Type>(_Begin, _End, _Begin < _End ? _Type(1) : _Type(-1));
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::IsArithmeticValue<_Type>>>
decltype(auto) Ranges(_Type _End)
{
	return NumberRanges<_Type>(_Type(0), _End, _End > _Type(0) ? _Type(1) : _Type(-1));
}

_D_Dragonian_Lib_Template_Library_Space_End

_D_Dragonian_Lib_Space_Begin

enum class Device
{
	CPU = 0,
	CUDA,
	HIP,
	DIRECTX,
	CUSTOM
};

static inline size_t NopID = size_t(-1);

namespace DragonianLibSTL
{
	using namespace _D_Dragonian_Lib_Namespace TemplateLibrary;
}

_D_Dragonian_Lib_Space_End




