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

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasCRange<_Type>>>
decltype(auto) CBegin(const _Type& _Container)
{
	if constexpr (TypeTraits::HasCLRange<_Type>)
		return _Container.cbegin();
	else if constexpr (TypeTraits::HasCHRange<_Type>)
		return _Container.CBegin();
}
template <typename _Type, typename = std::enable_if_t<TypeTraits::HasCRange<_Type>>>
decltype(auto) CEnd(const _Type& _Container)
{
	if constexpr (TypeTraits::HasCLRange<_Type>)
		return _Container.cend();
	else if constexpr (TypeTraits::HasCHRange<_Type>)
		return _Container.CEnd();
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Begin(_Type&& _Container)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).begin();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).Begin();
	else
		return CBegin(std::forward<_Type>(_Container));
}
template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) End(_Type&& _Container)
{
	if constexpr (TypeTraits::HasLRange<_Type>)
		return std::forward<_Type>(_Container).end();
	else if constexpr (TypeTraits::HasHRange<_Type>)
		return std::forward<_Type>(_Container).End();
	else
		return CEnd(std::forward<_Type>(_Container));
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

	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp() = delete;
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
		if constexpr (TypeTraits::HasIntegerSubtraction<MyIterTypeBeg>)
			return static_cast<UInt64>(_MyEnd - _MyBegin);
		else if constexpr (TypeTraits::HasFrontIncrement<MyIterTypeBeg>)
		{
			auto _Tmp = _MyBegin;
			UInt64 _Size = 0;
			while (_Tmp != _MyEnd)
			{
				++_Tmp;
				++_Size;
			}
			return _Size;
		}
		else 			
			_D_Dragonian_Lib_Throw_Exception("Could not get size!");
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

	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(RangesWrp&& _RRight) noexcept = delete;
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(const RangesWrp& _Right) = delete;

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

	template <typename _Type2, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<MyValueType, _Type2>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline RangesWrp& operator=(const _Type2& _Right)
	{
		for (size_t i = 0; i < Size(); ++i)
			operator[](i) = static_cast<MyValueType>(_Right);
		return *this;
	}

	RangesWrp<const MyValueType*, const MyValueType*> RawConst() const
	{
		return RangesWrp<const MyValueType*, const MyValueType*>(_MyBegin, _MyEnd);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Raw() const noexcept
	{
		return RangesWrp<decltype(&*_MyBegin), decltype(&*_MyEnd)>{ &*_MyBegin, &* _MyEnd };
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

protected:
	MyIterTypeBeg _MyBegin = nullptr;
	MyIterTypeEnd _MyEnd = nullptr;
};

template <typename _ValueType>
using ConstantRanges = RangesWrp<const _ValueType*, const _ValueType*>;
template <typename _ValueType>
using MutableRanges = RangesWrp<_ValueType*, _ValueType*>;

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
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
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
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
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

template <typename _Type, typename _IntegerType = Int64,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
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

template <typename _IntegerType = Int64, typename _Type,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Enumrate(_Type& _Value)
{
	return MutableEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _IntegerType = Int64, typename _Type,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Enumrate(const _Type& _Value)
{
	return ConstEnumrate<_Type, _IntegerType>(_Value);
}

template <typename _IntegerType = Int64, typename _Type,
	typename = std::enable_if_t<TypeTraits::IsIntegerValue<_IntegerType>>,
	typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Enumrate(_Type&& _Value)
{
	return ObjectiveEnumrate<_Type, _IntegerType>(std::forward<_Type>(_Value));
}

template <typename _IteratorTypeBeg, typename _IteratorTypeEnd>
decltype(auto) Ranges(_IteratorTypeBeg _Begin, _IteratorTypeEnd _End)
{
	return RangesWrp(_Begin, _End);
}

template <typename _Type, typename = std::enable_if_t<TypeTraits::HasRange<_Type>>>
decltype(auto) Ranges(_Type&& _Rng)
{
	return RangesWrp(Begin(std::forward<_Type>(_Rng)), End(std::forward<_Type>(_Rng)));
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




