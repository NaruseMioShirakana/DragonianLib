#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Operator_Binary_Function_Def(_Function, Unfold, AvxThroughput) \
namespace BinaryOperators \
{ \
	namespace _Function##Binary \
	{ \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) { _D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l); }; \
		template <class _ValueType> \
		concept HasInplaceOperatorValue = HasOperatorValue<_ValueType> && std::is_copy_assignable_v<_ValueType>; \
		 \
	} \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Scalar( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>) \
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<Vectorized<_Type>>), BinaryOperators::_Function##<Vectorized<_Type>>, TypeDef::ConstantOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo), \
			nullptr, \
			nullptr, \
			std::make_shared<_Type>(_Value), \
			Continuous \
		); \
	else \
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, TypeDef::ConstantOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo), \
			nullptr, \
			nullptr, \
			std::make_shared<_Type>(_Value), \
			Continuous \
		); \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Tensor( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src1, \
	const OperatorParameter<_NRank>& _SrcInfo1, \
	const _Type* _Src2, \
	const OperatorParameter<_NRank>& _SrcInfo2, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>) \
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<Vectorized<_Type>>), BinaryOperators::_Function##<Vectorized<_Type>>, TypeDef::BinaryOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src1, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo1), \
			_Src2, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo2), \
			nullptr, \
			Continuous \
		); \
	else \
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, TypeDef::BinaryOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src1, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo1), \
			_Src2, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo2), \
			nullptr, \
			Continuous \
		); \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace BinaryOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { {_Left + _Right}->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Add(const _Type& _Left, const _Type& _Right)
	{
		return _Left + _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left - _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sub(const _Type& _Left, const _Type& _Right)
	{
		return _Left - _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left * _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mul(const _Type& _Left, const _Type& _Right)
	{
		return _Left * _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left / _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Div(const _Type& _Left, const _Type& _Right)
	{
		return _Left / _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { std::fmod(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left, _Type & _Right) { { _Left% _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mod(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (IsFloatingPointValue<_Type>)
			return std::fmod(_Left, _Right);
		else
			return _Left % _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left && _Right }; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		And(const _Type& _Left, const _Type& _Right)
	{
		return _Type(_Left && _Right);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left || _Right }; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Or(const _Type& _Left, const _Type& _Right)
	{
		return _Type(_Left || _Right);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left & _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryAnd(_Type& _Left, const _Type& _Right)
	{
		return _Left & _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left | _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryOr(_Type& _Left, const _Type& _Right)
	{
		return _Left | _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left ^ _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Xor(const _Type& _Left, const _Type& _Right)
	{
		return _Left ^ _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left << _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LShift(const _Type& _Left, const _Type& _Right)
	{
		return _Left << _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		(requires(_Type& _Left, _Type& _Right) { { _Left >> _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; })
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		RShift(const _Type& _Left, const _Type& _Right)
	{
		return _Left >> _Right;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left, _Type& _Right) { { _Left.Pow(_Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type& _Left, _Type& _Right) { { std::pow(_Left, _Right) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Pow(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Left.Pow(_Right);
		else
			return std::pow(_Left, _Right);
	}
}

_D_Dragonian_Lib_Operator_Binary_Function_Def(Add, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Sub, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mul, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Div, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mod, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(And, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Or, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Xor, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BinaryOr, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BinaryAnd, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(LShift, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(RShift, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Pow, 8, 2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Function_Def