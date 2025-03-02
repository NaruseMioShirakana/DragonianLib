#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Operator_Binary_Function_Def(_Function, Unfold, AvxThroughput) \
namespace BinaryOperators \
{ \
	namespace _Function##Binary \
	{ \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) { {_D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
		template <class _ValueType> \
		concept HasVectorOperatorValue = requires(Vectorized<_ValueType> & __r, Vectorized<_ValueType> & __l) { {_D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
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

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Add(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A + _B }; })
			return _Left + _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Add(_B) }; })
			return _Left.Add(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.add(_B) }; })
			return _Left.add(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sub(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A - _B }; })
			return _Left - _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Sub(_B) }; })
			return _Left.Sub(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.sub(_B) }; })
			return _Left.sub(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mul(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A* _B }; })
			return _Left * _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Mul(_B) }; })
			return _Left.Mul(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.mul(_B) }; })
			return _Left.mul(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Div(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A / _B }; })
			return _Left / _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Div(_B) }; })
			return _Left.Div(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.div(_B) }; })
			return _Left.div(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Mod(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (IsFloatingPointValue<_Type>)
			return std::fmod(_Left, _Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A% _B }; })
			return _Left % _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Mod(_B) }; })
			return _Left.Mod(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.mod(_B) }; })
			return _Left.mod(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		And(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A& _B }; })
			return _Left & _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.And(_B) }; })
			return _Left.And(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Or(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A | _B }; })
			return _Left | _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Or(_B) }; })
			return _Left.Or(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryAnd(_Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A& _B }; })
			return _Left & _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.BinaryAnd(_B) }; })
			return _Left.BinaryAnd(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BinaryOr(_Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A | _B }; })
			return _Left | _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.BinaryOr(_B) }; })
			return _Left.BinaryOr(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Xor(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A^ _B }; })
			return _Left ^ _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Xor(_B) }; })
			return _Left.Xor(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LShift(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A << _B }; })
			return _Left << _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.LShift(_B) }; })
			return _Left.LShift(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.lshift(_B) }; })
			return _Left.lshift(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		RShift(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { _A >> _B }; })
			return _Left >> _Right;
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.RShift(_B) }; })
			return _Left.RShift(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.rshift(_B) }; })
			return _Left.rshift(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Pow(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { std::pow(_A, _B) }; })
			return std::pow(_Left, _Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Pow(_B) }; })
			return _Left.Pow(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.pow(_B) }; })
			return _Left.pow(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Max(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { std::max(_A, _B) }; })
			return std::max(_Left, _Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Max(_B) }; })
			return _Left.Max(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.max(_B) }; })
			return _Left.max(_Right);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Min(const _Type& _Left, const _Type& _Right)
	{
		if constexpr (requires(_Type & _A, _Type & _B) { { std::min(_A, _B) }; })
			return std::min(_Left, _Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Min(_B) }; })
			return _Left.Min(_Right);
		else if constexpr (requires(_Type & _A, _Type & _B) { { _A.min(_B) }; })
			return _Left.min(_Right);
		else
			return std::nullopt;
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
_D_Dragonian_Lib_Operator_Binary_Function_Def(Max, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Min, 8, 2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Function_Def