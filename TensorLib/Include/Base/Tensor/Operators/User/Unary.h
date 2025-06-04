#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/OperatorBase.h"

#define _D_Dragonian_Lib_Operator_Unary_Traits(_Function) \
namespace UnaryOperators \
{ \
	namespace _Function##Unary { \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r) { {_D_Dragonian_Lib_Namespace Operators::UnaryOperators::_Function(__r)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
		template <class _ValueType> \
		concept HasVectorOperatorValue = requires(Vectorized<_ValueType> & __r) { {_D_Dragonian_Lib_Namespace Operators::UnaryOperators::_Function(__r)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
	} \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace UnaryOperators
{
	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Negative(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Negative(); })
			return static_cast<_Type>(_Value.Negative());
		else if constexpr (requires(_Type & _Left) { _Left.negative(); })
			return static_cast<_Type>(_Value.negative());
		else if constexpr (requires(_Type & _Left) { -_Left; })
			return static_cast<_Type>(-_Value);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sqrt(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Sqrt(); })
			return static_cast<_Type>(_Value.Sqrt());
		else if constexpr (requires(_Type & _Left) { _Left.sqrt(); })
			return static_cast<_Type>(_Value.sqrt());
		else if constexpr (requires(_Type & _Left) { std::sqrt(_Left); })
			return static_cast<_Type>(std::sqrt(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		RSqrt(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.RSqrt(); })
			return static_cast<_Type>(_Value.RSqrt());
		else if constexpr (requires(_Type & _Left) { _Left.rsqrt(); })
			return static_cast<_Type>(_Value.rsqrt());
		else if constexpr (requires(_Type & _Left) { _Type(1) / _Left.Sqrt(); })
			return static_cast<_Type>(_Type(1) / _Value.Sqrt());
		else if constexpr (requires(_Type & _Left) { _Type(1) / _Left.sqrt(); })
			return static_cast<_Type>(_Type(1) / _Value.sqrt());
		else if constexpr (requires(_Type & _Left) { _Type(1) / std::sqrt(_Left); })
			return static_cast<_Type>(_Type(1) / std::sqrt(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Reciprocal(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Reciprocal(); })
			return static_cast<_Type>(_Value.Reciprocal());
		else if constexpr (requires(_Type & _Left) { _Left.reciprocal(); })
			return static_cast<_Type>(_Value.reciprocal());
		else if constexpr (requires(_Type & _Left) { _Type(1) / _Left; })
			return static_cast<_Type>(_Type(1) / _Value);
		else
			return std::nullopt;
	}

	template <typename _Type>
	decltype(auto) CAbs(const std::complex<_Type>& _Value)
	{
		return std::complex<_Type>{ std::abs(_Value) };
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Abs(const _Type& _Value)
	{
		if constexpr (IsComplexValue<_Type>)
			return CAbs(_Value);
		else if constexpr (requires(_Type & _Left) { _Left.Abs(); })
			return static_cast<_Type>(_Value.Abs());
		else if constexpr (requires(_Type & _Left) { _Left.abs(); })
			return static_cast<_Type>(_Value.abs());
		else if constexpr (requires(_Type & _Left) { std::abs(_Left); })
			return static_cast<_Type>(std::abs(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Force_Inline decltype(auto)
		Polar(const _Type& _Value)
	{
		if constexpr (IsComplexValue<_Type>)
			return std::polar(_Value.real(), _Value.imag());
		else if constexpr (IsAnyOfValue<_Type, Vectorized<Complex32>, Vectorized<Complex64>>)
			return _Value.Polar();
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Force_Inline decltype(auto)
		ATan2(const _Type& _Value)
	{
		if constexpr (IsComplexValue<_Type>)
			return _Type{ std::abs(_Value), std::atan2(_Value.imag(), _Value.real()) };
		else if constexpr (IsAnyOfValue<_Type, Vectorized<Complex32>, Vectorized<Complex64>>)
			return _Value.ATan2();
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sin(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Sin(); })
			return static_cast<_Type>(_Value.Sin());
		else if constexpr (requires(_Type & _Left) { _Left.sin(); })
			return static_cast<_Type>(_Value.sin());
		else if constexpr (requires(_Type & _Left) { std::sin(_Left); })
			return static_cast<_Type>(std::sin(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Cos(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Cos(); })
			return static_cast<_Type>(_Value.Cos());
		else if constexpr (requires(_Type & _Left) { _Left.cos(); })
			return static_cast<_Type>(_Value.cos());
		else if constexpr (requires(_Type & _Left) { std::cos(_Left); })
			return static_cast<_Type>(std::cos(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Tan(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Tan(); })
			return static_cast<_Type>(_Value.Tan());
		else if constexpr (requires(_Type & _Left) { _Left.tan(); })
			return static_cast<_Type>(_Value.tan());
		else if constexpr (requires(_Type & _Left) { std::tan(_Left); })
			return static_cast<_Type>(std::tan(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ASin(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ASin(); })
			return static_cast<_Type>(_Value.ASin());
		else if constexpr (requires(_Type & _Left) { _Left.asin(); })
			return static_cast<_Type>(_Value.asin());
		else if constexpr (requires(_Type & _Left) { std::asin(_Left); })
			return static_cast<_Type>(std::asin(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ACos(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ACos(); })
			return static_cast<_Type>(_Value.ACos());
		else if constexpr (requires(_Type & _Left) { _Left.acos(); })
			return static_cast<_Type>(_Value.acos());
		else if constexpr (requires(_Type & _Left) { std::acos(_Left); })
			return static_cast<_Type>(std::acos(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ATan(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ATan(); })
			return static_cast<_Type>(_Value.ATan());
		else if constexpr (requires(_Type & _Left) { _Left.atan(); })
			return static_cast<_Type>(_Value.atan());
		else if constexpr (requires(_Type & _Left) { std::atan(_Left); })
			return static_cast<_Type>(std::atan(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sinh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Sinh(); })
			return static_cast<_Type>(_Value.Sinh());
		else if constexpr (requires(_Type & _Left) { _Left.sinh(); })
			return static_cast<_Type>(_Value.sinh());
		else if constexpr (requires(_Type & _Left) { std::sinh(_Left); })
			return static_cast<_Type>(std::sinh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Cosh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Cosh(); })
			return static_cast<_Type>(_Value.Cosh());
		else if constexpr (requires(_Type & _Left) { _Left.cosh(); })
			return static_cast<_Type>(_Value.cosh());
		else if constexpr (requires(_Type & _Left) { std::cosh(_Left); })
			return static_cast<_Type>(std::cosh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Tanh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Tanh(); })
			return static_cast<_Type>(_Value.Tanh());
		else if constexpr (requires(_Type & _Left) { _Left.tanh(); })
			return static_cast<_Type>(_Value.tanh());
		else if constexpr (requires(_Type & _Left) { std::tanh(_Left); })
			return static_cast<_Type>(std::tanh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ASinh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ASinh(); })
			return static_cast<_Type>(_Value.ASinh());
		else if constexpr (requires(_Type & _Left) { _Left.asinh(); })
			return static_cast<_Type>(_Value.asinh());
		else if constexpr (requires(_Type & _Left) { std::asinh(_Left); })
			return static_cast<_Type>(std::asinh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ACosh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ACosh(); })
			return static_cast<_Type>(_Value.ACosh());
		else if constexpr (requires(_Type & _Left) { _Left.acosh(); })
			return static_cast<_Type>(_Value.acosh());
		else if constexpr (requires(_Type & _Left) { std::acosh(_Left); })
			return static_cast<_Type>(std::acosh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ATanh(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.ATanh(); })
			return static_cast<_Type>(_Value.ATanh());
		else if constexpr (requires(_Type & _Left) { _Left.atanh(); })
			return static_cast<_Type>(_Value.atanh());
		else if constexpr (requires(_Type & _Left) { std::atanh(_Left); })
			return static_cast<_Type>(std::atanh(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Exp(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Exp(); })
			return static_cast<_Type>(_Value.Exp());
		else if constexpr (requires(_Type & _Left) { _Left.exp(); })
			return static_cast<_Type>(_Value.exp());
		else if constexpr (requires(_Type & _Left) { std::exp(_Left); })
			return static_cast<_Type>(std::exp(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Exp2(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Exp2(); })
			return static_cast<_Type>(_Value.Exp2());
		else if constexpr (requires(_Type & _Left) { _Left.exp2(); })
			return static_cast<_Type>(_Value.exp2());
		else if constexpr (requires(_Type & _Left) { std::exp2(_Left); })
			return static_cast<_Type>(std::exp2(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Exp10(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Exp10(); })
			return static_cast<_Type>(_Value.Exp10());
		else if constexpr (requires(_Type & _Left) { _Left.exp10(); })
			return static_cast<_Type>(_Value.exp10());
		else if constexpr (requires(_Type & _Left) { std::pow(10, _Left); })
			return static_cast<_Type>(std::pow(10, _Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Log(); })
			return static_cast<_Type>(_Value.Log());
		else if constexpr (requires(_Type & _Left) { _Left.log(); })
			return static_cast<_Type>(_Value.log());
		else if constexpr (requires(_Type & _Left) { std::log(_Left); })
			return static_cast<_Type>(std::log(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log2(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Log2(); })
			return static_cast<_Type>(_Value.Log2());
		else if constexpr (requires(_Type & _Left) { _Left.log2(); })
			return static_cast<_Type>(_Value.log2());
		else if constexpr (requires(_Type & _Left) { std::log2(_Left); })
			return static_cast<_Type>(std::log2(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log10(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Log10(); })
			return static_cast<_Type>(_Value.Log10());
		else if constexpr (requires(_Type & _Left) { _Left.log10(); })
			return static_cast<_Type>(_Value.log10());
		else if constexpr (requires(_Type & _Left) { std::log10(_Left); })
			return static_cast<_Type>(std::log10(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Ceil(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Ceil(); })
			return static_cast<_Type>(_Value.Ceil());
		else if constexpr (requires(_Type & _Left) { _Left.ceil(); })
			return static_cast<_Type>(_Value.ceil());
		else if constexpr (requires(_Type & _Left) { std::ceil(_Left); })
			return static_cast<_Type>(std::ceil(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Floor(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Floor(); })
			return static_cast<_Type>(_Value.Floor());
		else if constexpr (requires(_Type & _Left) { _Left.floor(); })
			return static_cast<_Type>(_Value.floor());
		else if constexpr (requires(_Type & _Left) { std::floor(_Left); })
			return static_cast<_Type>(std::floor(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Round(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Round(); })
			return static_cast<_Type>(_Value.Round());
		else if constexpr (requires(_Type & _Left) { _Left.round(); })
			return static_cast<_Type>(_Value.round());
		else if constexpr (requires(_Type & _Left) { std::round(_Left); })
			return static_cast<_Type>(std::round(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Trunc(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Trunc(); })
			return static_cast<_Type>(_Value.Trunc());
		else if constexpr (requires(_Type & _Left) { _Left.trunc(); })
			return static_cast<_Type>(_Value.trunc());
		else if constexpr (requires(_Type & _Left) { std::trunc(_Left); })
			return static_cast<_Type>(std::trunc(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Frac(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Frac(); })
			return static_cast<_Type>(_Value.Frac());
		else if constexpr (requires(_Type & _Left) { _Left.frac(); })
			return static_cast<_Type>(_Value.frac());
		else if constexpr (requires(_Type & _Left) { _Left - _Left.Floor(); })
			return static_cast<_Type>(_Value - _Value.Floor());
		else if constexpr (requires(_Type & _Left) { _Left - _Left.floor(); })
			return static_cast<_Type>(_Value - _Value.floor());
		else if constexpr (requires(_Type & _Left) { _Left - std::floor(_Left); })
			return static_cast<_Type>(_Value - std::floor(_Value));
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BitwiseNot(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.BitwiseNot(); })
			return static_cast<_Type>(_Value.BitwiseNot());
		else if constexpr (requires(_Type & _Left) { _Left.bitwise_not(); })
			return static_cast<_Type>(_Value.bitwise_not());
		else if constexpr (requires(_Type & _Left) { ~_Left; })
			return static_cast<_Type>(~_Value);
		else
			return std::nullopt;
	}

	template <typename _Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Not(const _Type& _Value)
	{
		if constexpr (requires(_Type & _Left) { _Left.Not(); })
			return static_cast<_Type>(_Value.Not());
		else if constexpr (requires(_Type & _Left) { !_Left; })
			return static_cast<_Type>(!_Value);
		else
			return std::nullopt;
	}
}

_D_Dragonian_Lib_Operator_Unary_Traits(Sqrt);
_D_Dragonian_Lib_Operator_Unary_Traits(RSqrt);
_D_Dragonian_Lib_Operator_Unary_Traits(Reciprocal);
_D_Dragonian_Lib_Operator_Unary_Traits(Abs);
_D_Dragonian_Lib_Operator_Unary_Traits(Sin);
_D_Dragonian_Lib_Operator_Unary_Traits(Cos);
_D_Dragonian_Lib_Operator_Unary_Traits(Tan);
_D_Dragonian_Lib_Operator_Unary_Traits(ASin);
_D_Dragonian_Lib_Operator_Unary_Traits(ACos);
_D_Dragonian_Lib_Operator_Unary_Traits(ATan);
_D_Dragonian_Lib_Operator_Unary_Traits(Sinh);
_D_Dragonian_Lib_Operator_Unary_Traits(Cosh);
_D_Dragonian_Lib_Operator_Unary_Traits(Tanh);
_D_Dragonian_Lib_Operator_Unary_Traits(ASinh);
_D_Dragonian_Lib_Operator_Unary_Traits(ACosh);
_D_Dragonian_Lib_Operator_Unary_Traits(ATanh);
_D_Dragonian_Lib_Operator_Unary_Traits(Exp);
_D_Dragonian_Lib_Operator_Unary_Traits(Exp2);
_D_Dragonian_Lib_Operator_Unary_Traits(Log);
_D_Dragonian_Lib_Operator_Unary_Traits(Log2);
_D_Dragonian_Lib_Operator_Unary_Traits(Log10);
_D_Dragonian_Lib_Operator_Unary_Traits(Ceil);
_D_Dragonian_Lib_Operator_Unary_Traits(Floor);
_D_Dragonian_Lib_Operator_Unary_Traits(Round);
_D_Dragonian_Lib_Operator_Unary_Traits(Trunc);
_D_Dragonian_Lib_Operator_Unary_Traits(Frac);
_D_Dragonian_Lib_Operator_Unary_Traits(Negative);
_D_Dragonian_Lib_Operator_Unary_Traits(BitwiseNot);
_D_Dragonian_Lib_Operator_Unary_Traits(Not);
_D_Dragonian_Lib_Operator_Unary_Traits(Polar);
_D_Dragonian_Lib_Operator_Unary_Traits(ATan2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Unary_Traits