#pragma once
#include "CPU.h"
#include "Libraries/Util/Logger.h"

#define _D_Dragonian_Lib_Operator_Unary_Function_Def(_Function, Unfold, AvxThroughput) \
namespace UnaryOperators \
{ \
	namespace _Function##Unary { \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r) { _D_Dragonian_Lib_Namespace Operators::UnaryOperators::_Function(__r); }; \
	} \
} \
template <typename _Type> \
template<typename _ResultType, size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Unary( \
	_ResultType* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type> && TypeTraits::IsSameTypeValue<_Type, _ResultType>) \
		ImplMultiThreadBasic<decltype(UnaryOperators::_Function##<_Type>), UnaryOperators::_Function##<_Type>, decltype(UnaryOperators::_Function##<Vectorized<_Type>>), UnaryOperators::_Function##<Vectorized<_Type>>, TypeDef::UnaryOperatorType, false, Unfold, AvxThroughput, _NRank, _ResultType, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo), \
			nullptr, \
			nullptr, \
			nullptr, \
			Continuous \
		); \
	else \
		ImplMultiThreadBasic<decltype(UnaryOperators::_Function##<_Type>), UnaryOperators::_Function##<_Type>, decltype(UnaryOperators::_Function##<_Type>), UnaryOperators::_Function##<_Type>, TypeDef::UnaryOperatorType, false, Unfold, AvxThroughput, _NRank, _ResultType, _Type, _Type>(\
			_Dest, \
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo), \
			_Src, \
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo), \
			nullptr, \
			nullptr, \
			nullptr, \
			Continuous \
		); \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace UnaryOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { -_Left }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Negative(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			_D_Dragonian_Lib_Simd_Not_Implemented_Error;
		else
			return -_Value;
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Sqrt() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::sqrt(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sqrt(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sqrt();
		else
			return static_cast<_Type>(std::sqrt(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.RSqrt() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { 1 / std::sqrt(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		RSqrt(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.RSqrt();
		else
			return static_cast<_Type>(1 / std::sqrt(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Reciprocal() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { 1 / _Left }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Reciprocal(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Reciprocal();
		else
			return static_cast<_Type>(1 / _Value);
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Abs(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::abs(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Abs(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Abs(_mm256_set1_epi32(0x7FFFFFFF));
		else
			return static_cast<_Type>(std::abs(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Sin() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::sin(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sin(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sin();
		else
			return static_cast<_Type>(std::sin(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Cos() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::cos(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Cos(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Cos();
		else
			return static_cast<_Type>(std::cos(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Tan() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::tan(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Tan(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Tan();
		else
			return static_cast<_Type>(std::tan(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ASin() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::asin(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ASin(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ASin();
		else
			return static_cast<_Type>(std::asin(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ACos() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::acos(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ACos(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ACos();
		else
			return static_cast<_Type>(std::acos(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ATan() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::atan(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ATan(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ATan();
		else
			return static_cast<_Type>(std::atan(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Sinh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::sinh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Sinh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Sinh();
		else
			return static_cast<_Type>(std::sinh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Cosh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::cosh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Cosh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Cosh();
		else
			return static_cast<_Type>(std::cosh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Tanh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::tanh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Tanh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Tanh();
		else
			return static_cast<_Type>(std::tanh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ASinh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::asinh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ASinh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ASinh();
		else
			return static_cast<_Type>(std::asinh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ACosh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::acosh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ACosh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ACosh();
		else
			return static_cast<_Type>(std::acosh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.ATanh() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::atanh(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ATanh(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.ATanh();
		else
			return static_cast<_Type>(std::atanh(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Exp() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::exp(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Exp(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Exp();
		else
			return static_cast<_Type>(std::exp(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Log() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::log(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log();
		else
			return static_cast<_Type>(std::log(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Log2() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::log2(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log2(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log2();
		else
			return static_cast<_Type>(std::log2(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Log10() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::log10(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Log10(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Log10();
		else
			return static_cast<_Type>(std::log10(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Ceil() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::ceil(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Ceil(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Ceil();
		else
			return static_cast<_Type>(std::ceil(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Floor() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::floor(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Floor(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Floor();
		else
			return static_cast<_Type>(std::floor(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Round() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::round(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Round(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Round();
		else
			return static_cast<_Type>(std::round(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Trunc() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { std::trunc(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Trunc(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Trunc();
		else
			return static_cast<_Type>(std::trunc(_Value));
	}

	template <typename _Type, typename = std::enable_if_t <
		requires(_Type& _Left) { { _Left.Frac() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; } ||
		requires(_Type & _Left) { { (_Left)-std::trunc(_Left) }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<_Type>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Frac(const _Type& _Value)
	{
		if constexpr (IsVectorizedValue<_Type>)
			return _Value.Frac();
		else
			return static_cast<_Type>(static_cast<decltype(std::trunc(_Value))>(_Value) - std::trunc(_Value));
	}

}

_D_Dragonian_Lib_Operator_Unary_Function_Def(Negative, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sqrt, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(RSqrt, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Reciprocal, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Abs, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sin, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Cos, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Tan, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ASin, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ACos, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ATan, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Sinh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Cosh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Tanh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ASinh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ACosh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ATanh, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Exp, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log2, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log10, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Ceil, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Floor, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Round, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Trunc, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Frac, 8, 2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Unary_Function_Def