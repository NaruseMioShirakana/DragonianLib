#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(_Function, Unfold, AvxThroughput) \
namespace ComparisonOperators \
{ \
	namespace _Function##Binary \
	{ \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) { _D_Dragonian_Lib_Namespace Operators::ComparisonOperators::_Function(__r, __l); }; \
	} \
} \
 \
template <typename _Type> \
template<size_t _NRank> \
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##Scalar( \
	bool* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>) \
		ImplMultiThreadBasic<decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, decltype(ComparisonOperators::_Function##<Vectorized<_Type>>), ComparisonOperators::_Function##<Vectorized<_Type>>, TypeDef::ConstantOperatorType, true, Unfold, AvxThroughput, _NRank, bool, _Type, _Type>(\
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
		ImplMultiThreadBasic<decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, TypeDef::ConstantOperatorType, true, Unfold, AvxThroughput, _NRank, bool, _Type, _Type>(\
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
	bool* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src1, \
	const OperatorParameter<_NRank>& _SrcInfo1, \
	const _Type* _Src2, \
	const OperatorParameter<_NRank>& _SrcInfo2, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>) \
		ImplMultiThreadBasic<decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, decltype(ComparisonOperators::_Function##<Vectorized<_Type>>), ComparisonOperators::_Function##<Vectorized<_Type>>, TypeDef::BinaryOperatorType, true, Unfold, AvxThroughput, _NRank, bool, _Type, _Type>(\
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
		ImplMultiThreadBasic<decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, decltype(ComparisonOperators::_Function##<_Type>), ComparisonOperators::_Function##<_Type>, TypeDef::BinaryOperatorType, true, Unfold, AvxThroughput, _NRank, bool, _Type, _Type>(\
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

namespace ComparisonOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

	template <typename Type, typename = std::enable_if_t <
		requires(Type& _Left, Type& _Right) { { std::fabs(_Left - _Right) <= std::numeric_limits<Type>::epsilon() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left == _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left == _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Equal(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) <= std::numeric_limits<Type>::epsilon();
		else
			return A == B;
	}

	template <typename Type, typename = std::enable_if_t <
		(requires(Type& _Left, Type& _Right) { { std::fabs(_Left - _Right) > std::numeric_limits<Type>::epsilon() }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; }) ||
		requires(Type & _Left, Type & _Right) { { _Left != _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left != _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		NotEqual(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) > std::numeric_limits<Type>::epsilon();
		else
			return A != B;
	}

	template <typename Type, typename = std::enable_if_t <
		(requires(Type& _Left, Type& _Right) { { _Left > _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; }) ||
		(requires(Type & _Left, Type & _Right) { { _Left > _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; })
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Greater(const Type& A, const Type& B)
	{
		return A > B;
	}

	template <typename Type, typename = std::enable_if_t <
		requires(Type& _Left, Type& _Right) { { _Left >= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left >= _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		GreaterEqual(const Type& A, const Type& B)
	{
		return A >= B;
	}

	template <typename Type, typename = std::enable_if_t <
		requires(Type& _Left, Type& _Right) { { _Left < _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left < _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Less(const Type& A, const Type& B)
	{
		return A < B;
	}

	template <typename Type, typename = std::enable_if_t <
		requires(Type& _Left, Type& _Right) { { _Left <= _Right }->_D_Dragonian_Lib_Namespace TypeTraits::IsType<bool>; } ||
		requires(Type & _Left, Type & _Right) { { _Left <= _Right }->_D_Dragonian_Lib_Namespace Operators::SimdTypeTraits::IsSimdVector<>; }
	>> _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LessEqual(const Type& A, const Type& B)
	{
		return A <= B;
	}
}

_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Equal, 8, 2)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(NotEqual, 8, 2)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Greater, 8, 2)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(GreaterEqual, 8, 2)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(Less, 8, 2)
_D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(LessEqual, 8, 2)

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Bool_Function_Def