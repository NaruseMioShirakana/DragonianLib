/**
 * @file Comparison.h
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
 * @brief Comparison operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "CPU.h"

#define _D_Dragonian_Lib_Operator_Binary_Bool_Function_Def(_Function, Unfold, AvxThroughput) \
namespace ComparisonOperators \
{ \
	namespace _Function##Binary \
	{ \
		template <class _ValueType> \
		concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) { {_D_Dragonian_Lib_Namespace Operators::ComparisonOperators::_Function(__r, __l)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
		template <class _ValueType> \
		concept HasVectorOperatorValue = requires(Vectorized<_ValueType> & __r, Vectorized<_ValueType> & __l) { {_D_Dragonian_Lib_Namespace Operators::ComparisonOperators::_Function(__r, __l)} -> TypeTraits::NotType<decltype(std::nullopt)>; }; \
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
	if constexpr (ComparisonOperators::##_Function##Binary::HasVectorOperatorValue<_Type>) \
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
	if constexpr (ComparisonOperators::##_Function##Binary::HasVectorOperatorValue<_Type>) \
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

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Equal(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) <= std::numeric_limits<Type>::epsilon();
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left == _Right }; })
			return A == B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Equal(_Right) }; })
			return A.Equal(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.equal(_Right) }; })
			return A.equal(B);
		else
			return std::nullopt;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		NotEqual(const Type& A, const Type& B)
	{
		if constexpr (IsAnyOfValue<Type, float, double>)
			return std::fabs(A - B) > std::numeric_limits<Type>::epsilon();
		else if constexpr (requires(Type & _Left, Type & _Right) { {_Left != _Right}; })
			return A != B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.NotEqual(_Right) }; })
			return A.NotEqual(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.not_equal(_Right) }; })
			return A.not_equal(B);
		else
			return std::nullopt;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Greater(const Type& A, const Type& B)
	{
		if constexpr (requires(Type & _Left, Type & _Right) { {_Left > _Right}; })
			return A > B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Greater(_Right) }; })
			return A.Greater(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.greater(_Right) }; })
			return A.greater(B);
		else
			return std::nullopt;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		GreaterEqual(const Type& A, const Type& B)
	{
		if constexpr (requires(Type & _Left, Type & _Right) { {_Left >= _Right}; })
			return A >= B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.GreaterEqual(_Right) }; })
			return A.GreaterEqual(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.greater_equal(_Right) }; })
			return A.greater_equal(B);
		else
			return std::nullopt;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		Less(const Type& A, const Type& B)
	{
		if constexpr (requires(Type & _Left, Type & _Right) { {_Left < _Right}; })
			return A < B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Less(_Right) }; })
			return A.Less(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.less(_Right) }; })
			return A.less(B);
		else
			return std::nullopt;
	}

	template <typename Type>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		LessEqual(const Type& A, const Type& B)
	{
		if constexpr (requires(Type & _Left, Type & _Right) { {_Left <= _Right}; })
			return A <= B;
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.LessEqual(_Right) }; })
			return A.LessEqual(B);
		else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.less_equal(_Right) }; })
			return A.less_equal(B);
		else
			return std::nullopt;
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