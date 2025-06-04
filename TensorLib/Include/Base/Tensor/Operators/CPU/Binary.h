/**
 * @file Binary.h
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
 * @brief Binary operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"
#include "TensorLib/Include/Base/Tensor/Operators/User/Binary.h"

#define _D_Dragonian_Lib_Operator_Binary_Function_Def(_Function, Unfold, AvxThroughput) \
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
void OperatorsBase<_Type, Device::CPU>::Impl##_Function##ReverseScalar( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
) \
{ \
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>) \
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<Vectorized<_Type>>), BinaryOperators::_Function##<Vectorized<_Type>>, TypeDef::ReversedConstantOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
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
		ImplMultiThreadBasic<decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, decltype(BinaryOperators::_Function##<_Type>), BinaryOperators::_Function##<_Type>, TypeDef::ReversedConstantOperatorType, false, Unfold, AvxThroughput, _NRank, _Type, _Type, _Type>(\
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

_D_Dragonian_Lib_Operator_Binary_Function_Def(Add, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Sub, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mul, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Div, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Mod, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(And, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Or, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Xor, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BitwiseOr, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(BitwiseAnd, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(LShift, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(RShift, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Pow, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Max, 8, 2);
_D_Dragonian_Lib_Operator_Binary_Function_Def(Min, 8, 2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Function_Def