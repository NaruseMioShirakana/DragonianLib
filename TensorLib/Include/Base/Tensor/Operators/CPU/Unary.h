/**
 * @file Unary.h
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
 * @brief Unary operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"
#include "TensorLib/Include/Base/Tensor/Operators/User/Unary.h"

#define _D_Dragonian_Lib_Operator_Unary_Function_Def(_Function, Unfold, AvxThroughput) \
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
	if constexpr (UnaryOperators::##_Function##Unary::HasVectorOperatorValue<_Type> && TypeTraits::IsSameTypeValue<_Type, _ResultType>) \
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
_D_Dragonian_Lib_Operator_Unary_Function_Def(Exp2, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log2, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Log10, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Ceil, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Floor, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Round, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Trunc, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Frac, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Negative, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(BitwiseNot, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Not, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(Polar, 8, 2);
_D_Dragonian_Lib_Operator_Unary_Function_Def(ATan2, 8, 2);

_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Unary_Function_Def