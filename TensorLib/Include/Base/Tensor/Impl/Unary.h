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
 * @brief Unary ops for DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Impl/Grad/Unary.h"

#pragma region u_impl_mc
#define _D_Dragonian_Lib_Operator_Unary_Function_Define(_Function) \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##Inplace() \
	requires (Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	WaitingAsResult(); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, *this); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<ValueType, _MyDevice>::Impl##_Function##Unary \
	( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return *this; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##() \
const requires (std::is_default_constructible_v<ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	auto Ret = Tensor::New(_MyShape, _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, Ret); \
	Operators::OperatorsBase<ValueType, _MyDevice>::Impl##_Function##Unary \
	( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		!IsBroadCasted() && IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _BufferRank> \
Tensor<_TensorType, _BufferRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	Tensor<ValueType, _BufferRank, _MyDevice>& _Buffer \
) const requires (Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_BufferRank >= _NRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCasted, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	Operators::OperatorsBase<ValueType, _MyDevice>::Impl##_Function##Unary \
	( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
	return _Buffer; \
}
#pragma endregion

_D_Dragonian_Lib_Space_Begin

_D_Dragonian_Lib_Operator_Unary_Function_Define(Sqrt);
_D_Dragonian_Lib_Operator_Unary_Function_Define(RSqrt);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Reciprocal);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Abs);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Sin);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Cos);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Tan);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ASin);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ACos);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ATan);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Sinh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Cosh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Tanh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ASinh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ACosh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ATanh);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Exp);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Exp2);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Log);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Log2);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Log10);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Ceil);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Floor);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Round);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Trunc);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Frac);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Negative);
_D_Dragonian_Lib_Operator_Unary_Function_Define(BitwiseNot);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Not);
_D_Dragonian_Lib_Operator_Unary_Function_Define(ATan2);
_D_Dragonian_Lib_Operator_Unary_Function_Define(Polar);

_D_Dragonian_Lib_Space_End
