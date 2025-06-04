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
 * @brief Binary ops for DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Impl/Grad/Binary.h"

#pragma region b_impl_mc
#define _D_Dragonian_Lib_Operator_Binary_Function_Define(_Function) \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const ValueType& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	auto Ret = New(_MyShape, _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, _Right, Ret); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Right, \
		!IsBroadCasted() && IsContiguous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::__##_Function##( \
	const ValueType& _Left \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	auto Ret = New(_MyShape, _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, _Left, *this, Ret); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Left, \
		!IsBroadCasted() && IsContiguous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##Inplace( \
	const ValueType& _Right \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, _Right, *this); \
	WaitingAsResult(); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		_Right, \
		!IsBroadCasted() && IsContiguous() \
	); \
	return *this; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<_TensorType, _NRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::__##_Function##Inplace( \
	const ValueType& _Left \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)) \
{ \
	ThrowOnNotEnabled(); \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, _Left, *this, *this); \
	WaitingAsResult(); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		_Left, \
		!IsBroadCasted() && IsContiguous() \
	); \
	return *this; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const ValueType& _Right, \
	Tensor<_TensorType, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank >= _NRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = _Buffer.BroadCast(*this); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCasted, _Right, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Right, \
		_Buffer.IsContiguous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContiguous() \
	); \
	return _Buffer; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::__##_Function##( \
	const ValueType& _Left, \
	Tensor<_TensorType, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank >= _NRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = _Buffer.BroadCast(*this); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, _Left, BroadCasted, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Left, \
		_Buffer.IsContiguous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContiguous() \
	); \
	return _Buffer; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<_TensorType, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_NRank >= _MyOpRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	auto [BroadCastedLeft, BroadCastedRight] = BroadCast(*this, _Right); \
	auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(BroadCastedLeft.Shape(), _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, Ret); \
	Ret.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContiguous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContiguous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_NRank < _MyOpRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	auto [BroadCastedLeft, BroadCastedRight] = BroadCast(*this, _Right); \
	auto Ret = Tensor<_TensorType, _MyOpRank, _MyDevice>::New(BroadCastedRight.Shape(), _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, Ret); \
	Ret.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContiguous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContiguous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<_TensorType, _NRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##Inplace( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_NRank >= _MyOpRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = BroadCast(_Right); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, BroadCasted, *this); \
	WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		IsContiguous() && BroadCasted.IsContiguous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted() \
	); \
	return *this; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank1, size_t _MyOpRank2> \
Tensor<_TensorType, _MyOpRank2, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<_TensorType, _MyOpRank1, _MyDevice>& _Right, \
	Tensor<_TensorType, _MyOpRank2, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCastedLeft = _Buffer.Broadcast(*this); \
	auto BroadCastedRight = _Buffer.Broadcast(_Right); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContiguous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContiguous() && \
		_Buffer.IsContiguous() \
	); \
	return _Buffer; \
}

#define _D_Dragonian_Lib_Operator_Compare_Function_Define(_Function) \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<bool, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const ValueType& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType>) \
{ \
	ThrowOnNotEnabled(); \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape, _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, *this, _Right, Ret); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		(bool*)Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Right, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename> \
Tensor<bool, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::__##_Function##( \
	const ValueType& _Left \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType>) \
{ \
	ThrowOnNotEnabled(); \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape, _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, _Left, *this, Ret); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		(bool*)Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Left, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const ValueType& _Right, \
	Tensor<bool, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank >= _NRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCasted, _Right, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		(bool*)_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Right, \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
	return _Buffer; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::__##_Function##( \
	const ValueType& _Left, \
	Tensor<bool, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank >= _NRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, _Left, BroadCasted, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCasted.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		(bool*)_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Left, \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
	return _Buffer; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<bool, _NRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_NRank >= _MyOpRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	auto [BroadCastedLeft, BroadCastedRight] = BroadCast(*this, _Right); \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(BroadCastedLeft.Shape(), _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, Ret); \
	Ret.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		(bool*)Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContinuous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice> Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_NRank < _MyOpRank)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	auto [BroadCastedLeft, BroadCastedRight] = BroadCast(*this, _Right); \
	auto Ret = Tensor<bool, _MyOpRank, _MyDevice>::New(BroadCastedLeft.Shape(), _MyAllocator); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, Ret); \
	Ret.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		(bool*)Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContinuous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _TensorType, size_t _NRank, Device _MyDevice> \
template <typename, size_t _MyOpRank1, size_t _MyOpRank2> \
Tensor<bool, _MyOpRank2, _MyDevice>& Tensor<_TensorType, _NRank, _MyDevice>::##_Function##( \
	const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, \
	Tensor<bool, _MyOpRank2, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)) \
{ \
	ThrowOnNotEnabled(); \
	_Right.ThrowOnNotEnabled(); \
	_Buffer.ThrowOnNotEnabled(); \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	auto BroadCastedLeft = _Buffer.Broadcast(*this); \
	auto BroadCastedRight = _Buffer.Broadcast(_Right); \
	_D_Dragonian_Lib_Auto_Grad(##_Function##, BroadCastedLeft, BroadCastedRight, _Buffer); \
	_Buffer.WaitingAsResult(); \
	BroadCastedLeft.WaitingAsArgument(); \
	BroadCastedRight.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCastedLeft.Data(), \
		BroadCastedLeft.GetDefaultOperatorParameter(), \
		BroadCastedRight.Data(), \
		BroadCastedRight.GetDefaultOperatorParameter(), \
		!BroadCastedLeft.IsBroadCasted() && BroadCastedLeft.IsContinuous() && \
		!BroadCastedRight.IsBroadCasted() && BroadCastedRight.IsContinuous() && \
		_Buffer.IsContinuous() \
	); \
	return _Buffer; \
}
#pragma endregion

_D_Dragonian_Lib_Space_Begin

_D_Dragonian_Lib_Operator_Binary_Function_Define(Add);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Sub);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Mul);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Div);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Mod);
_D_Dragonian_Lib_Operator_Binary_Function_Define(And);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Or);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Xor);
_D_Dragonian_Lib_Operator_Binary_Function_Define(LShift);
_D_Dragonian_Lib_Operator_Binary_Function_Define(RShift);
_D_Dragonian_Lib_Operator_Binary_Function_Define(BitwiseOr);
_D_Dragonian_Lib_Operator_Binary_Function_Define(BitwiseAnd);
_D_Dragonian_Lib_Operator_Binary_Function_Define(Pow);

_D_Dragonian_Lib_Operator_Compare_Function_Define(Equal);
_D_Dragonian_Lib_Operator_Compare_Function_Define(NotEqual);
_D_Dragonian_Lib_Operator_Compare_Function_Define(Greater);
_D_Dragonian_Lib_Operator_Compare_Function_Define(Less);
_D_Dragonian_Lib_Operator_Compare_Function_Define(GreaterEqual);
_D_Dragonian_Lib_Operator_Compare_Function_Define(LessEqual);

_D_Dragonian_Lib_Space_End

#undef _D_Dragonian_Lib_Operator_Compare_Function_Define
#undef _D_Dragonian_Lib_Operator_Binary_Function_Define