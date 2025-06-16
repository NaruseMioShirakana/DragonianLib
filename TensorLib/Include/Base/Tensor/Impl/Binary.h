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

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _MaskType, typename>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::MaskedFill(
	const Tensor<_MaskType, _NRank, _MyDevice>& _Mask,
	const ValueType& _Value
) requires (std::is_copy_assignable_v<ValueType> && TypeTraits::CouldBeConvertedFromValue<bool, _MaskType>)
{
	ThrowOnNotEnabled();
	_Mask.ThrowOnNotEnabled();
	auto MaskBroadCasted = BroadCast(_Mask);
	_D_Dragonian_Lib_Auto_Grad(MaskedFill, *this, MaskBroadCasted, _Value);
	WaitingAsResult();
	MaskBroadCasted.WaitingAsArgument();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMaskedAssignScalar(
		_MyData,
		GetDefaultOperatorParameter(),
		MaskBroadCasted.Data(),
		MaskBroadCasted.GetDefaultOperatorParameter(),
		_Value,
		!IsBroadCasted() && !MaskBroadCasted.IsBroadCasted() && IsContiguous() && MaskBroadCasted.IsContiguous()
	);
	return *this;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _MaskType, size_t _TRank, typename>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::MaskedFill(
	const Tensor<_MaskType, _NRank, _MyDevice>& _Mask,
	const Tensor<ValueType, _TRank, _MyDevice>& _Value
) requires (std::is_copy_assignable_v<ValueType> && TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank))
{
	ThrowOnNotEnabled();
	_Mask.ThrowOnNotEnabled();
	_Value.ThrowOnNotEnabled();
	auto MaskBroadCasted = BroadCast(_Mask);
	auto BroadCasted = BroadCast(_Value);
	_D_Dragonian_Lib_Auto_Grad(MaskedFill, *this, MaskBroadCasted, BroadCasted);
	WaitingAsResult();
	MaskBroadCasted.WaitingAsArgument();
	BroadCasted.WaitingAsArgument();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMaskedAssign(
		_MyData,
		GetDefaultOperatorParameter(),
		BroadCasted.Data(),
		BroadCasted.GetDefaultOperatorParameter(),
		MaskBroadCasted.Data(),
		MaskBroadCasted.GetDefaultOperatorParameter(),
		!IsBroadCasted() && !MaskBroadCasted.IsBroadCasted() &&
		!BroadCasted.IsBroadCasted() && IsContiguous() &&
		MaskBroadCasted.IsContiguous() && BroadCasted.IsContiguous()
	);
	return *this;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _MaskType, typename _FunTy, size_t _TRank, typename _ArgType, typename _VectorizedFnTy>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::MaskedInplace(
	const Tensor<_MaskType, _TRank, _MyDevice>& _Mask,
	const _ArgType& _Value,
	_FunTy _ScalarFun,
	_VectorizedFnTy _VectorizedFn
) requires (TypeTraits::IsInvocableValue<std::decay_t<_FunTy>, ValueType&, const _ArgType&> && TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank))
{
	if (RequiresGrad() || _Mask.RequiresGrad())
		_D_Dragonian_Lib_Throw_Exception("MaskedInplace does not support gradients!");

	ThrowOnNotEnabled();
	_Mask.ThrowOnNotEnabled();
	auto MaskBroadCasted = BroadCast(_Mask);
	WaitingAsResult();
	MaskBroadCasted.WaitingAsArgument();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMaskedInplaceScalar(
		_MyData,
		GetDefaultOperatorParameter(),
		MaskBroadCasted.Data(),
		MaskBroadCasted.GetDefaultOperatorParameter(),
		_Value,
		_ScalarFun,
		_VectorizedFn,
		!IsBroadCasted() && !MaskBroadCasted.IsBroadCasted() && IsContiguous() && MaskBroadCasted.IsContiguous()
	);
	return *this;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _ArgType, typename _MaskType, typename _FunTy, size_t _TRank1, size_t _TRank2, typename _VectorizedFnTy>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::MaskedInplace(
	const Tensor<_MaskType, _TRank1, _MyDevice>& _Mask,
	const Tensor<_ArgType, _TRank2, _MyDevice>& _Value,
	_FunTy _ScalarFun,
	_VectorizedFnTy _VectorizedFn
) requires (TypeTraits::IsInvocableValue<TypeTraits::RemoveReferenceType<_FunTy>, ValueType&, const _ArgType&> && TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank1) && (_NRank >= _TRank2))
{
	if (RequiresGrad() || _Mask.RequiresGrad() || _Value.RequiresGrad())
		_D_Dragonian_Lib_Throw_Exception("MaskedInplace does not support gradients!");

	ThrowOnNotEnabled();
	_Mask.ThrowOnNotEnabled();
	_Value.ThrowOnNotEnabled();
	auto MaskBroadCasted = BroadCast(_Mask);
	auto BroadCasted = BroadCast(_Value);
	WaitingAsResult();
	MaskBroadCasted.WaitingAsArgument();
	BroadCasted.WaitingAsArgument();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMaskedInplace(
		_MyData,
		GetDefaultOperatorParameter(),
		BroadCasted.Data(),
		BroadCasted.GetDefaultOperatorParameter(),
		MaskBroadCasted.Data(),
		MaskBroadCasted.GetDefaultOperatorParameter(),
		_ScalarFun,
		_VectorizedFn,
		!IsBroadCasted() && !MaskBroadCasted.IsBroadCasted() &&
		!BroadCasted.IsBroadCasted() && IsContiguous() &&
		MaskBroadCasted.IsContiguous() && BroadCasted.IsContiguous()
	);
	return *this;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Assign(
	const ValueType& _Value
) requires (std::is_copy_assignable_v<ValueType>)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	_D_Dragonian_Lib_Auto_Grad(Assign, *this, _Value);
	WaitingAsResult();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignScalar(
		_MyData,
		GetDefaultOperatorParameter(),
		_Value,
		!IsBroadCasted() && IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Assign(
	const ValueType* _Buffer,
	SizeType _Count
) requires (std::is_copy_assignable_v<ValueType>)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	if (_Count != ElementCount())
		_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
	WaitingAsResult();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignBuffer(
		_MyData,
		GetDefaultOperatorParameter(),
		_Buffer,
		_Count,
		!IsBroadCasted() && IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::MoveAssign(
	const ValueType* _Buffer,
	SizeType _Count
) requires (std::is_move_assignable_v<ValueType>)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	if (_Count != ElementCount())
		_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
	if (RequiresGrad())
		_D_Dragonian_Lib_Throw_Exception("Could Not Has Grad!");
	WaitingAsResult();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMoveBuffer(
		_MyData,
		GetDefaultOperatorParameter(),
		_Buffer,
		_Count,
		!IsBroadCasted() && IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename, size_t _TRank>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Assign(
	const Tensor<ValueType, _TRank, _MyDevice>& _Val
) requires (std::is_copy_assignable_v<ValueType>)
{
	ThrowOnNotEnabled();
	_Val.ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	_Val.WaitingAsArgument();
	if (_Val.IsScalar())
		return Assign(_Val.Item());
	
	Tensor BroadCasted = BroadCast(_Val);
	_D_Dragonian_Lib_Auto_Grad(Assign, *this, BroadCasted);
	WaitingAsResult();

	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignTensor(
		_MyData,
		GetDefaultOperatorParameter(),
		BroadCasted.Data(),
		BroadCasted.GetDefaultOperatorParameter(),
		!IsBroadCasted() && !BroadCasted.IsBroadCasted() && IsContiguous() && BroadCasted.IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::AssignRand(
	const ValueType& Min,
	const ValueType& Max
) requires (TypeTraits::IsArithmeticValue<ValueType>)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	WaitingAsResult();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRand(
		_MyData,
		GetDefaultOperatorParameter(),
		Min, Max,
		!IsBroadCasted() && IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::AssignRandn(
	double _Mean,
	double _Sigma
) requires (TypeTraits::IsArithmeticValue<ValueType>)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
	WaitingAsResult();
	Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRandn(
		_MyData,
		GetDefaultOperatorParameter(),
		_Mean,
		_Sigma,
		!IsBroadCasted() && IsContiguous()
	);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <SizeType _Axis, typename, typename _IndexType>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Gather(
	const Tensor<_IndexType, _NRank, _MyDevice>& _Indices
) const requires ((_Axis < _NRank) && (_Axis > -_NRank - 1) && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
{
	ThrowOnNotEnabled();
	_Indices.ThrowOnNotEnabled();
	for (SizeType i = 0; std::cmp_less(i, _NRank); ++i)
		if (i != _Axis && _MyShape[i] != _Indices.Shape()[i])
			_D_Dragonian_Lib_Throw_Exception("Shape Mismatch!");

	_Indices.WaitingAsArgument();
	WaitingAsArgument();
	constexpr auto _Dim = TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank));
	auto Ret = New(_Indices.Shape(), _MyAllocator);
	Ret.WaitingAsResult();
	Operators::OperatorsBase<ValueType, _MyDevice>::template ImplGather<_IndexType, _NRank, _Dim>
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Indices.Data(),
			_Indices.GetDefaultOperatorParameter()
		);
	_D_Dragonian_Lib_Auto_Grad(Gather, *this, _Indices, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <SizeType _Axis, typename, typename _IndexType>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Gather(
	const Tensor<_IndexType, _NRank, _MyDevice>& _Indices,
	Tensor<_IndexType, _NRank, _MyDevice>& _Buffer
) requires ((_Axis < _NRank) && (_Axis > -_NRank - 1) && std::is_copy_assignable_v<ValueType>)
{
	ThrowOnNotEnabled();
	_Indices.ThrowOnNotEnabled();
	_Buffer.ThrowOnNotEnabled();
	for (SizeType i = 0; std::cmp_less(i, _NRank); ++i)
		if ((i != _Axis && _MyShape[i] != _Indices.Shape()[i]) || (_Buffer.Shape()[i] != _Indices.Shape()[i]))
			_D_Dragonian_Lib_Throw_Exception("Shape Mismatch!");

	_Indices.WaitingAsArgument();
	WaitingAsArgument();
	constexpr auto _Dim = TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank));
	_Buffer.WaitingAsResult();
	Operators::OperatorsBase<ValueType, _MyDevice>::template ImplGather<_IndexType, _NRank, _Dim>
		(
			_Buffer.Data(),
			_Buffer.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Indices.Data(),
			_Indices.GetDefaultOperatorParameter()
		);
	_D_Dragonian_Lib_Auto_Grad(Gather, *this, _Indices, _Buffer);
	return _Buffer;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _Type>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Cast() const
	requires (TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>)
{
	ThrowOnNotEnabled();
	WaitingAsArgument();
	if constexpr (TypeTraits::IsSameTypeValue<_Type, ValueType>)
		return View();
	else
	{
		Tensor<_Type, _NRank, _MyDevice> Ret = Tensor<_Type, _NRank, _MyDevice>::New(_MyShape, _MyAllocator);
		Ret.WaitingAsResult();
		Operators::OperatorsBase<_Type, _MyDevice>::template ImplCast<ValueType>
			(
				Ret.Data(),
				Ret.GetDefaultOperatorParameter(),
				Data(),
				GetDefaultOperatorParameter(),
				IsContiguous() && !IsBroadCasted()
			);
		_D_Dragonian_Lib_Auto_Grad(Cast, *this, Ret);
		return Ret;
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _Type>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Cast(
	Tensor<_Type, _NRank, _MyDevice>& _Buffer
) const requires (TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>)
{
	ThrowOnNotEnabled();
	_Buffer.ThrowOnNotEnabled();
	WaitingAsArgument();
	_Buffer.WaitingAsResult();
	auto BroadCasted = _Buffer.Broadcast(*this);
	Operators::OperatorsBase<_Type, _MyDevice>::template ImplCast<ValueType>
		(
			_Buffer.Data(),
			_Buffer.GetDefaultOperatorParameter(),
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			BroadCasted.IsContiguous() && !BroadCasted.IsBroadCasted() && _Buffer.IsContiguous()
		);
	_D_Dragonian_Lib_Auto_Grad(Cast, BroadCasted, _Buffer);
	return _Buffer;
}

_D_Dragonian_Lib_Space_End

#undef _D_Dragonian_Lib_Operator_Compare_Function_Define
#undef _D_Dragonian_Lib_Operator_Binary_Function_Define