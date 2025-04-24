/**
 * @file OperatorMarco.h
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
 * @brief Macro for operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once

#define _D_Dragonian_Lib_Operator_Binary_Define(_Function) \
template<size_t _NRank> \
static void Impl##_Function##Tensor(\
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src1, \
	const OperatorParameter<_NRank>& _SrcInfo1, \
	const _Type* _Src2, \
	const OperatorParameter<_NRank>& _SrcInfo2, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Binary_Define_Scalar(_Function) \
template<size_t _NRank> \
static void Impl##_Function##Scalar( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Comparison_Define(_Function) \
template<size_t _NRank> \
static void Impl##_Function##Tensor(\
	bool* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src1, \
	const OperatorParameter<_NRank>& _SrcInfo1, \
	const _Type* _Src2, \
	const OperatorParameter<_NRank>& _SrcInfo2, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Comparison_Define_Scalar(_Function) \
template<size_t _NRank> \
static void Impl##_Function##Scalar( \
	bool* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	const _Type& _Value, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Unary_Define(_Function) \
template<typename _ResultType, size_t _NRank> \
static void Impl##_Function##Unary( \
	_ResultType* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Unary_St_Define(_Function) \
template<size_t _NRank> \
static void Impl##_Function##Unary( \
	_Type* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Unary_With_Extra_Define(_Function, ExtraTemplate) \
template<typename _ResultType, size_t _NRank, ExtraTemplate> \
static void Impl##_Function##Unary( \
	_ResultType* _Dest, \
	const OperatorParameter<_NRank>& _DestInfo, \
	const _Type* _Src, \
	const OperatorParameter<_NRank>& _SrcInfo, \
	bool Continuous \
)

#define _D_Dragonian_Lib_Operator_Unary_Function_Define(_Function) \
template <typename _CurValueType = ValueType> \
decltype(auto) _Function##Inplace() requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&&(std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	WaitingAsResult(); \
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
template <typename _CurValueType = ValueType> \
decltype(auto) _Function() const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	auto Ret = Tensor::New(_MyShape, _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
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
template <typename _CurValueType = ValueType, size_t _BufferRank> \
decltype(auto) _Function(Tensor<ValueType, _BufferRank, _MyDevice>& _Buffer) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_BufferRank >= _NRank)) \
{ \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_Buffer.WaitingAsResult(); \
	WaitingAsArgument(); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	Operators::OperatorsBase<ValueType, _MyDevice>::Impl##_Function##Unary \
	( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
	return _Buffer; \
} \
struct _D_Dragonian_Lib_Unary##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Binary_Function_Define(_Function) \
template <typename _CurValueType = ValueType> \
decltype(auto) _Function(const ValueType& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	auto Ret = New(_MyShape, _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Right, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _CurValueType = ValueType> \
decltype(auto) __##_Function(const ValueType& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	auto Ret = New(_MyShape, _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		_MyData, \
		GetDefaultOperatorParameter(), \
		_Right, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _CurValueType = ValueType> \
decltype(auto) _Function##Inplace(const ValueType& _Right) requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	WaitingAsResult(); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		_Right, \
		!IsBroadCasted() && IsContinuous() \
	); \
	return *this; \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank> \
decltype(auto) _Function(const ValueType& _Right, Tensor<ValueType, _MyOpRank, _MyDevice>& _Buffer) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_MyOpRank >= _NRank)) \
{ \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_Buffer.WaitingAsResult(); \
	WaitingAsArgument(); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Right, \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
	return _Buffer; \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)) \
{ \
	auto BroadCasted = BroadCast(*this, _Right); \
	auto Ret = Tensor<ValueType, MaxOf(_NRank, _MyOpRank), _MyDevice>::New(BroadCasted.first.Shape(), _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	_Right.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCasted.first.Data(), \
		BroadCasted.first.GetDefaultOperatorParameter(), \
		BroadCasted.second.Data(), \
		BroadCasted.second.GetDefaultOperatorParameter(), \
		!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() && \
		!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank> \
decltype(auto) _Function##Inplace(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)&& (_NRank >= _MyOpRank)) \
{ \
	if (IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	WaitingAsResult(); \
	_Right.WaitingAsArgument(); \
	auto BroadCasted = BroadCast(_Right); \
	const auto MyParameter = GetDefaultOperatorParameter(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		_MyData, \
		MyParameter, \
		_MyData, \
		MyParameter, \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted() \
	); \
	return *this; \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank1, size_t _MyOpRank2> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, Tensor<ValueType, _MyOpRank2, _MyDevice>& _Buffer) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)) \
{ \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_Buffer.WaitingAsResult(); \
	WaitingAsArgument(); \
	_Right.WaitingAsArgument(); \
	auto BroadCastedLeft = _Buffer.Broadcast(*this); \
	auto BroadCastedRight = _Buffer.Broadcast(_Right); \
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
} \
struct _D_Dragonian_Lib_Binary##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Compare_Function_Define(_Function) \
template <typename _CurValueType = ValueType> \
decltype(auto) _Function(const ValueType& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<bool>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType>) \
{ \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape, _MyAllocator); \
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
template <typename _CurValueType = ValueType> \
decltype(auto) __##_Function(const ValueType& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<bool>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType>) \
{ \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape, _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##ReverseScalar( \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<bool>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType>) \
{ \
	auto BroadCasted = BroadCast(*this, _Right); \
	auto Ret = Tensor<bool, MaxOf(_NRank, _MyOpRank), _MyDevice>::New(BroadCasted.first.Shape(), _MyAllocator); \
	Ret.WaitingAsResult(); \
	WaitingAsArgument(); \
	_Right.WaitingAsArgument(); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Tensor( \
		(bool*)Ret.Data(), \
		Ret.GetDefaultOperatorParameter(), \
		BroadCasted.first.Data(), \
		BroadCasted.first.GetDefaultOperatorParameter(), \
		BroadCasted.second.Data(), \
		BroadCasted.second.GetDefaultOperatorParameter(), \
		!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() && \
		!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous() \
	); \
	return Ret; \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank> \
decltype(auto) _Function(const ValueType& _Right, Tensor<bool, _MyOpRank, _MyDevice>& _Buffer) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType> && (_MyOpRank >= _NRank)) \
{ \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_Buffer.WaitingAsResult(); \
	WaitingAsArgument(); \
	auto BroadCasted = _Buffer.Broadcast(*this); \
	Operators::OperatorsBase<_TensorType, _MyDevice>::Impl##_Function##Scalar( \
		(bool*)_Buffer.Data(), \
		_Buffer.GetDefaultOperatorParameter(), \
		BroadCasted.Data(), \
		BroadCasted.GetDefaultOperatorParameter(), \
		_Right, \
		_Buffer.IsContinuous() && !BroadCasted.IsBroadCasted() && BroadCasted.IsContinuous() \
	); \
} \
 \
template <typename _CurValueType = ValueType, size_t _MyOpRank1, size_t _MyOpRank2> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, Tensor<bool, _MyOpRank2, _MyDevice>& _Buffer) const requires (TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType> && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)) \
{ \
	if (_Buffer.IsBroadCasted()) \
		_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!"); \
	_Buffer.WaitingAsResult(); \
	WaitingAsArgument(); \
	_Right.WaitingAsArgument(); \
	auto BroadCastedLeft = _Buffer.Broadcast(*this); \
	auto BroadCastedRight = _Buffer.Broadcast(_Right); \
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
} \
struct _D_Dragonian_Lib_Compare##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator(_Function, _Operator, _Condition) \
template <typename _CurValueType = ValueType> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const requires ((_Condition)&& std::is_default_constructible_v<_CurValueType>) \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType> \
	decltype(auto) operator##_Operator##=(const ValueType& _Right) requires (_Condition) \
{ \
	return _Function##Inplace(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const requires ((_Condition)&& std::is_default_constructible_v<_CurValueType>) \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) requires ((_Condition)&& (_NRank >= _TRank)) \
{ \
	return _Function##Inplace(_Right); \
} \
template <typename _CurValueType = ValueType> \
friend decltype(auto) operator##_Operator##(const ValueType& _Left, const Tensor& _Right) requires ((_Condition)&& std::is_default_constructible_v<_CurValueType>) \
{ \
	return _Right.__##_Function(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(_Function, _Operator, _Condition) \
template <typename _CurValueType = ValueType> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const requires (_Condition) \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const requires (_Condition) \
{ \
	return _Function(_Right); \
} \
template <typename _CurValueType = ValueType> \
friend decltype(auto) operator##_Operator##(const ValueType& _Left, const Tensor& _Right) requires (_Condition) \
{ \
	return _Right.__##_Function(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Reduce_Function_Body(_FunctionName, _Function) \
{ \
	if constexpr (_NRank == 1) \
		return UnSqueeze(0).template _FunctionName##<false>(-1).Squeeze(0); \
	else \
	{ \
		_Axis = CalcIndex(_Axis, Rank()); \
		auto TensorTmp = AxisFromTo(_Axis, -1); \
		TensorTmp.WaitingAsArgument(); \
		Dimensions<_NRank - 1> OutShape; \
		OutShape.Assign(TensorTmp.Shape().Data()); \
		auto Ret = Tensor<_TensorType, _NRank - 1, _MyDevice>::New(OutShape, _MyAllocator); \
		Ret.WaitingAsResult(); \
		auto RetView = Ret.UnSqueeze(-1); \
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplReduce##_Function##Unary \
		( \
			RetView.Data(), \
			RetView.GetDefaultOperatorParameter(), \
			TensorTmp.Data(), \
			TensorTmp.GetDefaultOperatorParameter(), \
			RetView.IsContinuous() && TensorTmp.IsContinuous() \
		); \
		if constexpr (KeepDim) \
			return RetView; \
		else \
			return Ret; \
	} \
} \
struct _D_Dragonian_Lib_Reduce##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Reduce_Function_Body_T(_FunctionName, _Function, _RetType) \
{ \
	if constexpr (_NRank == 1) \
		return UnSqueeze(0).template _FunctionName##<false>(-1).Squeeze(0); \
	else \
	{ \
		_Axis = CalcIndex(_Axis, Rank()); \
		auto TensorTmp = AxisFromTo(_Axis, -1); \
		TensorTmp.WaitingAsArgument(); \
		Dimensions<_NRank - 1> OutShape; \
		OutShape.Assign(TensorTmp.Shape().Data()); \
		auto Ret = Tensor<_RetType, _NRank - 1, _MyDevice>::New(OutShape, _MyAllocator); \
		Ret.WaitingAsResult(); \
		auto RetView = Ret.UnSqueeze(-1); \
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplReduce##_Function##Unary \
		( \
			RetView.Data(), \
			RetView.GetDefaultOperatorParameter(), \
			TensorTmp.Data(), \
			TensorTmp.GetDefaultOperatorParameter(), \
			RetView.IsContinuous() && TensorTmp.IsContinuous() \
		); \
		if constexpr (KeepDim) \
			return RetView; \
		else \
			return Ret; \
	} \
} \
struct _D_Dragonian_Lib_Reduce##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Cumulate_Function_Body(_Function) \
{ \
	if constexpr (_NRank == 1) \
		return UnSqueeze(0)._Function##(-1).Squeeze(0); \
	else \
	{ \
		_Axis = CalcIndex(_Axis, Rank()); \
		if (Shape()[_Axis] == 1) \
			return View(); \
		auto TensorTmp = AxisFromTo(_Axis, -1); \
		TensorTmp.WaitingAsArgument(); \
		auto Result = Tensor<_TensorType, _NRank, _MyDevice>::New(Shape(), _MyAllocator); \
		auto ResultView = Result.AxisFromTo(_Axis, -1); \
		ResultView.WaitingAsResult(); \
		Operators::OperatorsBase<ValueType, _MyDevice>::Impl##_Function##Unary \
		( \
			ResultView.Data(), \
			ResultView.GetDefaultOperatorParameter(), \
			TensorTmp.Data(), \
			TensorTmp.GetDefaultOperatorParameter(), \
			ResultView.IsContinuous() && TensorTmp.IsContinuous() \
		); \
		return Result; \
	} \
} \
struct _D_Dragonian_Lib_Cumulate##_Function##_Defined_Tag