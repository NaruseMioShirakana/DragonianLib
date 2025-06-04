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

//********************************************************************************************//
#define _D_Dragonian_Lib_Operator_Binary_Function_Declare(_Function) \
template <typename = ValueType> \
Tensor<_TensorType, _NRank, _MyDevice> _Function( \
	const ValueType& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)); \
 \
template <typename = ValueType> \
Tensor<_TensorType, _NRank, _MyDevice> __##_Function( \
	const ValueType& _Left \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)); \
 \
template <typename = ValueType> \
Tensor<_TensorType, _NRank, _MyDevice>& _Function##Inplace( \
	const ValueType& _Right \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)); \
 \
template <typename = ValueType> \
Tensor<_TensorType, _NRank, _MyDevice>& __##_Function##Inplace( \
	const ValueType& _Left \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice>& _Function( \
	const ValueType& _Right, \
	Tensor<_TensorType, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank >= _NRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice>& __##_Function( \
	const ValueType& _Left, \
	Tensor<_TensorType, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank >= _NRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<_TensorType, _NRank, _MyDevice> _Function( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_NRank >= _MyOpRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<_TensorType, _MyOpRank, _MyDevice> _Function( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) const requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_NRank < _MyOpRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<_TensorType, _NRank, _MyDevice>& _Function##Inplace( \
	const Tensor<_TensorType, _MyOpRank, _MyDevice>& _Right \
) requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)&& (_NRank >= _MyOpRank)); \
 \
template <typename = ValueType, size_t _MyOpRank1, size_t _MyOpRank2> \
Tensor<_TensorType, _MyOpRank2, _MyDevice>& _Function( \
	const Tensor<_TensorType, _MyOpRank1, _MyDevice>& _Right, \
	Tensor<_TensorType, _MyOpRank2, _MyDevice>& _Buffer \
) const requires (Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1))

//********************************************************************************************//
#define _D_Dragonian_Lib_Operator_Compare_Function_Declare(_Function) \
template <typename = ValueType> \
Tensor<bool, _NRank, _MyDevice> _Function( \
	const ValueType& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType>); \
 \
template <typename = ValueType> \
Tensor<bool, _NRank, _MyDevice> __##_Function( \
	const ValueType& _Left \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType>); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice>& _Function( \
	const ValueType& _Right, \
	Tensor<bool, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank >= _NRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice>& __##_Function( \
	const ValueType& _Left, \
	Tensor<bool, _MyOpRank, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank >= _NRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<bool, _NRank, _MyDevice> _Function( \
	const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_NRank >= _MyOpRank)); \
 \
template <typename = ValueType, size_t _MyOpRank> \
Tensor<bool, _MyOpRank, _MyDevice> _Function( \
	const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_NRank < _MyOpRank)); \
 \
template <typename = ValueType, size_t _MyOpRank1, size_t _MyOpRank2> \
Tensor<bool, _MyOpRank2, _MyDevice>& _Function( \
	const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, \
	Tensor<bool, _MyOpRank2, _MyDevice>& _Buffer \
) const requires (Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<ValueType> && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1))

//********************************************************************************************//
#define _D_Dragonian_Lib_Operator_Unary_Function_Declare(_Function) \
template <typename = ValueType> \
Tensor& _Function##Inplace() requires (Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType>&&(std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>));\
 \
template <typename = ValueType> \
Tensor _Function() const requires (std::is_default_constructible_v<ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>));\
 \
template <typename = ValueType, size_t _BufferRank> \
Tensor<_TensorType, _BufferRank, _MyDevice>& _Function(Tensor<ValueType, _BufferRank, _MyDevice>& _Buffer) const requires (Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<ValueType>&& (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>) && (_BufferRank >= _NRank))

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator(_Function, _Operator, _Condition) \
template <typename = ValueType> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const requires ((_Condition)&& std::is_default_constructible_v<ValueType>) \
{ \
	return _Function(_Right); \
} \
 \
template <typename = ValueType> \
	decltype(auto) operator##_Operator##=(const ValueType& _Right) requires (_Condition) \
{ \
	return _Function##Inplace(_Right); \
} \
 \
template <typename = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const requires ((_Condition)&& std::is_default_constructible_v<ValueType>) \
{ \
	return _Function(_Right); \
} \
 \
template <typename = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) requires ((_Condition)&& (_NRank >= _TRank)) \
{ \
	return _Function##Inplace(_Right); \
} \
template <typename = ValueType> \
friend decltype(auto) operator##_Operator##(const ValueType& _Left, const Tensor& _Right) requires ((_Condition)&& std::is_default_constructible_v<ValueType>) \
{ \
	return _Right.__##_Function(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(_Function, _Operator, _Condition) \
template <typename = ValueType> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const requires (_Condition) \
{ \
	return _Function(_Right); \
} \
 \
template <typename = ValueType, size_t _TRank> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const requires (_Condition) \
{ \
	return _Function(_Right); \
} \
template <typename = ValueType> \
friend decltype(auto) operator##_Operator##(const ValueType& _Left, const Tensor& _Right) requires (_Condition) \
{ \
	return _Right.__##_Function(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

//********************************************************************************************
#define _D_Dragonian_Lib_Operator_Reduce_Function_Body(_FunctionName, _Function) \
{ \
	ThrowOnNotEnabled(); \
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
	ThrowOnNotEnabled(); \
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
	ThrowOnNotEnabled(); \
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