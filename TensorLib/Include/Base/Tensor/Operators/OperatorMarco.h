﻿#pragma once

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
template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&&(std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>> \
decltype(auto) _Function##Inplace() \
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
template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>> \
decltype(auto) _Function() const \
{ \
	auto Ret = Tensor::New(_MyShape); \
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
template <typename _CurValueType = ValueType, size_t _BufferRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::##_Function##Unary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_BufferRank >= _NRank)>> \
decltype(auto) _Function(Tensor<ValueType, _BufferRank, _MyDevice>& _Buffer) const \
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

#define _D_Dragonian_Lib_Operator_Binary_Function_Define(_Function) \
template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>> \
decltype(auto) _Function(const ValueType& _Right) const \
{ \
	auto Ret = New(_MyShape); \
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
template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>> \
decltype(auto) _Function##Inplace(const ValueType& _Right) \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_MyOpRank >= _NRank)>> \
decltype(auto) _Function(const ValueType& _Right, Tensor<ValueType, _MyOpRank, _MyDevice>& _Buffer) const \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) const \
{ \
	auto BroadCasted = BroadCast(*this, _Right); \
	auto Ret = Tensor<ValueType, MaxOf(_NRank, _MyOpRank), _MyDevice>::New(BroadCasted.first.Shape()); \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)&& (_NRank >= _MyOpRank)>> \
decltype(auto) _Function##Inplace(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank1, size_t _MyOpRank2, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::BinaryOperators::##_Function##Binary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>) && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)>> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, Tensor<ValueType, _MyOpRank2, _MyDevice>& _Buffer) const \
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

#define _D_Dragonian_Lib_Operator_Compare_Function_Define(_Function) \
template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<bool>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType>>> \
decltype(auto) _Function(const ValueType& _Right) const \
{ \
	auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape); \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<bool>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType>>> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank, _MyDevice>& _Right) const \
{ \
	auto BroadCasted = BroadCast(*this, _Right); \
	auto Ret = Tensor<bool, MaxOf(_NRank, _MyOpRank), _MyDevice>::New(BroadCasted.first.Shape()); \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType> && (_MyOpRank >= _NRank)>> \
decltype(auto) _Function(const ValueType& _Right, Tensor<bool, _MyOpRank, _MyDevice>& _Buffer) const \
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
template <typename _CurValueType = ValueType, size_t _MyOpRank1, size_t _MyOpRank2, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::ComparisonOperators::##_Function##Binary::HasOperatorValue<_CurValueType> && (_MyOpRank2 >= _NRank) && (_MyOpRank2 >= _MyOpRank1)>> \
decltype(auto) _Function(const Tensor<ValueType, _MyOpRank1, _MyDevice>& _Right, Tensor<bool, _MyOpRank2, _MyDevice>& _Buffer) const \
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

#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator(_Function, _Operator, _Condition) \
template <typename _CurValueType = ValueType, typename = std::enable_if_t <(_Condition)&& std::is_default_constructible_v<_CurValueType>>> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType, typename = std::enable_if_t <(_Condition)>> \
	decltype(auto) operator##_Operator##=(const ValueType& _Right) \
{ \
	return _Function##Inplace(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t <(_Condition)&& std::is_default_constructible_v<_CurValueType>>> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t <(_Condition)&& (_NRank >= _TRank)>> \
	decltype(auto) operator##_Operator##=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) \
{ \
	return _Function##Inplace(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

#define _D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(_Function, _Operator, _Condition) \
template <typename _CurValueType = ValueType, typename = std::enable_if_t <(_Condition)>> \
decltype(auto) operator##_Operator##(const ValueType& _Right) const \
{ \
	return _Function(_Right); \
} \
 \
template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t <(_Condition)>> \
	decltype(auto) operator##_Operator##(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const \
{ \
	return _Function(_Right); \
} \
struct _D_Dragonian_Lib_Operator_##_Function##_Defined_Tag

#define _D_Dragonian_Lib_Operator_Reduce_Function_Body(_FunctionName, _Function, ...) \
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
		auto Ret = Tensor<_TensorType, _NRank - 1, _MyDevice>::New(OutShape); \
		Ret.WaitingAsResult(); \
		auto RetView = Ret.UnSqueeze(-1); \
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplReduce##_Function##Unary<__VA_ARGS__> \
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