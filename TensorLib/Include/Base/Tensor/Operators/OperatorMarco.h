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