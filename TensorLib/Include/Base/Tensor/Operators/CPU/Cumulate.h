#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumSumUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{

}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumProdUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumMaxUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{

}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumMinUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{

}

_D_Dragonian_Lib_Operator_Space_End