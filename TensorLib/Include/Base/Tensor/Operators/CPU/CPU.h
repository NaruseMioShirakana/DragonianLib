#pragma once
#include "Simd.h"

DragonianLibOperatorSpaceBegin

template<typename _Type>
class OperatorsBase<_Type, Device::CPU>
{
	OperatorsBase() = delete;
public:
	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		const _Type* _Src,
		const TensorShapeInfo& _SrcInfo,
		bool Continuous
	);

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		_Type _Value,
		bool Continuous
	);

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	);

	static void ImplAssignRandn(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	);

	static void ImplAssignRand(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		bool Continuous
	);
};

DragonianLibOperatorSpaceEnd