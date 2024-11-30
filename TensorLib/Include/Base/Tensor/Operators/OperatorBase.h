#pragma once
#include <future>
#include <random>
#include "../TensorBase.h"
#include "OperatorMarco.h"
#include "CPU/Unary.h"

#define _D_Dragonian_Lib_Operator_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Operators {
#define _D_Dragonian_Lib_Operator_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Operator_Space_Begin

using namespace TypeTraits;

template<SizeType Rank>
struct OperatorParameterND
{
	SizeType Shape[Rank], ViewStep[Rank], ViewLeft[Rank], ViewStride[Rank];
	bool IsContinuousND[Rank];
	std::shared_ptr<std::deque<std::future<void>>> ThreadPool;
	SizeType ViewRank = 1;
	OperatorParameterND()
	{
		for (SizeType i = 0; i < Rank; i++)
		{
			Shape[i] = 0;
			ViewStep[i] = 0;
			ViewLeft[i] = 0;
			ViewStride[i] = 0;
			IsContinuousND[i] = false;
		}
	}
	SizeType GetSize() const
	{
		SizeType Size = 1;
		for (SizeType i = 0; i < Rank; ++i) Size *= Shape[i];
		return Size;
	}
	constexpr static SizeType StructRank = Rank;
};

template<size_t _NRank>
struct OperatorParameter
{
	using _MyMultiThreadSyncT = std::deque<std::pair<std::future<void>, std::vector<std::shared_ptr<void>>>>;
	using _MyMultiThreadSyncP = std::shared_ptr<_MyMultiThreadSyncT>;

	IDLArray<SizeType, _NRank> Shape; ///< Shape: The [view end/shape] of the tensor.
	IDLArray<SizeType, _NRank> Begin; ///< Begin: The [view begin] of the tensor.
	IDLArray<SizeType, _NRank> ViewStep; ///< ViewStep: The step of the view.
	IDLArray<SizeType, _NRank> ViewLeft; ///< ViewLeft: The left of the view.
	IDLArray<SizeType, _NRank> ViewStride; ///< ViewStride: The stride of the view.
	IDLArray<bool, _NRank> IsContinuous; ///< IsContinuous: The continuous flag of the view.
	_MyMultiThreadSyncP ThreadPool = nullptr; ///< ThreadPool: The futures of the tensor.
	void* UserParameter = nullptr; ///< UserParameter: The user parameter.
	std::shared_ptr<void> Data = nullptr; ///< Data: The data of the tensor (prevent from the data being released while the tensor is used by an operator).
	SizeType GetSize(size_t RangeBegin = 0, size_t RangeEnd = _NRank) const
	{
		SizeType Size = 1;
		RangeEnd = std::min(RangeEnd, _NRank);
		for (size_t i = RangeBegin; i < RangeEnd; ++i) Size *= Shape[i];
		return Size;
	}
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline static SizeType GetRank() { return (SizeType)_NRank; }
};

template<typename _Type, Device _Device>
class OperatorsBase
{
	OperatorsBase() = delete;
public:

	template<typename _TypeSrc, size_t _NRank>
	static void ImplCast(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _TypeSrc* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignBuffer(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignRandn(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignRand(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Min,
		const _Type& _Max,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplArange(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Start,
		const _Type& _Step,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	_D_Dragonian_Lib_Operator_Binary_Define(Add) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Sub) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Mul) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Div) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Mod) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(And) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Or) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Xor) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(LShift) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(RShift) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Pow) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(BinaryOr) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(BinaryAnd) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Add) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Sub) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Mul) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Div) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Mod) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(And) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Or) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Xor) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(LShift) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(RShift) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Pow) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryOr) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryAnd) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Comparison_Define(Equal) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define(NotEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define(Greater) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define(GreaterEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define(Less) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define(LessEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Equal) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(NotEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Greater) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Less) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessEqual) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Unary_Define(Sqrt) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(RSqrt) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Reciprocal) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Abs) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Sin) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Cos) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Tan) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ASin) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ACos) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ATan) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Sinh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Cosh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Tanh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ASinh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ACosh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ATanh) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Exp) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log2) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log10) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Ceil) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Floor) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Round) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Trunc) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Frac) { _D_Dragonian_Lib_Not_Implemented_Error; }
};

_D_Dragonian_Lib_Operator_Space_End
