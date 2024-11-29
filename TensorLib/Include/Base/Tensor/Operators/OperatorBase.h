#pragma once
#include <future>
#include <random>
#include "../TensorBase.h"
#include "OperatorMarco.h"

#define _D_Dragonian_Lib_Operator_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Operators {
#define _D_Dragonian_Lib_Operator_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Operator_Space_Begin

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
	static void ImplAddScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplSubScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplMulScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplDivScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAddTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplSubTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplMulTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplDivTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplEqualScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplNotEqualScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplGreaterScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplGreaterEqualScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplLessScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplLessEqualScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplEqualTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplNotEqualTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplGreaterTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplGreaterEqualTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplLessTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplLessEqualTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAndScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplOrScalar(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAndTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplOrTensor(
		bool* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplPowScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplPowTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter<_NRank>& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter<_NRank>& _SrcInfo2,
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

	template<typename _ResultType, size_t _NRank> static void UnaryOperator(
		_ResultType* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}
};

_D_Dragonian_Lib_Operator_Space_End
