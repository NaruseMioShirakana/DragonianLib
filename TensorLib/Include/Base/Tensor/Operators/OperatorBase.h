#pragma once
#include <future>
#include <random>
#include "Tensor/TensorBase.h"
#include "Tensor/Operators/OperatorMarco.h"

#define _D_Dragonian_Lib_Operator_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Operators {
#define _D_Dragonian_Lib_Operator_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Operator_Space_Begin

static inline SizeType _Impl_Global_Seed = 114;
static inline std::uniform_int_distribution RandomInt64Distribution(INT64_MIN, INT64_MAX);
static inline std::uniform_int_distribution RandomInt32Distribution(INT32_MIN, INT32_MAX);
static inline std::uniform_int_distribution<int16_t> RandomInt16Distribution(INT16_MIN, INT16_MAX);
static inline std::uniform_real_distribution RandomDoubleDistribution(-DBL_MAX, DBL_MAX);
static inline std::uniform_real_distribution RandomFloatDistribution(-FLT_MAX, FLT_MAX);

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

struct OperatorParameter
{
	Vector<SizeType> Shape; ///< Shape: The [view end/shape] of the tensor.
	Vector<SizeType> Begin; ///< Begin: The [view begin] of the tensor.
	Vector<SizeType> ViewStep; ///< ViewStep: The step of the view.
	Vector<SizeType> ViewLeft; ///< ViewLeft: The left of the view.
	Vector<SizeType> ViewStride; ///< ViewStride: The stride of the view.
	Vector<bool> IsContinuous; ///< IsContinuous: The continuous flag of the view.
	std::shared_ptr<std::deque<std::future<void>>> ThreadPool; ///< ThreadPool: The futures of the tensor.
	void* UserParameter = nullptr; ///< UserParameter: The user parameter.
	std::shared_ptr<void> Data = nullptr; ///< Data: The data of the tensor (prevent from the data being released while the tensor is used by an operator).

	OperatorParameter() = default;
	OperatorParameter(
		Vector<SizeType>&& _Shape, Vector<SizeType>&& _Begin, Vector<SizeType>&& _ViewStep,
		Vector<SizeType>&& _ViewLeft, Vector<SizeType>&& _ViewStride
	)
		: Shape(std::move(_Shape)), Begin(std::move(_Begin)), ViewStep(std::move(_ViewStep)),
		ViewLeft(std::move(_ViewLeft)), ViewStride(std::move(_ViewStride)) { }
	SizeType GetSize(size_t RangeBegin = 0, size_t RangeEnd = UINT64_MAX) const
	{
		SizeType Size = 1;
		RangeEnd = std::min(RangeEnd, Shape.Size());
		for (size_t i = RangeBegin; i < RangeEnd; ++i) Size *= Shape[i];
		return Size;
	}
	SizeType GetRank() const { return (SizeType)Shape.Size(); }
};

template<typename _Type, Device _Device>
class OperatorsBase
{
	OperatorsBase() = delete;
public:
	template<typename _TypeSrc>
	static void ImplCast(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _TypeSrc* _Src,
		const OperatorParameter& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignBuffer(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		_Type _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignRandn(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignRand(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}
	
};

_D_Dragonian_Lib_Operator_Space_End
