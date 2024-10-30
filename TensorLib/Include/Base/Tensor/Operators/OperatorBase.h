#pragma once
#include <random>
#include "Tensor/TensorBase.h"
#include "OperatorMarco.h"
#define _D_Dragonian_Lib_Operator_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Operators {
#define _D_Dragonian_Lib_Operator_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Operator_Space_Begin

static inline SizeType _Impl_Global_Seed = 114;
static inline std::mt19937_64 RandomEngine(114);
static inline std::uniform_int_distribution RandomInt64Distribution(INT64_MIN, INT64_MAX);
static inline std::uniform_int_distribution RandomInt32Distribution(INT32_MIN, INT32_MAX);
static inline std::uniform_int_distribution<int16_t> RandomInt16Distribution(INT16_MIN, INT16_MAX);
static inline std::uniform_real_distribution RandomDoubleDistribution(-DBL_MAX, DBL_MAX);
static inline std::uniform_real_distribution RandomFloatDistribution(-FLT_MAX, FLT_MAX);

template<size_t Rank>
struct TensorShapeInfoND
{
	SizeType Shape[Rank], ViewStep[Rank], ViewLeft[Rank], ViewStride[Rank];
	bool IsContinuousND[Rank];
	SizeType ViewRank = 1;
	TensorShapeInfoND()
	{
		for (size_t i = 0; i < Rank; i++)
		{
			Shape[i] = 0;
			ViewStep[i] = 0;
			ViewLeft[i] = 0;
			ViewStride[i] = 0;
			IsContinuousND[i] = false;
		}
	}
};
using TensorShapeInfo = TensorShapeInfoND<6>;

template<typename _Type, Device _Device>
class OperatorsBase
{
	OperatorsBase() = delete;
public:
	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		const _Type* _Src,
		const TensorShapeInfo& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		_Type _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignRandn(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	static void ImplAssignRand(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}
	
};

_D_Dragonian_Lib_Operator_Space_End
