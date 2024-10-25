#pragma once
#include <random>
#include "Tensor/TensorBase.h"
#define DragonianLibOperatorSpaceBegin DragonianLibSpaceBegin namespace Operators {
#define DragonianLibOperatorSpaceEnd } DragonianLibSpaceEnd

DragonianLibOperatorSpaceBegin

static inline std::mt19937_64 RandomEngine(std::random_device{}());
static inline std::uniform_int_distribution RandomInt64Distribution(INT64_MIN, INT64_MAX);
static inline std::uniform_int_distribution RandomInt32Distribution(INT32_MIN, INT32_MAX);
static inline std::uniform_int_distribution<int16_t> RandomInt16Distribution(INT16_MIN, INT16_MAX);
static inline std::uniform_real_distribution RandomDoubleDistribution(-DBL_MAX, DBL_MAX);
static inline std::uniform_real_distribution RandomFloatDistribution(-FLT_MAX, FLT_MAX);

struct TensorShapeInfo
{
	SizeType Shape[6], ViewStep[6], ViewLeft[6], ViewStride[6];
};

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
		DragonianLibNotImplementedError;
	}

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	)
	{
		DragonianLibNotImplementedError;
	}

	static void ImplAssign(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		_Type _Value,
		bool Continuous
	)
	{
		DragonianLibNotImplementedError;
	}

	static void ImplAssignRandn(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	)
	{
		DragonianLibNotImplementedError;
	}

	static void ImplAssignRand(
		_Type* _Dest,
		const TensorShapeInfo& _DestInfo,
		bool Continuous
	)
	{
		DragonianLibNotImplementedError;
	}
	
};

DragonianLibOperatorSpaceEnd
