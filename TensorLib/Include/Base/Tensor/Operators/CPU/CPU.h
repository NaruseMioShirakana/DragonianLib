#pragma once
#include "Simd.h"
#include "Libraries/Util/ThreadPool.h"

_D_Dragonian_Lib_Operator_Space_Begin

inline ThreadPool _Valdef_My_Thread_Pool{ 1 };
inline SizeType _Valdef_Global_Max_Task_Count_Per_Operator = 1;
inline bool _Flag_Instant_Run = true;
inline std::atomic_uint64_t _Valdef_Global_Random_Device_Id = 0;

template<typename _Type>
class OperatorsBase<_Type, Device::CPU>
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
	);

	static void ImplAssignTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		bool Continuous
	);

	static void ImplAssignScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplAssignBuffer(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	);

	static void ImplAssignRandn(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	);

	static void ImplAssignRand(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type& _Min,
		const _Type& _Max,
		bool Continuous
	);

	static void ImplAddScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplSubScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplMulScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplDivScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplAddTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplSubTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplMulTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplDivTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplEqualScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplNotEqualScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplGreaterScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplGreaterEqualScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplLessScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplLessEqualScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplEqualTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplNotEqualTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplGreaterTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplGreaterEqualTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplLessTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplLessEqualTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplAndScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplOrScalar(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplAndTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplOrTensor(
		bool* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplPowScalar(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src,
		const OperatorParameter& _SrcInfo,
		const _Type& _Value,
		bool Continuous
	);

	static void ImplPowTensor(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type* _Src1,
		const OperatorParameter& _SrcInfo1,
		const _Type* _Src2,
		const OperatorParameter& _SrcInfo2,
		bool Continuous
	);

	static void ImplArange(
		_Type* _Dest,
		const OperatorParameter& _DestInfo,
		const _Type& _Start,
		const _Type& _Step,
		bool Continuous
	);
};

template <typename _Type>
struct RandomSettings
{
	using RandomNormalType = _Impl_Dragonian_Lib_Conditional_t<sizeof(_Type) >= sizeof(double), double, float>;
	using NormalDistributionType = std::normal_distribution<RandomNormalType>;
	using RandomType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<sizeof(_Type) == sizeof(char), Int16, _Type>;
	using RandomDistributionType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<
		_Impl_Dragonian_Lib_Is_Integer_v<_Type>,
		std::uniform_int_distribution<RandomType>,
		std::uniform_real_distribution<RandomType>
	>;

	RandomType _Min;
	RandomType _Max;
	RandomNormalType _Mean;
	RandomNormalType _Sigma;
	size_t _ThreadId = 0;
};
template <typename _Type>
struct RandomSettings<std::complex<_Type>>
{
	using RandomNormalType = _Impl_Dragonian_Lib_Conditional_t<sizeof(_Type) >= sizeof(double), double, float>;
	using NormalDistributionType = std::normal_distribution<RandomNormalType>;
	using RandomType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<sizeof(_Type) == sizeof(char), Int16, _Type>;
	using RandomDistributionType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<
		_Impl_Dragonian_Lib_Is_Integer_v<_Type>,
		std::uniform_int_distribution<RandomType>,
		std::uniform_real_distribution<RandomType>
	>;

	RandomType _Min;
	RandomType _Max;
	RandomNormalType _Mean;
	RandomNormalType _Sigma;
	size_t _ThreadId = 0;
};
template <typename _Type>
using _Impl_Dragonian_Lib_Random_Type = typename RandomSettings<_Type>::RandomType;
template <typename _Type>
using _Impl_Dragonian_Lib_Random_Distribution_Type = typename RandomSettings<_Type>::RandomDistributionType;
template <typename _Type>
using _Impl_Dragonian_Lib_Random_Normal_Type = typename RandomSettings<_Type>::RandomNormalType;
template <typename _Type>
using _Impl_Dragonian_Lib_Normal_Distribution_Type = typename RandomSettings<_Type>::NormalDistributionType;

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> SingleTensorLoop(
	int64_t Value,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step, const int64_t* __restrict Left, const int64_t* __restrict Stride,
	_Fn _Func
)
{
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val = Value + ((i * *Stride) + *Left) * *Step;
			SingleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val,
				Shape + 1, LoopBegin + 1,
				Step + 1, Left + 1, Stride + 1,
				_Func
			);
		}
	else
	{
		int64_t i = *LoopBegin;
		while (i < *Shape - LoopUnfold)
		{
			for (int64_t j = 0; j < LoopUnfold; ++j)
			{
				const auto Val = Value + ((i * *Stride) + *Left) * *Step;
				_Func(Val);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val = Value + ((i * *Stride) + *Left) * *Step;
			_Func(Val);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> DoubleTensorLoop(
	int64_t Value1, int64_t Value2,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step1, const int64_t* __restrict Left1, const int64_t* __restrict Stride1,
	const int64_t* __restrict Step2, const int64_t* __restrict Left2, const int64_t* __restrict Stride2,
	_Fn _Func
)
{
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
			const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
			DoubleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val1, Val2,
				Shape + 1, LoopBegin + 1,
				Step1 + 1, Left1 + 1, Stride1 + 1,
				Step2 + 1, Left2 + 1, Stride2 + 1,
				_Func
			);
		}
	else
	{
		int64_t i = *LoopBegin;
		while (i < *Shape - LoopUnfold)
		{
			for (int64_t j = 0; j < LoopUnfold; ++j)
			{
				const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
				const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
				_Func(Val1, Val2);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
			const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
			_Func(Val1, Val2);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> TripleTensorLoop(
	int64_t Value1, int64_t Value2, int64_t Value3,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step1, const int64_t* __restrict Left1, const int64_t* __restrict Stride1,
	const int64_t* __restrict Step2, const int64_t* __restrict Left2, const int64_t* __restrict Stride2,
	const int64_t* __restrict Step3, const int64_t* __restrict Left3, const int64_t* __restrict Stride3,
	_Fn _Func
)
{
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
			const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
			const auto Val3 = Value3 + ((i * *Stride3) + *Left3) * *Step3;
			TripleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val1, Val2, Val3,
				Shape + 1, LoopBegin + 1,
				Step1 + 1, Left1 + 1, Stride1 + 1,
				Step2 + 1, Left2 + 1, Stride2 + 1,
				Step3 + 1, Left3 + 1, Stride3 + 1,
				_Func
			);
		}
	else
	{
		int64_t i = *LoopBegin;
		while (i < *Shape - LoopUnfold)
		{
			for (int64_t j = 0; j < LoopUnfold; ++j)
			{
				const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
				const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
				const auto Val3 = Value3 + ((i * *Stride3) + *Left3) * *Step3;
				_Func(Val1, Val2, Val3);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + ((i * *Stride1) + *Left1) * *Step1;
			const auto Val2 = Value2 + ((i * *Stride2) + *Left2) * *Step2;
			const auto Val3 = Value3 + ((i * *Stride3) + *Left3) * *Step3;
			_Func(Val1, Val2, Val3);
			++i;
		}
	}
}

/**
 * @brief Multithreaded single tensor operation.
 * @tparam _Type Type of the tensor.
 * @tparam _Parameter Parameter of the operator.
 * @tparam _Fn Incontinuous function (pointer, shape parameter, operator parameter).
 * @tparam _ContFn Continuous function, (pointer, size, operator parameter) if operator dims = 0, (pointer, shape parameter, operator parameter) otherwise.
 * @tparam OperatorDims Number of operator dimensions, rank - operator dims = batch dims.
 * @param _Dest Data pointer of the destination tensor.
 * @param _DestParameter Parameter of the destination tensor.
 * @param _UserParameter User parameter.
 * @param Continuous Whether the operation is continuous.
 * @param _Function Incontinuous function.
 * @param _ContFunction Continuous function.
 * @return void.
 */
template<typename _Type, typename _Parameter, typename _Fn, typename _ContFn, SizeType OperatorDims = 0> 
std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> ImplMultiThreadSingle(
	_Type* _Dest,
	const OperatorParameter& _DestParameter,
	_Parameter _UserParameter,
	bool Continuous,
	_Fn _Function,
	_ContFn _ContFunction
)
{
	const auto TotalRank = _DestParameter.GetRank();
	const auto BatchDims = TotalRank - OperatorDims;
	const auto BatchCount = _DestParameter.GetSize(0, BatchDims);
	const auto OperatorUnfoldCount = _DestParameter.GetSize(BatchDims);
	const auto DataSize = BatchCount * OperatorUnfoldCount;

	if constexpr (_Impl_Dragonian_Lib_Is_Callable_v<_ContFn> && OperatorDims == 0)
	{
		if (Continuous)
		{
			if (DataSize < DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
				_DestParameter.ThreadPool->emplace_back(
					_Valdef_My_Thread_Pool.Commit(
						_ContFunction,
						_Dest,
						DataSize,
						_UserParameter
					),
					std::vector{
						_DestParameter.Data,
					}
				);
			else
			{
				const auto ThreadCount = std::min(
					std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
					_Valdef_Global_Max_Task_Count_Per_Operator
				);

				auto SplitSize = DataSize / ThreadCount / DRAGONIANLIB_ALLOC_ALIG * DRAGONIANLIB_ALLOC_ALIG;
				if (SplitSize == 0) SplitSize = 1;
				const auto TaskCount = DataSize / SplitSize;
				const auto Remainder = DataSize % SplitSize;

				SizeType i = 0;
				for (; i < TaskCount; ++i)
				{
					if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Impl_Dragonian_Lib_Remove_ARPCV_t<_Parameter>, RandomSettings<_Type>>)
						_UserParameter._ThreadId = _Valdef_Global_Random_Device_Id++;
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							SplitSize,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
						}
					);
				}
				if (Remainder)
				{
					if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Impl_Dragonian_Lib_Remove_ARPCV_t<_Parameter>, RandomSettings<_Type>>)
						_UserParameter._ThreadId = _Valdef_Global_Random_Device_Id++;
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							Remainder,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
						}
					);
				}
			}
			return;
		}
	}

	if (DataSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NTasks = std::min(
			std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
			_Valdef_Global_Max_Task_Count_Per_Operator
		);
		SizeType TotalTaskCount = -1, TaskDim = -1;
		for (SizeType i = 0; i < BatchDims; ++i)
			if (_DestParameter.Shape[i] >= NTasks)
			{
				TotalTaskCount = _DestParameter.Shape[i];
				TaskDim = i;
				break;
			}
		if (TotalTaskCount != -1)
		{
			auto TaskPerSlice = TotalTaskCount / NTasks;
			if (TaskPerSlice == 0) TaskPerSlice = 1;
			const auto Remainder = TotalTaskCount % TaskPerSlice;

			SizeType ShapeIndex = 0;
			while (ShapeIndex < TotalTaskCount - Remainder)
			{
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;

				if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Impl_Dragonian_Lib_Remove_ARPCV_t<_Parameter>, RandomSettings<_Type>>)
					_UserParameter._ThreadId = _Valdef_Global_Random_Device_Id++;
				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
						}
					);
				}
				ShapeIndex += TaskPerSlice;
			}
			if (Remainder)
			{
				if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_Impl_Dragonian_Lib_Remove_ARPCV_t<_Parameter>, RandomSettings<_Type>>)
					_UserParameter._ThreadId = _Valdef_Global_Random_Device_Id++;
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + Remainder;

				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
						}
					);
				}
			}
			return;
		}
	}

	if (Continuous)
	{
		if constexpr (OperatorDims != 0)
			_DestParameter.ThreadPool->emplace_back(
				_Valdef_My_Thread_Pool.Commit(
					_ContFunction,
					_Dest,
					std::make_shared<OperatorParameter>(_DestParameter),
					_UserParameter
				),
				std::vector{
					_DestParameter.Data,
				}
			);
	}
	else
	{
		_DestParameter.ThreadPool->emplace_back(
			_Valdef_My_Thread_Pool.Commit(
				_Function,
				_Dest,
				std::make_shared<OperatorParameter>(_DestParameter),
				_UserParameter
			),
			std::vector{
				_DestParameter.Data,
			}
		);
	}
}

/**
 * @brief Multithreaded single tensor operation.
 * @tparam _DstType Type of the destination tensor.
 * @tparam _SrcType Type of the source tensor.
 * @tparam _Parameter Parameter of the operator.
 * @tparam _Fn Incontinuous function (pointer, shape parameter, operator parameter).
 * @tparam _ContFn Continuous function, (pointer, size, operator parameter) if operator dims = 0, (pointer, shape parameter, operator parameter) otherwise.
 * @tparam OperatorDims Number of operator dimensions, rank - operator dims = batch dims.
 * @param _Dest Data pointer of the destination tensor.
 * @param _DestParameter Parameter of the destination tensor.
 * @param _Src Data pointer of the source tensor.
 * @param _SrcParameter Parameter of the source tensor.
 * @param _UserParameter User parameter.
 * @param Continuous Whether the operation is continuous.
 * @param _Function Incontinuous function.
 * @param _ContFunction Continuous function.
 * @return void.
 */
template<typename _DstType, typename _SrcType, typename _Parameter, typename _Fn, typename _ContFn, SizeType OperatorDims = 0>
std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> ImplMultiThreadDouble(
	_DstType* _Dest,
	const OperatorParameter& _DestParameter,
	const _SrcType* _Src,
	const OperatorParameter& _SrcParameter,
	_Parameter _UserParameter,
	bool Continuous,
	_Fn _Function,
	_ContFn _ContFunction
)
{
	const auto TotalRank = _DestParameter.GetRank();
	const auto BatchDims = TotalRank - OperatorDims;
	const auto BatchCount = _DestParameter.GetSize(0, BatchDims);
	const auto OperatorUnfoldCount = _DestParameter.GetSize(BatchDims);
	const auto DataSize = BatchCount * OperatorUnfoldCount;

	if constexpr (_Impl_Dragonian_Lib_Is_Callable_v<_ContFn> && OperatorDims == 0)
	{
		if (Continuous)
		{
			if (DataSize < DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
				_DestParameter.ThreadPool->emplace_back(
					_Valdef_My_Thread_Pool.Commit(
						_ContFunction,
						_Dest,
						_Src,
						DataSize,
						_UserParameter
					),
					std::vector{
						_DestParameter.Data,
						_SrcParameter.Data,
					}
				);
			else
			{
				const auto ThreadCount = std::min(
					std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
					_Valdef_Global_Max_Task_Count_Per_Operator
				);
				auto SplitSize = DataSize / ThreadCount / DRAGONIANLIB_ALLOC_ALIG * DRAGONIANLIB_ALLOC_ALIG;
				if (SplitSize == 0) SplitSize = 1;
				const auto TaskCount = DataSize / SplitSize;
				const auto Remainder = DataSize % SplitSize;

				SizeType i = 0;
				for (; i < TaskCount; ++i)
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							_Src + i * SplitSize,
							SplitSize,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_SrcParameter.Data,
						}
					);
				}
				if (Remainder)
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							_Src + i * SplitSize,
							Remainder,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_SrcParameter.Data,
						}
					);
				}
			}
			return;
		}
	}

	if (DataSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NTasks = std::min(
			std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
			_Valdef_Global_Max_Task_Count_Per_Operator
		);
		SizeType TotalTaskCount = -1, TaskDim = -1;
		for (SizeType i = 0; i < BatchDims; ++i)
			if (_DestParameter.Shape[i] >= NTasks)
			{
				TotalTaskCount = _DestParameter.Shape[i];
				TaskDim = i;
				break;
			}
		if (TotalTaskCount != -1)
		{
			auto TaskPerSlice = TotalTaskCount / NTasks;
			if (TaskPerSlice == 0) TaskPerSlice = 1;
			const auto Remainder = TotalTaskCount % TaskPerSlice;

			SizeType ShapeIndex = 0;
			while (ShapeIndex < TotalTaskCount - Remainder)
			{
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;

				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_Src,
								std::make_shared<OperatorParameter>(_SrcParameter),
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
								_SrcParameter.Data,
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_Src,
							std::make_shared<OperatorParameter>(_SrcParameter),
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_SrcParameter.Data,
						}
					);
				}

				ShapeIndex += TaskPerSlice;
			}
			if (Remainder)
			{
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + Remainder;

				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_Src,
								std::make_shared<OperatorParameter>(_SrcParameter),
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
								_SrcParameter.Data,
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_Src,
							std::make_shared<OperatorParameter>(_SrcParameter),
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_SrcParameter.Data,
						}
					);
				}
			}
			return;
		}
	}

	if (Continuous)
	{
		if constexpr (OperatorDims != 0)
			_DestParameter.ThreadPool->emplace_back(
				_Valdef_My_Thread_Pool.Commit(
					_ContFunction,
					_Dest,
					std::make_shared<OperatorParameter>(_DestParameter),
					_Src,
					std::make_shared<OperatorParameter>(_SrcParameter),
					_UserParameter
				),
				std::vector{
					_DestParameter.Data,
					_SrcParameter.Data,
				}
			);
	}
	else
	{
		_DestParameter.ThreadPool->emplace_back(
			_Valdef_My_Thread_Pool.Commit(
				_Function,
				_Dest,
				std::make_shared<OperatorParameter>(_DestParameter),
				_Src,
				std::make_shared<OperatorParameter>(_SrcParameter),
				_UserParameter
			),
			std::vector{
				_DestParameter.Data,
				_SrcParameter.Data,
			}
		);
	}
}

/**
 * @brief Multithreaded single tensor operation.
 * @tparam _DstType Type of the destination tensor.
 * @tparam _Src1Type Type of the first source tensor.
 * @tparam _Src2Type Type of the second source tensor.
 * @tparam _Parameter Parameter of the operator.
 * @tparam _Fn Incontinuous function (pointer, shape parameter, operator parameter).
 * @tparam _ContFn Continuous function, (pointer, size, operator parameter) if operator dims = 0, (pointer, shape parameter, operator parameter) otherwise.
 * @tparam OperatorDims Number of operator dimensions, rank - operator dims = batch dims.
 * @param _Dest Data pointer of the destination tensor.
 * @param _DestParameter Parameter of the destination tensor.
 * @param _Src1 Data pointer of the first source tensor.
 * @param _Src1Parameter Parameter of the first source tensor.
 * @param _Src2 Data pointer of the second source tensor.
 * @param _Src2Parameter Parameter of the second source tensor.
 * @param _UserParameter User parameter.
 * @param Continuous Whether the operation is continuous.
 * @param _Function Incontinuous function.
 * @param _ContFunction Continuous function.
 * @return void.
 */
template<typename _DstType, typename _Src1Type, typename _Src2Type, typename _Parameter, typename _Fn, typename _ContFn, SizeType OperatorDims = 0>
std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<_Fn>> ImplMultiThreadTriple(
	_DstType* _Dest,
	const OperatorParameter& _DestParameter,
	const _Src1Type* _Src1,
	const OperatorParameter& _Src1Parameter,
	const _Src2Type* _Src2,
	const OperatorParameter& _Src2Parameter,
	_Parameter _UserParameter,
	bool Continuous,
	_Fn _Function,
	_ContFn _ContFunction
)
{
	const auto TotalRank = _DestParameter.GetRank();
	const auto BatchDims = TotalRank - OperatorDims;
	const auto BatchCount = _DestParameter.GetSize(0, BatchDims);
	const auto OperatorUnfoldCount = _DestParameter.GetSize(BatchDims);
	const auto DataSize = BatchCount * OperatorUnfoldCount;

	if constexpr (_Impl_Dragonian_Lib_Is_Callable_v<_ContFn> && OperatorDims == 0)
	{
		if (Continuous)
		{
			if (DataSize < DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
				_DestParameter.ThreadPool->emplace_back(
					_Valdef_My_Thread_Pool.Commit(
						_ContFunction,
						_Dest,
						_Src1,
						_Src2,
						DataSize,
						_UserParameter
					),
					std::vector{
						_DestParameter.Data,
						_Src1Parameter.Data,
						_Src2Parameter.Data
					}
				);
			else
			{
				const auto ThreadCount = std::min(
					std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
					_Valdef_Global_Max_Task_Count_Per_Operator
				);
				auto SplitSize = DataSize / ThreadCount / DRAGONIANLIB_ALLOC_ALIG * DRAGONIANLIB_ALLOC_ALIG;
				if (SplitSize == 0) SplitSize = 1;
				const auto TaskCount = DataSize / SplitSize;
				const auto Remainder = DataSize % SplitSize;

				SizeType i = 0;
				for (; i < TaskCount; ++i)
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							_Src1 + i * SplitSize,
							_Src2 + i * SplitSize,
							SplitSize,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_Src1Parameter.Data,
							_Src2Parameter.Data
						}
					);
				}
				if (Remainder)
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_ContFunction,
							_Dest + i * SplitSize,
							_Src1 + i * SplitSize,
							_Src2 + i * SplitSize,
							Remainder,
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_Src1Parameter.Data,
							_Src2Parameter.Data
						}
					);
				}
			}
			return;
		}
	}

	if (DataSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NTasks = std::min(
			std::max(_Valdef_My_Thread_Pool.GetThreadCount(), 1ll),
			_Valdef_Global_Max_Task_Count_Per_Operator
		);
		SizeType TotalTaskCount = -1, TaskDim = -1;
		for (SizeType i = 0; i < BatchDims; ++i)
			if (_DestParameter.Shape[i] >= NTasks)
			{
				TotalTaskCount = _DestParameter.Shape[i];
				TaskDim = i;
				break;
			}
		if (TotalTaskCount != -1)
		{
			auto TaskPerSlice = TotalTaskCount / NTasks;
			if (TaskPerSlice == 0) TaskPerSlice = 1;
			const auto Remainder = TotalTaskCount % TaskPerSlice;

			SizeType ShapeIndex = 0;
			while (ShapeIndex < TotalTaskCount - Remainder)
			{
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;

				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_Src1,
								std::make_shared<OperatorParameter>(_Src1Parameter),
								_Src2,
								std::make_shared<OperatorParameter>(_Src2Parameter),
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
								_Src1Parameter.Data,
								_Src2Parameter.Data
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_Src1,
							std::make_shared<OperatorParameter>(_Src1Parameter),
							_Src2,
							std::make_shared<OperatorParameter>(_Src2Parameter),
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_Src1Parameter.Data,
							_Src2Parameter.Data
						}
					);
				}

				ShapeIndex += TaskPerSlice;
			}
			if (Remainder)
			{
				auto Info = std::make_shared<OperatorParameter>(
					Vector{ _DestParameter.Shape.Begin(), _DestParameter.Shape.End(), _DestParameter.Shape.GetAllocator() },
					Vector{ _DestParameter.Begin.Begin(), _DestParameter.Begin.End(), _DestParameter.Begin.GetAllocator() },
					Vector{ _DestParameter.ViewStep.Begin(), _DestParameter.ViewStep.End(), _DestParameter.ViewStep.GetAllocator() },
					Vector{ _DestParameter.ViewLeft.Begin(), _DestParameter.ViewLeft.End(), _DestParameter.ViewLeft.GetAllocator() },
					Vector{ _DestParameter.ViewStride.Begin(), _DestParameter.ViewStride.End(), _DestParameter.ViewStride.GetAllocator() }
				);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + Remainder;

				if (Continuous)
				{
					if constexpr (OperatorDims != 0)
						_DestParameter.ThreadPool->emplace_back(
							_Valdef_My_Thread_Pool.Commit(
								_ContFunction,
								_Dest,
								Info,
								_Src1,
								std::make_shared<OperatorParameter>(_Src1Parameter),
								_Src2,
								std::make_shared<OperatorParameter>(_Src2Parameter),
								_UserParameter
							),
							std::vector{
								_DestParameter.Data,
								_Src1Parameter.Data,
								_Src2Parameter.Data
							}
						);
				}
				else
				{
					_DestParameter.ThreadPool->emplace_back(
						_Valdef_My_Thread_Pool.Commit(
							_Function,
							_Dest,
							Info,
							_Src1,
							std::make_shared<OperatorParameter>(_Src1Parameter),
							_Src2,
							std::make_shared<OperatorParameter>(_Src2Parameter),
							_UserParameter
						),
						std::vector{
							_DestParameter.Data,
							_Src1Parameter.Data,
							_Src2Parameter.Data
						}
					);
				}
			}
			return;
		}
	}

	if (Continuous)
	{
		if constexpr (OperatorDims != 0)
			_DestParameter.ThreadPool->emplace_back(
				_Valdef_My_Thread_Pool.Commit(
					_ContFunction,
					_Dest,
					std::make_shared<OperatorParameter>(_DestParameter),
					_Src1,
					std::make_shared<OperatorParameter>(_Src1Parameter),
					_Src2,
					std::make_shared<OperatorParameter>(_Src2Parameter),
					_UserParameter
				),
				std::vector{
					_DestParameter.Data,
					_Src1Parameter.Data,
					_Src2Parameter.Data
				}
			);
	}
	else
	{
		_DestParameter.ThreadPool->emplace_back(
			_Valdef_My_Thread_Pool.Commit(
				_Function,
				_Dest,
				std::make_shared<OperatorParameter>(_DestParameter),
				_Src1,
				std::make_shared<OperatorParameter>(_Src1Parameter),
				_Src2,
				std::make_shared<OperatorParameter>(_Src2Parameter),
				_UserParameter
			),
			std::vector{
				_DestParameter.Data,
				_Src1Parameter.Data,
				_Src2Parameter.Data
			}
		);
	}
}

_D_Dragonian_Lib_Operator_Space_End