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
	template<typename _TypeSrc, size_t _NRank>
	static void ImplCast(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _TypeSrc* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplAssignTensor(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplAssignScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Value,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplAssignBuffer(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplAssignRandn(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		double _Mean,
		double _Sigma,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplAssignRand(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Min,
		const _Type& _Max,
		bool Continuous
	);

	template<size_t _NRank>
	static void ImplArange(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Start,
		const _Type& _Step,
		bool Continuous
	);

	_D_Dragonian_Lib_Operator_Binary_Define(Add);
	_D_Dragonian_Lib_Operator_Binary_Define(Sub);
	_D_Dragonian_Lib_Operator_Binary_Define(Mul);
	_D_Dragonian_Lib_Operator_Binary_Define(Div);
	_D_Dragonian_Lib_Operator_Binary_Define(Mod);
	_D_Dragonian_Lib_Operator_Binary_Define(And);
	_D_Dragonian_Lib_Operator_Binary_Define(Or);
	_D_Dragonian_Lib_Operator_Binary_Define(Xor);
	_D_Dragonian_Lib_Operator_Binary_Define(LShift);
	_D_Dragonian_Lib_Operator_Binary_Define(RShift);
	_D_Dragonian_Lib_Operator_Binary_Define(Pow);
	_D_Dragonian_Lib_Operator_Binary_Define(BinaryOr);
	_D_Dragonian_Lib_Operator_Binary_Define(BinaryAnd);

	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Add);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Sub);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Mul);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Div);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Mod);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(And);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Or);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Xor);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(LShift);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(RShift);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Pow);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryOr);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryAnd);

	_D_Dragonian_Lib_Operator_Comparison_Define(Equal);
	_D_Dragonian_Lib_Operator_Comparison_Define(NotEqual);
	_D_Dragonian_Lib_Operator_Comparison_Define(Greater);
	_D_Dragonian_Lib_Operator_Comparison_Define(GreaterEqual);
	_D_Dragonian_Lib_Operator_Comparison_Define(Less);
	_D_Dragonian_Lib_Operator_Comparison_Define(LessEqual);

	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Equal);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(NotEqual);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Greater);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterEqual);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(Less);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessEqual);

	_D_Dragonian_Lib_Operator_Unary_Define(Sqrt);
	_D_Dragonian_Lib_Operator_Unary_Define(RSqrt);
	_D_Dragonian_Lib_Operator_Unary_Define(Reciprocal);
	_D_Dragonian_Lib_Operator_Unary_Define(Abs);
	_D_Dragonian_Lib_Operator_Unary_Define(Sin);
	_D_Dragonian_Lib_Operator_Unary_Define(Cos);
	_D_Dragonian_Lib_Operator_Unary_Define(Tan);
	_D_Dragonian_Lib_Operator_Unary_Define(ASin);
	_D_Dragonian_Lib_Operator_Unary_Define(ACos);
	_D_Dragonian_Lib_Operator_Unary_Define(ATan);
	_D_Dragonian_Lib_Operator_Unary_Define(Sinh);
	_D_Dragonian_Lib_Operator_Unary_Define(Cosh);
	_D_Dragonian_Lib_Operator_Unary_Define(Tanh);
	_D_Dragonian_Lib_Operator_Unary_Define(ASinh);
	_D_Dragonian_Lib_Operator_Unary_Define(ACosh);
	_D_Dragonian_Lib_Operator_Unary_Define(ATanh);
	_D_Dragonian_Lib_Operator_Unary_Define(Exp);
	_D_Dragonian_Lib_Operator_Unary_Define(Log);
	_D_Dragonian_Lib_Operator_Unary_Define(Log2);
	_D_Dragonian_Lib_Operator_Unary_Define(Log10);
	_D_Dragonian_Lib_Operator_Unary_Define(Ceil);
	_D_Dragonian_Lib_Operator_Unary_Define(Floor);
	_D_Dragonian_Lib_Operator_Unary_Define(Round);
	_D_Dragonian_Lib_Operator_Unary_Define(Trunc);
	_D_Dragonian_Lib_Operator_Unary_Define(Frac);

};

template <typename _Type>
struct RandomSettings
{
	using RandomNormalType = ConditionalType<sizeof(_Type) >= sizeof(double), double, float>;
	using NormalDistributionType = std::normal_distribution<RandomNormalType>;
	using RandomType = ConditionalType<sizeof(_Type) == sizeof(char), Int16, _Type>;
	using RandomDistributionType = ConditionalType<
		IsIntegerValue<_Type>,
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
	using RandomNormalType = ConditionalType<sizeof(_Type) >= sizeof(double), double, float>;
	using NormalDistributionType = std::normal_distribution<RandomNormalType>;
	using RandomType = ConditionalType<sizeof(_Type) == sizeof(char), Int16, _Type>;
	using RandomDistributionType = ConditionalType<
		IsIntegerValue<_Type>,
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
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsCallableValue<_Fn>> SingleTensorLoop(
	int64_t Value,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step, const int64_t* __restrict Left, const int64_t* __restrict Stride,
	_Fn _Func
)
{
	Value += *Left * *Step;
	const auto StepStride = *Stride * *Step;
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val = Value + i * StepStride;
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
				const auto Val = Value + i * StepStride;
				_Func(Val);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val = Value + i * StepStride;
			_Func(Val);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsCallableValue<_Fn>> DoubleTensorLoop(
	int64_t Value1, int64_t Value2,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step1, const int64_t* __restrict Left1, const int64_t* __restrict Stride1,
	const int64_t* __restrict Step2, const int64_t* __restrict Left2, const int64_t* __restrict Stride2,
	_Fn _Func
)
{
	Value1 += *Left1 * *Step1;
	Value2 += *Left2 * *Step2;
	const auto StepStride1 = *Stride1 * *Step1;
	const auto StepStride2 = *Stride2 * *Step2;
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * StepStride1;
			const auto Val2 = Value2 + i * StepStride2;
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
				const auto Val1 = Value1 + i * StepStride1;
				const auto Val2 = Value2 + i * StepStride2;
				_Func(Val1, Val2);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * StepStride1;
			const auto Val2 = Value2 + i * StepStride2;
			_Func(Val1, Val2);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<IsCallableValue<_Fn>> TripleTensorLoop(
	int64_t Value1, int64_t Value2, int64_t Value3,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Step1, const int64_t* __restrict Left1, const int64_t* __restrict Stride1,
	const int64_t* __restrict Step2, const int64_t* __restrict Left2, const int64_t* __restrict Stride2,
	const int64_t* __restrict Step3, const int64_t* __restrict Left3, const int64_t* __restrict Stride3,
	_Fn _Func
)
{
	Value1 += *Left1 * *Step1;
	Value2 += *Left2 * *Step2;
	Value3 += *Left3 * *Step3;
	const auto StepStride1 = *Stride1 * *Step1;
	const auto StepStride2 = *Stride2 * *Step2;
	const auto StepStride3 = *Stride3 * *Step3;
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * StepStride1;
			const auto Val2 = Value2 + i * StepStride2;
			const auto Val3 = Value3 + i * StepStride3;
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
				const auto Val1 = Value1 + i * StepStride1;
				const auto Val2 = Value2 + i * StepStride2;
				const auto Val3 = Value3 + i * StepStride3;
				_Func(Val1, Val2, Val3);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * StepStride1;
			const auto Val2 = Value2 + i * StepStride2;
			const auto Val3 = Value3 + i * StepStride3;
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
template<typename _Type, typename _Parameter, size_t _NRank, typename _Fn, typename _ContFn, SizeType OperatorDims = 0>
std::enable_if_t<IsCallableValue<_Fn>> ImplMultiThreadSingle(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestParameter,
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

	if constexpr (IsCallableValue<_ContFn> && OperatorDims == 0)
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
					if constexpr (IsSameTypeValue<RemoveARPCVType<_Parameter>, RandomSettings<_Type>>)
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
					if constexpr (IsSameTypeValue<RemoveARPCVType<_Parameter>, RandomSettings<_Type>>)
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;

				if constexpr (IsSameTypeValue<RemoveARPCVType<_Parameter>, RandomSettings<_Type>>)
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
				if constexpr (IsSameTypeValue<RemoveARPCVType<_Parameter>, RandomSettings<_Type>>)
					_UserParameter._ThreadId = _Valdef_Global_Random_Device_Id++;
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
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
					std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
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
				std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
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
template<typename _DstType, typename _SrcType, typename _Parameter, size_t _NRank, typename _Fn, typename _ContFn, SizeType OperatorDims = 0>
std::enable_if_t<IsCallableValue<_Fn>> ImplMultiThreadDouble(
	_DstType* _Dest,
	const OperatorParameter<_NRank>& _DestParameter,
	const _SrcType* _Src,
	const OperatorParameter<_NRank>& _SrcParameter,
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

	if constexpr (IsCallableValue<_ContFn> && OperatorDims == 0)
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
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
								std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
							std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
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
								std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
							std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
					std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
					_Src,
					std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
				std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
				_Src,
				std::make_shared<OperatorParameter<_NRank>>(_SrcParameter),
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
template<typename _DstType, typename _Src1Type, typename _Src2Type, typename _Parameter, size_t _NRank, typename _Fn, typename _ContFn, SizeType OperatorDims = 0>
std::enable_if_t<IsCallableValue<_Fn>> ImplMultiThreadTriple(
	_DstType* _Dest,
	const OperatorParameter<_NRank>& _DestParameter,
	const _Src1Type* _Src1,
	const OperatorParameter<_NRank>& _Src1Parameter,
	const _Src2Type* _Src2,
	const OperatorParameter<_NRank>& _Src2Parameter,
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

	if constexpr (IsCallableValue<_ContFn> && OperatorDims == 0)
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
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
								std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
								_Src2,
								std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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
							std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
							_Src2,
							std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(_DestParameter);
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
								std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
								_Src2,
								std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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
							std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
							_Src2,
							std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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
					std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
					_Src1,
					std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
					_Src2,
					std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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
				std::make_shared<OperatorParameter<_NRank>>(_DestParameter),
				_Src1,
				std::make_shared<OperatorParameter<_NRank>>(_Src1Parameter),
				_Src2,
				std::make_shared<OperatorParameter<_NRank>>(_Src2Parameter),
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