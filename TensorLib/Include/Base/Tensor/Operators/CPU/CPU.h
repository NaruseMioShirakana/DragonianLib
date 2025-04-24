/**
 * @file CPU.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief CPU operators base
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Simd.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/ThreadPool.h"
#include "Libraries/Util/StringPreprocess.h"

#define _D_Dragonian_Lib_Operator_Fndecl(_Function) decltype(_Function), _Function

_D_Dragonian_Lib_Operator_Space_Begin

ThreadPool& GetThreadPool();
ThreadPool& GetTaskPool();
SizeType GetMaxTaskCountPerOperator();
void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount);
bool GetInstantRunFlag();
void SetInstantRunFlag(bool _Flag);
std::atomic_uint64_t& GetRandomDeviceId();
void SetTaskPoolSize(SizeType _Size);

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
	static void ImplMoveBuffer(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		SizeType _Count,
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

	template<typename _IndexType, size_t _NRank, size_t _Dim>
	static void ImplGather(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _IndexType* _Index,
		const OperatorParameter<_NRank>& _IndexInfo
	);

	template<typename _MaskType, size_t _NRank>
	static void ImplMaskedAssign(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const _MaskType* _Mask,
		const OperatorParameter<_NRank>& _MaskInfo,
		bool Continuous
	);

	template<typename _MaskType, size_t _NRank>
	static void ImplMaskedAssignScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _MaskType* _Mask,
		const OperatorParameter<_NRank>& _MaskInfo,
		const _Type& _Value,
		bool Continuous
	);

	template<size_t _NRank>
	static void MatMul(
		_Type* _OutFeature, const OperatorParameter<_NRank>& _OutFeatureInfo,
		const _Type* _InFeature, const OperatorParameter<_NRank>& _InFeatureInfo,
		const _Type* _Weight, const OperatorParameter<_NRank>& _WeightInfo,
		const _Type* _Bias, std::shared_ptr<OperatorParameter<_NRank>> _BiasInfo,
		_Type Alpha, _Type AlphaBias,
		bool _Conj
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
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(AddReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(SubReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MulReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(DivReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(ModReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(AndReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(OrReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(XorReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(LShiftReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(RShiftReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(PowReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryOrReverse);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryAndReverse);

	_D_Dragonian_Lib_Operator_Binary_Define(Max);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Max);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MaxReverse);
	_D_Dragonian_Lib_Operator_Binary_Define(Min);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Min);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MinReverse);

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
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(EqualReverse);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(NotEqualReverse);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterReverse);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterEqualReverse);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessReverse);
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessEqualReverse);

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
	_D_Dragonian_Lib_Operator_Unary_Define(Exp2);
	_D_Dragonian_Lib_Operator_Unary_Define(Log);
	_D_Dragonian_Lib_Operator_Unary_Define(Log2);
	_D_Dragonian_Lib_Operator_Unary_Define(Log10);
	_D_Dragonian_Lib_Operator_Unary_Define(Ceil);
	_D_Dragonian_Lib_Operator_Unary_Define(Floor);
	_D_Dragonian_Lib_Operator_Unary_Define(Round);
	_D_Dragonian_Lib_Operator_Unary_Define(Trunc);
	_D_Dragonian_Lib_Operator_Unary_Define(Frac);
	_D_Dragonian_Lib_Operator_Unary_Define(Negative);
	_D_Dragonian_Lib_Operator_Unary_Define(BitwiseNot);
	_D_Dragonian_Lib_Operator_Unary_Define(Not);
	_D_Dragonian_Lib_Operator_Unary_Define(Polar);
	_D_Dragonian_Lib_Operator_Unary_Define(ATan2);

	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceSum);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceProd);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMax);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMin);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMean);
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(ReduceLp);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceLogSum);
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceLogSumExp);
	_D_Dragonian_Lib_Operator_Unary_Define(ReduceArgMax);
	_D_Dragonian_Lib_Operator_Unary_Define(ReduceArgMin);

	_D_Dragonian_Lib_Operator_Unary_St_Define(CumSum);
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumSub);
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumProd);
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumDiv);
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumMax);
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumMin);
	_D_Dragonian_Lib_Operator_Unary_St_Define(Diff);

	template <InterpolateMode _Mode, size_t _NRank>
	static void ImplInterpolate(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const InterpolateParam<_Mode>& _Param,
		bool Continuous
	);

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
_D_Dragonian_Lib_Constexpr_Force_Inline void SingleTensorLoop(
	int64_t Value,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Stride,
	_Fn _Func
) requires (IsCallableValue<_Fn>)
{
	if constexpr (LoopCount == 0)
		_Func(Value);
	else if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val = Value + i * *Stride;
			SingleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val,
				Shape + 1, LoopBegin + 1,
				Stride + 1,
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
				const auto Val = Value + i * *Stride;
				_Func(Val);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val = Value + i * *Stride;
			_Func(Val);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline void DoubleTensorLoop(
	int64_t Value1, int64_t Value2,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Stride1, const int64_t* __restrict Stride2,
	_Fn _Func
) requires (IsCallableValue<_Fn>)
{
	if constexpr (LoopCount == 0)
		_Func(Value1, Value2);
	else if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			DoubleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val1, Val2,
				Shape + 1, LoopBegin + 1,
				Stride1 + 1, Stride2 + 1,
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
				const auto Val1 = Value1 + i * *Stride1;
				const auto Val2 = Value2 + i * *Stride2;
				_Func(Val1, Val2);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			_Func(Val1, Val2);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline void TripleTensorLoop(
	int64_t Value1, int64_t Value2, int64_t Value3,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Stride1, const int64_t* __restrict Stride2, const int64_t* __restrict Stride3,
	_Fn _Func
) requires (IsCallableValue<_Fn>)
{
	if constexpr (LoopCount == 0)
		_Func(Value1, Value2, Value3);
	else if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			TripleTensorLoop<LoopCount - 1, LoopUnfold>(
				Val1, Val2, Val3,
				Shape + 1, LoopBegin + 1,
				Stride1 + 1, Stride2 + 1, Stride3 + 1,
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
				const auto Val1 = Value1 + i * *Stride1;
				const auto Val2 = Value2 + i * *Stride2;
				const auto Val3 = Value3 + i * *Stride3;
				_Func(Val1, Val2, Val3);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			_Func(Val1, Val2, Val3);
			++i;
		}
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline void InlinedTensorLoop(
	int64_t Value1, int64_t Value2, int64_t Value3, int64_t Value4,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Stride1, const int64_t* __restrict Stride2, const int64_t* __restrict Stride3, const int64_t* __restrict Stride4,
	_Fn _Func
) requires (IsCallableValue<_Fn>)
{
	if constexpr (LoopCount == 0)
		_Func(Value1, Value2, Value3, Value4);
	else if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			const auto Val4 = Value4 + i * *Stride4;
			InlinedTensorLoop<LoopCount - 1, LoopUnfold>(
				Val1, Val2, Val3, Val4,
				Shape + 1, LoopBegin + 1,
				Stride1 + 1, Stride2 + 1, Stride3 + 1, Stride4 + 1,
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
				const auto Val1 = Value1 + i * *Stride1;
				const auto Val2 = Value2 + i * *Stride2;
				const auto Val3 = Value3 + i * *Stride3;
				const auto Val4 = Value4 + i * *Stride4;
				_Func(Val1, Val2, Val3, Val4);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			const auto Val4 = Value4 + i * *Stride4;
			_Func(Val1, Val2, Val3, Val4);
			++i;
		}
	}
}

template <
	typename _RetType, typename _InputType, typename _ParameterType,
	typename _FunctionType, _FunctionType _Function,
	TypeDef::OperatorType _OType,
	size_t _NRank, size_t _Unfold
> void BasicOperators(
	_RetType* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _IDestInfoOld,
	const _InputType* _Src1,
	std::shared_ptr<OperatorParameter<_NRank>> _ISrc1InfoOld,
	const _InputType* _Src2,
	std::shared_ptr<OperatorParameter<_NRank>> _ISrc2InfoOld,
	std::shared_ptr<_ParameterType> _IValue
)
{
	auto _DestInfoOld = std::move(_IDestInfoOld);
	auto _Src1InfoOld = std::move(_ISrc1InfoOld);
	auto _Src2InfoOld = std::move(_ISrc2InfoOld);
	auto _Value = std::move(_IValue);

	const auto& _DestInfo = *_DestInfoOld;
	const auto& _Src1Info = *_Src1InfoOld;

	const SizeType* __restrict DestShape = _DestInfo.Shape.Data();
	const SizeType* __restrict DestBegin = _DestInfo.Begin.Data();
	const SizeType* __restrict DestViewStride = _DestInfo.ViewStride.Data();
	const SizeType* __restrict Src1ViewStride = _Src1Info.ViewStride.Data();

	if constexpr (_OType == TypeDef::UnaryOperatorType)
	{
		DoubleTensorLoop<_NRank, _Unfold>(
			0, 0,
			DestShape, DestBegin,
			DestViewStride, Src1ViewStride,
			[&](int64_t _IndexA, int64_t _IndexB)
			{
				_Dest[_IndexA] = (_RetType)_Function(_Src1[_IndexB]);
			}
		);
	}
	else if constexpr (_OType == TypeDef::BinaryOperatorType)
	{
		const auto& _Src2Info = *_Src2InfoOld;
		const SizeType* __restrict Src2ViewStride = _Src2Info.ViewStride.Data();
		TripleTensorLoop<_NRank, _Unfold>(
			0, 0, 0,
			DestShape, DestBegin,
			DestViewStride, Src1ViewStride, Src2ViewStride,
			[&](int64_t _IndexA, int64_t _IndexB, int64_t _IndexC)
			{
				_Dest[_IndexA] = (_RetType)_Function(_Src1[_IndexB], _Src2[_IndexC]);
			}
		);
	}
	else if constexpr (_OType == TypeDef::ConstantOperatorType)
	{
		const auto& _ParameterValue = *_Value;
		auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				_Dest[_IndexA] = (_RetType)_Function(_Src1[_IndexB], _ParameterValue);
			};
		DoubleTensorLoop<_NRank, _Unfold>(
			0, 0,
			DestShape, DestBegin,
			DestViewStride, Src1ViewStride,
			Func
		);
	}
	else if constexpr (_OType == TypeDef::ReversedConstantOperatorType)
	{
		const auto& _ParameterValue = *_Value;
		auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				_Dest[_IndexA] = (_RetType)_Function(_ParameterValue, _Src1[_IndexB]);
			};
		DoubleTensorLoop<_NRank, _Unfold>(
			0, 0,
			DestShape, DestBegin,
			DestViewStride, Src1ViewStride,
			Func
		);
	}
}

template<
	typename _FunctionType, _FunctionType _Function,
	typename _VectorizedFunctionType, _VectorizedFunctionType _VectorizedFunction,
	TypeDef::OperatorType _OType, bool _IsCompare,
	size_t _Unfold, Int64 _OpThroughput, size_t _NRank,
	typename _RetType, typename _InputType, typename _ParameterType
> void ImplMultiThreadBasic(
	_RetType* _Dest,
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _InputType* _Src1,
	const std::shared_ptr<OperatorParameter<_NRank>> _Src1InfoOld,
	const _InputType* _Src2,
	const std::shared_ptr<OperatorParameter<_NRank>> _Src2InfoOld,
	const std::shared_ptr<_ParameterType> _Value,
	bool Continuous
)
{
	if constexpr (!IsCallableValue<_FunctionType>)
	{
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->LogWarn(L"This Op Is Not Implemented, So It Will Has No Effect.");
		return;
	}

	const auto TotalDataSize = _DestInfoOld->GetSize();
	TemplateLibrary::Array<std::shared_ptr<void>, 5> _DataPointer{
		nullptr, nullptr, nullptr, nullptr, nullptr
	};
	_DataPointer[0] = _DestInfoOld->Data;
	_DataPointer[1] = _Src1InfoOld->Data;
	if constexpr (_OType == TypeDef::BinaryOperatorType)
		_DataPointer[2] = _Src2InfoOld->Data;

	auto CreateTask = [&](const std::shared_future<void>& TaskFuture)
		{
			_DestInfoOld->ResultDependency->emplace_back(TaskFuture, _DataPointer);
			if (_Src1InfoOld->ArgumentDependency != _DestInfoOld->ArgumentDependency)
				_Src1InfoOld->ArgumentDependency->emplace_back(TaskFuture, _DataPointer);
			if constexpr (_OType == TypeDef::BinaryOperatorType)
				if (_Src1InfoOld->ArgumentDependency != _Src2InfoOld->ArgumentDependency)
					_Src2InfoOld->ArgumentDependency->emplace_back(TaskFuture, _DataPointer);
		};

	if (Continuous)
	{
		auto ContinuousFn = [=](Int64 _Offset, Int64 _Size)
			{
				if constexpr (IsSameTypeValue<_FunctionType, _VectorizedFunctionType>)
					ContiguousFunction<_RetType, _InputType, _ParameterType, _FunctionType, _Function, _OType, _OpThroughput>(
						_Dest + _Offset, _Size, _Src1 + _Offset, _Src2 + _Offset, _Value
					);
				else
					VectorizedFunction<_RetType, _InputType, _ParameterType, _FunctionType, _Function, _VectorizedFunctionType, _VectorizedFunction, _OType, _IsCompare, _OpThroughput>(
						_Dest + _Offset, _Size, _Src1 + _Offset, _Src2 + _Offset, _Value
					);
			};

		if (TotalDataSize < DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
			CreateTask(GetThreadPool().Commit(ContinuousFn, 0, TotalDataSize));
		else
		{
			const auto ThreadCount = std::min(
				std::max(GetThreadPool().GetThreadCount(), 1ll),
				GetMaxTaskCountPerOperator()
			);
			auto SplitSize = TotalDataSize / ThreadCount;
			if (SplitSize == 0) SplitSize = 1;
			const auto TaskCount = TotalDataSize / SplitSize;
			const auto Remainder = TotalDataSize % SplitSize;

			SizeType i = 0;
			for (; i < TaskCount; ++i)
				CreateTask(GetThreadPool().Commit(ContinuousFn, i * SplitSize, SplitSize));
			if (Remainder)
				CreateTask(GetThreadPool().Commit(ContinuousFn, i * SplitSize, Remainder));
		}
		return;
	}

	auto InContinuousFn = [=](auto& Info)
		{
			BasicOperators<_RetType, _InputType, _ParameterType, _FunctionType, _Function, _OType, _NRank, _Unfold>(
				_Dest, Info, _Src1, _Src1InfoOld, _Src2, _Src2InfoOld, _Value
			);
		};

	if (TotalDataSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NTasks = std::min(
			std::max(GetThreadPool().GetThreadCount(), 1ll),
			GetMaxTaskCountPerOperator()
		);
		SizeType TotalTaskCount = -1, TaskDim = -1;
		for (SizeType i = 0; std::cmp_less(i, _NRank); ++i)
			if (_DestInfoOld->Shape[i] >= NTasks)
			{
				TotalTaskCount = _DestInfoOld->Shape[i];
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(*_DestInfoOld);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;
				CreateTask(GetThreadPool().Commit(InContinuousFn, Info));
				ShapeIndex += TaskPerSlice;
			}
			if (Remainder)
			{
				auto Info = std::make_shared<OperatorParameter<_NRank>>(*_DestInfoOld);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + Remainder;
				CreateTask(GetThreadPool().Commit(InContinuousFn, Info));
			}
			return;
		}
	}

	CreateTask(GetThreadPool().Commit(InContinuousFn, _DestInfoOld));
}

template<
	size_t _ArgCount, size_t _NRank, SizeType OperatorDims,
	typename _Src2Type, typename _Src1Type, typename _DstType,
	typename _ParameterType, typename _FunctionType, typename _ContinuousFunctionType
>
void ImplMultiThreadCaller(
	_DstType* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _IDestInfoOld,
	const _Src1Type* _Src1,
	std::shared_ptr<OperatorParameter<_NRank>> _ISrc1InfoOld,
	const _Src2Type* _Src2,
	std::shared_ptr<OperatorParameter<_NRank>> _ISrc2InfoOld,
	std::shared_ptr<_ParameterType> _IUserParameter,
	bool Continuous,
	_FunctionType _Function,
	_ContinuousFunctionType _ContFunction
)
{
	static_assert((_ArgCount < 4) && (_ArgCount > 0));

	auto _DestInfoOld = std::move(_IDestInfoOld);
	auto _Src1InfoOld = std::move(_ISrc1InfoOld);
	auto _Src2InfoOld = std::move(_ISrc2InfoOld);
	auto _UserParameter = std::move(_IUserParameter);

	const auto TotalRank = _DestInfoOld->GetRank();
	const auto BatchDims = TotalRank - OperatorDims;
	const auto BatchCount = _DestInfoOld->GetSize(0, BatchDims);
	const auto OperatorUnfoldCount = _DestInfoOld->GetSize(BatchDims);
	const auto DataSize = BatchCount * OperatorUnfoldCount;

	TemplateLibrary::Array<std::shared_ptr<void>, 5> _DataPointer{
		nullptr, nullptr, nullptr, nullptr, nullptr
	};
	_DataPointer[0] = _DestInfoOld->Data;
	if constexpr (_ArgCount > 1)
		_DataPointer[1] = _Src1InfoOld->Data;
	if constexpr (_ArgCount > 2)
		_DataPointer[2] = _Src2InfoOld->Data;

	auto CreateTask = [&](const std::shared_future<void>& TaskFuture)
		{
			_DestInfoOld->ResultDependency->emplace_back(TaskFuture, _DataPointer);
			if constexpr (_ArgCount > 1)
				if (_Src1InfoOld->ArgumentDependency != _DestInfoOld->ArgumentDependency)
					_Src1InfoOld->ArgumentDependency->emplace_back(TaskFuture, _DataPointer);
			if constexpr (_ArgCount > 2)
				if (_Src1InfoOld->ArgumentDependency != _Src2InfoOld->ArgumentDependency)
					_Src2InfoOld->ArgumentDependency->emplace_back(TaskFuture, _DataPointer);
		};

	if constexpr (IsCallableValue<_ContinuousFunctionType>)
	{
		if (Continuous)
		{
			if (DataSize < DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
			{
				if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
					_UserParameter->_ThreadId = GetRandomDeviceId().fetch_add(1);
				if constexpr (_ArgCount == 1)
					CreateTask(GetThreadPool().Commit(_ContFunction, _Dest, BatchCount, _UserParameter));
				else if constexpr (_ArgCount == 2)
					CreateTask(GetThreadPool().Commit(_ContFunction, _Dest, _Src1, BatchCount, _UserParameter));
				else if constexpr (_ArgCount == 3)
					CreateTask(GetThreadPool().Commit(_ContFunction, _Dest, _Src1, _Src2, BatchCount, _UserParameter));
			}
			else
			{
				const auto ThreadCount = std::min(
					std::max(GetThreadPool().GetThreadCount(), 1ll),
					GetMaxTaskCountPerOperator()
				);
				auto SplitSize = BatchCount / ThreadCount / DRAGONIANLIB_ALLOC_ALIG * DRAGONIANLIB_ALLOC_ALIG;
				if (SplitSize == 0) SplitSize = 1;
				const auto TaskCount = BatchCount / SplitSize;
				const auto Remainder = BatchCount % SplitSize;

				SizeType i = 0;
				for (; i < TaskCount; ++i)
				{
					auto _CDest = _Dest + i * SplitSize;
					auto _CSrc1 = _Src1 + i * SplitSize;
					auto _CSrc2 = _Src2 + i * SplitSize;
					if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
					{
						auto _Param = std::make_shared<RandomSettings<_DstType>>(*_UserParameter);
						_Param->_ThreadId = GetRandomDeviceId().fetch_add(1);
						if constexpr (_ArgCount == 1)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, SplitSize, _Param));
						else if constexpr (_ArgCount == 2)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, SplitSize, _Param));
						else if constexpr (_ArgCount == 3)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, _CSrc2, SplitSize, _Param));
					}
					else
					{
						if constexpr (_ArgCount == 1)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, SplitSize, _UserParameter));
						else if constexpr (_ArgCount == 2)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, SplitSize, _UserParameter));
						else if constexpr (_ArgCount == 3)
							CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, _CSrc2, SplitSize, _UserParameter));
					}
				}
				if (Remainder)
				{
					auto _CDest = _Dest + i * SplitSize;
					auto _CSrc1 = _Src1 + i * SplitSize;
					auto _CSrc2 = _Src2 + i * SplitSize;
					if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
						_UserParameter->_ThreadId = GetRandomDeviceId().fetch_add(1);
					if constexpr (_ArgCount == 1)
						CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, Remainder, _UserParameter));
					else if constexpr (_ArgCount == 2)
						CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, Remainder, _UserParameter));
					else if constexpr (_ArgCount == 3)
						CreateTask(GetThreadPool().Commit(_ContFunction, _CDest, _CSrc1, _CSrc2, Remainder, _UserParameter));
				}
			}
			return;
		}
	}

	if constexpr (!IsCallableValue<_FunctionType>)
	{
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->LogWarn(L"This Op Is Not Implemented, So It Will Has No Effect.");
		return;
	}

	if (DataSize > DRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE)
	{
		const auto NTasks = std::min(
			std::max(GetThreadPool().GetThreadCount(), 1ll),
			GetMaxTaskCountPerOperator()
		);
		SizeType TotalTaskCount = -1, TaskDim = -1;
		for (SizeType i = 0; i < BatchDims; ++i)
			if (_DestInfoOld->Shape[i] >= NTasks)
			{
				TotalTaskCount = _DestInfoOld->Shape[i];
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
				auto Info = std::make_shared<OperatorParameter<_NRank>>(*_DestInfoOld);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + TaskPerSlice;
				if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
				{
					auto _Param = std::make_shared<RandomSettings<_DstType>>(*_UserParameter);
					_Param->_ThreadId = GetRandomDeviceId().fetch_add(1);
					if constexpr (_ArgCount == 1)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Param));
					else if constexpr (_ArgCount == 2)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _Param));
					else if constexpr (_ArgCount == 3)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _Src2, _Src2InfoOld, _Param));
				}
				else
				{
					if constexpr (_ArgCount == 1)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _UserParameter));
					else if constexpr (_ArgCount == 2)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _UserParameter));
					else if constexpr (_ArgCount == 3)
						CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _Src2, _Src2InfoOld, _UserParameter));
				}

				ShapeIndex += TaskPerSlice;
			}
			if (Remainder)
			{
				auto Info = std::make_shared<OperatorParameter<_NRank>>(*_DestInfoOld);
				Info->Begin[TaskDim] = ShapeIndex;
				Info->Shape[TaskDim] = ShapeIndex + Remainder;

				if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
					_UserParameter->_ThreadId = GetRandomDeviceId().fetch_add(1);
				if constexpr (_ArgCount == 1)
					CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _UserParameter));
				else if constexpr (_ArgCount == 2)
					CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _UserParameter));
				else if constexpr (_ArgCount == 3)
					CreateTask(GetThreadPool().Commit(_Function, _Dest, Info, _Src1, _Src1InfoOld, _Src2, _Src2InfoOld, _UserParameter));
			}
			return;
		}
	}

	if constexpr (IsSameTypeValue<RemoveARPCVType<_ParameterType>, RandomSettings<_DstType>>)
		_UserParameter->_ThreadId = GetRandomDeviceId().fetch_add(1);
	if constexpr (_ArgCount == 1)
		CreateTask(GetThreadPool().Commit(_Function, _Dest, _DestInfoOld, _UserParameter));
	else if constexpr (_ArgCount == 2)
		CreateTask(GetThreadPool().Commit(_Function, _Dest, _DestInfoOld, _Src1, _Src1InfoOld, _UserParameter));
	else if constexpr (_ArgCount == 3)
		CreateTask(GetThreadPool().Commit(_Function, _Dest, _DestInfoOld, _Src1, _Src1InfoOld, _Src2, _Src2InfoOld, _UserParameter));
}

_D_Dragonian_Lib_Operator_Space_End