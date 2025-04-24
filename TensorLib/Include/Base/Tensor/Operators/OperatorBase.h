/**
 * @file OperatorBase.h
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
 * @brief Base class of operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include <future>
#include <random>
#include <deque>

#include "OperatorMarco.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"

#include "../TensorBase.h"

#define _D_Dragonian_Lib_Operator_Space_Begin _D_Dragonian_Lib_Space_Begin namespace Operators {
#define _D_Dragonian_Lib_Operator_Space_End } _D_Dragonian_Lib_Space_End

_D_Dragonian_Lib_Operator_Space_Begin

using namespace TypeTraits;

_D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcIndexOp(SizeType _Index, SizeType _Max)
{
	if (_Index < 0)
		_Index += _Max;
	if (_Index >= _Max || _Index < 0)
		_D_Dragonian_Lib_Throw_Exception("Index Out Of Range!");
	return _Index;
}

/**
* @brief Enum class representing interpolation mode.
*/
enum class InterpolateMode
{
	Nearest, ///< Nearest neighbor interpolation (1D)
	Nearest2D, ///< Nearest neighbor interpolation (2D)
	Nearest3D, ///< Nearest neighbor interpolation (3D)
	Linear, ///< Linear interpolation
	Bilinear, ///< Bilinear interpolation
	Bicubic, ///< Bicubic interpolation
	Trilinear, ///< Trilinear interpolation
	Area, ///< Area interpolation
};

template <InterpolateMode _Mode>
constexpr size_t GetInterpolateModeRank = 0;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Nearest> = 1;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Nearest2D> = 2;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Nearest3D> = 3;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Linear> = 1;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Bilinear> = 2;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Bicubic> = 2;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Trilinear> = 3;
template <>
constexpr size_t GetInterpolateModeRank<InterpolateMode::Area> = 2;

template <InterpolateMode _Mode>
struct InterpolateParam
{
	static constexpr auto _MyMode = _Mode;
	static constexpr auto _IsLinear =
		_MyMode == InterpolateMode::Linear || _MyMode == InterpolateMode::Bilinear || _MyMode == InterpolateMode::Trilinear ||
		_MyMode == InterpolateMode::Nearest || _MyMode == InterpolateMode::Nearest2D || _MyMode == InterpolateMode::Nearest3D;
	static constexpr auto _MyRank = GetInterpolateModeRank<_MyMode>;
	using DoubleArrayT = IDLArray<double, _MyRank>;
	using SizeTypeArrayT = IDLArray<SizeType, _MyRank>;

	std::optional<DoubleArrayT> _MyScale;
	bool _MyAlignCorners = false;
	std::optional<SizeTypeArrayT> _MySize;

	InterpolateParam(const DoubleArrayT& _Scale, bool _AlignCorners = false)
		: _MyScale(_Scale), _MyAlignCorners(_AlignCorners), _MySize(std::nullopt)
	{

	}

	InterpolateParam(const SizeTypeArrayT& _Size, bool _AlignCorners = false)
		: _MyScale(std::nullopt), _MyAlignCorners(_AlignCorners), _MySize(_Size)
	{

	}

	InterpolateParam(DoubleArrayT&& _Scale, bool _AlignCorners = false)
		: _MyScale(std::move(_Scale)), _MyAlignCorners(_AlignCorners), _MySize(std::nullopt)
	{

	}

	InterpolateParam(SizeTypeArrayT&& _Size, bool _AlignCorners = false)
		: _MyScale(std::nullopt), _MyAlignCorners(_AlignCorners), _MySize(std::move(_Size))
	{

	}
};

template<size_t _NRank>
struct OperatorParameter
{
	using DependencyChainDataPointers = TemplateLibrary::Array<std::shared_ptr<void>, 5>;
	using DependencyChainPair = std::pair<std::shared_future<void>, DependencyChainDataPointers>;
	using DependencyChainType = std::deque<DependencyChainPair>;
	using DependencyChainPointer = std::shared_ptr<DependencyChainType>;

	IDLArray<SizeType, _NRank> Shape; ///< Shape: The [view end/shape] of the tensor.
	IDLArray<SizeType, _NRank> Begin; ///< Begin: The [view begin] of the tensor.
	IDLArray<SizeType, _NRank> ViewStride; ///< ViewStep: The step of the view.
	DependencyChainPointer ResultDependency = nullptr; ///< Dependency: Block All the operations until the dependency is finished.
	DependencyChainPointer ArgumentDependency = nullptr; ///< InplaceLock: Block the inplace operation until the dependency is finished.
	void* UserParameter = nullptr; ///< UserParameter: The user parameter.
	std::shared_ptr<void> Data = nullptr; ///< Data: The data of the tensor (prevent from the data being released while the tensor is used by an operator).
	SizeType GetSize(size_t RangeBegin = 0, size_t RangeEnd = _NRank) const
	{
		SizeType Size = 1;
		RangeEnd = std::min(RangeEnd, _NRank);
		for (size_t i = RangeBegin; i < RangeEnd; ++i) Size *= Shape[i];
		return Size;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline static SizeType GetRank() { return (SizeType)_NRank; }
};

template<typename _Type, Device _Device>
class OperatorsBase
{
	OperatorsBase() = delete;
public:
	template<typename _TypeSrc, size_t _NRank>
	static void ImplCast(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _TypeSrc* _Src, const OperatorParameter<_NRank>& _SrcInfo, 
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignTensor(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src, const OperatorParameter<_NRank>& _SrcInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplMoveBuffer(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src, SizeType _Count, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignBuffer(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src, SizeType _Count, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignScalar(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Value, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignRandn(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		double _Mean, double _Sigma, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplAssignRand(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Min, const _Type& _Max, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void ImplArange(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type& _Start, const _Type& _Step, bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<typename _IndexType, size_t _NRank, size_t _Dim>
	static void ImplGather(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src, const OperatorParameter<_NRank>& _SrcInfo,
		const _IndexType* _Index, const OperatorParameter<_NRank>& _IndexInfo
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<typename _MaskType, size_t _NRank>
	static void ImplMaskedAssign(
		_Type* _Dest, const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src, const OperatorParameter<_NRank>& _SrcInfo,
		const _MaskType* _Mask, const OperatorParameter<_NRank>& _MaskInfo,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<typename _MaskType, size_t _NRank>
	static void ImplMaskedAssignScalar(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _MaskType* _Mask,
		const OperatorParameter<_NRank>& _MaskInfo,
		const _Type& _Value,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template<size_t _NRank>
	static void MatMul(
		_Type* _OutFeature, const OperatorParameter<_NRank>& _OutFeatureInfo,
		const _Type* _InFeature, const OperatorParameter<_NRank>& _InFeatureInfo,
		const _Type* _Weight, const OperatorParameter<_NRank>& _WeightInfo,
		const _Type* _Bias, std::shared_ptr<OperatorParameter<_NRank>> _BiasInfo,
		_Type Alpha, _Type AlphaBias,
		bool _Conj
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
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(AddReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(SubReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MulReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(DivReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(ModReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(AndReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(OrReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(XorReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(LShiftReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(RShiftReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(PowReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryOrReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(BinaryAndReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Binary_Define(Max) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Max) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MaxReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define(Min) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(Min) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(MinReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }

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
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(EqualReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(NotEqualReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(GreaterEqualReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Comparison_Define_Scalar(LessEqualReverse) { _D_Dragonian_Lib_Not_Implemented_Error; }

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
	_D_Dragonian_Lib_Operator_Unary_Define(Exp2) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log2) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Log10) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Ceil) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Floor) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Round) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Trunc) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Frac) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Negative) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(BitwiseNot) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Not) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(Polar) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ATan2) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceSum) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceProd) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMax) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMin) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceMean) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Binary_Define_Scalar(ReduceLp) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceLogSum) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(ReduceLogSumExp) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ReduceArgMax) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_Define(ReduceArgMin) { _D_Dragonian_Lib_Not_Implemented_Error; }

	_D_Dragonian_Lib_Operator_Unary_St_Define(CumSum) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumSub) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumProd) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumDiv) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumMax) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(CumMin) { _D_Dragonian_Lib_Not_Implemented_Error; }
	_D_Dragonian_Lib_Operator_Unary_St_Define(Diff) { _D_Dragonian_Lib_Not_Implemented_Error; }

	template <InterpolateMode _Mode, size_t _NRank>
	static void ImplInterpolate(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		const InterpolateParam<_Mode>& _Param,
		bool Continuous
	)
	{
		_D_Dragonian_Lib_Not_Implemented_Error;
	}

};

_D_Dragonian_Lib_Operator_Space_End
