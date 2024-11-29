﻿/**
 * FileName: Tensor.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once

#include <deque>
#include <ranges>
#include <mdspan>
#include "Operators.h"
#include "Libraries/Util/ThreadPool.h"

_D_Dragonian_Lib_Space_Begin

static inline double DoubleZero = 0.; ///< Static inline double zero

/**
 * @brief Struct representing a range with begin, step, and end values.
 */
struct Range
{
	SizeType Begin = 0; ///< Begin value
	SizeType Step = 1; ///< Step value
	SizeType End = 0; ///< End value
	bool IsNone = true; ///< Flag indicating if it is none

	/**
	 * @brief Constructor for a none range.
	 * @param _NoneVal The none value to initialize the range.
	 */
	Range(NoneType _NoneVal) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End), IsNone(false) {}

	/**
	 * @brief Constructor for a range with none, step, and end values.
	 * @param _NoneVal The none value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	Range(NoneType _NoneVal, SizeType _Step, SizeType _End) :Begin(_Step > 0 ? 0 : -1), Step(_Step), End(_End), IsNone(false)
	{
		UNUSED(_NoneVal);
	}

	/**
	 * @brief Constructor for a range with begin, step, and none values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _NoneVal The none value.
	 */
	Range(SizeType _Begin, SizeType _Step, NoneType _NoneVal) :Begin(_Begin), Step(_Step), End(_Step > 0 ? -1 : 0), IsNone(false)
	{
		UNUSED(_NoneVal);
	}

	/**
	 * @brief Constructor for a range with begin and end values.
	 * @param _Begin The begining value.
	 * @param _End The end value.
	 */
	Range(SizeType _Begin, SizeType _End) :Begin(_Begin), End(_End), IsNone(false) {}

	/**
	 * @brief Constructor for a range with none and end values.
	 * @param _NoneVal The none value.
	 * @param _End The end value.
	 */
	Range(NoneType _NoneVal, SizeType _End) :End(_End), IsNone(false) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin and none values.
	 * @param _Begin The begining value.
	 * @param _NoneVal The none value.
	 */
	Range(SizeType _Begin, NoneType _NoneVal) :Begin(_Begin), End(-1), IsNone(false) { UNUSED(_NoneVal); }

	/**
	 * @brief Reverse the range.
	 */
	void Reverse() { std::swap(Begin, End); Step = -Step; }

	/**
	 * @brief Equality operator for NoneType.
	 * @param _NoneVal The none value.
	 * @return True if the range is none, false otherwise.
	 */
	bool operator==(const NoneType& _NoneVal) const { UNUSED(_NoneVal); return IsNone; }
};

/**
 * @brief Enum class representing padding types.
 */
enum class PaddingType
{
	Zero, ///< Zero padding
	Constant, ///< Constant padding
	Reflect, ///< Reflect padding
	Cicular, ///< Circular padding
	Replicate ///< Replicate padding
};

/**
 * @brief Enum class representing interpolation types.
 */
enum class InterpolateType
{
	Nearest1D, ///< Nearest neighbor interpolation for 1D
	Nearest2D, ///< Nearest neighbor interpolation for 2D
	Linear, ///< Linear interpolation
	Bilinear, ///< Bilinear interpolation
	Bicubic, ///< Bicubic interpolation
	Trilinear, ///< Trilinear interpolation
	Area, ///< Area interpolation
};

template <size_t _NRank>
using SliceOptions = IDLArray<Range, _NRank>; ///< Alias for vector of ranges
template <size_t _NRank>
using VRanges = IDLArray<Range, _NRank>; ///< Alias for vector of ranges
template <size_t _NRank>
using Dimensions = IDLArray<SizeType, _NRank>;

/**
 * @brief Set the random seed.
 * @param _Seed The seed value.
 */
void SetRandomSeed(SizeType _Seed);

/**
 * @brief Set the number of worker threads.
 * @param _ThreadCount The number of worker threads.
 */
void SetWorkerCount(SizeType _ThreadCount);

/**
 * @brief Set the maximum task count per operator.
 * @param _MaxTaskCount The maximum task count per operator.
 */
void SetMaxTaskCountPerOperator(SizeType _MaxTaskCount);

/**
 * @brief Enable the time logger in thread pool.
 * @param _Enable True to enable, false to disable.
 */
void EnableTimeLogger(bool _Enable);

/**
 * @brief Enable the instant run in thread pool.
 * @param _Enable True to enable, false to disable.
 */
void EnableInstantRun(bool _Enable);

template <size_t _NRank>
std::string ToString(const Dimensions<_NRank>& _Dimensions)
{
	if constexpr (_Dimensions.Empty())
		return "()";
	std::string Ret = "(";
	for (const auto& Dim : _Dimensions)
		Ret += std::to_string(Dim) + ", ";
	Ret.pop_back(); Ret.pop_back();
	Ret += ")";
	return Ret;
}

template <typename _Type>
constexpr const _Type& MaxOf(const _Type& _Left, const _Type& _Right) { return _Left > _Right ? _Left : _Right; }
template <typename _Type>
constexpr const _Type& MinOf(const _Type& _Left, const _Type& _Right) { return _Left < _Right ? _Left : _Right; }

/**
 * @class Tensor
 * @brief Tensor with a specified value type and device.
 * @tparam _TensorType The value type of the tensor.
 * @tparam _NRank The rank of the tensor.
 * @tparam _MyDevice The device of the tensor.
 */
template <typename _TensorType = float, size_t _NRank = 2, Device _MyDevice = Device::CPU>
class Tensor : public Value
{
public:
	static_assert(_NRank > 0, "The rank of the tensor must be greater than 0!");

	template <typename _TensorType_, size_t _NRank_, Device _MyDevice_>
	friend class Tensor;
	using ValueTypeSrcImpl = std::remove_reference_t<_TensorType>;
	using ValueType = _Impl_Dragonian_Lib_Conditional_t<
		_Impl_Dragonian_Lib_Is_Bool_v<ValueTypeSrcImpl>, Int8, ValueTypeSrcImpl
	>;
	using Pointer = std::shared_ptr<void>;
	using RawPointer = ValueType*;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;
	static_assert(!_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<ValueType, _D_Dragonian_Lib_Namespace Value>);

	using _MyMultiThreadSyncT = typename Operators::OperatorParameter<_NRank>::_MyMultiThreadSyncT;
	using _MyMultiThreadSyncP = typename Operators::OperatorParameter<_NRank>::_MyMultiThreadSyncP;
	static constexpr auto _Device = _MyDevice;
	static constexpr auto _DType = _Impl_Dragonian_Lib_Decldtype_v<_TensorType>;

	void Eval() const
	{
		if (_MyFutures)
		{
			if (!_MyFutures->empty() && !Operators::_Flag_Instant_Run)
				Operators::_Valdef_My_Thread_Pool.Notify(_MyFutures->size());
			while (!_MyFutures->empty())
			{
				_MyFutures->front().first.get();
				_MyFutures->pop_front();
			}
		}
	}
	Tensor& Eval()
	{
		if (_MyFutures)
		{
			if (!_MyFutures->empty() && !Operators::_Flag_Instant_Run)
				Operators::_Valdef_My_Thread_Pool.Notify(_MyFutures->size());
			while (!_MyFutures->empty())
			{
				_MyFutures->front().first.get();
				_MyFutures->pop_front();
			}
		}
		return *this;
	}

protected:
	Pointer _MyFirst = nullptr;
	RawPointer _MyLast = nullptr;
	RawPointer _MyData = nullptr;
	Dimensions<_NRank> _MyShape;
	Dimensions<_NRank> _MyViewStep;
	Dimensions<_NRank> _MyViewLeft;
	Dimensions<_NRank> _MyViewStride;
	_MyMultiThreadSyncP _MyFutures = nullptr;
	bool IsBroadCasted_ = false;

private:

	template <size_t _TmpTank = _NRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
		(_TmpTank > 1) && _TmpTank == _NRank,
		Tensor<_TensorType, _TmpTank - 1, _MyDevice>>
		ViewFirstDim(SizeType _Index) const
	{
		const auto Idx = CalcIndex(_Index, _MyShape.Front());
		Tensor<_TensorType, _TmpTank - 1, _MyDevice> Ret;

		Ret._MyShape.Assign(_MyShape.begin() + 1);
		Ret._MyViewStep.Assign(_MyViewStep.begin() + 1);
		Ret._MyViewLeft.Assign(_MyViewLeft.begin() + 1);
		Ret._MyViewStride.Assign(_MyViewStride.begin() + 1);

		auto Index = (_MyViewLeft.Front() + (Idx * _MyViewStride.Front())) * _MyViewStep.Front();
		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData + Index;
		Ret._MyLast = _MyLast;
		Ret._MyFutures = _MyFutures;
		Ret.IsBroadCasted_ = IsBroadCasted_;
		return Ret;
	}

	template <size_t _TRank>
	constexpr std::enable_if_t<
		(_NRank > _TRank),
		Tensor<_TensorType, _NRank - _TRank, _MyDevice>> ViewDimensions(const Dimensions<_TRank>& _Indice) const

	{
		Tensor<_TensorType, _NRank - _TRank, _MyDevice> Ret;
		Ret._MyShape.Assign(_MyShape.begin() + _TRank);
		Ret._MyViewStep.Assign(_MyViewStep.begin() + _TRank);
		Ret._MyViewLeft.Assign(_MyViewLeft.begin() + _TRank);
		Ret._MyViewStride.Assign(_MyViewStride.begin() + _TRank);

		Ret._MyFirst = _MyFirst;
		Ret._MyData = Data(_Indice);
		Ret._MyLast = _MyLast;
		Ret._MyFutures = _MyFutures;
		Ret.IsBroadCasted_ = IsBroadCasted_;
		return Ret;
	}

	template <size_t _Rank1, size_t _Rank2>
	static std::pair<
		Tensor<_TensorType, MaxOf(_Rank1, _Rank2), _MyDevice>,
		Tensor<_TensorType, MaxOf(_Rank1, _Rank2), _MyDevice>
	> BroadCast(
		const Tensor<_TensorType, _Rank1, _MyDevice>& _A,
		const Tensor<_TensorType, _Rank2, _MyDevice>& _B,
		bool Inplace = false
	)
	{
		constexpr auto CurrentRank = MaxOf(_Rank1, _Rank2);
		std::pair<
			Tensor<_TensorType, CurrentRank, _MyDevice>,
			Tensor<_TensorType, CurrentRank, _MyDevice>
		> Ret{ {},{} };

		auto& First = Ret.first;
		auto& Second = Ret.second;
		First._MyShape.AssignConstant(1);				Second._MyShape.AssignConstant(1);
		First._MyViewStep.AssignConstant(1);			Second._MyViewStep.AssignConstant(1);
		First._MyViewLeft.AssignConstant(0);			Second._MyViewLeft.AssignConstant(0);
		First._MyViewStride.AssignConstant(0);			Second._MyViewStride.AssignConstant(0);
		First._MyFirst = _A._MyFirst;						Second._MyFirst = _B._MyFirst;
		First._MyLast = _A._MyLast;							Second._MyLast = _B._MyLast;
		First._MyFutures = _A._MyFutures;					Second._MyFutures = _B._MyFutures;
		First._MyData = _A._MyData;							Second._MyData = _B._MyData;
		if constexpr (CurrentRank != _Rank1)
			First.IsBroadCasted_ = true;
		if constexpr (CurrentRank != _Rank2)
			Second.IsBroadCasted_ = true;

		for (size_t CurrentIndex = 0; CurrentIndex < CurrentRank; ++CurrentIndex)
		{
			//const auto i = CurrentRank - CurrentIndex - 1;
			const auto idx = CurrentRank - CurrentIndex - 1;
			auto XSize = 1ll, YSize = 1ll;
			if (CurrentIndex < _Rank1)
			{
				const auto i = _Rank1 - CurrentIndex - 1;
				First._MyShape[idx] = _A._MyShape[i];		First._MyViewStride[idx] = _A._MyViewStride[i];
				First._MyViewStep[idx] = _A._MyViewStep[i];	First._MyViewLeft[idx] = _A._MyViewLeft[i];
				XSize = _A._MyShape[i];
			}
			if (CurrentIndex < _Rank2)
			{
				const auto i = _Rank2 - CurrentIndex - 1;
				Second._MyShape[idx] = _B._MyShape[i];		Second._MyViewStride[idx] = _B._MyViewStride[i];
				Second._MyViewStep[idx] = _B._MyViewStep[i]; Second._MyViewLeft[idx] = _B._MyViewLeft[i];
				YSize = _B._MyShape[i];
			}
			if (XSize == YSize)
				continue;
			if (XSize == 1)
			{
				if (Inplace)
					_D_Dragonian_Lib_Throw_Exception(
						"Could Not Inplace Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
						"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
						"] From " + std::to_string(XSize) + " To " + std::to_string(YSize) + "!"
					);

				First._MyShape[idx] = YSize;					First._MyViewStride[idx] = 0;
				First.IsBroadCasted_ = true;
			}
			else if (YSize == 1)
			{
				Second._MyShape[idx] = XSize;					Second._MyViewStride[idx] = 0;
				Second.IsBroadCasted_ = true;
			}
			else
				_D_Dragonian_Lib_Throw_Exception(
					"Could Not Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
					"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
					"] With Value [" + std::to_string(XSize) + ", " + std::to_string(YSize) + "]!"
				);
		}
		return Ret;
	}

	template <size_t _Rank2>
	std::enable_if_t<
		_Rank2 <= _NRank,
		Tensor> BroadCast(const Tensor<_TensorType, _Rank2, _MyDevice>& _Other) const
	{
		decltype(auto) Bd = BroadCast(*this, _Other, true);
		return std::move(Bd.second);
	}

public:
	~Tensor() override = default;
	Tensor(const Tensor& Left) = default;
	Tensor(Tensor&& Right) noexcept = default;
	constexpr Tensor& operator=(Tensor&& _Right) noexcept = default;

	/**
	 * @brief Copy the data of a tensor
	 * @param _Left Source tensor
	 * @return Reference of this
	 */
	template <size_t _TRank>
	constexpr Tensor& operator=(const Tensor<ValueType, _TRank, _MyDevice>& _Left)
	{
		if constexpr (_Impl_Dragonian_Lib_Could_Be_Converted_From_v<ValueType, ValueType> && std::is_copy_assignable_v<ValueType>)
		{
			if ((const void*)this != (const void*)&_Left)
				Assign(_Left);
			return *this;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

	constexpr Tensor& operator=(const Tensor& _Left)
	{
		if constexpr (_Impl_Dragonian_Lib_Could_Be_Converted_From_v<ValueType, ValueType> && std::is_copy_assignable_v<ValueType>)
		{
			if ((const void*)this != (const void*)&_Left)
				Assign(_Left);
			return *this;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

	/**
	 * @brief Assign the tensor with a scalar value.
	 * @param _Val The scalar value.
	 * @return Reference of this.
	 */
	constexpr Tensor& operator=(const ValueType& _Val)
	{
		if constexpr (_Impl_Dragonian_Lib_Could_Be_Converted_From_v<ValueType, ValueType> && std::is_copy_assignable_v<ValueType>)
		{
			Assign(_Val);
			return *this;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 2D tensor, then tensor[0] will return the 1st row of the tensor.
	 * @param _Index The index of the element tensor.
	 * @return The element tensor.
	 */
	template <size_t _TmpTank = _NRank>
	constexpr std::enable_if_t<
		(_TmpTank > 1) && _TmpTank == _NRank,
		Tensor<_TensorType, _TmpTank - 1, _MyDevice>> operator[](SizeType _Index) const
	{
		return ViewFirstDim(_Index);
	}

	template <size_t _TmpTank = _NRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline std::enable_if_t<
		(_TmpTank <= 1) && _TmpTank == _NRank,
		ValueType&> operator[](SizeType _Index) const
	{
		return Get(_Index);
	}

	/**
	 * @brief Get a sliced tensor of the tensor.
	 * @param _SliceOptions The slice options of the tensor.
	 * @return The sliced tensor.
	 */
	constexpr Tensor operator[](const SliceOptions<_NRank>& _SliceOptions) const
	{
		return Slice(_SliceOptions);
	}

	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 3D tensor, then tensor[{0, 0}] will return the 1st row of the 1st matrix of the tensor.
	 * @param _Indice
	 * @return
	 */
	template <size_t _TRank>
	constexpr std::enable_if_t<
		(_NRank > _TRank),
		Tensor<_TensorType, _NRank - _TRank, _MyDevice>> operator[](const Dimensions<_TRank>& _Indice) const
	{
		return ViewDimensions(_Indice);
	}

	ValueType& operator[](const Dimensions<_NRank>& _Indice) const
	{
		return *Data(_Indice);
	}

	//****************************************************Constructor****************************************************//

	/**
	 * @brief Create a new tensor with the specified shape.
	 * @param MyShape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_trivial_v<_CurValueType> ||
		std::is_constructible_v<_CurValueType>>,
		Tensor> New(const Dimensions<_NRank>& MyShape)
	{
		return Tensor(MyShape);
	}

	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_trivial_v<_CurValueType> ||
		std::is_constructible_v<_CurValueType>>,
		Tensor> New(const SizeType(&MyShape)[_NRank])
	{
		return Tensor(MyShape);
	}

	template <typename _CurValueType = ValueType, typename _First, typename ...Rest>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_constructible_v<_CurValueType, _First, Rest...>>,
		Tensor> New(const Dimensions<_NRank>& MyShape, _First Arg0, Rest ...Args)
	{
		return Tensor(MyShape, Arg0, Args...);
	}

	/**
	 * @brief Create an empty new tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_trivial_v<_CurValueType>>,
		Tensor> New()
	{
		return Tensor();
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with ones.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(1)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> Ones(const Dimensions<_NRank>& _Shape)
	{
		Tensor Ret(_Shape);
		Ret.Assign(ValueType(1));
		Ret.Eval();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with zeros.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(0)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> Zeros(const Dimensions<_NRank>& _Shape)
	{
		Tensor Ret(_Shape);
		Ret.Assign(ValueType(0));
		Ret.Eval();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with a constant value.
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> ConstantOf(const Dimensions<_NRank>& _Shape, const ValueType& _Val)
	{
		Tensor Ret(_Shape);
		Ret.Assign(_Val);
		Ret.Eval();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor> Rand(const Dimensions<_NRank>& _Shape, const ValueType& Min, const ValueType& Max)
	{
		Tensor Ret(_Shape);
		Ret.AssignRand(Min, Max);
		Ret.Eval();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor> Randn(const Dimensions<_NRank>& _Shape, double _Mean = 0., double _Sigma = 1.)
	{
		Tensor Ret(_Shape);
		Ret.AssignRandn(_Mean, _Sigma);
		Ret.Eval();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with ones.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(1)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> OnesLike(const Tensor& _ShapeReference)
	{
		return Ones(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with zeros.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(0)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> ZerosLike(const Tensor& _ShapeReference)
	{
		return Zeros(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with a constant value.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> ConstantLike(const Tensor& _ShapeReference, const ValueType& _Val)
	{
		return ConstantOf(_ShapeReference.Shape(), _Val);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor> RandLike(const Tensor& _ShapeReference, const ValueType& Min, const ValueType& Max)
	{
		return Rand(_ShapeReference.Shape(), Min, Max);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor> RandnLike(const Tensor& _ShapeReference, double _Mean = 0., double _Sigma = 1.)
	{
		return Randn(_ShapeReference.Shape(), _Mean, _Sigma);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_trivial_v<_CurValueType>>,
		Tensor> Empty(const Dimensions<_NRank>& _Shape)
	{
		return Tensor(_Shape);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_trivial_v<_CurValueType>>,
		Tensor> EmptyLike(const Tensor& _ShapeReference)
	{
		return Tensor(_ShapeReference._MyShape);
	}

	template <typename _CurValueType = ValueType>
	static constexpr std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Add_Operator_v<_CurValueType>&&
		Operators::_Impl_Dragonian_Lib_Has_Mul_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>&&
		std::is_constructible_v<ValueType>>,
		Tensor> Arange(ValueType _Begin, ValueType _End, ValueType _Step)
	{
		if (_Step == ValueType(0))
			_D_Dragonian_Lib_Throw_Exception("Step Can't Be Zero!");
		auto _Count = static_cast<SizeType>((_End - _Begin) / _Step);
		if (_Count <= 0)
			_D_Dragonian_Lib_Throw_Exception("End Must Be Greater Than Begin!");
		if constexpr (_Impl_Dragonian_Lib_Is_Floating_Point_v<ValueType>)
			if (std::isnan(_Count))
				_D_Dragonian_Lib_Throw_Exception("Invalid Range!");
		Tensor Ret = New({ _Count });
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplArange(
			Ret._MyData,
			Ret.GetDefaultOperatorParameter(),
			_Begin, _Step,
			!Ret.IsBroadCasted() && Ret.IsContinuous()
		);
		return Ret;
	}

private:
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool AllocateMemory(
		const Dimensions<_NRank>& MyShape, Allocator MyAlloc
	)
	{
		if (MyShape.Empty())
			return false;
		const auto Size = MyShape.Multiply();
		_MyFirst = Pointer(
			MyAlloc->Allocate(std::max(Size * sizeof(ValueType), 256ull)),
			[MyAlloc](void* _Pointer) { MyAlloc->Free(_Pointer); }
		);
		_MyData = (RawPointer)_MyFirst.get();
		_MyLast = _MyData + Size;
		return true;
	}

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void ConstructViewInfo(
		const Dimensions<_NRank>& MyShape
	)
	{
		_MyShape = MyShape;
		auto _Begin = _MyViewStep.ReversedBegin();
		auto _End = _MyViewStep.ReversedEnd();
		auto _Iter = _MyShape.ReversedBegin();
		*_Begin-- = 1;
		while (_Begin != _End)
		{
			*_Begin = *(_Begin + 1) * *_Iter--;
			--_Begin;
		}
		_MyViewLeft.AssignConstant(0ll);
		_MyViewStride.AssignConstant(1ll);
	}

	Tensor() = default;

	Tensor(const Dimensions<_NRank>& MyShape) : _MyFutures(new _MyMultiThreadSyncT)
	{
		if (AllocateMemory(MyShape, GetMemoryProvider(_MyDevice)))
		{
			ConstructViewInfo(MyShape);
			if constexpr (!std::is_trivial_v<ValueType> && std::is_constructible_v<ValueType>)
			{
				auto IterData = _MyData;
				while (IterData != _MyLast)
					new (IterData++) ValueType();
			}
		}
	}

	template <typename _First, typename ...Rest>
	Tensor(const Dimensions<_NRank>& MyShape, _First Arg0, Rest ...Args) : _MyFutures(new _MyMultiThreadSyncT)
	{
		if (AllocateMemory(MyShape, GetMemoryProvider(_MyDevice)))
		{
			ConstructViewInfo(MyShape);
			if constexpr (std::is_constructible_v<ValueType, _First, Rest...>)
			{
				auto IterData = _MyData;
				while (IterData != _MyLast)
					new (IterData++) ValueType(Arg0, Args...);
			}
		}
	}

	template <typename _CurValueType = ValueType>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>
		> Assign(const ValueType& _Value)
	{
		Eval();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignScalar(
			_MyData,
			GetDefaultOperatorParameter(),
			_Value,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>
		> Assign(const ValueType* _Buffer, SizeType _Count)
	{
		Eval();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignBuffer(
			_MyData,
			GetDefaultOperatorParameter(),
			_Buffer,
			_Count,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>
		> Assign(const Tensor<ValueType, _TRank, _MyDevice>& _Val)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		_Val.Eval();
		Eval();
		if (_Val.IsScalar())
			return Assign(_Val.Item());
		Tensor BroadCasted = BroadCast(_Val);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignTensor(
			_MyData,
			GetDefaultOperatorParameter(),
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			!IsBroadCasted() && !BroadCasted.IsBroadCasted() && IsContinuous() && BroadCasted.IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>
		> AssignRand(const ValueType& Min, const ValueType& Max)
	{
		Eval();
		//std::unique_lock lg(Operators::_Valdef_My_Thread_Pool.GetSeedMutex(), std::try_to_lock);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRand(
			_MyData,
			GetDefaultOperatorParameter(),
			Min, Max,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>
		> AssignRandn(double _Mean = 0., double _Sigma = 1.)
	{
		Eval();
		//std::unique_lock lg(Operators::_Valdef_My_Thread_Pool.GetSeedMutex(), std::try_to_lock);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRandn(
			_MyData,
			GetDefaultOperatorParameter(),
			_Mean,
			_Sigma,
			!IsBroadCasted() && IsContinuous()
		);
	}

public:

	//********************************************************Info********************************************************//

	/**
	 * @brief Get the alignment size of the value type.
	 * @return The alignment size of the value type.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		SizeType GetAlignSize()
	{
		return alignof(ValueType);
	}

	/**
	 * @brief Get the device of the tensor.
	 * @return The device of the tensor.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		Device GetDevice()
	{
		return _Device;
	}

	/**
	 * @brief Get the allocator of the tensor.
	 * @return The allocator of the tensor.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		Allocator GetAllocator()
	{
		return GetMemoryProvider(_MyDevice);
	}

	/**
	 * @brief Get the buffer of the tensor.
	 * @return The buffer of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Buffer()
	{
		return _MyFirst;
	}

	/**
	 * @brief Get the data pointer of the tensor.
	 * @return The data pointer of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Data() const
	{
		if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>)
			return (bool*)_MyData;
		else
			return _MyData;
	}

	/**
	 * @brief Get the data pointer of the tensor with the specified indices.
	 * @param _Indices The indices of the tensor.
	 * @return The data pointer of the tensor.
	 */
	template <size_t _TRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Data(const Dimensions<_TRank>& _Indices) const
	{
		static_assert(_TRank <= _NRank, "The rank of the indices must be less than or equal to the rank of the tensor!");
		SizeType Index = 0;
		for (size_t i = 0; i < _Indices.Size(); ++i)
		{
			const SizeType Idx = CalcIndex(_Indices[i], _MyShape[i]);
			Index += ((Idx * _MyViewStride[i]) + _MyViewLeft[i]) * _MyViewStep[i];
		}
		if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>)
			return (bool*)(_MyData + Index);
		else
			return _MyData + Index;
	}

	/**
	 * @brief Get a val of the tensor with the specified index.
	 * @param Index The index.
	 * @return The val.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Get(SizeType Index) const
	{
		if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>)
			return *(bool*)Data<1>({ Index });
		else
			return *Data<1>({ Index });
	}

	/**
	 * @brief Get a val of the tensor with the specified indices.
	 * @param _Indices The indices.
	 * @return The val.
	 */
	template <size_t _TRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Item(const Dimensions<_TRank>& _Indices) const
	{
		return *Data(_Indices);
	}

	/**
	 * @brief Get the first val of the tensor.
	 * @return The val.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) Item() const
	{
		if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>)
			return *(bool*)(_MyData + _MyViewLeft[0] * _MyViewStep[0]);
		else
			return *(_MyData + _MyViewLeft[0] * _MyViewStep[0]);
	}

	/**
	 * @brief Get the pointer of the first val of the tensor.
	 * @return The pointer.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		decltype(auto) ItemPointer() const
	{
		return _MyData + _MyViewLeft[0] * _MyViewStep[0];
	}

	//******************************************************Operator******************************************************//

	/**
	 * @brief Assign the tensor with ones.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(1)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor&> FixOnes()
	{
		Assign(ValueType(1));
		return *this;
	}

	/**
	 * @brief Assign the tensor with zeros.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, decltype(0)>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor&> FixZeros()
	{
		Assign(ValueType(0));
		return *this;
	}

	/**
	 * @brief Assign the tensor with a constant value.
	 * @param _Val The constant value.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor&> Fix(const ValueType& _Val)
	{
		Assign(_Val);
		return *this;
	}

	/**
	 * @brief Fix the tensor with a buffer.
	 * @param _Buffer The buffer.
	 * @param _Count Data count of the buffer.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor&> Fix(const ValueType* _Buffer, SizeType _Count)
	{
		Assign(_Buffer, _Count);
		return *this;
	}

	/**
	 * @brief Assign the tensor with random values.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor&> RandFix(const ValueType& Min = ValueType(0), const ValueType& Max = ValueType(1))
	{
		AssignRand(Min, Max);
		return *this;
	}

	/**
	 * @brief Assign the tensor with random values.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Is_Arithmetic_v<_CurValueType>>,
		Tensor&> RandnFix(double _Mean = 0., double _Sigma = 1.)
	{
		AssignRandn(_Mean, _Sigma);
		return *this;
	}

	//*************************************************Binary Operator*************************************************//

	template <typename _CurValueType = ValueType, size_t >
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Add_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> operator+(const ValueType& _Right) const
	{
		Eval();
		auto Ret = New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAddScalar(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Sub_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> operator-(const ValueType& _Right) const
	{
		Eval();
		auto Ret = New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplSubScalar(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Mul_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> operator*(const ValueType& _Right) const
	{
		Eval();
		auto Ret = New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMulScalar(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Div_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> operator/(const ValueType& _Right) const
	{
		Eval();
		auto Ret = New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplDivScalar(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Add_Inplace_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> operator+=(const ValueType& _Right)
	{
		Eval();
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAddScalar(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Sub_Inplace_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> operator-=(const ValueType& _Right)
	{
		Eval();
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplSubScalar(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Mul_Inplace_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> operator*=(const ValueType& _Right)
	{
		Eval();
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMulScalar(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Div_Inplace_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> operator/=(const ValueType& _Right)
	{
		Eval();
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplDivScalar(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_NotEqual_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator!=(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplNotEqualScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Equal_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator==(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplEqualScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Less_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator<(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplLessScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Greater_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator>(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplGreaterScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;

	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_LessEqual_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator<=(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplLessEqualScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_GreaterEqual_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator>=(const ValueType& _Right) const
	{
		Eval();
		auto Ret = Tensor<bool, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplGreaterEqualScalar(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			_Right,
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Add_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
		> operator+(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAddTensor(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Sub_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
		> operator-(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplSubTensor(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Mul_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
		> operator*(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMulTensor(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Div_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
		> operator/(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplDivTensor(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Add_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>&&
		_TRank <= _NRank>,
		Tensor> operator+=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		auto BroadCasted = BroadCast(_Right);
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAddTensor(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Sub_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>&&
		_TRank <= _NRank>,
		Tensor> operator-=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		auto BroadCasted = BroadCast(_Right);
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplSubTensor(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Mul_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>&&
		_TRank <= _NRank>,
		Tensor> operator*=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		auto BroadCasted = BroadCast(_Right);
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMulTensor(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Div_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>&&
		_TRank <= _NRank>,
		Tensor> operator/=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		auto BroadCasted = BroadCast(_Right);
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplDivTensor(
			_MyData,
			MyParameter,
			_MyData,
			MyParameter,
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<bool, _TensorType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator&&(const Tensor<bool, _TRank, _MyDevice>& _Right)
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAndTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<bool, _TensorType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator||(const Tensor<bool, _TRank, _MyDevice>& _Right)
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplOrTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_NotEqual_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
	> operator!=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplNotEqualTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Equal_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator==(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplEqualTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Less_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator<(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplLessTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Greater_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator>(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplGreaterTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_LessEqual_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator<=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplLessEqualTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_GreaterEqual_Operator_v<_CurValueType>>,
		Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>
		> operator>=(const Tensor<ValueType, _TRank, _MyDevice>& _Right) const
	{
		Eval();
		_Right.Eval();
		auto BroadCasted = BroadCast(*this, _Right);
		auto Ret = Tensor<bool, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplGreaterEqualTensor(
			(bool*)Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			!BroadCasted.first.IsBroadCasted() && BroadCasted.first.IsContinuous() &&
			!BroadCasted.second.IsBroadCasted() && BroadCasted.second.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	static std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
	> Pow(const Tensor& _InputA, const Tensor& _InputB)
	{
		_InputA.Eval();
		_InputB.Eval();
		auto BroadCasted = BroadCast(_InputA, _InputB);
		auto Ret = Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>::New(BroadCasted.first.Shape());
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplPowTensor
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			BroadCasted.first.Data(),
			BroadCasted.first.GetDefaultOperatorParameter(),
			BroadCasted.second.Data(),
			BroadCasted.second.GetDefaultOperatorParameter(),
			BroadCasted.first.IsContinuous() && !BroadCasted.first.IsBroadCasted() &&
			BroadCasted.second.IsContinuous() && !BroadCasted.second.IsBroadCasted()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	static std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> Pow(const Tensor& _InputA, ValueType _Val)
	{
		_InputA.Eval();
		auto Ret = Tensor<ValueType, _NRank, _MyDevice>::New(_InputA.Shape());
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplPowScalar
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_InputA.Data(),
			_InputA.GetDefaultOperatorParameter(),
			_Val,
			_InputA.IsContinuous() && !_InputA.IsBroadCasted()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor<ValueType, MaxOf(_NRank, _TRank), _MyDevice>
		> Pow(const Tensor& _InputB) const
	{
		return Pow(*this, _InputB);
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> Pow(ValueType _Val) const
	{
		return Pow(*this, _Val);
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> PowInplace(const Tensor& _InputB)
	{
		Eval();
		_InputB.Eval();
		if (_InputB.IsScalar())
			return PowInplace(_InputB.Item());
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		auto BroadCasted = BroadCast(_InputB);
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplPowTensor
		(
			Data(),
			MyParameter,
			Data(),
			MyParameter,
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			IsContinuous() && !IsBroadCasted() &&
			IsContinuous() && BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && !IsBroadCasted()
		);
		return *this;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Pow_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor&> PowInplace(ValueType _Val)
	{
		Eval();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		const auto MyParameter = GetDefaultOperatorParameter();
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplPowScalar
		(
			Data(),
			MyParameter,
			Data(),
			MyParameter,
			_Val,
			IsContinuous() && !IsBroadCasted()
		);
		return *this;
	}

	//****************************************************Unary Operator****************************************************//

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Sub_Inplace_Operator_v<_CurValueType>&&
		std::is_move_assignable_v<_CurValueType>>,
		Tensor> operator-() const
	{
		Eval();
		auto Ret = ZerosLike(*this);
		Ret -= *this;
		return Ret;
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		Operators::_Impl_Dragonian_Lib_Has_Equal_Operator_v<_CurValueType>>,
		Tensor<bool, _NRank, _MyDevice>> operator!() const
	{
		Eval();
		auto Ret = Zeros(_MyShape);
		return Ret == *this;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Sqrt() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Sqrt::UnaryOperatorSqrt
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> RSqrt() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::RSqrt::UnaryOperatorRSqrt
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Reciprocal() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Reciprocal::UnaryOperatorReciprocal
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Abs() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Abs::UnaryOperatorAbs
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Sin() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Sin::UnaryOperatorSin
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Cos() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Cos::UnaryOperatorCos
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Tan() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Tan::UnaryOperatorTan
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ASin() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ASin::UnaryOperatorASin
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ACos() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ACos::UnaryOperatorACos
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ATan() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ATan::UnaryOperatorATan
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Sinh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Sinh::UnaryOperatorSinh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Cosh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Cosh::UnaryOperatorCosh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Tanh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Tanh::UnaryOperatorTanh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ASinh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ASinh::UnaryOperatorASinh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ACosh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ACosh::UnaryOperatorACosh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> ATanh() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::ATanh::UnaryOperatorATanh
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Exp() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Exp::UnaryOperatorExp
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Log() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Log::UnaryOperatorLog
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Log2() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Log2::UnaryOperatorLog2
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Log10() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Log10::UnaryOperatorLog10
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Ceil() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Ceil::UnaryOperatorCeil
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Floor() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Floor::UnaryOperatorFloor
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Round() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Round::UnaryOperatorRound
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Trunc() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Trunc::UnaryOperatorTrunc
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, Device _CurDevice = Device::CPU>
	std::enable_if_t <
		_Impl_Dragonian_Lib_And_v <
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		std::is_copy_assignable_v<_CurValueType>&&
		_CurDevice == _MyDevice &&
		_CurDevice == Device::CPU>,
		Tensor> Frac() const
	{
		Eval();
		auto Ret = Tensor::New(_MyShape);
		Operators::Frac::UnaryOperatorFrac
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			_MyData,
			GetDefaultOperatorParameter(),
			!IsBroadCasted() && IsContinuous()
		);
		return Ret;
	}

	//*********************************************************Info*********************************************************//

	/**
	 * @brief Get the shape info of the tensor.
	 * @tparam _Begin The start axis.
	 * @tparam _End The end axis.
	 * @return The shape info of the tensor.
	 */
	template <size_t _Begin = 0, size_t _End = _NRank>
	Operators::OperatorParameter<_End - _Begin> GetDefaultOperatorParameter() const
	{
		constexpr auto CurrentRank = _End - _Begin;
		if constexpr (CurrentRank <= 0)
			_D_Dragonian_Lib_Throw_Exception("The Rank Of The Tensor Is Too Low!");
		if constexpr (CurrentRank > Rank())
			_D_Dragonian_Lib_Throw_Exception("The Rank Of The Info Is Too High!");
		Operators::OperatorParameter<CurrentRank> Ret;
		Ret.Begin.AssignConstant(0);
		Ret.Shape.Assign(_MyShape.Data() + _Begin);
		Ret.ViewStride.Assign(_MyViewStride.Data() + _Begin);
		Ret.ViewStep.Assign(_MyViewStep.Data() + _Begin);
		Ret.ViewLeft.Assign(_MyViewLeft.Data() + _Begin);
		for (size_t i = 0; i < CurrentRank; ++i)
			Ret.IsContinuous[i] = IsContinuous(_Begin + i, _End);
		Ret.ThreadPool = _MyFutures;
		Ret.Data = _MyFirst;
		return Ret;
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const Dimensions<_NRank>& Shape() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Shape(SizeType _Index) const
	{
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const Dimensions<_NRank>& Size() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the total size of the tensor.
	 * @return The total size of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType TotalSize() const
	{
		return _MyShape.Multiply();
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Size(SizeType _Index) const
	{
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the rank of the tensor.
	 * @return The rank of the tensor.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Rank()
	{
		return _NRank;
	}

	/**
	 * @brief Get the strides of the tensor.
	 * @return The strides of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const Dimensions<_NRank>& ViewStrides() const
	{
		return _MyViewStride;
	}

	/**
	 * @brief Get the steps of the tensor.
	 * @return The steps of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const Dimensions<_NRank>& ViewSteps() const
	{
		return _MyViewStep;
	}

	/**
	 * @brief Get the left indices of the tensor.
	 * @return The left indices of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline const Dimensions<_NRank>& ViewLeft() const
	{
		return _MyViewLeft;
	}

	/**
	 * @brief Get the default slice vector of the tensor.
	 * @return The default slice vector of the tensor.
	 */
	SliceOptions<_NRank> GetDefaultSliceVector() const
	{
		SliceOptions<_NRank> Ret;
		Ret.Reserve(_MyShape.Size());
		for (auto i : _MyShape)
			Ret.EmplaceBack(0, i);
		return Ret;
	}

	/**
	 * @brief Get the continuous access order of the tensor.
	 * @return The continuous access order of the tensor.
	 */
	Dimensions<_NRank> CalcContinuousAccessOrder() const
	{
		const auto Dims = Rank();
		if (Dims == 1)
			return Dimensions<_NRank>{};
		std::vector<std::pair<SizeType, SizeType>> Ret;
		Ret.reserve(Dims);
		for (SizeType i = 0; i < Dims; ++i)
			Ret.emplace_back(_MyViewStep[i], i);
		std::ranges::sort(Ret);
		std::ranges::reverse(Ret);
		Dimensions<_NRank> Rtn;
		size_t Index_ = 0;
		for (const auto& i : Ret | std::views::values)
			Rtn[Index_++] = i;
		return Rtn;
	}

	//********************************************************Check********************************************************//

	/**
	 * @brief Check if the tensor is enabled.
	 * @return True if the tensor is enabled, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsEnabled() const
	{
		return _MyData != nullptr;
	}

	/**
	 * @brief Check if the tensor is scalar.
	 * @return True if the tensor is scalar, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsScalar() const
	{
		return _MyShape.Size() == 1 && _MyShape[0] == 1;
	}

	/**
	 * @brief Check if the tensor is vector.
	 * @return True if the tensor is vector, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsVector() const
	{
		return _MyShape.Size() == 1;
	}

	/**
	 * @brief Check if the tensor is continuous in the specified range.
	 * @tparam _Begin start axis
	 * @tparam _End end axis
	 * @return True if the tensor is continuous, false otherwise.
	 */
	template <size_t _Begin = 0, size_t _End = _NRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsContinuous() const
	{
		for (size_t i = _Begin; i < _End; ++i)
			if (_MyViewStride[i] != 1 || _MyViewLeft[i] != 0)
				return false;
		for (size_t i = _Begin + 1; i < _End; ++i)
			if (_MyViewStep[i - 1] / _MyShape[i] != _MyViewStep[i])
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is continuous in the specified range.
	 * @param _Begin start axis
	 * @param _End end axis
	 * @return True if the tensor is continuous, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsContinuous(SizeType _Begin = 0, SizeType _End = _NRank) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcRange(_End, Rank());

		for (SizeType i = _Begin; i < _End; ++i)
			if (_MyViewStride[i] != 1 || _MyViewLeft[i] != 0)
				return false;
		for (SizeType i = _Begin + 1; i < _End; ++i)
			if (_MyViewStep[i - 1] / _MyShape[i] != _MyViewStep[i])
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is not sliced in the specified range.
	 * @tparam _Begin start axis
	 * @tparam _End end axis
	 * @return True if the tensor is not sliced, false otherwise.
	 */
	template <size_t _Begin = 0, size_t _End = _NRank>
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsNotSliced() const
	{
		for (size_t i = _Begin; i < _End; ++i)
			if (_MyViewStride[i] != 1 || _MyViewLeft[i] != 0)
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is not sliced in the specified range.
	 * @param _Begin start axis
	 * @param _End end axis
	 * @return True if the tensor is not sliced, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsNotSliced(SizeType _Begin = 0, SizeType _End = _NRank) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcRange(_End, Rank());

		for (SizeType i = _Begin; i < _End; ++i)
			if (_MyViewStride[i] != 1 || _MyViewLeft[i] != 0)
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is view.
	 * @return True if the tensor is view, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsView() const
	{
		return _MyData != (RawPointer)_MyFirst.get() || !IsContinuous<0, _NRank>();
	}

	/**
	 * @brief Check if the tensor is broadcasted.
	 * @return True if the tensor is broadcasted, false otherwise.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline bool IsBroadCasted() const
	{
		return IsBroadCasted_;
	}

private:

	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline void ThrowOnNotEnabled() const
	{
		if (!IsEnabled())
			_D_Dragonian_Lib_Fatal_Error;
	}

public:
	//*******************************************************Iterator*******************************************************//

	/**
	 * @brief Get the begining iterator of the tensor.
	 * @return The begining iterator of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		TensorIterator<
		_Impl_Dragonian_Lib_Conditional_t<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>
		, bool, ValueType>, _NRank> Begin() const
	{
		ThrowOnNotEnabled();
		using RetType = _Impl_Dragonian_Lib_Conditional_t<
			_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>
			, bool, ValueType>;
		return TensorIterator<RetType, _NRank>(
			(RetType*)_MyData, _MyShape.Data(), _MyViewStep.Data(), _MyViewLeft.Data(), _MyViewStride.Data()
		);
	}

	/**
	 * @brief Get the ending iterator of the tensor.
	 * @return The ending iterator of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		TensorIterator<
		_Impl_Dragonian_Lib_Conditional_t<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>
		, bool, ValueType>, _NRank> End() const
	{
		ThrowOnNotEnabled();
		return Begin() + _MyShape[0];
	}

	/**
	 * @brief Get the begining iterator of the tensor.
	 * @return The begining iterator of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		TensorIterator<
		_Impl_Dragonian_Lib_Conditional_t<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>
		, bool, ValueType>, _NRank> begin() const
	{
		return Begin();
	}

	/**
	 * @brief Get the ending iterator of the tensor.
	 * @return The ending iterator of the tensor.
	 */
	_D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline
		TensorIterator<
		_Impl_Dragonian_Lib_Conditional_t<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TensorType, bool>
		, bool, ValueType>, _NRank> end() const
	{
		return End();
	}

	/**
	 * @brief Add 1 to the indices of a loop iterator.
	 * @param _Indices The indices of the loop iterator.
	 * @return Reference of _Indices
	 */
	Dimensions<_NRank>& IteratorAdd(Dimensions<_NRank>& _Indices) const
	{
		auto Val = _Indices.Data() + _Indices.Size() - 1;
		const auto ShapePtr = _MyShape.Data();
		for (size_t i = _Indices.Size() - 1; ; --i)
		{
			const auto Ret = *Val + 1;
			if (Ret < *(ShapePtr + i))
			{
				*Val = Ret;
				return _Indices;
			}
			if (i == 0)
				return _Indices;
			*Val = 0;
			--Val;
		}
	}

	/**
	 * @brief Sub 1 to the indices of a loop iterator.
	 * @param _Indices The indices of the loop iterator.
	 * @return Reference of _Indices
	 */
	Dimensions<_NRank>& IteratorSub(Dimensions<_NRank>& _Indices) const
	{
		auto Val = _Indices.Data() + _Indices.Size() - 1;
		const auto ShapePtr = _MyShape.Data();

		for (size_t i = _Indices.Size() - 1; ; --i)
		{
			const auto Ret = *Val - 1;
			if (Ret >= 0)
			{
				*Val = Ret;
				return _Indices;
			}
			if (i == 0)
				return _Indices;
			*Val = (*(ShapePtr + i) - 1);
			--Val;
		}
	}

	/**
	 * @brief Transform the index which is negative to the positive index and check if it is out of range.
	 * @param _Index The index to transform.
	 * @param _Max The max index.
	 * @return The transformed index.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType CalcIndex(SizeType _Index, SizeType _Max)
	{
		if (_Index < 0)
			_Index += _Max;
		if (_Index >= _Max || _Index < 0)
			_D_Dragonian_Lib_Throw_Exception("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Transform the range index which is negative to the positive range index and check if it is out of range.
	 * @param _Index The index to transform.
	 * @param _Max The max index.
	 * @return The transformed index.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType CalcRange(SizeType _Index, SizeType _Max)
	{
		if (_Index < 0)
			_Index += _Max + 1;
		if (_Index > _Max || _Index < -1)
			_D_Dragonian_Lib_Throw_Exception("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Calculate the ceil of the division of two numbers.
	 * @param _Left The left number.
	 * @param _Right The right number.
	 * @return The ceil of the division.
	 */
	static _D_Dragonian_Lib_Member_Function_Constexpr_Force_Inline SizeType Ceil(SizeType _Left, SizeType _Right)
	{
		auto Mul = _Left / _Right;
		if (_Left > (Mul * _Right))
			++Mul;
		return Mul;
	}

	//*********************************************************View*********************************************************//

	/**
	 * @brief Slice the tensor, the order of the axes is ([0, 1, ... , N_DIMS - 1]).
	 * @param _SliceOptions A [[begin, step, end]/null, ...] array of all sliced axes, null means no slice.
	 * @return A sliced tensor(view).
	 */
	Tensor Slice(const SliceOptions<_NRank>& _SliceOptions) const
	{
		ThrowOnNotEnabled();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("Broad Casted Could Not Be Sliced!");
		if (_MyShape.Empty() || _SliceOptions.Size() > _MyShape.Size())
			_D_Dragonian_Lib_Throw_Exception("Axis Out Of Range!");

		Tensor Ret = View();
		for (size_t i = 0; i < _SliceOptions.Size(); ++i)
		{
			if (_SliceOptions[i].IsNone)
				continue;
			const auto SliceBeginPos = CalcIndex(_SliceOptions[i].Begin, _MyShape[i]);
			auto SliceEndPos = CalcRange(_SliceOptions[i].End, _MyShape[i]);
			const auto SliceLength = SliceEndPos - SliceBeginPos;
			if (SliceLength == 0)
				_D_Dragonian_Lib_Throw_Exception("Slice Length Must > 0");
			if (SliceLength > 0 && _SliceOptions[i].Step < 0 ||
				SliceLength < 0 && _SliceOptions[i].Step > 0)
				_D_Dragonian_Lib_Throw_Exception("Step And (SliceEnd - SliceBegin) Should Have The Same Sign!");
			Ret._MyViewLeft[i] += SliceBeginPos * Ret._MyViewStride[i];
			Ret._MyShape[i] = Ceil(abs(SliceLength), abs(_SliceOptions[i].Step));
			Ret._MyViewStride[i] *= _SliceOptions[i].Step;
		}
		return Ret;
	}

	/**
	 * @brief Slice the tensor, the order of the axes is reversed ([-1, -2, ... , -N_DIMS]).
	 * @param _SliceOptions A [[begin, end, step]/none, ...] array of all sliced axes, none means no slice.
	 * @return A sliced tensor(view).
	 */
	Tensor ReversedSlice(const SliceOptions<_NRank>& _SliceOptions) const
	{
		auto NewRange = _SliceOptions;
		std::ranges::reverse(NewRange);
		return Slice(NewRange);
	}

	/**
	 * @brief Permute the order of axes of a tensor, the order of original axes is ([0, 1, ... , N_DIMS - 1]). for example, we have a tensor with [N, H, C] shape, we can permute it to [N, C, H] shape with Permute([0, 2, 1])
	 * @param _PremuteOrder The new order of axes.
	 * @return A permuted tensor(view).
	 */
	Tensor Permute(const Dimensions<_NRank>& _PremuteOrder) const
	{
		ThrowOnNotEnabled();
		if (_MyShape.Empty() || _PremuteOrder.Size() != _MyShape.Size())
			_D_Dragonian_Lib_Throw_Exception("N_DIMS MisMatch!");
		Tensor Ret = View();
		Dimensions<_NRank> TransposedDims = _PremuteOrder;
		std::ranges::sort(TransposedDims);
		if (TransposedDims[0] != 0)
			_D_Dragonian_Lib_Throw_Exception("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");
		for (size_t i = 1; i < TransposedDims.Size(); ++i)
			if (TransposedDims[i] != TransposedDims[i - 1] + 1)
				_D_Dragonian_Lib_Throw_Exception("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");

		for (size_t i = 0; i < _PremuteOrder.Size(); ++i)
		{
			Ret._MyShape[i] = _MyShape[_PremuteOrder[i]];
			Ret._MyViewStep[i] = _MyViewStep[_PremuteOrder[i]];
			Ret._MyViewLeft[i] = _MyViewLeft[_PremuteOrder[i]];
			Ret._MyViewStride[i] = _MyViewStride[_PremuteOrder[i]];
		}
		return Ret;
	}

	/**
	 * @brief Transpose the tensor, swap the axes at the specified positions. for example, we have a tensor with [N, C, H] shape, we can transpose it with Transpose(1, 2) to get a tensor with [N, H, C] shape.
	 * @param _Axis1 The first axis.
	 * @param _Axis2 The second axis.
	 * @return A transposed tensor(view).
	 */
	Tensor Transpose(SizeType _Axis1, SizeType _Axis2) const
	{
		ThrowOnNotEnabled();
		const auto AxisCount = (SizeType)_MyShape.Size();
		_Axis1 = CalcIndex(_Axis1, AxisCount);
		_Axis2 = CalcIndex(_Axis2, AxisCount);
		Tensor Ret = View();
		if (_Axis1 == _Axis2)
			return Ret;
		Ret._MyShape[_Axis2] = _MyShape[_Axis1];
		Ret._MyViewStep[_Axis2] = _MyViewStep[_Axis1];
		Ret._MyViewLeft[_Axis2] = _MyViewLeft[_Axis1];
		Ret._MyViewStride[_Axis2] = _MyViewStride[_Axis1];
		Ret._MyShape[_Axis1] = _MyShape[_Axis2];
		Ret._MyViewStep[_Axis1] = _MyViewStep[_Axis2];
		Ret._MyViewLeft[_Axis1] = _MyViewLeft[_Axis2];
		Ret._MyViewStride[_Axis1] = _MyViewStride[_Axis2];
		return Ret;
	}

	/**
	 * @brief Unsqueeze the tensor, add a new axis at the specified position. for example, we have a tensor with [N, C, H] shape, we can unsqueeze it at the 1st axis with UnSqueeze(1) to get a tensor with [N, 1, C, H] shape.
	 * @param _Dim The specified position.
	 * @return An unsqueezed tensor(view).
	 */
	Tensor<_TensorType, _NRank + 1, _MyDevice> UnSqueeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		Tensor<_TensorType, _NRank + 1, _MyDevice> Ret;
		_Dim = CalcRange(_Dim, Rank());
		const auto _Value = _Dim == Rank() ? 1 : _MyViewStep[_Dim] * _MyShape[_Dim];
		Ret._MyShape = _MyShape.Insert(1, _Dim);
		Ret._MyViewStep = _MyViewStep.Insert(_Value, _Dim);
		Ret._MyViewLeft = _MyViewLeft.Insert(0, _Dim);
		Ret._MyViewStride = _MyViewStride.Insert(1, _Dim);
		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData;
		Ret._MyLast = _MyLast;
		Ret._MyFutures = _MyFutures;
		Ret.IsBroadCasted_ = IsBroadCasted_;
		return Ret;
	}

	/**
	 * @brief Squeeze the tensor, remove the axis with size 1 at the specified position. for example, we have a tensor with [N, 1, C, H] shape, we can squeeze it at the 1st axis with Squeeze(1) to get a tensor with [N, C, H] shape.
	 * @param _Dim The specified position.
	 * @return A squeezed tensor(view).
	 */
	template <size_t _TRank = _NRank>
	std::enable_if_t<(_TRank > 1) && _TRank == _NRank, Tensor<_TensorType, _NRank - 1, _MyDevice>> Squeeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		Tensor<_TensorType, _NRank - 1, _MyDevice> Ret;
		_Dim = CalcIndex(_Dim, SizeType(Ret._MyShape.Size()));
		if (Ret._MyShape[_Dim] != 1)
			_D_Dragonian_Lib_Throw_Exception("The Dim Must Be 1!");

		if (Ret._MyViewLeft[_Dim])
			_MyData += _MyViewLeft[_Dim] * _MyViewStep[_Dim];
		Ret._MyShape.Erase(Ret._MyShape.begin() + _Dim);
		Ret._MyViewStep.Erase(Ret._MyViewStep.begin() + _Dim);
		Ret._MyViewLeft.Erase(Ret._MyViewLeft.begin() + _Dim);
		Ret._MyViewStride.Erase(Ret._MyViewStride.begin() + _Dim);
		return Ret;
	}

	/**
	 * @brief Create a view of the tensor.
	 * @return A viewed tensor(view).
	 */
	Tensor View() const
	{
		return Tensor(*this);
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View([N, -1]) to get a tensor with [N, C * H * W] shape.
	 * @param _ViewShape The specified shape.
	 * @return A viewed tensor(view).
	 */
	template <size_t _TRank>
	Tensor<_TensorType, _TRank, _MyDevice> View(const Dimensions<_TRank>& _ViewShape) const
	{
		if (!IsContinuous())
			_D_Dragonian_Lib_Throw_Exception("View Should Be Continuous!");
		if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), -1) > 1)
			_D_Dragonian_Lib_Throw_Exception("Count Of Dynamic Axis Should <= 1!");
		for (const auto i : _ViewShape)
			if (i <= 0 && i != -1)
				_D_Dragonian_Lib_Throw_Exception("Count Of Size Should > 0 Or = -1 (Dynamic Axis)!");
		Tensor<_TensorType, _TRank, _MyDevice> Ret(_ViewShape);
		const auto SrcSize = Ret._MyShape.Multiply();
		const auto DstSize = _ViewShape.Multiply();
		if ((DstSize < 0 && (SrcSize % abs(DstSize)) != 0) || (DstSize > 0 && (SrcSize != DstSize)))
			_D_Dragonian_Lib_Throw_Exception("Size MisMatch!");
		const auto DynamicAxes = SrcSize / DstSize;

		Ret._MyShape = _ViewShape;
		for (auto& i : Ret._MyShape)
			if (i == -1)
			{
				i = abs(DynamicAxes);
				break;
			}
		Ret._MyViewStep.Resize(Ret._MyShape.Size());
		auto _Begin = Ret._MyViewStep.ReversedBegin();
		auto _End = Ret._MyViewStep.ReversedEnd();
		auto _Iter = Ret._MyShape.ReversedBegin();
		*_Begin-- = 1;
		while (_Begin != _End) *_Begin-- = *_Iter--;
		_MyViewLeft.AssignConstant(0ll);
		_MyViewStride.AssignConstant(1ll);
		return Ret;
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View(N, -1) to get a tensor with [N, C * H * W] shape.
	 * @tparam _Args The specified shape.
	 * @param _Shape0 The first shape.
	 * @param _Shape The rest shapes.
	 * @return A viewed tensor(view).
	 */
	template <typename... _Args>
	Tensor View(SizeType _Shape0, _Args... _Shape) const
	{
		Dimensions _ViewShape{ _Shape0, _Shape... };
		return View(_ViewShape);
	}

	/**
	 * @brief Clone this tensor, if the tensor is not continuous, make output continuous.
	 * @return New tensor.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> Clone() const
	{
		Tensor Ret{ this->_MyShape };
		Ret = *this;
		return Ret;
	}

	/**
	 * @brief If the tensor is not continuous, make output continuous.
	 * @return New tensor (view or clone).
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> Continuous() const
	{
		if (IsContinuous())
			return View();
		return Clone();
	}

	/**
	 * @brief Make this tensor continuous.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor&> MakeContinuous()
	{
		if (IsContinuous())
			return *this;
		return *this = Clone();
	}

	//********************************************************Operation********************************************************//

	template <size_t _UnfoldDim, size_t _UnfoldCount, typename InvokeFnType>
	static std::enable_if_t<_Impl_Dragonian_Lib_Is_Callable_v<InvokeFnType>> Invoke(Tensor& _Tensor, InvokeFnType _Fn)
	{
		const auto Parameter = _Tensor.GetDefaultOperatorParameter();
		auto Data = _Tensor.Data();
		auto Function = [=](int64_t _Index)
			{
				_Fn(Data + _Index);
			};
		Operators::SingleTensorLoop<_UnfoldDim, _UnfoldCount>(
			0,
			Parameter.Shape.Data(), Parameter.Begin.Data(),
			Parameter.ViewStep.Data(), Parameter.ViewLeft.Data(),
			Parameter.ViewStride.Data(), Function);
	}

	template <typename _CurValueType = ValueType>
	std::enable_if_t<
		_Impl_Dragonian_Lib_And_v<
		_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_CurValueType, ValueType>,
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_CurValueType, _CurValueType>&&
		std::is_copy_assignable_v<_CurValueType>>,
		Tensor> Gather(
			const Tensor& _Indices,
			SizeType _Axis = 0
		) const;

	template <typename _Type>
	std::enable_if_t<
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_Type, ValueType>&&
		_Impl_Dragonian_Lib_Could_Be_Converted_From_v<_Type, _Type>&&
		std::is_copy_assignable_v<_Type>,
		Tensor<_Type, _NRank, _MyDevice>> Cast() const
	{
		Eval();
		Tensor<_Type, _NRank, _MyDevice> Ret = Tensor<_Type, _NRank, _MyDevice>::New(_MyShape);
		Operators::OperatorsBase<_Type, _MyDevice>::template ImplCast<ValueType>
			(
				Ret.Data(),
				Ret.GetDefaultOperatorParameter(),
				Data(),
				GetDefaultOperatorParameter(),
				IsContinuous() && !IsBroadCasted()
			);
		return Ret;
	}

	/*

	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		const ValueType& _Val
	);

	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		const ValueType& _Val
	);

	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero
	);

	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero
	);

	static Tensor Repeat(
		const Tensor& _Input,
		const Vector<std::pair<SizeType, SizeType>>& _Repeat
	);

	static Tensor Stack(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0
	);

	static Tensor Cat(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0
	);

	static Tensor Gather(
		const Tensor& _Input,
		const Tensor& _Indices,
		SizeType _Axis = 0
	);

	static Tensor Sum(
		const Tensor& _Input,
		SizeType _Axis = 0
	);

	static Tensor CumSum(
		const Tensor& _Input,
		SizeType _Axis = 0
	);

	static Tensor Diff(
		const Tensor& _Input,
		SizeType _Axis = 0
	);

	static Tensor CumProd(
		const Tensor& _Input,
		SizeType _Axis = 0
	);

*/

/*DragonianLibTensorFnDef(Abs);
DragonianLibTensorFnDef(Sin);
DragonianLibTensorFnDef(Sinh);
DragonianLibTensorFnDef(Cos);
DragonianLibTensorFnDef(Cosh);
DragonianLibTensorFnDef(Tan);
DragonianLibTensorFnDef(Tanh);
DragonianLibTensorFnDef(ASin);
DragonianLibTensorFnDef(ACos);
DragonianLibTensorFnDef(ATan);
DragonianLibTensorFnDef(ASinh);
DragonianLibTensorFnDef(ACosh);
DragonianLibTensorFnDef(ATanh);
DragonianLibTensorFnDef(Exp);
DragonianLibTensorFnDef(Exp2);
DragonianLibTensorFnDef(Exp10);
DragonianLibTensorFnDef(Log);
DragonianLibTensorFnDef(Log2);
DragonianLibTensorFnDef(Log10);
DragonianLibTensorFnDef(Floor);
DragonianLibTensorFnDef(Ceil);
DragonianLibTensorFnDef(Round);*/
};

_D_Dragonian_Lib_Space_End