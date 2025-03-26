﻿/**
 * @file Tensor.h
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
 * @brief Tensor of DragonianLib
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once

#include <deque>
#include <ranges>
#include <mdspan>
#include "Operators.h"
#include "Libraries/Util/StringPreprocess.h"
#include "Libraries/Util/ThreadPool.h"

_D_Dragonian_Lib_Space_Begin

static inline constexpr SizeType RangeBeginPos = INT64_MAX; ///< Begin index
static inline constexpr SizeType RangeEndPos = INT64_MIN; ///< End index
static inline SizeType ZeroConstantVal = 0; ///< None index

namespace Functional
{
	class FunctionalImpl;
}

/**
 * @brief Struct representing a range with begin, step, and end values.
 */
struct Range
{
	SizeType Begin = 0; ///< Begin value
	SizeType Step = 1; ///< Step value
	SizeType End = RangeEndPos; ///< End value

	Range() = default;

	/**
	 * @brief Constructor for a none range.
	 * @param _NoneVal The none value to initialize the range.
	 */
	Range(NoneType _NoneVal) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const char* _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const wchar_t* _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const std::string& _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _RangeArgs The range arguments.
	 */
	Range(const std::wstring& _RangeArgs);

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End) {}

	/**
	 * @brief Constructor for a range with begin and end values.
	 * @param _Begin The begining value.
	 * @param _End The end value.
	 */
	Range(SizeType _Begin, SizeType _End) :Begin(_Begin), End(_End) {}

	/**
	 * @brief Constructor for a range with none and end values.
	 * @param _NoneVal The none value.
	 * @param _End The end value.
	 */
	Range(NoneType _NoneVal, SizeType _End) :End(_End) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin and none values.
	 * @param _Begin The begining value.
	 * @param _NoneVal The none value.
	 */
	Range(SizeType _Begin, NoneType _NoneVal) :Begin(_Begin) { UNUSED(_NoneVal); }

	/**
	 * @brief Reverse the range.
	 */
	void Reverse() { std::swap(Begin, End); Step = -Step; }

	std::string ToString() const
	{
		return "[" + std::to_string(Begin) + ":" + std::to_string(Step) + ":" +
			(End == RangeEndPos ? std::string("EndPos") : std::to_string(End)) +
			"]";
	}

	std::wstring ToWString() const
	{
		return L"[" + std::to_wstring(Begin) + L":" + std::to_wstring(Step) + L":" +
			(End == RangeEndPos ? std::wstring(L"EndPos") : std::to_wstring(End)) +
			L"]";
	}

	Range operator-() const { return { End, -Step, Begin }; }

	static Range Idx(SizeType Idx) { return { Idx, Idx, Idx }; }
};

namespace TypeTraits
{
	template <typename _Type>
	struct IsRange : std::false_type {};
	template <>
	struct IsRange<Range> : std::true_type {};
	template <typename _Type>
	constexpr bool IsRangeValue = IsRange<std::remove_cv_t<_Type>>::value;
}

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
 * @brief Set the task pool size.
 * @param _Size The size of the task pool.
 */
void SetTaskPoolSize(SizeType _Size);

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
template <typename _TensorType, size_t _NRank, Device _MyDevice>
class Tensor : public DlibValue
{
public:
	static_assert(_NRank > 0, "The rank of the tensor must be greater than 0!");

	template <typename _TensorType_, size_t _NRank_, Device _MyDevice_>
	friend class Tensor;
	friend class Functional::FunctionalImpl;
	using ValueType = std::remove_reference_t<_TensorType>;
	static_assert(!Operators::SimdTypeTraits::IsVectorizedValue<ValueType>, "Vectorized value type is not supported!");
	using Pointer = std::shared_ptr<void>;
	using RawPointer = ValueType*;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;
	static_assert(!TypeTraits::IsSameTypeValue<ValueType, _D_Dragonian_Lib_Namespace DlibValue>);

	using DependencyChainDataPointers = typename Operators::OperatorParameter<_NRank>::DependencyChainDataPointers;
	using DependencyChainPair = typename Operators::OperatorParameter<_NRank>::DependencyChainPair;
	using DependencyChainType = typename Operators::OperatorParameter<_NRank>::DependencyChainType;
	using DependencyChainPointer = typename Operators::OperatorParameter<_NRank>::DependencyChainPointer;
	static constexpr auto _Device = _MyDevice;
	static constexpr auto _DType = Type2TensorType<_TensorType>;
	using Allocator = TemplateLibrary::GetAllocatorType<_MyDevice>;

	auto CreateShared() const { return std::make_shared<Tensor<_TensorType, _NRank, _MyDevice>>(*this); }

	Tensor() = default;

	//Waiting for all the tasks which dependent on this tensor.
	void WaitingForTheInplaceLock() const
	{
		if (_MyFuturesAsArgument)
		{
			if (!_MyFuturesAsArgument->empty() && !Operators::GetInstantRunFlag())
				Operators::GetThreadPool().Notify(_MyFuturesAsArgument->size());
			while (!_MyFuturesAsArgument->empty())
			{
				_MyFuturesAsArgument->front().first.get();
				_MyFuturesAsArgument->pop_front();
			}
		}
	}
	//Waiting for all the tasks which change the data of this tensor. (waiting for the result)
	void WaitingForTheOperationLock() const
	{
		if (_MyFuturesAsResult)
		{
			if (!_MyFuturesAsResult->empty() && !Operators::GetInstantRunFlag())
				Operators::GetThreadPool().Notify(_MyFuturesAsResult->size());
			while (!_MyFuturesAsResult->empty())
			{
				_MyFuturesAsResult->front().first.get();
				_MyFuturesAsResult->pop_front();
			}
		}
	}
	//Check write permission.
	void WaitingAsResult() const
	{
		WaitingForTheInplaceLock();
		WaitingForTheOperationLock();
	}
	//Check read permission.
	void WaitingAsArgument() const
	{
		WaitingForTheOperationLock();
	}
	//Check read and write permission.
	void WaitingForAllLocks() const
	{
		WaitingForTheInplaceLock();
		WaitingForTheOperationLock();
	}

	template <typename _ThisType>
	decltype(auto) Evaluate(this _ThisType&& Self)
	{
		std::forward<_ThisType>(Self).WaitingForAllLocks();
		return std::forward<_ThisType>(Self);
	}

	template <typename _ValType = _TensorType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_ValType, _TensorType>>>
	decltype(auto) CastToString(bool _Fold = true) const
	{
		return CastToString(TotalSize(), _Fold);
	}

	template <typename _ValType = _TensorType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_ValType, _TensorType>>>
	decltype(auto) to_string() const
	{
		return CastToString(TotalSize());
	}

	template <typename _ValType = _TensorType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_ValType, _TensorType>>>
	decltype(auto) CastToWideString(bool _Fold = true) const
	{
		return UTF8ToWideString(CastToString(TotalSize()), _Fold);
	}

	TemplateLibrary::Vector<ValueType> ToVectorView() const
	{
		if (IsContinuous())
			return TemplateLibrary::Vector<ValueType>::CreateView(_MyData, TotalSize(), GetAllocator());
		_D_Dragonian_Lib_Throw_Exception("Could Not Convert Non-Continuous Tensor To Vector View!");
	}

	template <typename _Type2, size_t _Rank2, Device _Device2, typename = std::enable_if_t<_Rank2 <= _NRank>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BroadCast2AndCpy(const Tensor<_Type2, _Rank2, _Device2>& _Other) const
	{
		decltype(auto) Bd = BroadCast(*this, _Other, false);
		return Bd.first.Continuous();
	}

	template <typename _ThisType, typename _TFn, typename = std::enable_if_t<TypeTraits::IsInvokeableValue<_TFn>>>
	decltype(auto) AppendTask(this _ThisType&& _Self, _TFn&& _Fn)
	{
		DependencyChainDataPointers _DataPointer{ std::forward<_ThisType>(_Self)._MyFirst, nullptr, nullptr };
		if (std::forward<_ThisType>(_Self)._MyFuturesAsResult)
			std::forward<_ThisType>(_Self)._MyFuturesAsResult->emplace_back(
				Operators::GetTaskPool().Commit(std::forward<_TFn>(_Fn)), _DataPointer
			);
		else
			std::forward<_TFn>(_Fn)();
		return std::forward<_ThisType>(_Self);
	}

protected:
	Pointer _MyFirst = nullptr;
	RawPointer _MyLast = nullptr;
	RawPointer _MyData = nullptr;
	Dimensions<_NRank> _MyShape;
	Dimensions<_NRank> _MyViewStride;
	DependencyChainPointer _MyFuturesAsResult = nullptr;
	DependencyChainPointer _MyFuturesAsArgument = nullptr;
	Allocator _MyAllocator;
	bool _MyShapeIsBroadCasted = false;

private:

	template <typename _ValType = _TensorType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_ValType, _TensorType>>>
	decltype(auto) CastToString(SizeType _MyTotalSize, bool _Fold = true) const
	{
		if constexpr (_NRank > 1)
		{
			if (_MyShape.Front() > 10 && _MyTotalSize > 1000 && _Fold)
				return "[" +
				operator[](0).CastToString(_MyTotalSize) + ",\n" +
				operator[](1).CastToString(_MyTotalSize) + ",\n" +
				operator[](2).CastToString(_MyTotalSize) + "\n" +
				"[...]\n" +
				operator[](-3).CastToString(_MyTotalSize) + ",\n" +
				operator[](-2).CastToString(_MyTotalSize) + ",\n" +
				operator[](-1).CastToString(_MyTotalSize) + "]";
			std::string Ret = "[";
			for (SizeType i = 0; i < _MyShape.Front(); ++i)
				Ret += operator[](i).CastToString(_MyTotalSize) + ",\n";
			Ret.pop_back(); Ret.pop_back();
			Ret += "]";
			return Ret;
		}
		else
		{
			if (_MyShape.Front() > 10 && _MyTotalSize > 1000 && _Fold)
				return "[" +
				CvtToString(Get(0)) + ", " +
				CvtToString(Get(1)) + ", " +
				CvtToString(Get(2)) + ", ... , " +
				CvtToString(Get(-3)) + ", " +
				CvtToString(Get(-2)) + ", " +
				CvtToString(Get(-1)) +
				"]";
			std::string Ret = "[";
			for (SizeType i = 0; i < _MyShape.Front(); ++i)
				Ret += CvtToString(Get(i)) + ", ";
			Ret.pop_back(); Ret.pop_back();
			Ret += "]";
			return Ret;
		}
	}

	template <size_t _TmpTank = _NRank, typename = std::enable_if_t<(_TmpTank > 1) && _TmpTank == _NRank>>
		_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		ViewFirstDim(SizeType _Index) const
	{
		const auto Idx = CalcIndex(_Index, _MyShape.Front());
		Tensor<_TensorType, _TmpTank - 1, _MyDevice> Ret;

		Ret._MyShape.Assign(_MyShape.begin() + 1);
		Ret._MyViewStride.Assign(_MyViewStride.begin() + 1);

		auto Index = Idx * _MyViewStride.Front();
		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData + Index;
		Ret._MyLast = _MyLast;
		Ret._MyFuturesAsResult = _MyFuturesAsResult;
		Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
		Ret._MyShapeIsBroadCasted = _MyShapeIsBroadCasted;
		return Ret;
	}

	template <size_t _TRank, typename = std::enable_if_t<(_NRank > _TRank)>>
	constexpr decltype(auto) ViewDimensions(const Dimensions<_TRank>& _Indice) const

	{
		Tensor<_TensorType, _NRank - _TRank, _MyDevice> Ret;
		Ret._MyShape.Assign(_MyShape.begin() + _TRank);
		Ret._MyViewStride.Assign(_MyViewStride.begin() + _TRank);

		Ret._MyFirst = _MyFirst;
		Ret._MyData = Data(_Indice);
		Ret._MyLast = _MyLast;
		Ret._MyFuturesAsResult = _MyFuturesAsResult;
		Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
		Ret._MyShapeIsBroadCasted = _MyShapeIsBroadCasted;
		return Ret;
	}

	template <typename _Type1, typename _Type2, size_t _Rank1, size_t _Rank2, Device _Device1, Device _Device2>
	static std::pair<
		Tensor<_Type1, MaxOf(_Rank1, _Rank2), _Device1>,
		Tensor<_Type2, MaxOf(_Rank1, _Rank2), _Device2>
	> BroadCast(
		const Tensor<_Type1, _Rank1, _Device1>& _A,
		const Tensor<_Type2, _Rank2, _Device2>& _B,
		bool Inplace = false
	)
	{
		constexpr auto CurrentRank = MaxOf(_Rank1, _Rank2);
		std::pair<
			Tensor<_Type1, CurrentRank, _Device1>,
			Tensor<_Type2, CurrentRank, _Device2>
		> Ret{ {},{} };

		auto& First = Ret.first;		auto& Second = Ret.second;
		First._MyShape.AssignConstant(1);					Second._MyShape.AssignConstant(1);
		First._MyViewStride.AssignConstant(0);				Second._MyViewStride.AssignConstant(0);
		First._MyFirst = _A._MyFirst;							Second._MyFirst = _B._MyFirst;
		First._MyLast = _A._MyLast;								Second._MyLast = _B._MyLast;
		First._MyFuturesAsResult = _A._MyFuturesAsResult;		Second._MyFuturesAsResult = _B._MyFuturesAsResult;
		First._MyFuturesAsArgument = _A._MyFuturesAsArgument;	Second._MyFuturesAsArgument = _B._MyFuturesAsArgument;
		First._MyData = _A._MyData;								Second._MyData = _B._MyData;
		if constexpr (CurrentRank != _Rank1)
			First._MyShapeIsBroadCasted = true;
		if constexpr (CurrentRank != _Rank2)
			Second._MyShapeIsBroadCasted = true;

		for (size_t CurrentIndex = 0; CurrentIndex < CurrentRank; ++CurrentIndex)
		{
			//const auto i = CurrentRank - CurrentIndex - 1;
			const auto idx = CurrentRank - CurrentIndex - 1;
			auto XSize = 1ll, YSize = 1ll;
			if (CurrentIndex < _Rank1)
			{
				const auto i = _Rank1 - CurrentIndex - 1;
				First._MyShape[idx] = _A._MyShape[i];
				First._MyViewStride[idx] = _A._MyViewStride[i];
				XSize = _A._MyShape[i];
			}
			if (CurrentIndex < _Rank2)
			{
				const auto i = _Rank2 - CurrentIndex - 1;
				Second._MyShape[idx] = _B._MyShape[i];
				Second._MyViewStride[idx] = _B._MyViewStride[i];
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
				First._MyShapeIsBroadCasted = true;
			}
			else if (YSize == 1)
			{
				Second._MyShape[idx] = XSize;					Second._MyViewStride[idx] = 0;
				Second._MyShapeIsBroadCasted = true;
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

	template <typename _Type2, size_t _Rank2, Device _Device2, typename = std::enable_if_t<_Rank2 <= _NRank>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto)
		BroadCast(const Tensor<_Type2, _Rank2, _Device2>& _Other, bool Inplace = true) const
	{
		auto [_Self, _That] = BroadCast(*this, _Other, Inplace);
		return _That;
	}

public:
	Tensor(const Tensor& Left) = default;
	Tensor(Tensor&& Right) noexcept = default;
	constexpr Tensor& operator=(Tensor&& _Right) noexcept = default;

	template <size_t _TRank, typename = std::enable_if_t<(_NRank >= _TRank)>>
	Tensor& TensorAssign(const Tensor<ValueType, _TRank, _MyDevice>& _Left)
	{
		if constexpr (std::is_copy_assignable_v<ValueType>)
		{
			if ((const void*)this != (const void*)&_Left)
				Assign(_Left);
			return *this;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

	template <size_t _TRank, typename = std::enable_if_t<(_NRank >= _TRank)>>
	Tensor& Inplace(const Tensor<ValueType, _TRank, _MyDevice>& _Left)
	{
		return TensorAssign(_Left);
	}

	constexpr Tensor& operator=(const Tensor& _Left) = delete;

	/**
	 * @brief Assign the tensor with a scalar value.
	 * @param _Val The scalar value.
	 * @return Reference of this.
	 */
	Tensor& operator=(const ValueType& _Val)
	{
		if constexpr (std::is_copy_assignable_v<ValueType>)
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
	template <size_t _TmpTank = _NRank, typename = std::enable_if_t<(_TmpTank > 1) && _TmpTank == _NRank>>
	constexpr decltype(auto) operator[](SizeType _Index) const
	{
		return ViewFirstDim(_Index);
	}

	template <size_t _TmpTank = _NRank>
	_D_Dragonian_Lib_Constexpr_Force_Inline std::enable_if_t<
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
	constexpr decltype(auto) operator[](const SliceOptions<_NRank>& _SliceOptions) const
	{
		return Slice(_SliceOptions);
	}

	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 3D tensor, then tensor[{0, 0}] will return the 1st row of the 1st matrix of the tensor.
	 * @param _Indice
	 * @return
	 */
	template <size_t _TRank, typename = std::enable_if_t<(_NRank > _TRank)>>
	constexpr decltype(auto) operator[](const Dimensions<_TRank>& _Indice) const
	{
		return ViewDimensions(_Indice);
	}

	template <size_t _SliceDim = 0, typename _FirstType, typename ..._ArgTypes,
		typename = std::enable_if_t<(sizeof...(_ArgTypes) < _NRank) &&
		TypeTraits::IsIntegerValue<_FirstType> && (_SliceDim < _NRank)>>
		decltype(auto) operator()(_FirstType _Index, _ArgTypes ..._Args) const
	{
		if constexpr (TypeTraits::IsStringValue<_FirstType>)
			return operator() < _SliceDim > (Range(_Index), _Args...);
		else if constexpr (_SliceDim)
		{
			_Index = static_cast<_FirstType>(CalcIndex(static_cast<SizeType>(_Index), _MyShape[_SliceDim]));
			return operator() < _SliceDim > (Range{ static_cast<SizeType>(_Index), static_cast<SizeType>(_Index + 1) }, _Args...);
		}
		else if constexpr (_NRank == 1)
			return Get(static_cast<SizeType>(_Index));
		else if constexpr (sizeof...(_ArgTypes))
			return operator[](static_cast<SizeType>(_Index)).template operator() < 0 > (_Args...);
		else
			return operator[](static_cast<SizeType>(_Index));
	}

	template <size_t _SliceDim = 0, typename ..._ArgTypes,
		typename = std::enable_if_t<(sizeof...(_ArgTypes) < _NRank) && (_SliceDim < _NRank)>>
		decltype(auto) operator()(Range _Range, _ArgTypes ..._Args) const
	{
		SliceOptions<_NRank> SliceOptions;
		SliceOptions[_SliceDim] = _Range;
		if constexpr (sizeof...(_ArgTypes))
			return Slice(SliceOptions).template operator() < _SliceDim + 1 > (_Args...);
		else
			return Slice(SliceOptions);
	}

	//****************************************************Constructor****************************************************//

	/**
	 * @brief Create a new tensor with the specified shape.
	 * @param MyShape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_trivially_copy_assignable_v<_CurValueType> ||
		std::is_default_constructible_v<_CurValueType>)>>
	static constexpr decltype(auto) New(const Dimensions<_NRank>& MyShape)
	{
		return Tensor(MyShape);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_trivially_copy_assignable_v<_CurValueType> ||
		std::is_default_constructible_v<_CurValueType>)>>
		static constexpr decltype(auto) NewVector(SizeType MySize)
	{
		Dimensions<_NRank> MyShape;
		for (size_t i = 0; i < _NRank; ++i)
			MyShape[i] = 1;
		MyShape.Back() = MySize;
		return Tensor(MyShape);
	}

	template <typename _CurValueType = ValueType, typename _First, typename ..._Rest, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_constructible_v<_CurValueType, _First, _Rest...>>>
	static constexpr decltype(auto) New(const Dimensions<_NRank>& MyShape, _First&& Arg0, _Rest&& ...Args)
	{
		return Tensor(MyShape, std::forward<_First>(Arg0), std::forward<_Rest>(Args)...);
	}

	/**
	 * @brief Create an empty new tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_trivially_copy_assignable_v<_CurValueType>>>
	static constexpr decltype(auto) New()
	{
		return Tensor();
	}

	static constexpr decltype(auto) New(const Dimensions<_NRank>& MyShape, const Pointer& Buffer, size_t BufferSize)
	{
		return Tensor(MyShape, Buffer, BufferSize);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with ones.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(1)>>>
	static constexpr decltype(auto) Ones(const Dimensions<_NRank>& _Shape)
	{
		Tensor Ret(_Shape);
		Ret.Assign(ValueType(1));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with zeros.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(0)>>>
	static constexpr decltype(auto) Zeros(const Dimensions<_NRank>& _Shape)
	{
		Tensor Ret(_Shape);
		Ret.Assign(ValueType(0));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with a constant value.
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	static constexpr decltype(auto) ConstantOf(const Dimensions<_NRank>& _Shape, const ValueType& _Val)
	{
		Tensor Ret(_Shape);
		Ret.Assign(_Val);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	static constexpr decltype(auto) Rand(const Dimensions<_NRank>& _Shape, const ValueType& Min, const ValueType& Max)
	{
		Tensor Ret(_Shape);
		Ret.AssignRand(Min, Max);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	static constexpr decltype(auto) Randn(const Dimensions<_NRank>& _Shape, double _Mean = 0., double _Sigma = 1.)
	{
		Tensor Ret(_Shape);
		Ret.AssignRandn(_Mean, _Sigma);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with ones.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(1)>>>
	static constexpr decltype(auto) OnesLike(const Tensor& _ShapeReference)
	{
		return Ones(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with zeros.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(0)>>>
	static constexpr decltype(auto) ZerosLike(const Tensor& _ShapeReference)
	{
		return Zeros(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with a constant value.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	static constexpr decltype(auto) ConstantLike(const Tensor& _ShapeReference, const ValueType& _Val)
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
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	static constexpr decltype(auto) RandLike(const Tensor& _ShapeReference, const ValueType& Min, const ValueType& Max)
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
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	static constexpr decltype(auto) RandnLike(const Tensor& _ShapeReference, double _Mean = 0., double _Sigma = 1.)
	{
		return Randn(_ShapeReference.Shape(), _Mean, _Sigma);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_trivially_copy_assignable_v<_CurValueType>>>
	static constexpr decltype(auto) Empty(const Dimensions<_NRank>& _Shape)
	{
		return Tensor(_Shape);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_trivially_copy_assignable_v<_CurValueType>>>
	static constexpr decltype(auto) EmptyLike(const Tensor& _ShapeReference)
	{
		return Tensor(_ShapeReference._MyShape);
	}

	template <typename _CurValueType = ValueType, size_t _TRank = _NRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& _TRank == _NRank && _TRank == 1 && Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<ValueType>>>
	static constexpr decltype(auto) Arange(ValueType _Begin, ValueType _End, ValueType _Step)
	{
		if (_Step == ValueType(0))
			_D_Dragonian_Lib_Throw_Exception("Step Can't Be Zero!");
		auto _Count = static_cast<SizeType>((_End - _Begin) / _Step);
		if (_Count <= 0)
			_D_Dragonian_Lib_Throw_Exception("End Must Be Greater Than Begin!");
		if constexpr (TypeTraits::IsFloatingPointValue<ValueType>)
			if (std::isnan(_Count))
				_D_Dragonian_Lib_Throw_Exception("Invalid Range!");
		Tensor Ret = New({ _Count });
		Ret.WaitingAsResult();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplArange(
			Ret._MyData,
			Ret.GetDefaultOperatorParameter(),
			_Begin, _Step,
			!Ret.IsBroadCasted() && Ret.IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank = _NRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& _TRank == _NRank && _TRank == 1 && Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<ValueType>>>
	static constexpr decltype(auto) Linspace(ValueType _Begin, ValueType _End, size_t _Count, bool _EndPoint = false)
	{
		if (_EndPoint)
		{
			const auto Step = (_End - _Begin) / ValueType(_Count - 1);
			return Arange(_Begin, _End + (Step * 1.01), Step);
		}
		const auto Step = (_End - _Begin) / ValueType(_Count);
		return Arange(_Begin, _End + Step * 0.01, Step);
	}

	static constexpr Tensor FromBuffer(const Dimensions<_NRank>& MyShape, ValueType* Buffer, size_t BufferSize, Allocator Alloc)
	{
		return Tensor(MyShape, Buffer, BufferSize, Alloc);
	}

	static constexpr Tensor FromBuffer(const Dimensions<_NRank>& MyShape, ValueType* Buffer, size_t BufferSize)
	{
		return Tensor(MyShape, Buffer, BufferSize);
	}

	~Tensor() override = default;

private:
	_D_Dragonian_Lib_Constexpr_Force_Inline bool AllocateMemory(const Dimensions<_NRank>& MyShape, Allocator MyAlloc)
	{
		if (MyShape.Empty())
			return false;
		const auto Size = MyShape.Multiply();
		_MyFirst = Pointer(
			MyAlloc.allocate(std::max(Size * sizeof(ValueType), 256ull)),
			[MyAlloc, Size](void* _Pointer)
			{
				auto _DataPointer = static_cast<ValueType*>(_Pointer);
				TemplateLibrary::_Impl_Dragonian_Lib_Destroy_Range(_DataPointer, _DataPointer + Size);
				MyAlloc.deallocate(_Pointer);
			}
		);
		_MyData = (RawPointer)_MyFirst.get();
		_MyLast = _MyData + Size;
		return true;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline void ConstructViewInfo(const Dimensions<_NRank>& MyShape)
	{
		_MyShape = MyShape;
		auto _Begin = _MyViewStride.ReversedBegin();
		const auto _End = _MyViewStride.ReversedEnd();
		auto _Iter = _MyShape.ReversedBegin();
		*_Begin-- = 1;
		while (_Begin != _End)
		{
			*_Begin = *(_Begin + 1) * *_Iter--;
			--_Begin;
		}
	}

	Tensor(const Dimensions<_NRank>& MyShape) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType)
	{
		if (AllocateMemory(MyShape, Allocator()))
		{
			ConstructViewInfo(MyShape);
			if constexpr (!std::is_trivially_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
			{
				auto IterData = _MyData;
				while (IterData != _MyLast)
					TemplateLibrary::_Impl_Dragonian_Lib_Construct_At(*IterData++);
			}
		}
	}

	template <typename _First, typename ..._Rest>
	Tensor(const Dimensions<_NRank>& MyShape, _First&& Arg0, _Rest&& ...Args) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType)
	{
		if (AllocateMemory(MyShape, Allocator()))
		{
			ConstructViewInfo(MyShape);
			if constexpr (std::is_constructible_v<ValueType, _First, _Rest...>)
			{
				auto NewObj = ValueType(std::forward<_First>(Arg0), std::forward<_Rest>(Args)...);
				auto IterData = _MyData;
				while (IterData != _MyLast)
					TemplateLibrary::_Impl_Dragonian_Lib_Construct_At(*IterData++, NewObj);
			}
		}
	}

	Tensor(const Dimensions<_NRank>& MyShape, ValueType* Buffer, size_t BufferSize, Allocator Alloc) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType)
	{
		auto TSize = static_cast<size_t>(MyShape.Multiply());
		if (MyShape.Empty())
			return;
		if (BufferSize < TSize)
			_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
		if (BufferSize > TSize)
			_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
		_MyFirst = Pointer(
			Buffer,
			[Alloc](void* _Pointer) { Alloc.deallocate(_Pointer); }
		);
		_MyData = (RawPointer)_MyFirst.get();
		_MyLast = _MyData + BufferSize;
		ConstructViewInfo(MyShape);
	}

	Tensor(const Dimensions<_NRank>& MyShape, ValueType* Buffer, size_t BufferSize) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType)
	{
		auto TSize = static_cast<size_t>(MyShape.Multiply());
		if (MyShape.Empty())
			return;
		if (BufferSize < TSize)
			_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
		if (BufferSize > TSize)
			_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
		_MyFirst = Pointer(
			Buffer,
			[](void*) {}
		);
		_MyData = (RawPointer)_MyFirst.get();
		_MyLast = _MyData + BufferSize;
		ConstructViewInfo(MyShape);
	}

	Tensor(const Dimensions<_NRank>& MyShape, const Pointer& Buffer, size_t BufferSize) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType)
	{
		auto TSize = static_cast<size_t>(MyShape.Multiply());
		if (MyShape.Empty())
			return;
		if (BufferSize < TSize)
			_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
		if (BufferSize > TSize)
			_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
		_MyFirst = Buffer;
		_MyData = (RawPointer)_MyFirst.get();
		_MyLast = _MyData + BufferSize;
		ConstructViewInfo(MyShape);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(const ValueType& _Value)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		WaitingAsResult();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignScalar(
			_MyData,
			GetDefaultOperatorParameter(),
			_Value,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(const ValueType* _Buffer, SizeType _Count)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		if (_Count != ElementCount())
			_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
		WaitingAsResult();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignBuffer(
			_MyData,
			GetDefaultOperatorParameter(),
			_Buffer,
			_Count,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_move_assignable_v<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) MoveAssign(const ValueType* _Buffer, SizeType _Count)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		if(_Count != ElementCount())
			_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
		WaitingAsResult();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplMoveBuffer(
			_MyData,
			GetDefaultOperatorParameter(),
			_Buffer,
			_Count,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(const Tensor<ValueType, _TRank, _MyDevice>& _Val)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		_Val.WaitingAsArgument();
		if (_Val.IsScalar())
			return Assign(_Val.Item());
		WaitingAsResult();
		Tensor BroadCasted = BroadCast(_Val);
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignTensor(
			_MyData,
			GetDefaultOperatorParameter(),
			BroadCasted.Data(),
			BroadCasted.GetDefaultOperatorParameter(),
			!IsBroadCasted() && !BroadCasted.IsBroadCasted() && IsContinuous() && BroadCasted.IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) AssignRand(const ValueType& Min, const ValueType& Max)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		WaitingAsResult();
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRand(
			_MyData,
			GetDefaultOperatorParameter(),
			Min, Max,
			!IsBroadCasted() && IsContinuous()
		);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) AssignRandn(double _Mean = 0., double _Sigma = 1.)
	{
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("You Can't Assign To a BroadCasted Tensor!");
		WaitingAsResult();
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
	static _D_Dragonian_Lib_Constexpr_Force_Inline
		SizeType GetAlignSize()
	{
		return alignof(ValueType);
	}

	/**
	 * @brief Get the device of the tensor.
	 * @return The device of the tensor.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline
		Device GetDevice()
	{
		return _Device;
	}

	/**
	 * @brief Get the allocator of the tensor.
	 * @return The allocator of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) GetAllocator() const
	{
		return _MyAllocator;
	}

	/**
	 * @brief Get the buffer of the tensor.
	 * @return The buffer of the tensor.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Buffer(this _ThisType&& Self)
	{
		return std::forward<_ThisType>(Self)._MyFirst;
	}

	/**
	 * @brief Get the data pointer of the tensor.
	 * @return The data pointer of the tensor.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Data(this _ThisType&& Self)
	{
		return std::forward<_ThisType>(Self)._MyData;
	}

	/**
	 * @brief Get the data pointer of the tensor with the specified indices.
	 * @return The data pointer of the tensor.
	 */
	template <size_t _TRank, typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Data(this _ThisType&& Self, const Dimensions<_TRank>& _Indices)
	{
		static_assert(_TRank <= _NRank, "The rank of the indices must be less than or equal to the rank of the tensor!");
		SizeType Index = 0;
		for (size_t i = 0; i < _Indices.Size(); ++i)
		{
			const SizeType Idx = CalcIndex(_Indices[i], std::forward<_ThisType>(Self)._MyShape[i]);
			Index += (Idx * std::forward<_ThisType>(Self)._MyViewStride[i]);
		}
		return std::forward<_ThisType>(Self)._MyData + Index;
	}

	/**
	 * @brief Get a val of the tensor with the specified index.
	 * @return The val.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Get(this _ThisType&& Self, SizeType Index)
	{
		return *(std::forward<_ThisType>(Self).template Data<1>({ Index }));
	}

	/**
	 * @brief Get a val of the tensor with the specified indices.
	 * @return The val.
	 */
	template <size_t _TRank, typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Item(this _ThisType&& Self, const Dimensions<_TRank>& _Indices)
	{
		return *(std::forward<_ThisType>(Self).Data(_Indices));
	}

	/**
	 * @brief Get the first val of the tensor.
	 * @return The val.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Item(this _ThisType&& Self)
	{
		return *std::forward<_ThisType>(Self)._MyData;
	}

	/**
	 * @brief Get the pointer of the first val of the tensor.
	 * @return The pointer.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) ItemPointer(this _ThisType&& Self)
	{
		return std::forward<_ThisType>(Self)._MyData;
	}

	decltype(auto) GetShared()const
	{
		auto Shared = _MyFirst;
		return Shared;
	}

	//******************************************************Operator******************************************************//

	/**
	 * @brief Assign the tensor with ones.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(1)>>>
	decltype(auto) FixOnes()
	{
		Assign(ValueType(1));
		return *this;
	}

	/**
	 * @brief Assign the tensor with zeros.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_constructible_v<_CurValueType, decltype(0)>>>
	decltype(auto) FixZeros()
	{
		Assign(ValueType(0));
		return *this;
	}

	/**
	 * @brief Assign the tensor with a constant value.
	 * @param _Val The constant value.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	decltype(auto) Fix(const ValueType& _Val)
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
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	decltype(auto) Fix(const ValueType* _Buffer, SizeType _Count)
	{
		Assign(_Buffer, _Count);
		return *this;
	}

	/**
	 * @brief Fix the tensor with a buffer (move).
	 * @param _Buffer The buffer.
	 * @param _Count Data count of the buffer.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_move_assignable_v<_CurValueType>>>
	decltype(auto) MoveFix(const ValueType* _Buffer, SizeType _Count)
	{
		MoveAssign(_Buffer, _Count);
		return *this;
	}

	/**
	 * @brief Assign the tensor with random values.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	decltype(auto) RandFix(const ValueType& Min = ValueType(0), const ValueType& Max = ValueType(1))
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
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::IsArithmeticValue<_CurValueType>>>
	decltype(auto) RandnFix(double _Mean = 0., double _Sigma = 1.)
	{
		AssignRandn(_Mean, _Sigma);
		return *this;
	}

	//*************************************************Binary Operator*************************************************//

	_D_Dragonian_Lib_Operator_Binary_Function_Define(Add);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Add, +, (Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Sub);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Sub, -, (Operators::BinaryOperators::SubBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Mul);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Mul, *, (Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Div);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Div, / , (Operators::BinaryOperators::DivBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Mod);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Mod, %, (Operators::BinaryOperators::ModBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(And);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(And, &&, (Operators::BinaryOperators::AndBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Or);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Or, || , (Operators::BinaryOperators::OrBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Xor);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Xor, ^, (Operators::BinaryOperators::XorBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(LShift);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(LShift, << , (Operators::BinaryOperators::LShiftBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(RShift);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(RShift, >> , (Operators::BinaryOperators::RShiftBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(BinaryOr);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(BinaryOr, | , (Operators::BinaryOperators::BinaryOrBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));
	_D_Dragonian_Lib_Operator_Binary_Function_Define(BinaryAnd);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(BinaryAnd, &, (Operators::BinaryOperators::BinaryAndBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType> && (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)));

	_D_Dragonian_Lib_Operator_Compare_Function_Define(Equal);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Equal, == , (Operators::ComparisonOperators::EqualBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	_D_Dragonian_Lib_Operator_Compare_Function_Define(NotEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(NotEqual, != , (Operators::ComparisonOperators::NotEqualBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	_D_Dragonian_Lib_Operator_Compare_Function_Define(Greater);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Greater, > , (Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	_D_Dragonian_Lib_Operator_Compare_Function_Define(Less);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Less, < , (Operators::ComparisonOperators::LessBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	_D_Dragonian_Lib_Operator_Compare_Function_Define(GreaterEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(GreaterEqual, >= , (Operators::ComparisonOperators::GreaterEqualBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	_D_Dragonian_Lib_Operator_Compare_Function_Define(LessEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(LessEqual, <= , (Operators::ComparisonOperators::LessEqualBinary::HasOperatorValue<_CurValueType>&& TypeTraits::IsSameTypeValue<_CurValueType, ValueType>));
	
	_D_Dragonian_Lib_Operator_Binary_Function_Define(Pow);

	//****************************************************Unary Operator****************************************************//

	_D_Dragonian_Lib_Operator_Unary_Function_Define(Sqrt);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(RSqrt);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Reciprocal);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Abs);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Sin);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Cos);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Tan);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ASin);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ACos);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ATan);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Sinh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Cosh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Tanh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ASinh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ACosh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(ATanh);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Exp);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Exp2);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Log);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Log2);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Log10);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Ceil);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Floor);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Round);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Trunc);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Frac);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Negative);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(BitwiseNot);
	_D_Dragonian_Lib_Operator_Unary_Function_Define(Not);

	template <typename _CurValueType = ValueType,
		typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::NegativeUnary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>>
	decltype(auto) operator-() const
	{
		return Negative();
	}

	template <typename _CurValueType = ValueType,
		typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::NotUnary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>>
	decltype(auto) operator!() const
	{
		return Not();
	}

	template <typename _CurValueType = ValueType,
		typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& Operators::UnaryOperators::BitwiseNotUnary::HasOperatorValue<_CurValueType>&& (std::is_copy_assignable_v<_CurValueType> || std::is_move_assignable_v<_CurValueType>)>>
	decltype(auto) operator~() const
	{
		return BitwiseNot();
	}

	//*********************************************************Info*********************************************************//

	/**
	 * @brief Get the shape info of the tensor.
	 * @tparam _Begin The start axis.
	 * @tparam _End The end axis.
	 * @return The shape info of the tensor.
	 */
	template <size_t _Begin = 0, size_t _End = _NRank>
	Operators::OperatorParameter<_End - _Begin> GetDefaultOperatorParameter(bool _CheckIsContinuous = false) const
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
		Ret.ResultDependency = _MyFuturesAsResult;
		Ret.ArgumentDependency = _MyFuturesAsArgument;
		Ret.Data = _MyFirst;
		return Ret;
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline const Dimensions<_NRank>& Shape() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Shape(SizeType _Index) const
	{
		_Index = CalcIndex(_Index, static_cast<SizeType>(_NRank));
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline const Dimensions<_NRank>& Size() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the total size of the tensor.
	 * @return The total size of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType TotalSize() const
	{
		return _MyShape.Multiply();
	}

	/**
	 * @brief Get the element count of the tensor.
	 * @return The element count of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType ElementCount() const
	{
		return _MyShape.Multiply();
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index The index of the axis.
	 * @return The shape of the specified axis of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Size(SizeType _Index) const
	{
		_Index = CalcIndex(_Index, static_cast<SizeType>(_NRank));
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the rank of the tensor.
	 * @return The rank of the tensor.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType Rank()
	{
		return _NRank;
	}

	/**
	 * @brief Get the strides of the tensor.
	 * @return The strides of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline const Dimensions<_NRank>& ViewStrides() const
	{
		return _MyViewStride;
	}

	/**
	 * @brief Whether the tensor is empty (null).
	 * @return true if the tensor is empty, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool Empty() const
	{
		return _MyFirst == nullptr;
	}

	/**
	 * @brief Whether the tensor is null (empty).
	 * @return true if the tensor is null, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool Null() const
	{
		return _MyData == nullptr;
	}

	/**
	 * @brief Reset the tensor to null.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline void Clear()
	{
		_MyFirst = nullptr;
		_MyData = nullptr;
		_MyLast = nullptr;
		_MyShapeIsBroadCasted = false;
		_MyFuturesAsResult = nullptr;
		_MyFuturesAsArgument = nullptr;
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
			Ret.emplace_back(_MyViewStride[i], i);
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
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsEnabled() const
	{
		return _MyData != nullptr;
	}

	/**
	 * @brief Check if the tensor is scalar.
	 * @return True if the tensor is scalar, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsScalar() const
	{
		return _MyShape.Size() == 1 && _MyShape[0] == 1;
	}

	/**
	 * @brief Check if the tensor is vector.
	 * @return True if the tensor is vector, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsVector() const
	{
		return _MyShape.Size() == 1;
	}

	/**
	 * @brief Check if the tensor is continuous in the specified range.
	 * @return True if the tensor is continuous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContinuous() const
	{
		if (IsBroadCasted())
			return false;

		if (_MyViewStride[_NRank - 1] != 1)
			return false;

		const auto Diff = _MyData - (const ValueType*)_MyFirst.get();
		for (SizeType i = 1; i < _NRank; ++i)
			if (_MyViewStride[i - 1] / _MyShape[i] != _MyViewStride[i] || Diff % _MyShape[i])
				return false;
		return true;
		
	}

	/**
	 * @brief Check if the tensor is continuous in the specified range.
	 * @return True if the tensor is continuous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContinuous(SizeType _Begin, SizeType _End) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcIterator(_End, Rank());

		const auto Diff = _MyData - (const ValueType*)_MyFirst.get();
		for (SizeType i = _Begin; i < _End; ++i)
			if (_MyViewStride[i - 1] / _MyShape[i] != _MyViewStride[i] || Diff % _MyShape[i])
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is not sliced.
	 * @return True if the tensor is not sliced, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool NotSliced(SizeType _Begin, SizeType _End) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcIterator(_End, Rank());

		const auto Diff = _MyData - (const ValueType*)_MyFirst.get();
		for (SizeType i = _Begin; i < _End; ++i)
			if (Diff % _MyShape[i])
				return false;
		return true;
	}

	/**
	 * @brief Check if the tensor is view.
	 * @return True if the tensor is view, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsView() const
	{
		return _MyData != (RawPointer)_MyFirst.get() || !IsContinuous();
	}

	/**
	 * @brief Check if the tensor is broadcasted.
	 * @return True if the tensor is broadcasted, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsBroadCasted() const
	{
		return _MyShapeIsBroadCasted;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsBroadCasted_(SizeType _Begin = 0, SizeType _End = _NRank) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcIterator(_End, Rank());

		for (SizeType i = _Begin; i < _End; ++i)
			if (!_MyViewStride[i])
				return true;

		return false;
	}

private:

	_D_Dragonian_Lib_Constexpr_Force_Inline void ThrowOnNotEnabled() const
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
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) Begin() const
	{
		ThrowOnNotEnabled();
		return TensorIterator<ValueType, _NRank>{_MyData, _MyShape.Data(), _MyViewStride.Data()};
	}

	/**
	 * @brief Get the ending iterator of the tensor.
	 * @return The ending iterator of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) End() const
	{
		return Begin() + _MyShape[0];
	}

	/**
	 * @brief Get the begining iterator of the tensor.
	 * @return The begining iterator of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) begin() const
	{
		return Begin();
	}

	/**
	 * @brief Get the ending iterator of the tensor.
	 * @return The ending iterator of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline
		decltype(auto) end() const
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
	 * @return The transformed index. (0 ~ (Max - 1))
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcIndex(SizeType _Index, SizeType _Max)
	{
		if (_Index < 0)
			_Index += _Max;
		if (_Index >= _Max || _Index < 0)
			_D_Dragonian_Lib_Throw_Exception("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Transform the range index which is negative to the positive end index and check if it is out of range.
	 * this function has reverse condition, so the index should be in the range of [-1, Max].
	 * e.g. if a container has 5 elements[|(begin)|v1|v2|v3|v4|v5|(end)],
	 * it has 7 index positions[-1(begin)0(v1)1(v2)2(v3)3(v4)4(v5)5(end)](as '|'),
	 * _Max means size of the container(5), _Index means the index position(-5 ~ 4 | RangeEndPos | RangeBeginPos),
	 * if _Index is RangeEndPos, it means the end of the container(5(end)),
	 * if _Index is RangeBeginPos, it means the end of the reversed container(-1(begin)),
	 * if _Index is negative, it means the position from the end of the element(4(v5)),
	 * @param _Index The index to transform.
	 * @param _Max The max index.
	 * @return The transformed index. (-1 ~ Max)
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcEndPos(SizeType _Index, SizeType _Max)
	{
		if (_Index == RangeEndPos)
			return _Max;
		if (_Index == RangeBeginPos)
			return -1;
		if (_Index == _Max)
			return _Max;
		return CalcIndex(_Index, _Max);
	}

	/**
	 * @brief Transform the iterator index which is negative to the positive index and check if it is out of range,
	 * this function has no reverse condition, so the index should be in the range of [0, Max].
	 * e.g. if a container has 5 elements[|v1|v2|v3|v4|v5|end],
	 * it has 6 iterator positions[0(v1)1(v2)2(v3)3(v4)4(v5)5(end)](as '|'),
	 * _Max means size of the container(5), _Index means the iterator position(-6 ~ 5),
	 * if _Index is negative, it means the position from the end of the container,
	 * so -1 means back of v5(5), -2 means back of v4(4), and so on.
	 * @param _Index The index to transform.
	 * @param _Max The max index.
	 * @return The transformed index. (0 ~ Max)
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcIterator(SizeType _Index, SizeType _Max)
	{
		if (_Index == RangeEndPos)
			return _Max;
		if (_Index == _Max)
			return _Max;

		if (_Index < 0)
			_Index += _Max + 1;
		if (_Index > _Max || _Index < 0)
			_D_Dragonian_Lib_Throw_Exception("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Calculate the ceil of the division of two numbers.
	 * @param _Left The left number.
	 * @param _Right The right number.
	 * @return The ceil of the division.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType Ceil(SizeType _Left, SizeType _Right)
	{
		auto Div = _Left / _Right;
		if (_Left > (Div * _Right))
			++Div;
		return Div;
	}

	//*********************************************************View*********************************************************//

	/**
	 * @brief Slice the tensor, the order of the axes is ([0, 1, ... , N_DIMS - 1]).
	 * @param _SliceOptions A [[begin, step, end]/null, ...] array of all sliced axes, null means no slice.
	 * @return A sliced tensor(view).
	 */
	decltype(auto) Slice(const SliceOptions<_NRank>& _SliceOptions) const
	{
		ThrowOnNotEnabled();
		if (IsBroadCasted())
			_D_Dragonian_Lib_Throw_Exception("Broad Casted Could Not Be Sliced!");
		if (_MyShape.Empty() || _SliceOptions.Size() > _MyShape.Size())
			_D_Dragonian_Lib_Throw_Exception("Axis Out Of Range!");

		Tensor Ret = View();
		for (size_t i = 0; i < _SliceOptions.Size(); ++i)
		{
			if (_SliceOptions[i].Begin == 0 && _SliceOptions[i].Step == 1 &&
				(_SliceOptions[i].End == RangeEndPos || _SliceOptions[i].End == _MyShape[i]))
				continue;

			SizeType SliceBeginPos, SliceStep, SliceEndPos;

			if (_SliceOptions[i].Begin == _SliceOptions[i].Step && _SliceOptions[i].Begin == _SliceOptions[i].End)
			{
				SliceBeginPos = CalcIndex(_SliceOptions[i].Step, _MyShape[i]);
				SliceStep = 1;
				SliceEndPos = SliceBeginPos + 1;
			}
			else
			{
				SliceStep = _SliceOptions[i].Step;
				if (SliceStep == 0)
					_D_Dragonian_Lib_Throw_Exception("SliceStep Should Not Be Zero!");
				SliceBeginPos = CalcIndex(_SliceOptions[i].Begin, _MyShape[i]);
				SliceEndPos = CalcEndPos(_SliceOptions[i].End, _MyShape[i]);
			}

			const auto SliceLength = SliceEndPos - SliceBeginPos;
			if (SliceLength == 0)
				_D_Dragonian_Lib_Throw_Exception("(SliceEnd - SliceBegin) Should Not Be Zero!");
			if (SliceLength / SliceStep < 0)
				_D_Dragonian_Lib_Throw_Exception("Step Error!");
			const auto SlicedShape = Ceil(abs(SliceLength), abs(SliceStep));
			if (SlicedShape < 0)
				_D_Dragonian_Lib_Throw_Exception("Step And (SliceEnd - SliceBegin) Should Have The Same Sign!");

			Ret._MyData += SliceBeginPos * Ret._MyViewStride[i];
			Ret._MyShape[i] = SlicedShape;
			Ret._MyViewStride[i] *= SliceStep;
		}
		return Ret;
	}

	/**
	 * @brief Slice the tensor, the order of the axes is reversed ([-1, -2, ... , -N_DIMS]).
	 * @param _SliceOptions A [[begin, end, step]/none, ...] array of all sliced axes, none means no slice.
	 * @return A sliced tensor(view).
	 */
	decltype(auto) ReversedSlice(const SliceOptions<_NRank>& _SliceOptions) const
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
	decltype(auto) Permute(const Dimensions<_NRank>& _PremuteOrder) const
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
			Ret._MyViewStride[i] = _MyViewStride[_PremuteOrder[i]];
		}
		return Ret;
	}

	/**
	 * @brief Permute the order of axes of a tensor, the order of original axes is ([0, 1, ... , N_DIMS - 1]). for example, we have a tensor with [N, H, C] shape, we can permute it to [N, C, H] shape with Permute([0, 2, 1])
	 * @param _Order The new order of axes.
	 * @return A permuted tensor(view).
	 */
	template <typename... _Args, typename = std::enable_if_t<sizeof...(_Args) == _NRank>>
	decltype(auto) Permute(_Args... _Order) const
	{
		return Permute(Dimensions<_NRank>{_Order...});
	}

	/**
	 * @brief Transpose the tensor, swap the axes at the specified positions. for example, we have a tensor with [N, C, H] shape, we can transpose it with Transpose(1, 2) to get a tensor with [N, H, C] shape.
	 * @param _Axis1 The first axis.
	 * @param _Axis2 The second axis.
	 * @return A transposed tensor(view).
	 */
	decltype(auto) Transpose(SizeType _Axis1 = -1, SizeType _Axis2 = -2) const
	{
		ThrowOnNotEnabled();
		const auto AxisCount = (SizeType)_MyShape.Size();
		_Axis1 = CalcIndex(_Axis1, AxisCount);
		_Axis2 = CalcIndex(_Axis2, AxisCount);
		Tensor Ret = View();
		if (_Axis1 == _Axis2)
			return Ret;
		Ret._MyShape[_Axis2] = _MyShape[_Axis1];
		Ret._MyViewStride[_Axis2] = _MyViewStride[_Axis1];
		Ret._MyShape[_Axis1] = _MyShape[_Axis2];
		Ret._MyViewStride[_Axis1] = _MyViewStride[_Axis2];
		return Ret;
	}

	decltype(auto) AxisFromTo(SizeType _Begin = -2, SizeType _End = -1) const
	{
		ThrowOnNotEnabled();
		const auto AxisCount = (SizeType)_MyShape.Size();
		_Begin = CalcIndex(_Begin, AxisCount);
		_End = CalcIterator(_End, AxisCount);
		if (_Begin > _End)
			_D_Dragonian_Lib_Throw_Exception("Begin Should Not Be Greater Than End!");
		Tensor Ret = View();
		for (SizeType i = _Begin; i < _End - 1; ++i)
		{
			std::swap(Ret._MyShape[i], Ret._MyShape[i + 1]);
			std::swap(Ret._MyViewStride[i], Ret._MyViewStride[i + 1]);
		}
		return Ret;
	}

	/**
	 * @brief Unsqueeze the tensor, add a new axis at the specified position. for example, we have a tensor with [N, C, H] shape, we can unsqueeze it at the 1st axis with UnSqueeze(1) to get a tensor with [N, 1, C, H] shape.
	 * @param _Dim The specified position.
	 * @return An unsqueezed tensor(view).
	 */
	decltype(auto) UnSqueeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		Tensor<_TensorType, _NRank + 1, _MyDevice> Ret;
		_Dim = CalcIterator(_Dim, Rank());
		const auto _Value = _Dim == Rank() ? 1 : _MyViewStride[_Dim] * _MyShape[_Dim];
		Ret._MyShape = _MyShape.Insert(1, _Dim);
		Ret._MyViewStride = _MyViewStride.Insert(_Value, _Dim);
		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData;
		Ret._MyLast = _MyLast;
		Ret._MyFuturesAsResult = _MyFuturesAsResult;
		Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
		Ret._MyShapeIsBroadCasted = Ret.IsBroadCasted_();
		return Ret;
	}

	/**
	 * @brief Squeeze the tensor, remove the axis with size 1 at the specified position. for example, we have a tensor with [N, 1, C, H] shape, we can squeeze it at the 1st axis with Squeeze(1) to get a tensor with [N, C, H] shape.
	 * @param _Dim The specified position.
	 * @return A squeezed tensor(view).
	 */
	template <size_t _TRank = _NRank, typename = std::enable_if_t<_TRank == _NRank>>
	decltype(auto) Squeeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		if constexpr (_TRank == 1)
			return View();
		else
		{
			_Dim = CalcIndex(_Dim, SizeType(_MyShape.Size()));
			if (_MyShape[_Dim] != 1)
				_D_Dragonian_Lib_Throw_Exception("The Shape Of Dim Must Be 1!");
			Tensor<_TensorType, _NRank - 1, _MyDevice> Ret;
			Ret._MyShape = _MyShape.Erase(_Dim);
			Ret._MyViewStride = _MyViewStride.Erase(_Dim);
			Ret._MyFirst = _MyFirst;
			Ret._MyData = _MyData;
			Ret._MyLast = _MyLast;
			Ret._MyFuturesAsResult = _MyFuturesAsResult;
			Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
			Ret._MyShapeIsBroadCasted = Ret.IsBroadCasted_();
			return Ret;
		}
	}

	/**
	 * @brief Create a view of the tensor.
	 * @return A viewed tensor(view).
	 */
	decltype(auto) View() const
	{
		return Tensor(*this);
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View([N, -1]) to get a tensor with [N, C * H * W] shape.
	 * @param _ViewShape The specified shape.
	 * @return A viewed tensor(view).
	 */
	template <size_t _TRank>
	decltype(auto) View(const Dimensions<_TRank>& _ViewShape) const
	{
		if (!IsContinuous())
			_D_Dragonian_Lib_Throw_Exception("View Should Be Continuous!");
		if (std::ranges::count(_ViewShape, -1) > 1)
			_D_Dragonian_Lib_Throw_Exception("Count Of Dynamic Axis Should Be <= 1!");
		for (const auto i : _ViewShape)
			if (i <= 0 && i != -1)
				_D_Dragonian_Lib_Throw_Exception("Count Of Size Should Be Greater Than 0 Or Equal -1 (Dynamic Axis)!");

		const auto SrcSize = _MyShape.Multiply();
		const auto DstSize = std::abs(_ViewShape.Multiply());

		const auto Remainder = SrcSize % DstSize;
		const auto DynamicAxes = SrcSize / DstSize;

		if (Remainder)
			_D_Dragonian_Lib_Throw_Exception("Could Not View The Tensor With Size["
				+ std::to_string(SrcSize) + "] To Size[" + std::to_string(DstSize) + "]!");

		Tensor<_TensorType, _TRank, _MyDevice> Ret;
		Ret._MyShape = _ViewShape;
		if (DynamicAxes >= 1)
			*std::ranges::find(Ret._MyShape, -1) = DynamicAxes;
		auto _Begin = Ret._MyViewStride.ReversedBegin();
		const auto _End = Ret._MyViewStride.ReversedEnd();
		auto _Iter = Ret._MyShape.ReversedBegin();
		*_Begin-- = 1;
		while (_Begin != _End)
		{
			*_Begin = *(_Begin + 1) * *_Iter--;
			--_Begin;
		}

		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData;
		Ret._MyLast = _MyLast;
		Ret._MyFuturesAsResult = _MyFuturesAsResult;
		Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
		Ret._MyShapeIsBroadCasted = Ret.IsBroadCasted_();

		return Ret;
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View(N, -1) to get a tensor with [N, C * H * W] shape.
	 * @tparam _Args The specified shape.
	 * @param _Shape The shapes.
	 * @return A viewed tensor(view).
	 */
	template <typename... _Args>
	decltype(auto) View(_Args... _Shape) const
	{
		Dimensions<sizeof...(_Args)> _ViewShape{ _Shape... };
		return View(_ViewShape);
	}

	/**
	 * @brief Reverse the tensor along the specified axis.
	 * @param _Axis The specified axis.
	 * @return A viewed tensor(view).
	 */
	decltype(auto) Reverse(SizeType _Axis = 0) const
	{
		_Axis = CalcIndex(_Axis, Rank());
		auto Ret = View();
		Ret._MyData += (_MyShape[_Axis] - 1) * _MyViewStride[_Axis];
		Ret._MyViewStride[_Axis] = -_MyViewStride[_Axis];
		return Ret;
	}

	/**
	 * @brief Clone this tensor, if the tensor is not continuous, make output continuous.
	 * @return New tensor.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Clone() const
	{
		auto Ret = New(_MyShape);
		Ret.TensorAssign(*this);
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	decltype(auto) Clone(Tensor<_CurValueType, _TRank, _MyDevice>& _Buffer) const
	{
		_Buffer.TensorAssign(*this);
		return _Buffer;
	}

	/**
	 * @brief If the tensor is not continuous, make output continuous.
	 * @return New tensor (view or clone).
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Continuous() const
	{
		if (IsContinuous())
			return View();
		return Clone();
	}

	/**
	 * @brief If the tensor is not contiguous, make output contiguous.
	 * @return New tensor (view or clone).
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Contiguous() const
	{
		return Continuous();
	}

	template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>>>
	decltype(auto) Continuous(Tensor<_CurValueType, _TRank, _MyDevice>& _Buffer) const
	{
		if (IsContinuous())
			return *this;
		return Clone(_Buffer);
	}

	/**
	 * @brief Make this tensor continuous.
	 * @return Reference of this.
	 */
	template <typename _CurValueType = ValueType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) MakeContinuous()
	{
		if (IsContinuous())
			return *this;
		return *this = Clone();
	}

	template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) ReShape(const Dimensions<_TRank>& _ViewShape) const
	{
		if (IsContinuous())
			return View(_ViewShape);
		return Clone().View(_ViewShape);
	}

	template <typename _CurValueType = ValueType, size_t _TRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) ReShape(const Dimensions<_TRank>& _ViewShape, Tensor<_CurValueType, _TRank, _MyDevice>& _Buffer) const
	{
		if (IsContinuous())
			return View(_ViewShape);
		return Clone(_Buffer).View(_ViewShape);
	}

	template <typename _CurValueType = ValueType, typename... _Args, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) ReShape(_Args... _Shape) const
	{
		Dimensions<sizeof...(_Args)> _ViewShape{ _Shape... };
		return ReShape(_ViewShape);
	}

	//********************************************************Operation********************************************************//

	template <size_t _UnfoldDim, size_t _UnfoldCount, typename InvokeFnType, typename = std::enable_if_t<TypeTraits::IsCallableValue<InvokeFnType>>>
	static decltype(auto) Invoke(Tensor& _Tensor, const InvokeFnType& _Fn)
	{
		const auto Parameter = _Tensor.GetDefaultOperatorParameter();
		auto Data = _Tensor.Data();
		const auto ShapeInfo = Parameter.Shape.Data();
		const auto BeginInfo = Parameter.Begin.Data();
		const auto ViewStrideInfo = Parameter.ViewStride.Data();
		auto Function = [=](int64_t _Index)
			{
				_Fn(Data + _Index, ShapeInfo, ViewStrideInfo);
			};
		Operators::SingleTensorLoop<_UnfoldDim, _UnfoldCount>(0, ShapeInfo, BeginInfo, ViewStrideInfo, Function);
	}

	template <SizeType _Axis = 0, typename _CurValueType = ValueType, typename _IndexType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&  TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank)) != -1 && std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Gather(const Tensor<_IndexType, _NRank, _MyDevice>& _Indices) const
	{
		for (size_t i = 0; i < _NRank; ++i)
			if (i != _Axis && _MyShape[i] != _Indices.Shape()[i])
				_D_Dragonian_Lib_Throw_Exception("Shape Mismatch!");

		_Indices.WaitingAsArgument();
		WaitingAsArgument();
		constexpr auto _Dim = TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank));
		auto Ret = New(_Indices.Shape());
		Ret.WaitingAsResult();
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplGather<_IndexType, _NRank, _Dim>
			(
				Ret.Data(),
				Ret.GetDefaultOperatorParameter(),
				Data(),
				GetDefaultOperatorParameter(),
				_Indices.Data(),
				_Indices.GetDefaultOperatorParameter()
			);
		return Ret;
	}

	template <SizeType _Axis = 0, typename _CurValueType = ValueType, typename _IndexType, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank)) != -1 && std::is_copy_assignable_v<_CurValueType>>>
	decltype(auto) Gather(const Tensor<_IndexType, _NRank, _MyDevice>& _Indices, Tensor<_IndexType, _NRank, _MyDevice>& _Buffer)
	{
		for (size_t i = 0; i < _NRank; ++i)
			if ((i != _Axis && _MyShape[i] != _Indices.Shape()[i]) || (_Buffer.Shape()[i] != _Indices.Shape()[i]))
				_D_Dragonian_Lib_Throw_Exception("Shape Mismatch!");

		_Indices.WaitingAsArgument();
		WaitingAsArgument();
		constexpr auto _Dim = TypeTraits::BTCalcIndex(_Axis, SizeType(_NRank));
		_Buffer.WaitingAsResult();
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplGather<_IndexType, _NRank, _Dim>
			(
				_Buffer.Data(),
				_Buffer.GetDefaultOperatorParameter(),
				Data(),
				GetDefaultOperatorParameter(),
				_Indices.Data(),
				_Indices.GetDefaultOperatorParameter()
			);
		return _Buffer;
	}

	template <typename _Type, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>>>
	decltype(auto) Cast() const
	{
		WaitingAsArgument();
		Tensor<_Type, _NRank, _MyDevice> Ret = Tensor<_Type, _NRank, _MyDevice>::New(_MyShape);
		Ret.WaitingAsResult();
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

	template <typename _Type, typename = std::enable_if_t<TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>>>
	decltype(auto) Cast(Tensor<_Type, _NRank, _MyDevice>& _Buffer) const
	{
		WaitingAsArgument();
		_Buffer.WaitingAsResult();
		auto BroadCasted = _Buffer.Broadcast(*this);
		Operators::OperatorsBase<_Type, _MyDevice>::template ImplCast<ValueType>
			(
				_Buffer.Data(),
				_Buffer.GetDefaultOperatorParameter(),
				BroadCasted.Data(),
				BroadCasted.GetDefaultOperatorParameter(),
				BroadCasted.IsContinuous() && !BroadCasted.IsBroadCasted() && _Buffer.IsContinuous()
			);
		return _Buffer;
	}

	template <typename _CurValueType = ValueType, size_t _TRank = _NRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& _TRank <= _NRank && std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Padding(
		const IDLArray<Range, _TRank>& _PaddingCount,
		PaddingType _Type,
		std::optional<ValueType> _Val = std::nullopt
	) const
	{
		auto Shape = _MyShape;
		SliceOptions<_NRank> NewTensorSlice;
		for (size_t i = 0; i < _TRank; ++i)
		{
			if (_PaddingCount[i].Begin == 0 && _PaddingCount[i].Step == 1 && _PaddingCount[i].End == RangeEndPos)
				continue;
			if ((_PaddingCount[i].Begin) < 0 || (_PaddingCount[i].End) < 0)
				_D_Dragonian_Lib_Throw_Exception("Padding Should Not Be Negative, Using Slice Instead!");
			Shape[i] += _PaddingCount[i].Begin + _PaddingCount[i].End;
			NewTensorSlice[i].Begin = _PaddingCount[i].Begin;
			NewTensorSlice[i].End = _PaddingCount[i].Begin + _MyShape[i];
		}
		
		auto Ret = New(Shape);

		Ret.WaitingAsResult();
		Ret[NewTensorSlice].TensorAssign(*this);
		if (_Type == PaddingType::Zero)
		{
			_Type = PaddingType::Constant;
			if (TypeTraits::CouldBeConvertedFromValue<ValueType, SizeType>)
				_Val = ValueType(ZeroConstantVal);
		}

		for (size_t i = _TRank - 1; i < _TRank; --i)
		{
			if (!(_PaddingCount[i].Begin == 0 && _PaddingCount[i].Step == 1 && _PaddingCount[i].End == RangeEndPos))
			{
				SliceOptions<_NRank> RngFront, RngBack;
				SliceOptions<_NRank> SrcFront, SrcBack;

				if (_Type == PaddingType::Constant)
				{
					if (_Val.has_value())
					{
						if (_PaddingCount[i].Begin > 0)
						{
							RngFront[i] = { 0, _PaddingCount[i].Begin };
							Ret[RngFront].Assign(_Val.value());
						}
						if (_PaddingCount[i].End > 0)
						{
							RngBack[i] = { _MyShape[i] + _PaddingCount[i].Begin, RangeEndPos };
							Ret[RngBack].Assign(_Val.value());
						}
					}
					else
						_D_Dragonian_Lib_Throw_Exception("Constant Padding Should Have A Value!");
				}
				else if (_Type == PaddingType::Replicate)
				{
					if (_PaddingCount[i].Begin > 0)
					{
						RngFront[i] = { 0, _PaddingCount[i].Begin };
						SrcFront[i] = { _PaddingCount[i].Begin, _PaddingCount[i].Begin + 1 };
						Ret[RngFront].Assign(Ret[SrcFront]);
					}
					if (_PaddingCount[i].End > 0)
					{
						RngBack[i] = { _MyShape[i] + _PaddingCount[i].Begin, RangeEndPos };
						SrcBack[i] = { _MyShape[i] + _PaddingCount[i].Begin - 1, _MyShape[i] + _PaddingCount[i].Begin };
						Ret[RngBack].Assign(Ret[SrcBack]);
					}
				}
				else if (_Type == PaddingType::Cicular)
				{
					if (_PaddingCount[i].Begin > 0)
					{
						auto PaddingPos = _PaddingCount[i].Begin;
						const auto ConstantPaddingPos = PaddingPos;
						RngFront[i] = { 0, 0 };
						while (PaddingPos)
						{
							const auto _ThisCount = PaddingPos < _MyShape[i] ? PaddingPos : _MyShape[i];
							const auto _Remainder = _MyShape[i] - _ThisCount;
							SrcFront[i] = { ConstantPaddingPos + _Remainder, ConstantPaddingPos + _Remainder + _ThisCount };
							RngFront[i].End = RngFront[i].Begin + _ThisCount;
							Ret[RngFront].Assign(Ret[SrcFront]);
							RngFront[i].Begin += _ThisCount;
							PaddingPos -= _ThisCount;
						}
					}
					if (_PaddingCount[i].End > 0)
					{
						auto PaddingPos = _PaddingCount[i].End;
						const auto ConstantPaddingPos = _PaddingCount[i].Begin;
						RngBack[i] = { _MyShape[i] + ConstantPaddingPos, 0 };
						while (PaddingPos)
						{
							const auto _ThisCount = PaddingPos < _MyShape[i] ? PaddingPos : _MyShape[i];
							SrcBack[i] = { ConstantPaddingPos, ConstantPaddingPos + _ThisCount };
							RngBack[i].End = RngBack[i].Begin + _ThisCount;
							Ret[RngBack].Assign(Ret[SrcBack]);
							RngBack[i].Begin += _ThisCount;
							PaddingPos -= _ThisCount;
						}
					}
				}
				else if (_Type == PaddingType::Reflect)
				{
					if (_PaddingCount[i].Begin >= _MyShape[i] || _PaddingCount[i].End >= _MyShape[i])
						_D_Dragonian_Lib_Throw_Exception("Reflect Padding Should Not Be Greater Than The Shape!");
					if (_PaddingCount[i].Begin > 0)
					{
						RngFront[i] = { 0, _PaddingCount[i].Begin };
						SrcFront[i] = { _PaddingCount[i].Begin * 2, -1, _PaddingCount[i].Begin };
						Ret[RngFront].Assign(Ret.Slice(SrcFront));
					}
					if (_PaddingCount[i].End > 0)
					{
						RngBack[i] = { _MyShape[i] + _PaddingCount[i].Begin, RangeEndPos };
						SrcBack[i] = { RngBack[i].Begin - 2, -1, RngBack[i].Begin - _PaddingCount[i].End - 2 };
						Ret[RngBack].Assign(Ret.Slice(SrcBack));
					}
				}
			}
		}
		return Ret;
	}

	template <typename _CurValueType = ValueType, size_t _TRank = _NRank, typename = std::enable_if_t<TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& _TRank <= _NRank && std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Pad(
		const IDLArray<Range, _TRank>& _PaddingCount,
		PaddingType _Type,
		std::optional<ValueType> _Val = std::nullopt
	) const
	{
		IDLArray<Range, _NRank> PaddingC;
		for (size_t i = 0; i < _TRank; ++i)
			PaddingC[_NRank - 1 - i] = _PaddingCount[i];
		return Padding(PaddingC, _Type, std::move(_Val));
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Repeat(const IDLArray<SizeType, _NRank>& _Repeat) const
	{
		IDLArray<Range, _NRank> _PaddingCount;
		for (size_t i = 0; i < _NRank; ++i)
		{
			if (_Repeat[i] <= 1)
				continue;
			_PaddingCount[i].End = (_Repeat[i] - 1) * _MyShape[i];
		}
		return Padding(_PaddingCount, PaddingType::Cicular);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_copy_assignable_v<_CurValueType>&& std::is_default_constructible_v<_CurValueType>>>
	decltype(auto) Repeat(SizeType _Axis, SizeType _Repeat) const
	{
		IDLArray<Range, _NRank> _PaddingCount;
		_Axis = CalcIndex(_Axis, Rank());
		_PaddingCount[_Axis].End = (_Repeat - 1) * _MyShape[_Axis];
		return Padding(_PaddingCount, PaddingType::Cicular);
	}

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) Sum(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(Sum, Sum);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) Prod(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(Prod, Prod);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::DivBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) Mean(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(Mean, Mean);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ReduceMax(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(ReduceMax, Max);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ReduceMin(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(ReduceMin, Min);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::UnaryOperators::LogUnary::HasOperatorValue<_CurValueType>>>
	decltype(auto) LogSum(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(LogSum, LogSum);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::UnaryOperators::ExpUnary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::UnaryOperators::LogUnary::HasOperatorValue<_CurValueType>>>
	decltype(auto) LogSumExp(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(LogSumExp, LogSumExp);

	template <typename RetType = Int32, bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ArgMax(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(ArgMax, ArgMax, RetType);

	template <typename RetType = Int32, bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ArgMin(SizeType _Axis) const _D_Dragonian_Lib_Operator_Reduce_Function_Body(ArgMin, ArgMin, RetType);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumSum(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumSum);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::SubBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumSub(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumSub);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumProd(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumProd);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::DivBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumDiv(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumDiv);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumMax(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumMax);

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) CumMin(SizeType _Axis) const _D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumMin);

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ReduceLp(SizeType _Axis,const ValueType& _P) const
	{
		if constexpr (_NRank == 1)
			return UnSqueeze(0).template LpNorm<false>(-1, _P).Squeeze(0);
		else
		{
			_Axis = CalcIndex(_Axis, Rank());
			auto TensorTmp = AxisFromTo(_Axis, -1);
			TensorTmp.WaitingAsArgument();
			Dimensions<_NRank - 1> OutShape;
			OutShape.Assign(TensorTmp.Shape().Data());
			auto Ret = Tensor<_TensorType, _NRank - 1, _MyDevice>::New(OutShape);
			Ret.WaitingAsResult();
			auto RetView = Ret.UnSqueeze(-1);
			Operators::OperatorsBase<ValueType, _MyDevice>::template ImplReduceLpScalar
			(
				RetView.Data(),
				RetView.GetDefaultOperatorParameter(),
				TensorTmp.Data(),
				TensorTmp.GetDefaultOperatorParameter(),
				_P,
				RetView.IsContinuous() && TensorTmp.IsContinuous()
			);
			if constexpr (KeepDim)
				return RetView;
			else
				return Ret;
		}
	}

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ReduceL1(SizeType _Axis) const
	{
		return ReduceLp<KeepDim>(_Axis, 1);
	}

	template <bool KeepDim = false, typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<_CurValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<_CurValueType>>>
	decltype(auto) ReduceL2(SizeType _Axis) const
	{
		return ReduceLp<KeepDim>(_Axis, 2);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t <TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&& std::is_default_constructible_v<_CurValueType>&& Operators::BinaryOperators::SubBinary::HasOperatorValue<_CurValueType>>>
	decltype(auto) Diff(SizeType _Axis) const
	{
		if constexpr (_NRank == 1)
			return UnSqueeze(0).Diff(-1).Squeeze(0);
		else
		{
			_Axis = CalcIndex(_Axis, Rank());
			if (Shape()[_Axis] == 1)
				return View();
			auto TensorTmp = AxisFromTo(_Axis, -1);
			TensorTmp.WaitingAsArgument();
			auto OutShape = Shape();
			OutShape[_Axis] -= 1;
			auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(OutShape);
			auto ResView = Ret.AxisFromTo(_Axis, -1);
			ResView.WaitingAsResult();
			Operators::OperatorsBase<ValueType, _MyDevice>::ImplDiffUnary
			(
				ResView.Data(),
				ResView.GetDefaultOperatorParameter(),
				TensorTmp.Data(),
				TensorTmp.GetDefaultOperatorParameter(),
				ResView.IsContinuous() && TensorTmp.IsContinuous()
			);
			return Ret;
		}
	}

	template <
		Operators::InterpolateMode _Mode = Operators::InterpolateMode::Nearest,
		typename _CurValueType = ValueType,
		typename = std::enable_if_t <
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>&&
		Operators::BinaryOperators::SubBinary::HasOperatorValue<_CurValueType>&&
		Operators::BinaryOperators::AddBinary::HasOperatorValue<_CurValueType>&&
		Operators::BinaryOperators::MulBinary::HasOperatorValue<_CurValueType
		>>>
	decltype(auto) Interpolate(const Dimensions<Operators::GetInterpolateModeRank<_Mode>>& _Dims, Operators::InterpolateParam<_Mode> _InterpParams) const
	{
		using ParamsType = Operators::InterpolateParam<_Mode>;

		auto OutShape = Shape();
		if (_InterpParams._MyScale.has_value())
		{
			if (!_InterpParams._MySize.has_value())
				_InterpParams._MySize = ParamsType::SizeTypeArrayT();
			auto& Scales = _InterpParams._MyScale.value();
			auto& Sizes = _InterpParams._MySize.value();
			for (size_t i = 0; i < _Dims.Size(); ++i)
			{
				const auto Axis = CalcIndex(_Dims[i], Rank());
				if (Scales[i] <= 0)
					_D_Dragonian_Lib_Throw_Exception("Scale Should Be Greater Than 0!");
				OutShape[Axis] = std::max(static_cast<SizeType>(double(OutShape[Axis]) * Scales[i]), 1ll);
				Sizes[i] = OutShape[Axis];
			}
		}
		else if (_InterpParams._MySize.has_value())
		{
			if (!_InterpParams._MyScale.has_value())
				_InterpParams._MyScale = ParamsType::DoubleArrayT();
			auto& Scales = _InterpParams._MyScale.value();
			auto& Sizes = _InterpParams._MySize.value();
			for (size_t i = 0; i < _Dims.Size(); ++i)
			{
				const auto Axis = CalcIndex(_Dims[i], Rank());
				if (Sizes[i] <= 0)
					_D_Dragonian_Lib_Throw_Exception("Size Should Be Greater Than 0!");
				Scales[i] = double(OutShape[Axis]) / double(Shape()[Axis]);
				OutShape[Axis] = Sizes[i];
			}
		}

		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(OutShape);
		auto RetView = Ret.View();
		auto MyView = View();
		for (size_t i = 0; i < _Dims.Size(); ++i)
		{
			const auto Axis = CalcIndex(_Dims[i], Rank());
			RetView = RetView.AxisFromTo(Axis, -1);
			MyView = MyView.AxisFromTo(Axis, -1);
		}
		
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplInterpolate<_Mode, _NRank>
			(
				RetView.Data(),
				RetView.GetDefaultOperatorParameter(),
				MyView.Data(),
				MyView.GetDefaultOperatorParameter(),
				_InterpParams,
				RetView.IsContinuous() && MyView.IsContinuous()
			);

		return Ret;
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&&
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>>>
		decltype(auto) ClampMin(ValueType _Min) const
	{
		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(_MyShape);
		Ret.WaitingAsResult();
		WaitingAsArgument();
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplMaxScalar
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Min,
			Ret.IsContinuous() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&&
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>>>
		decltype(auto) ClampMax(ValueType _Max) const
	{
		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(_MyShape);
		Ret.WaitingAsResult();
		WaitingAsArgument();
		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplMinScalar
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Max,
			Ret.IsContinuous() && IsContinuous()
		);
		return Ret;
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&&
		Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&&
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>>>
		decltype(auto) Clamp(ValueType _Min, ValueType _Max) const
	{
		return ClampMin(_Min).ClampMax(_Max);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&&
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>>>
		decltype(auto) Min(ValueType _Min) const
	{
		return ClampMax(_Min);
	}

	template <typename _CurValueType = ValueType, typename = std::enable_if_t<Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&&
		TypeTraits::IsSameTypeValue<_CurValueType, ValueType>&&
		std::is_default_constructible_v<_CurValueType>>>
		decltype(auto) Max(ValueType _Max) const
	{
		return ClampMin(_Max);
	}

	/*

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
};

template <typename _TensorType = float, Device _MyDevice = Device::CPU, size_t _NRank = 1>
using ITensor = Tensor<_TensorType, _NRank, _MyDevice>;

_D_Dragonian_Lib_Space_End