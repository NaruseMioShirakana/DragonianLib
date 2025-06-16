/**
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
#include "TensorLib/Include/Base/Tensor/Impl/Util.h"
#include "TensorLib/Include/Base/Tensor/Impl/AutoGrad.h"

#include "TensorLib/Include/Base/Tensor/Operators.h"

#include "TensorLib/Include/Base/Tensor/Operators/User/Binary.h"
#include "TensorLib/Include/Base/Tensor/Operators/User/Unary.h"

_D_Dragonian_Lib_Space_Begin

namespace Functional
{
	class FunctionalImpl;
}

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
#pragma region type_traits
public:
	static_assert(_NRank > 0, "The rank of the tensor must be greater than 0!");
	static_assert(!TypeTraits::IsReferenceValue<_TensorType>, "The value type of the tensor could not be a reference type!");

	template <typename _TensorType_, size_t _NRank_, Device _MyDevice_>
	friend class Tensor;
	friend class Functional::FunctionalImpl;

	using ValueType = std::remove_reference_t<_TensorType>;
	static_assert(!Operators::SimdTypeTraits::IsVectorizedValue<ValueType>, "Vectorized value type is not supported!");
	static_assert(!TypeTraits::IsSameTypeValue<ValueType, _D_Dragonian_Lib_Namespace DlibValue>);

	using Pointer = std::shared_ptr<void>;
	using RawPointer = ValueType*;
	using ConstRawPointer = const ValueType*;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;

	using DependencyChainDataPointers = Operators::DependencyChainTypes::DependencyChainDataPointers;
	using DependencyChainPair = Operators::DependencyChainTypes::DependencyChainPair;
	using DependencyChainType = Operators::DependencyChainTypes::DependencyChainType;
	using DependencyChainPointer = Operators::DependencyChainTypes::DependencyChainPointer;
	static constexpr auto _Device = _MyDevice;
	static constexpr auto _DType = Type2TensorType<_TensorType>;
	static constexpr auto _MyRank = _NRank;
	using Allocator = TemplateLibrary::GetAllocatorType<_MyDevice>;

	~Tensor() override = default;
#pragma endregion

#pragma region field
protected:
	Pointer _MyFirst = nullptr;
	RawPointer _MyLast = nullptr;
	RawPointer _MyData = nullptr;
	Dimensions<_NRank> _MyShape;
	Dimensions<_NRank> _MyViewStride;
	DependencyChainPointer _MyFuturesAsResult = nullptr;
	DependencyChainPointer _MyFuturesAsArgument = nullptr;
	Allocator _MyAllocator;
	std::shared_ptr<bool> _IgnoreDep = nullptr;
	AutoGrad::Function* _MyFunction = nullptr;
	AutoGrad::Graph* _MyGraph = nullptr;
#pragma endregion

#pragma region default
public:
	Tensor() = default;
	Tensor(std::nullopt_t) : Tensor() {}
	Tensor(NoneType) : Tensor() {}
	Tensor(const Tensor& Left) = default;
	Tensor(Tensor&& Right) noexcept = default;

	bool operator==(std::nullopt_t) const { return Null(); }
	bool operator!=(std::nullopt_t) const { return !Null(); }

	bool operator==(NoneType) const { return Null(); }
	bool operator!=(NoneType) const { return !Null(); }

	//operator bool() const { return !Null(); } ///< Implicit conversion to bool

	TemplateLibrary::Vector<ValueType> ToVectorView() const
	{
		if (IsContiguous())
			return TemplateLibrary::Vector<ValueType>::CreateView(_MyData, TotalSize(), GetAllocator());
		_D_Dragonian_Lib_Throw_Exception("Could Not Convert Non-Contiguous Tensor To Vector View!");
	}
	template<typename _ThisType>
	decltype(auto) operator*(this _ThisType&& _Self)
	{
		return *(std::forward<_ThisType>(_Self)._MyData);
	}
#pragma endregion

#pragma region constructor
private:
	_D_Dragonian_Lib_Constexpr_Force_Inline bool AllocateMemory(
		const Dimensions<_NRank>& MyShape,
		Allocator MyAlloc
	);

	_D_Dragonian_Lib_Constexpr_Force_Inline void ConstructViewInfo(
		const Dimensions<_NRank>& MyShape
	);

	_D_Dragonian_Lib_Constexpr_Force_Inline void ReConstructViewInfo();

	Tensor(
		const Dimensions<_NRank>& MyShape,
		Allocator Alloc = Allocator()
	) requires (std::is_trivially_copy_assignable_v<ValueType> || std::is_default_constructible_v<ValueType>);

	Tensor(
		const Dimensions<_NRank>& MyShape,
		ValueType* Buffer,
		size_t BufferSize,
		Allocator Alloc
	);

	Tensor(
		const Dimensions<_NRank>& MyShape,
		ValueType* Buffer,
		size_t BufferSize
	);

	Tensor(
		const Dimensions<_NRank>& MyShape,
		const Pointer& Buffer,
		size_t BufferSize
	);
#pragma endregion

#pragma region dep_chain
public:
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

	//Clean tasks that is already finished
	void CleanInplaceChain() const
	{
		if (_MyFuturesAsArgument)
		{
			auto [SubBegin, SubEnd] = std::ranges::remove_if(
				_MyFuturesAsArgument->begin(),
				_MyFuturesAsArgument->end(),
				[](const DependencyChainType::value_type& Iter) { return Iter.first._Is_ready(); }
			);
			_MyFuturesAsArgument->erase(SubBegin, SubEnd);
		}
	}

	//Clean tasks that is already finished
	void CleanOperationChain() const
	{
		if (_MyFuturesAsResult)
		{
			auto [SubBegin, SubEnd] = std::ranges::remove_if(
				_MyFuturesAsResult->begin(),
				_MyFuturesAsResult->end(),
				[](const DependencyChainType::value_type& Iter) { return Iter.first._Is_ready(); }
			);
			_MyFuturesAsResult->erase(SubBegin, SubEnd);
		}
	}

	//Check write permission.
	void WaitingAsResult() const
	{
		if (!(*_IgnoreDep))
		{
			WaitingForTheInplaceLock();
			WaitingForTheOperationLock();
		}
		else
		{
			(*_IgnoreDep) = false;
			CleanOperationChain();
			CleanInplaceChain();
		}
	}

	//Check read permission.
	void WaitingAsArgument() const
	{
		if (!(*_IgnoreDep))
			WaitingForTheOperationLock();
		else
		{
			(*_IgnoreDep) = false;
			CleanOperationChain();
		}
		CleanInplaceChain();
	}

	//Check read and write permission.
	void WaitingForAllLocks() const
	{
		if (!(*_IgnoreDep))
		{
			WaitingForTheInplaceLock();
			WaitingForTheOperationLock();
		}
		else
			(*_IgnoreDep) = false;
	}

	/**
	 * @brief Create a tensor that is a view of the current tensor, the tensor will detach from the dependency chain of the current tensor.
	 * @return A view that is detached from the dependency chain of the current tensor.
	 */
	decltype(auto) Detach() const
	{
		auto Ret = View();
		Ret._MyFuturesAsArgument = std::make_shared<DependencyChainType>();
		Ret._MyFuturesAsResult = std::make_shared<DependencyChainType>();
		Ret._IgnoreDep = std::make_shared<bool>(false);
		return Ret;
	}

	/**
	 * @brief Create a tensor that ignores the dependency chain of the current tensor one time. (The tensor will not wait for the tasks that depend on it before executing the next operator, but the tasks that depend on it will still wait for the current tensor, that means the current tensor will not be detached from the dependency chain. Evaluate will still wait for the tasks that depend on it.)
	 * @return A view of the current tensor that ignores the dependency chain.
	 */
	decltype(auto) Ignore() const
	{
		auto Ret = View();
		*Ret._IgnoreDep = true;
		return Ret;
	}

	/**
	 * @brief Wait for all the tasks that depend on this tensor.
	 * @return The current tensor.
	 */
	template <typename _ThisType>
	decltype(auto) Evaluate(this _ThisType&& Self)
	{
		std::forward<_ThisType>(Self).EvaluateTasks();
		return std::forward<_ThisType>(Self);
	}

	template <typename _ThisType, typename _TFn, typename... _ArgType>
	decltype(auto) AppendTask(
		this _ThisType&& _Self,
		_TFn&& _Fn,
		_ArgType&&... _Args
	) requires (TypeTraits::IsInvocableValue<_TFn, _ArgType...>)
	{
		DependencyChainDataPointers _DataPointer{ std::forward<_ThisType>(_Self)._MyFirst };
		if (std::forward<_ThisType>(_Self)._MyFuturesAsResult)
			std::forward<_ThisType>(_Self)._MyFuturesAsResult->emplace_back(
				Operators::GetTaskPool().Commit(
					std::forward<_TFn>(_Fn),
					std::forward<_ArgType>(_Args)...
				),
				_DataPointer
			);
		else
			std::forward<_TFn>(_Fn)(std::forward<_ArgType>(_Args)...);
		return std::forward<_ThisType>(_Self);
	}

private:
	void EvaluateTasks() const
	{
		WaitingForTheOperationLock();
		WaitingForTheInplaceLock();
		(*_IgnoreDep) = false;
	}
#pragma endregion

#pragma region grads
public:
	AutoGrad::Function* GetFunction() const
	{
		return _MyFunction;
	}

	AutoGrad::Graph* GetGraph() const
	{
		return _MyGraph;
	}

	decltype(auto) SetFunction(
		AutoGrad::Function* _Fn
	)
	{
		_MyFunction = _Fn;
		return *this;
	}

	decltype(auto) SetGraph(
		AutoGrad::Graph* _Graph
	)
	{
		_MyGraph = _Graph;
		return *this;
	}

	bool RequiresGrad() const
	{
		return _MyGraph != nullptr && _MyFunction != nullptr;
	}

	decltype(auto) NoGrad() const
	{
		auto Ret = View();
		Ret._MyGraph = nullptr;
		Ret._MyFunction = nullptr;
		return Ret;
	}

	decltype(auto) Grad() const
	{
		if (RequiresGrad())
			return _MyFunction->GetGrad<Tensor>();
		_D_Dragonian_Lib_Throw_Exception("This tensor has no grad!");
	}
#pragma endregion

#pragma region to_string
private:
	decltype(auto) CastToString(
		SizeType _MyTotalSize,
		bool _Fold = true
	) const
	{
		ThrowOnNotEnabled();
		if constexpr (_NRank > 1)
		{
			if (_MyShape.Front() > 10 && _MyTotalSize > 200 && _Fold)
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
			if (_MyShape.Front() > 10 && _MyTotalSize > 200 && _Fold)
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

public:
	decltype(auto) CastToAnsiString(
		bool _Fold = true
	) const
	{
		return CastToString(TotalSize(), _Fold);
	}

	decltype(auto) CastToWideString(
		bool _Fold = true
	) const
	{
		return UTF8ToWideString(CastToString(TotalSize()), _Fold);
	}

	decltype(auto) to_string() const
	{
		return CastToString(TotalSize());
	}

	decltype(auto) to_wstring() const
	{
		return UTF8ToWideString(to_string());
	}
#pragma endregion

#pragma region broad_cast
private:
	template <typename _Type1, typename _Type2, size_t _Rank1, size_t _Rank2, Device _Device1, Device _Device2>
	static std::pair<
		Tensor<_Type1, _Rank1, _Device1>,
		Tensor<_Type2, _Rank1, _Device2>
	> BroadCast(
		const Tensor<_Type1, _Rank1, _Device1>& _A,
		const Tensor<_Type2, _Rank2, _Device2>& _B,
		bool Inplace = false
	) requires (_Rank1 >= _Rank2);

	template <typename _Type1, typename _Type2, size_t _Rank1, size_t _Rank2, Device _Device1, Device _Device2>
	static std::pair<
		Tensor<_Type1, _Rank2, _Device1>,
		Tensor<_Type2, _Rank2, _Device2>
	> BroadCast(
		const Tensor<_Type1, _Rank1, _Device1>& _A,
		const Tensor<_Type2, _Rank2, _Device2>& _B,
		bool Inplace = false
	) requires (_Rank1 < _Rank2);

public:
	template <typename _Type2, size_t _Rank2, Device _Device2>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) BroadCast2AndCpy(
		const Tensor<_Type2, _Rank2, _Device2>& _Other
	) const requires (_Rank2 <= _NRank)
	{
		decltype(auto) Bd = BroadCast(*this, _Other, false);
		return Bd.first.Contiguous();
	}

private:
	template <typename _Type2, size_t _Rank2, Device _Device2>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) BroadCast(
		const Tensor<_Type2, _Rank2, _Device2>& _Other,
		bool Inplace = true
	) const requires (_Rank2 <= _NRank)
	{
		auto [_Self, _That] = BroadCast(*this, _Other, Inplace);
		return _That;
	}
#pragma endregion

#pragma region shape_ops
public:
	template <size_t _Axis, size_t = _NRank>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GatherAxis(
		SizeType _Index
	) const requires (_NRank > 0 && _Axis < _NRank);

private:
	template <size_t = _NRank>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ViewFirstAxis(
		SizeType _Index
	) const requires (_NRank > 0)
	{
		return GatherAxis<0>(_Index);
	}

	template <size_t _TRank>
	constexpr decltype(auto) ViewDimensions(
		const Dimensions<_TRank>& _Indice
	) const requires (_NRank >= _TRank);

public:
	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 2D tensor, then tensor[0] will return the 1st row of the tensor.
	 * @param _Index The index of the element tensor.
	 * @return The element tensor.
	 */
	template <size_t = _NRank>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator[](SizeType _Index) const
		requires (_NRank > 0)
	{
		return ViewFirstAxis(_Index);
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
	template <size_t _TRank>
	constexpr decltype(auto) operator[](const Dimensions<_TRank>& _Indice) const
		requires (_NRank >= _TRank)
	{
		return ViewDimensions(_Indice);
	}

	template <size_t _SliceDim = 0, typename _FirstType, typename ..._ArgTypes>
	decltype(auto) operator()(_FirstType _Index, _ArgTypes ..._Args) const
		requires ((sizeof...(_ArgTypes) < _NRank) && TypeTraits::IsIntegerValue<_FirstType> && (_SliceDim < _NRank));

	template <size_t _SliceDim = 0, typename ..._ArgTypes>
	decltype(auto) operator()(Range _Range, _ArgTypes ..._Args) const
		requires ((sizeof...(_ArgTypes) < _NRank) && (_SliceDim < _NRank));

	/**
	 * @brief Split the tensor into multiple tensors along the specified axis.
	 * @param _Size The size of each split tensor.
	 * @param _Axis The axis to split along.
	 * @return A vector of split tensors.
	 */
	template <size_t _Count>
	decltype(auto) Split(
		const Dimensions<_Count>& _Size,
		SizeType _Axis = 0
	) const;

	/**
	 * @brief Split the tensor into multiple tensors along the specified axis.
	 * @param _Size The size of each split tensor.
	 * @param _Axis The axis to split along.
	 * @return A vector of split tensors.
	 */
	decltype(auto) Split(
		SizeType _Size,
		SizeType _Axis = 0
	) const;

	/**
	 * @brief Slice the tensor, the order of the axes is ([0, 1, ... , N_DIMS - 1]).
	 * @param _SliceOptions A [[begin, step, end]/null, ...] array of all sliced axes, null means no slice.
	 * @return A sliced tensor(view).
	 */
	template <size_t _Rnk = _NRank>
	decltype(auto) Slice(
		const SliceOptions<_Rnk>& _SliceOptions
	) const requires (_Rnk <= _NRank && _Rnk > 0);

	/**
	 * @brief Slice the tensor, the order of the axes is reversed ([-1, -2, ... , -N_DIMS]).
	 * @param _SliceOptions A [[begin, end, step]/none, ...] array of all sliced axes, none means no slice.
	 * @return A sliced tensor(view).
	 */
	template <size_t _Rnk = _NRank>
	decltype(auto) ReversedSlice(
		const SliceOptions<_Rnk>& _SliceOptions
	) const requires (_Rnk <= _NRank && _Rnk > 0);

	/**
	 * @brief Permute the order of axes of a tensor, the order of original axes is ([0, 1, ... , N_DIMS - 1]). for example, we have a tensor with [N, H, C] shape, we can permute it to [N, C, H] shape with Permute([0, 2, 1])
	 * @param _PremuteOrder The new order of axes.
	 * @return A permuted tensor(view).
	 */
	decltype(auto) Permute(
		const Dimensions<_NRank>& _PremuteOrder
	) const;

	/**
	 * @brief Permute the order of axes of a tensor, the order of original axes is ([0, 1, ... , N_DIMS - 1]). for example, we have a tensor with [N, H, C] shape, we can permute it to [N, C, H] shape with Permute([0, 2, 1])
	 * @param _Order The new order of axes.
	 * @return A permuted tensor(view).
	 */
	template <typename... _Args>
	decltype(auto) Permute(
		_Args... _Order
	) const requires (sizeof...(_Args) == _NRank)
	{
		return Permute(Dimensions<_NRank>{_Order...});
	}

	/**
	 * @brief Transpose the tensor, swap the axes at the specified positions. for example, we have a tensor with [N, C, H] shape, we can transpose it with Transpose(1, 2) to get a tensor with [N, H, C] shape.
	 * @param _Axis1 The first axis.
	 * @param _Axis2 The second axis.
	 * @return A transposed tensor(view).
	 */
	decltype(auto) Transpose(
		SizeType _Axis1 = -1,
		SizeType _Axis2 = -2
	) const;

	decltype(auto) AxisFromTo(
		SizeType _Begin = -2,
		SizeType _End = -1
	) const;

	/**
	 * @brief Unsqueeze the tensor, add a new axis at the specified position. for example, we have a tensor with [N, C, H] shape, we can unsqueeze it at the 1st axis with UnSqueeze(1) to get a tensor with [N, 1, C, H] shape.
	 * @param _Dim The specified position.
	 * @return An unsqueezed tensor(view).
	 */
	decltype(auto) UnSqueeze(
		SizeType _Dim
	) const;

	/**
	 * @brief Squeeze the tensor, remove the axis with size 1 at the specified position. for example, we have a tensor with [N, 1, C, H] shape, we can squeeze it at the 1st axis with Squeeze(1) to get a tensor with [N, C, H] shape.
	 * @param _Dim The specified position.
	 * @return A squeezed tensor(view).
	 */
	decltype(auto) Squeeze(
		SizeType _Dim
	) const;

	/**
	 * @brief Create a view of the tensor.
	 * @return A viewed tensor(view).
	 */
	decltype(auto) View() const
	{
		auto Ret = Tensor(*this);
		return Ret;
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View([N, -1]) to get a tensor with [N, C * H * W] shape.
	 * @param _ViewShape The specified shape.
	 * @return A viewed tensor(view).
	 */
	template <size_t _TRank>
	decltype(auto) View(
		const Dimensions<_TRank>& _ViewShape
	) const;

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View(N, -1) to get a tensor with [N, C * H * W] shape.
	 * @tparam _Args The specified shape.
	 * @param _Shape0 The shapes.
	 * @param _Shape The shapes.
	 * @return A viewed tensor(view).
	 */
	template <typename... _Args>
	decltype(auto) View(
		SizeType _Shape0, _Args... _Shape
	) const
	{
		auto _ViewShape = Dimensions{ _Shape0, _Shape... };
		return View(_ViewShape);
	}

	template <typename... _Args>
	decltype(auto) AutoView(
		SizeType _Shape0, _Args... _Shape
	) const
	{
		auto _ViewShape = Dimensions{ _Shape0, _Shape... };
		for (auto& i : _ViewShape)
			if (i < 0)
				i = _MyShape[CalcIndex(i, Rank())];
		return View(_ViewShape);
	}

	/**
	 * @brief Reverse the tensor along the specified axis.
	 * @param _Axis The specified axis.
	 * @return A viewed tensor(view).
	 */
	decltype(auto) Reverse(
		SizeType _Axis = 0
	) const;

	template <typename _Type>
	decltype(auto) ViewAs() const requires (std::is_trivially_copy_assignable_v<_Type> && (bool(sizeof(_Type) % sizeof(ValueType)) || bool(sizeof(ValueType) % sizeof(_Type))));

	/**
	 * @brief Get the real part of the tensor.
	 * @return A viewed tensor(view).
	 */
	template <typename = _TensorType>
	decltype(auto) Real() const
		requires (TypeTraits::IsComplexValue<ValueType>)
	{
		ThrowOnNotEnabled();
		using RealType = typename ValueType::value_type;
		SliceOptions<_NRank> SliceVector;
		SliceVector[_NRank - 1].Step = 2;
		return ViewAs<RealType>().Slice(SliceVector);
	}

	/**
	 * @brief Get the imaginary part of the tensor.
	 * @return A viewed tensor(view).
	 */
	template <typename = _TensorType>
	decltype(auto) Imag() const
		requires (TypeTraits::IsComplexValue<ValueType>)
	{
		ThrowOnNotEnabled();
		using RealType = typename ValueType::value_type;
		SliceOptions<_NRank> SliceVector;
		SliceVector[_NRank - 1].Begin = 1;
		SliceVector[_NRank - 1].Step = 2;
		return ViewAs<RealType>().Slice(SliceVector);
	}

	/**
	 * @brief If the tensor is not Contiguous, make output Contiguous.
	 * @return New tensor (view or clone).
	 */
	template <typename = ValueType>
	decltype(auto) Contiguous() const
		requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		if (IsContiguous())
			return View();
		return Clone();
	}

	/**
	 * @brief If the tensor is not contiguous, make output contiguous.
	 * @return New tensor (view or clone).
	 */
	template <typename = ValueType>
	decltype(auto) Continuous() const
		requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		return Contiguous();
	}

	/**
	 * @brief If the tensor is not contiguous, make output contiguous.
	 * @param _Buffer The buffer to clone to.
	 * @return New tensor (view or clone).
	 */
	template <typename = ValueType, size_t _TRank>
	decltype(auto) Contiguous(
		Tensor<ValueType, _TRank, _MyDevice>& _Buffer
	) const requires (std::is_copy_assignable_v<ValueType>)
	{
		return Clone(_Buffer);
	}

	/**
	 * @brief If the tensor is not contiguous, make output contiguous.
	 * @param _Buffer The buffer to clone to.
	 * @return New tensor (view or clone).
	 */
	template <typename = ValueType, size_t _TRank>
	decltype(auto) Continuous(
		Tensor<ValueType, _TRank, _MyDevice>& _Buffer
	) const requires (std::is_copy_assignable_v<ValueType>)
	{
		return Clone(_Buffer);
	}

	/**
	 * @brief Make this tensor Contiguous.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) MakeContinuous()
		requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		if (IsContiguous())
			return *this;
		return *this = Clone();
	}

	/**
	 * @brief Make this tensor contiguous.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) MakeContiguous()
		requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		if (IsContiguous())
			return *this;
		return *this = Clone();
	}

	/**
	 * @brief Reshape the tensor to the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can reshape it to [N, C * H * W] shape with ReShape([N, -1]).
	 * @param _ViewShape The specified shape.
	 * @return A reshaped tensor(view).
	 */
	template <typename = ValueType, size_t _TRank>
	decltype(auto) ReShape(
		const Dimensions<_TRank>& _ViewShape
	) const requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		if (IsContiguous())
			return View(_ViewShape);
		return Clone().View(_ViewShape);
	}

	/**
	 * @brief Reshape the tensor to the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can reshape it to [N, C * H * W] shape with ReShape(N, -1).
	 * @param _ViewShape The specified shape.
	 * @param _Buffer The buffer to reshape to.
	 * @return A reshaped tensor(view).
	 */
	template <typename = ValueType, size_t _TRank>
	decltype(auto) ReShape(
		const Dimensions<_TRank>& _ViewShape,
		Tensor<ValueType, _TRank, _MyDevice>& _Buffer
	) const requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		return Clone(_Buffer).View(_ViewShape);
	}

	/**
	 * @brief Reshape the tensor to the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can reshape it to [N, C * H * W] shape with ReShape(N, -1).
	 * @param _Shape0 The shapes.
	 * @param _Shape The shapes.
	 * @return A reshaped tensor(view).
	 */
	template <typename = ValueType, typename... _Args>
	decltype(auto) ReShape(
		SizeType _Shape0,
		_Args... _Shape
	) const requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		auto _ViewShape = Dimensions{ _Shape0, _Shape... };
		return ReShape(_ViewShape);
	}

	template <typename... _Args>
	decltype(auto) AutoShape(
		SizeType _Shape0,
		_Args... _Shape
	) const
	{
		auto _ViewShape = Dimensions{ _Shape0, _Shape... };
		for (auto& i : _ViewShape)
			if (i < 0)
				i = _MyShape[CalcIndex(i, Rank())];
		return ReShape(_ViewShape);
	}

	template <typename = ValueType, typename... _Args>
	decltype(auto) ReSize(
		SizeType _Shape0,
		_Args... _Shape
	) const
	{
		return ReShape(_Shape0, _Shape...);
	}
#pragma endregion

#pragma region special_constructor
public:
	/**
	 * @brief Create a new tensor with the specified shape.
	 * @param _Shape The shape of the tensor.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) New(
		const Dimensions<_NRank>& _Shape,
		Allocator _Al = Allocator()
	) requires (std::is_trivially_copy_assignable_v<ValueType> || std::is_default_constructible_v<ValueType>)
	{
		return Tensor(_Shape, _Al);
	}

	template <typename = ValueType>
	static constexpr decltype(auto) NewVector(
		SizeType _Size,
		Allocator _Al = Allocator()
	) requires (std::is_trivially_copy_assignable_v<ValueType> || std::is_default_constructible_v<ValueType>)
	{
		auto MyShape = Dimensions<_NRank>::ConstantOf(1ll);
		MyShape.Back() = _Size;
		return Tensor(MyShape, _Al);
	}

	/**
	 * @brief Create an empty new tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) New()
	{
		return Tensor();
	}

	static constexpr decltype(auto) New(
		const Dimensions<_NRank>& MyShape,
		const Pointer& Buffer,
		size_t BufferSize
	)
	{
		return Tensor(MyShape, Buffer, BufferSize);
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with ones.
	 * @param _Shape The shape of the tensor.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) Ones(
		const Dimensions<_NRank>& _Shape,
		Allocator _Al = Allocator()
	) requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(1)>)
	{
		Tensor Ret(_Shape, _Al);
		Ret.Assign(ValueType(1));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with zeros.
	 * @param _Shape The shape of the tensor.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) Zeros(
		const Dimensions<_NRank>& _Shape,
		Allocator _Al = Allocator()
	) requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(0)>)
	{
		Tensor Ret(_Shape, _Al);
		Ret.Assign(ValueType(0));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with a constant value.
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value to fix the tensor.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) ConstantOf(
		const Dimensions<_NRank>& _Shape,
		const ValueType& _Val,
		Allocator _Al = Allocator()
	) requires (std::is_copy_assignable_v<ValueType>)
	{
		Tensor Ret(_Shape, _Al);
		Ret.Assign(_Val);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) Rand(
		const Dimensions<_NRank>& _Shape,
		const ValueType& Min,
		const ValueType& Max,
		Allocator _Al = Allocator()
	) requires (TypeTraits::IsArithmeticValue<ValueType>)
	{
		Tensor Ret(_Shape, _Al);
		Ret.AssignRand(Min, Max);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @param _Shape The shape of the tensor.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) Randn(
		const Dimensions<_NRank>& _Shape,
		double _Mean = 0.,
		double _Sigma = 1.,
		Allocator _Al = Allocator()
	) requires (TypeTraits::IsArithmeticValue<ValueType>)
	{
		Tensor Ret(_Shape, _Al);
		Ret.AssignRandn(_Mean, _Sigma);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with ones.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) OnesLike(
		const Tensor& _ShapeReference
	) requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(1)>)
	{
		_ShapeReference.ThrowOnNotEnabled();
		return Ones(_ShapeReference.Shape(), _ShapeReference.GetAllocator());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with zeros.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) ZerosLike(
		const Tensor& _ShapeReference
	) requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(0)>)
	{
		_ShapeReference.ThrowOnNotEnabled();
		return Zeros(_ShapeReference.Shape(), _ShapeReference.GetAllocator());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with a constant value.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) ConstantLike(
		const Tensor& _ShapeReference,
		const ValueType& _Val
	) requires (std::is_copy_assignable_v<ValueType>)
	{
		_ShapeReference.ThrowOnNotEnabled();
		return ConstantOf(_ShapeReference.Shape(), _Val, _ShapeReference.GetAllocator());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param Min The minimum value of the random values.
	 * @param Max The maximum value of the random values.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) RandLike(
		const Tensor& _ShapeReference,
		const ValueType& Min,
		const ValueType& Max
	) requires (TypeTraits::IsArithmeticValue<ValueType>)
	{
		_ShapeReference.ThrowOnNotEnabled();
		return Rand(_ShapeReference.Shape(), Min, Max, _ShapeReference.GetAllocator());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) RandnLike(
		const Tensor& _ShapeReference,
		double _Mean = 0.,
		double _Sigma = 1.
	) requires (TypeTraits::IsArithmeticValue<ValueType>)
	{
		_ShapeReference.ThrowOnNotEnabled();
		return Randn(_ShapeReference.Shape(), _Mean, _Sigma, _ShapeReference.GetAllocator());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _Shape The shape of the tensor.
	 * @param _Al The allocator of the tensor.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) Empty(
		const Dimensions<_NRank>& _Shape,
		Allocator _Al = Allocator()
	) requires (std::is_trivially_copy_assignable_v<ValueType>)
	{
		return Tensor(_Shape, _Al);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template <typename = ValueType>
	static constexpr decltype(auto) EmptyLike(
		const Tensor& _ShapeReference
	) requires (std::is_trivially_copy_assignable_v<ValueType>)
	{
		return Tensor(_ShapeReference._MyShape, _ShapeReference.GetAllocator());
	}

	template <typename = ValueType, size_t = _NRank>
	static constexpr decltype(auto) Arange(
		ValueType _Begin,
		ValueType _End,
		ValueType _Step,
		Allocator _Al = Allocator()
	) requires (_NRank == 1 && Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType> && Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType> && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>);

	template <typename = ValueType, size_t = _NRank>
	static constexpr decltype(auto) Linspace(
		ValueType _Begin,
		ValueType _End,
		size_t _Count,
		bool _EndPoint = false,
		Allocator _Al = Allocator()
	) requires (_NRank == 1 && Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType> && Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType> && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
	{
		if (_EndPoint)
		{
			const auto Step = (_End - _Begin) / ValueType(_Count - 1);
			return Arange(_Begin, _End + (Step * ValueType(1.01)), Step, _Al);
		}
		const auto Step = (_End - _Begin) / ValueType(_Count);
		return Arange(_Begin, _End + Step * ValueType(0.01), Step, _Al);
	}

	static constexpr Tensor FromBuffer(
		const Dimensions<_NRank>& MyShape,
		ValueType* Buffer,
		size_t BufferSize,
		Allocator Alloc
	)
	{
		return Tensor(MyShape, Buffer, BufferSize, Alloc);
	}

	static constexpr Tensor FromBuffer(
		const Dimensions<_NRank>& MyShape,
		ValueType* Buffer,
		size_t BufferSize
	)
	{
		return Tensor(MyShape, Buffer, BufferSize);
	}
#pragma endregion

#pragma region assign_ops
public:
	constexpr Tensor& operator=(const Tensor& _Left) = delete;
	constexpr Tensor& operator=(Tensor&& _Right) noexcept = default;

	template <size_t _TRank>
	Tensor& TensorAssign(
		const Tensor<ValueType, _TRank, _MyDevice>& _Left
	) requires (_NRank >= _TRank && std::is_copy_assignable_v<ValueType>)
	{
		if ((const void*)this != (const void*)&_Left)
			Assign(_Left);
		return *this;
	}

	template <size_t _TRank>
	Tensor& Inplace(
		const Tensor<ValueType, _TRank, _MyDevice>& _Left
	) requires (_NRank >= _TRank && std::is_copy_assignable_v<ValueType>)
	{
		return TensorAssign(_Left);
	}

	/**
	 * @brief Assign the tensor with a scalar value.
	 * @param _Val The scalar value.
	 * @return Reference of this.
	 */
	Tensor& operator=(const ValueType& _Val)
		requires (std::is_copy_assignable_v<ValueType>)
	{
		Assign(_Val);
		return *this;
	}

	/**
	 * @brief Assign the tensor with ones.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) FixOnes()
		requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(1)>)
	{
		Assign(ValueType(1));
		return *this;
	}

	/**
	 * @brief Assign the tensor with zeros.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) FixZeros()
		requires (std::is_copy_assignable_v<ValueType>&& std::is_constructible_v<ValueType, decltype(0)>)
	{
		Assign(ValueType(0));
		return *this;
	}

	/**
	 * @brief Assign the tensor with a constant value.
	 * @param _Val The constant value.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) Fix(
		const ValueType& _Val
	) requires (std::is_copy_assignable_v<ValueType>)
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
	template <typename = ValueType>
	decltype(auto) Fix(
		const ValueType* _Buffer,
		SizeType _Count
	) requires (std::is_copy_assignable_v<ValueType>)
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
	template <typename = ValueType>
	decltype(auto) MoveFix(
		const ValueType* _Buffer,
		SizeType _Count
	) requires (std::is_move_assignable_v<ValueType>)
	{
		MoveAssign(_Buffer, _Count);
		return *this;
	}

	/**
	 * @brief Assign the tensor with random values.
	 * @return Reference of this.
	 */
	template <typename = ValueType>
	decltype(auto) RandFix(
		const ValueType& Min = ValueType(0),
		const ValueType& Max = ValueType(1)
	) requires (TypeTraits::IsArithmeticValue<ValueType>)
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
	template <typename = ValueType>
		requires (TypeTraits::IsArithmeticValue<ValueType>)
	decltype(auto) RandnFix(
		double _Mean = 0.,
		double _Sigma = 1.
	)
	{
		AssignRandn(_Mean, _Sigma);
		return *this;
	}

	template <typename _MaskType, typename = ValueType>
	decltype(auto) MaskedFill(
		const Tensor<_MaskType, _NRank, _MyDevice>& _Mask,
		const ValueType& _Value
	) requires (std::is_copy_assignable_v<ValueType>&& TypeTraits::CouldBeConvertedFromValue<bool, _MaskType>);

	template <typename _MaskType, size_t _TRank, typename = ValueType>
	decltype(auto) MaskedFill(
		const Tensor<_MaskType, _NRank, _MyDevice>& _Mask,
		const Tensor<ValueType, _TRank, _MyDevice>& _Value
	) requires (std::is_copy_assignable_v<ValueType>&& TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank));

	template <typename _MaskType, typename _FunTy, size_t _TRank,
		typename _ArgType = std::nullptr_t, typename _VectorizedFnTy = nullptr_t
	> decltype(auto) MaskedInplace(
		const Tensor<_MaskType, _TRank, _MyDevice>& _Mask,
		const _ArgType& _Value,
		_FunTy _ScalarFun,
		_VectorizedFnTy _VectorizedFn = _VectorizedFnTy()
	) requires (TypeTraits::IsInvocableValue<std::decay_t<_FunTy>, ValueType&, const _ArgType&>&& TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank));

	template <typename _ArgType, typename _MaskType,
		typename _FunTy, size_t _TRank1, size_t _TRank2, typename _VectorizedFnTy = nullptr_t
	> decltype(auto) MaskedInplace(
		const Tensor<_MaskType, _TRank1, _MyDevice>& _Mask,
		const Tensor<_ArgType, _TRank2, _MyDevice>& _Value,
		_FunTy _ScalarFun,
		_VectorizedFnTy _VectorizedFn = _VectorizedFnTy()
	) requires (TypeTraits::IsInvocableValue<TypeTraits::RemoveReferenceType<_FunTy>, ValueType&, const _ArgType&>&& TypeTraits::CouldBeConvertedFromValue<bool, _MaskType> && (_NRank >= _TRank1) && (_NRank >= _TRank2));

	/**
	 * @brief Clone this tensor, if the tensor is not Contiguous, make output Contiguous.
	 * @return New tensor.
	 */
	template <typename = ValueType>
	decltype(auto) Clone() const
		requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
		auto Ret = New(_MyShape, _MyAllocator);
		Ret.TensorAssign(*this);
		return Ret;
	}

	/**
	 * @brief Clone this tensor, if the tensor is not Contiguous, make output Contiguous.
	 * @param _Buffer The buffer to clone to.
	 * @return New tensor.
	 */
	template <typename = ValueType, size_t _TRank>
	decltype(auto) Clone(
		Tensor<ValueType, _TRank, _MyDevice>& _Buffer
	) const requires (std::is_copy_assignable_v<ValueType>)
	{
		ThrowOnNotEnabled();
		_Buffer.ThrowOnNotEnabled();
		_Buffer.TensorAssign(*this);
		return _Buffer;
	}

	template <SizeType _Axis = 0, typename = ValueType, typename _IndexType>
	decltype(auto) Gather(
		const Tensor<_IndexType, _NRank, _MyDevice>& _Indices
	) const requires ((_Axis < _NRank) && (_Axis > -_NRank - 1) && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>);

	template <SizeType _Axis = 0, typename = ValueType, typename _IndexType>
	decltype(auto) Gather(
		const Tensor<_IndexType, _NRank, _MyDevice>& _Indices,
		Tensor<_IndexType, _NRank, _MyDevice>& _Buffer
	) requires ((_Axis < _NRank) && (_Axis > -_NRank - 1) && std::is_copy_assignable_v<ValueType>);

	template <typename _Type>
	decltype(auto) Cast() const
		requires (TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>);

	template <typename _Type>
	decltype(auto) Cast(
		Tensor<_Type, _NRank, _MyDevice>& _Buffer
	) const requires (TypeTraits::CouldBeConvertedFromValue<_Type, ValueType>&& TypeTraits::CouldBeConvertedFromValue<_Type, _Type>&& std::is_copy_assignable_v<_Type>&& std::is_default_constructible_v<_Type>);

private:
	template <typename = ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(
		const ValueType& _Value
	) requires (std::is_copy_assignable_v<ValueType>);

	template <typename = ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(
		const ValueType* _Buffer,
		SizeType _Count
	) requires (std::is_copy_assignable_v<ValueType>);

	template <typename = ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) MoveAssign(
		const ValueType* _Buffer,
		SizeType _Count
	) requires (std::is_move_assignable_v<ValueType>);

	template <typename = ValueType, size_t _TRank>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Assign(
		const Tensor<ValueType, _TRank, _MyDevice>& _Val
	) requires (std::is_copy_assignable_v<ValueType>);

	template <typename = ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) AssignRand(
		const ValueType& Min,
		const ValueType& Max
	) requires (TypeTraits::IsArithmeticValue<ValueType>);

	template <typename = ValueType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) AssignRandn(
		double _Mean = 0.,
		double _Sigma = 1.
	) requires (TypeTraits::IsArithmeticValue<ValueType>);
#pragma endregion

#pragma region attribute
public:
	/**
	 * @brief Get the rank of the tensor.
	 * @return The rank of the tensor.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType Rank()
	{
		return static_cast<SizeType>(_NRank);
	}

	/**
	 * @brief Get the alignment size of the value type.
	 * @return The alignment size of the value type.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType GetAlignSize()
	{
		return alignof(ValueType);
	}

	/**
	 * @brief Get the device of the tensor.
	 * @return The device of the tensor.
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline Device GetDevice()
	{
		return _Device;
	}

	decltype(auto) GetShared() const
	{
		auto Shared = _MyFirst;
		return Shared;
	}

	auto CreateShared() const
	{
		return std::make_shared<Tensor>(*this);
	}

	template <typename _ThisType>
	decltype(auto) GetRng(
		this _ThisType&& _Self
	)
	{
		if (std::forward<_ThisType>(_Self).IsContiguous())
			return TemplateLibrary::RangesWrp(
				std::forward<_ThisType>(_Self)._MyData,
				std::forward<_ThisType>(_Self)._MyData + std::forward<_ThisType>(_Self).ElementCount()
			);
		else
			_D_Dragonian_Lib_Throw_Exception("Could Not Get Range From Non-Contiguous Tensor!");
	}

	template <typename _ThisType>
	decltype(auto) GetCRng(
		this _ThisType&& _Self
	)
	{
		if (std::forward<_ThisType>(_Self).IsContiguous())
			return TemplateLibrary::ConstantRanges<ValueType>(
				std::forward<_ThisType>(_Self)._MyData,
				std::forward<_ThisType>(_Self)._MyData + std::forward<_ThisType>(_Self).ElementCount()
			);
		else
			_D_Dragonian_Lib_Throw_Exception("Could Not Get Range From Non-Contiguous Tensor!");
	}

	/**
	 * @brief Get the buffer of the tensor.
	 * @return The buffer of the tensor.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Buffer(
		this _ThisType&& Self
	)
	{
		return std::forward<_ThisType>(Self)._MyFirst;
	}

	/**
	 * @brief Get the data pointer of the tensor.
	 * @return The data pointer of the tensor.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data(
		this _ThisType&& Self
	)
	{
		return std::forward<_ThisType>(Self)._MyData;
	}

	/**
	 * @brief Get the data pointer of the tensor with the specified indices.
	 * @return The data pointer of the tensor.
	 */
	template <size_t _TRank, typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Data(
		this _ThisType&& Self,
		const Dimensions<_TRank>& _Indices
	)
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
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Get(
		this _ThisType&& Self,
		SizeType Index
	)
	{
		return *(std::forward<_ThisType>(Self).template Data<1>({ Index }));
	}

	/**
	 * @brief Get a val of the tensor with the specified indices.
	 * @return The val.
	 */
	template <size_t _TRank, typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Item(
		this _ThisType&& Self,
		const Dimensions<_TRank>& _Indices
	)
	{
		return *(std::forward<_ThisType>(Self).Data(_Indices));
	}

	/**
	 * @brief Get the first val of the tensor.
	 * @return The val.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Item(
		this _ThisType&& Self
	)
	{
		return *std::forward<_ThisType>(Self)._MyData;
	}

	/**
	 * @brief Get the pointer of the first val of the tensor.
	 * @return The pointer.
	 */
	template <typename _ThisType>
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) ItemPointer(
		this _ThisType&& Self
	)
	{
		return std::forward<_ThisType>(Self)._MyData;
	}

	/**
	 * @brief Get the allocator of the tensor.
	 * @return The allocator of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) GetAllocator() const
	{
		return _MyAllocator;
	}

	/**
	 * @brief Get the shape info of the tensor.
	 * @tparam _Begin The start axis.
	 * @tparam _End The end axis.
	 * @return The shape info of the tensor.
	 */
	template <size_t _Begin = 0, size_t _End = _NRank - _Begin>
	Operators::OperatorParameter<_End - _Begin> GetDefaultOperatorParameter(
		bool /*_CheckIsContiguous*/ = false
	) const
	{
		ThrowOnNotEnabled();
		constexpr auto CurrentRank = SizeType(_End - _Begin);
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
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline const Dimensions<_NRank>& Size() const
	{
		return _MyShape;
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
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Shape(
		SizeType _Index
	) const
	{
		_Index = CalcIndex(_Index, static_cast<SizeType>(_NRank));
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index The index of the axis.
	 * @return The shape of the specified axis of the tensor.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Size(
		SizeType _Index
	) const
	{
		_Index = CalcIndex(_Index, static_cast<SizeType>(_NRank));
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the stride of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline SizeType Stride(
		SizeType _Index
	) const
	{
		_Index = CalcIndex(_Index, static_cast<SizeType>(_NRank));
		return _MyViewStride[_Index];
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
	 * @brief Whether the tensor is not null (empty).
	 * @return true if the tensor is not null, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool HasValue() const
	{
		return _MyData != nullptr;
	}

	/**
	 * @brief Reset the tensor to null.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline void Clear()
	{
		_MyFirst = nullptr;
		_MyData = nullptr;
		_MyLast = nullptr;
		_MyFuturesAsResult = nullptr;
		_MyFuturesAsArgument = nullptr;
		_IgnoreDep = nullptr;
		_MyGraph = nullptr;
		_MyFunction = nullptr;
	}

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
	 * @brief Check if the tensor is Contiguous in the specified range.
	 * @return True if the tensor is Contiguous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContiguous() const
	{
		const auto Diff = _MyData - (const ValueType*)_MyFirst.get();

		auto _End = _NRank - 1;
		while (_End && _MyShape[_End] == 1)
			--_End;

		if (_MyViewStride[_End] != 1)
			return false;

		for (SizeType i = 1; std::cmp_less_equal(i, _End); ++i)
		{
			if (_MyViewStride[i - 1] <= 0 ||
				_MyViewStride[i - 1] % _MyShape[i] ||
				_MyViewStride[i - 1] / _MyShape[i] != _MyViewStride[i] ||
				Diff % _MyShape[i])
				return false;
		}

		return true;
	}

	/**
	 * @brief Check if the tensor is Contiguous in the specified range.
	 * @return True if the tensor is Contiguous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContinuous() const
	{
		return IsContiguous();
	}

	/**
	 * @brief Check if the tensor is view.
	 * @return True if the tensor is view, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsView() const
	{
		return _MyData != (RawPointer)_MyFirst.get() || !IsContiguous();
	}

	/**
	 * @brief Check if the tensor is broadcasted.
	 * @return True if the tensor is broadcasted, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsBroadCasted() const
	{
		return IsBroadCasted_();
	}

	/**
	 * @brief Check if the tensor is Contiguous in the specified range.
	 * @return True if the tensor is Contiguous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContiguous(
		SizeType _Begin,
		SizeType _End
	) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcIterator(_End, Rank());

		if (_MyViewStride[_End - 1] != 1)
			return false;

		const auto Diff = _MyData - (const ValueType*)_MyFirst.get();
		for (SizeType i = _Begin + 1; i < _End; ++i)
		{
			if (_MyViewStride[i - 1] <= 0 ||
				_MyViewStride[i - 1] % _MyShape[i] ||
				_MyViewStride[i - 1] / _MyShape[i] != _MyViewStride[i] ||
				Diff % _MyShape[i])
				return false;
		}
		return true;
	}

	/**
	 * @brief Check if the tensor is Contiguous in the specified range.
	 * @return True if the tensor is Contiguous, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsContinuous(
		SizeType _Begin,
		SizeType _End
	) const
	{
		return IsContiguous(_Begin, _End);
	}

	/**
	 * @brief Check if the tensor is not sliced.
	 * @return True if the tensor is not sliced, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool NotSliced(
		SizeType _Begin,
		SizeType _End
	) const
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
	 * @brief Check if the tensor is broadcasted.
	 * @return True if the tensor is broadcasted, false otherwise.
	 */
	_D_Dragonian_Lib_Constexpr_Force_Inline bool IsBroadCasted_(
		SizeType _Begin = 0,
		SizeType _End = _NRank
	) const
	{
		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcIterator(_End, Rank());

		for (SizeType i = _Begin; i < _End; ++i)
			if (!_MyViewStride[i])
				return true;

		return false;
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
	 * @brief Get the Contiguous access order of the tensor.
	 * @return The Contiguous access order of the tensor.
	 */
	Dimensions<_NRank> CalcContiguousAccessOrder() const
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
#pragma endregion

#pragma region binary_ops
	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Add);

	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Add, +, (Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Sub);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Sub, -, (Operators::BinaryOperators::SubBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Mul);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Mul, *, (Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Div);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Div, / , (Operators::BinaryOperators::DivBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Mod);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Mod, %, (Operators::BinaryOperators::ModBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(And);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(And, &&, (Operators::BinaryOperators::AndBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Or);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Or, || , (Operators::BinaryOperators::OrBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Xor);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(Xor, ^, (Operators::BinaryOperators::XorBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(LShift);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(LShift, << , (Operators::BinaryOperators::LShiftBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(RShift);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(RShift, >> , (Operators::BinaryOperators::RShiftBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(BitwiseOr);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(BitwiseOr, | , (Operators::BinaryOperators::BitwiseOrBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(BitwiseAnd);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator(BitwiseAnd, &, (Operators::BinaryOperators::BitwiseAndBinary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>)));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(Equal);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Equal, == , (Operators::ComparisonOperators::EqualBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(NotEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(NotEqual, != , (Operators::ComparisonOperators::NotEqualBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(Greater);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Greater, > , (Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(Less);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(Less, < , (Operators::ComparisonOperators::LessBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(GreaterEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(GreaterEqual, >= , (Operators::ComparisonOperators::GreaterEqualBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Compare_Function_Declare(LessEqual);
	_D_Dragonian_Lib_Operator_Bond_Function_2_Operator_Nip(LessEqual, <= , (Operators::ComparisonOperators::LessEqualBinary::HasOperatorValue<ValueType>));

	//*****************************************************************************************************************//
	_D_Dragonian_Lib_Operator_Binary_Function_Declare(Pow);

	//*****************************************************************************************************************//
	template <typename _Type>
	Tensor<std::complex<_Type>, _NRank, _MyDevice> operator*(const std::complex<_Type>& _Val) const
		requires (!TypeTraits::IsComplexValue<_TensorType>)
	{
		ThrowOnNotEnabled();
		Tensor<std::complex<_Type>, _NRank, _MyDevice> Ret{ Shape(), GetAllocator() };
		Ret.Real().Ignore().TensorAssign(Mul(_Val.real()));
		Ret.Imag().Ignore().TensorAssign(Mul(_Val.imag()));
		return Ret;
	}
	template <typename _Type>
	Tensor<std::complex<_Type>, _NRank, _MyDevice> operator*(const Tensor<std::complex<_Type>, _NRank, _MyDevice>& _Val) const
		requires (!TypeTraits::IsComplexValue<_TensorType>)
	{
		ThrowOnNotEnabled();
		_Val.ThrowOnNotEnabled();
		Tensor<std::complex<_Type>, _NRank, _MyDevice> Ret{ Shape(), GetAllocator() };
		Ret.Real().Ignore().TensorAssign(Mul(_Val.Real()));
		Ret.Imag().Ignore().TensorAssign(Mul(_Val.Imag()));
		return Ret;
	}
	//*****************************************************************************************************************//
#pragma endregion

#pragma region unary_ops
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Sqrt);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(RSqrt);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Reciprocal);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Abs);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Sin);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Cos);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Tan);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ASin);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ACos);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ATan);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Sinh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Cosh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Tanh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ASinh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ACosh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ATanh);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Exp);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Exp2);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Log);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Log2);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Log10);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Ceil);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Floor);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Round);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Trunc);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Frac);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Negative);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(BitwiseNot);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Not);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(ATan2);
	_D_Dragonian_Lib_Operator_Unary_Function_Declare(Polar);

	template <typename = ValueType>
	decltype(auto) operator-() const
		requires (Operators::UnaryOperators::NegativeUnary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>))
	{
		return Negative();
	}

	template <typename = ValueType>
	decltype(auto) operator!() const
		requires (Operators::UnaryOperators::NotUnary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>))
	{
		return Not();
	}

	template <typename = ValueType>
	decltype(auto) operator~() const
		requires (Operators::UnaryOperators::BitwiseNotUnary::HasOperatorValue<ValueType> && (std::is_copy_assignable_v<ValueType> || std::is_move_assignable_v<ValueType>))
	{
		return BitwiseNot();
	}
#pragma endregion

private:
	_D_Dragonian_Lib_Constexpr_Force_Inline void ThrowOnNotEnabled() const
	{
		if (!IsEnabled())
			_D_Dragonian_Lib_Throw_Exception("Null tensor could not be used!");
	}

#pragma region iterator
public:
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
	Dimensions<_NRank>& IteratorAdd(
		Dimensions<_NRank>& _Indices
	) const
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
	Dimensions<_NRank>& IteratorSub(
		Dimensions<_NRank>& _Indices
	) const
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
	 * @param _Strict Whether to throw exception when the index is out of range.
	 * @return The transformed index. (0 ~ (Max - 1))
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcIndex(
		SizeType _Index,
		SizeType _Max,
		bool _Strict = true
	)
	{
		if (_Index < 0)
			_Index += _Max;
		if (!_Strict)
			_Index = std::clamp(_Index, 0ll, _Max - 1);
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
	 * @param _Strict Whether to throw exception when the index is out of range.
	 * @return The transformed index. (-1 ~ Max)
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcEndPos(
		SizeType _Index,
		SizeType _Max,
		bool _Strict = true
	)
	{
		if (_Index == RangeEndPos)
			return _Max;
		if (_Index == RangeBeginPos)
			return -1;
		if (_Index == _Max || (!_Strict && _Index > _Max))
			return _Max;
		return CalcIndex(_Index, _Max, _Strict);
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
	 * @param _Strict Whether to throw exception when the index is out of range.
	 * @return The transformed index. (0 ~ Max)
	 */
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType CalcIterator(
		SizeType _Index,
		SizeType _Max,
		bool _Strict = true
	)
	{
		if (_Index == RangeEndPos)
			return _Max;
		if (_Index < 0)
			_Index += _Max + 1;
		if (!_Strict)
			_Index = std::clamp(_Index, 0ll, _Max);
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
	static _D_Dragonian_Lib_Constexpr_Force_Inline SizeType Ceil(
		SizeType _Left,
		SizeType _Right
	)
	{
		auto Div = _Left / _Right;
		if (_Left % _Right)
			return Div + 1;
		return Div;
	}
#pragma endregion

	//********************************************************Operation********************************************************//

	template <size_t _UnfoldDim, size_t _UnfoldCount, typename InvokeFnType>
	static decltype(auto) Invoke(
		Tensor& _Tensor,
		const InvokeFnType& _Fn
	) requires (TypeTraits::IsCallableValue<InvokeFnType>)
	{
		ThrowOnNotEnabled();
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

	template <typename = ValueType, size_t _TRank = _NRank>
	decltype(auto) Padding(
		const PaddingCounts<_TRank>& _PaddingCount,
		PaddingType _Type,
		std::optional<ValueType> _Val = std::nullopt
	) const requires (_TRank <= _NRank && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
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

		auto Ret = New(Shape, _MyAllocator);

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
							Ret[RngFront].Ignore().Assign(_Val.value());
						}
						if (_PaddingCount[i].End > 0)
						{
							RngBack[i] = { _MyShape[i] + _PaddingCount[i].Begin, RangeEndPos };
							Ret[RngBack].Ignore().Assign(_Val.value());
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

	template <typename = ValueType, size_t _TRank = _NRank>
	decltype(auto) Pad(
		const PaddingCounts<_TRank>& _PaddingCount,
		PaddingType _Type,
		std::optional<ValueType> _Val = std::nullopt
	) const requires (_TRank <= _NRank && std::is_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
	{
		PaddingCounts<_NRank> PaddingC;
		for (size_t i = 0; i < _TRank; ++i)
			PaddingC[_NRank - 1 - i] = _PaddingCount[i];
		return Padding(PaddingC, _Type, std::move(_Val));
	}

	template <typename = ValueType>
	decltype(auto) Repeat(
		const Dimensions<_NRank>& _Repeat
	) const requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
		PaddingCounts<_NRank> _PaddingCount;
		for (size_t i = 0; i < _NRank; ++i)
		{
			if (_Repeat[i] <= 1)
				continue;
			_PaddingCount[i].End = (_Repeat[i] - 1) * _MyShape[i];
		}
		return Padding(_PaddingCount, PaddingType::Cicular);
	}

	template <typename = ValueType>
	decltype(auto) Repeat(
		SizeType _Axis,
		SizeType _Repeat
	) const requires (std::is_copy_assignable_v<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
		PaddingCounts<_NRank> _PaddingCount;
		_Axis = CalcIndex(_Axis, Rank());
		_PaddingCount[_Axis].End = (_Repeat - 1) * _MyShape[_Axis];
		return Padding(_PaddingCount, PaddingType::Cicular);
	}

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) Sum(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(Sum, Sum);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) Prod(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(Prod, Prod);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) Mean(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::DivBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(Mean, Mean);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) ReduceMax(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(ReduceMax, Max);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) ReduceMin(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(ReduceMin, Min);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) LogSum(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::UnaryOperators::LogUnary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(LogSum, LogSum);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) LogSumExp(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::UnaryOperators::ExpUnary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::UnaryOperators::LogUnary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body(LogSumExp, LogSumExp);

	template <typename RetType = Int32, bool KeepDim = false, typename = ValueType>
	decltype(auto) ArgMax(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body_T(ArgMax, ArgMax, RetType);

	template <typename RetType = Int32, bool KeepDim = false, typename = ValueType>
	decltype(auto) ArgMin(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Reduce_Function_Body_T(ArgMin, ArgMin, RetType);

	template <typename = ValueType>
	decltype(auto) CumSum(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumSum);

	template <typename = ValueType>
	decltype(auto) CumSub(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::SubBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumSub);

	template <typename = ValueType>
	decltype(auto) CumProd(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumProd);

	template <typename = ValueType>
	decltype(auto) CumDiv(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::DivBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumDiv);

	template <typename = ValueType>
	decltype(auto) CumMax(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::GreaterBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumMax);

	template <typename = ValueType>
	decltype(auto) CumMin(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::ComparisonOperators::LessBinary::HasOperatorValue<ValueType>)
	_D_Dragonian_Lib_Operator_Cumulate_Function_Body(CumMin);

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) ReduceLp(SizeType _Axis, const ValueType& _P) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<ValueType>)
	{
		ThrowOnNotEnabled();
		if constexpr (_NRank == 1)
			return UnSqueeze(0).template LpNorm<false>(-1, _P).Squeeze(0);
		else
		{
			_Axis = CalcIndex(_Axis, Rank());
			auto TensorTmp = AxisFromTo(_Axis, -1);
			TensorTmp.WaitingAsArgument();
			Dimensions<_NRank - 1> OutShape;
			OutShape.Assign(TensorTmp.Shape().Data());
			auto Ret = Tensor<_TensorType, _NRank - 1, _MyDevice>::New(OutShape, _MyAllocator);
			Ret.WaitingAsResult();
			auto RetView = Ret.UnSqueeze(-1);
			Operators::OperatorsBase<ValueType, _MyDevice>::ImplReduceLpScalar
			(
				RetView.Data(),
				RetView.GetDefaultOperatorParameter(),
				TensorTmp.Data(),
				TensorTmp.GetDefaultOperatorParameter(),
				_P,
				RetView.IsContiguous() && TensorTmp.IsContiguous()
			);
			if constexpr (KeepDim)
				return RetView;
			else
				return Ret;
		}
	}

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) ReduceL1(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<ValueType>)
	{
		return ReduceLp<KeepDim>(_Axis, 1);
	}

	template <bool KeepDim = false, typename = ValueType>
	decltype(auto) ReduceL2(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::PowBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::UnaryOperators::AbsUnary::HasOperatorValue<ValueType>)
	{
		return ReduceLp<KeepDim>(_Axis, 2);
	}

	template <typename = ValueType>
	decltype(auto) Diff(SizeType _Axis) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::SubBinary::HasOperatorValue<ValueType>)
	{
		ThrowOnNotEnabled();
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
			auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(OutShape, _MyAllocator);
			auto ResView = Ret.AxisFromTo(_Axis, -1);
			ResView.WaitingAsResult();
			Operators::OperatorsBase<ValueType, _MyDevice>::ImplDiffUnary
			(
				ResView.Data(),
				ResView.GetDefaultOperatorParameter(),
				TensorTmp.Data(),
				TensorTmp.GetDefaultOperatorParameter(),
				ResView.IsContiguous() && TensorTmp.IsContiguous()
			);
			return Ret;
		}
	}

	template <Operators::InterpolateMode _Mode = Operators::InterpolateMode::Nearest, typename = ValueType>
	decltype(auto) Interpolate(const Dimensions<Operators::GetInterpolateModeRank<_Mode>>& _Dims, Operators::InterpolateParam<_Mode> _InterpParams) const
		requires (std::is_default_constructible_v<ValueType>&& Operators::BinaryOperators::SubBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::AddBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::MulBinary::HasOperatorValue<ValueType>)
	{
		ThrowOnNotEnabled();
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

		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(OutShape, _MyAllocator);
		auto _PermuteArgs = Dimensions<_NRank>::ConstantOf(-1);
		for (size_t i = 0; i < _Dims.Size(); ++i)
		{
			const auto Axis = CalcIndex(_Dims[i], Rank());
			_PermuteArgs[_NRank - _Dims.Size() + i] = Axis;
			//RetView = RetView.AxisFromTo(Axis, -1);
			//MyView = MyView.AxisFromTo(Axis, -1);
		}
		constexpr auto Dimm = Int64(_NRank - _Dims.Size());
		for (Int64 i = 0, j = 0; i < Rank() && j < Dimm; ++i)
		{
			while (std::ranges::contains(_PermuteArgs, i))
				++i;
			_PermuteArgs[j++] = i;
		}

		auto RetView = Ret.Permute(_PermuteArgs);
		auto MyView = Permute(_PermuteArgs);
		MyView.WaitingAsArgument();

		Operators::OperatorsBase<ValueType, _MyDevice>::template ImplInterpolate<_Mode, _NRank>
			(
				RetView.Data(),
				RetView.GetDefaultOperatorParameter(),
				MyView.Data(),
				MyView.GetDefaultOperatorParameter(),
				_InterpParams,
				RetView.IsContiguous() && MyView.IsContiguous()
			);

		return Ret;
	}

	template <typename = ValueType>
	decltype(auto) ClampMin(ValueType _Min) const
		requires (Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(_MyShape, _MyAllocator);
		Ret.WaitingAsResult();
		WaitingAsArgument();
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplMaxScalar
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Min,
			Ret.IsContiguous() && IsContiguous()
		);
		return Ret;
	}

	template <typename = ValueType>
	decltype(auto) ClampMax(ValueType _Max) const
		requires (Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		ThrowOnNotEnabled();
		auto Ret = Tensor<_TensorType, _NRank, _MyDevice>::New(_MyShape, _MyAllocator);
		Ret.WaitingAsResult();
		WaitingAsArgument();
		Operators::OperatorsBase<ValueType, _MyDevice>::ImplMinScalar
		(
			Ret.Data(),
			Ret.GetDefaultOperatorParameter(),
			Data(),
			GetDefaultOperatorParameter(),
			_Max,
			Ret.IsContiguous() && IsContiguous()
		);
		return Ret;
	}

	template <typename = ValueType>
	decltype(auto) Clamp(ValueType _Min, ValueType _Max) const
		requires (Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&& Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		return ClampMin(_Min).ClampMax(_Max);
	}

	template <typename = ValueType>
	decltype(auto) Min(ValueType _Min) const
		requires (Operators::BinaryOperators::MinBinary::HasOperatorValue<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		return ClampMax(_Min);
	}

	template <typename = ValueType>
	decltype(auto) Max(ValueType _Max) const
		requires (Operators::BinaryOperators::MaxBinary::HasOperatorValue<ValueType>&& std::is_default_constructible_v<ValueType>)
	{
		return ClampMin(_Max);
	}
};

template <typename _TensorType = float, Device _MyDevice = Device::CPU, size_t _NRank = 1>
using ITensor = Tensor<_TensorType, _NRank, _MyDevice>;

_D_Dragonian_Lib_Space_End