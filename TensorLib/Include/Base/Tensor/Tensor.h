/**
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
#include "Util/ThreadPool.h"

DragonianLibSpaceBegin

using Dimensions = Vector<SizeType>; ///< Alias for vector of size types
using ShapeIterator = Dimensions::Iterator; ///< Alias for iterator of shape type
static inline double DoubleZero = 0.; ///< Static inline double zero

/**
 * @brief Struct representing a range with begin, step, and end values.
 */
struct Range
{
	SizeType Begin = 0; ///< Begin value
	SizeType Step = 1; ///< Step value
	SizeType End = 0; ///< End value
	bool IsVal = false; ///< Flag indicating if it is a value
	bool IsNone = false; ///< Flag indicating if it is none

	/**
	 * @brief Constructor for a value range.
	 * @param _Val The value to initialize the range.
	 */
	Range(SizeType _Val) :Begin(_Val), Step(_Val), End(_Val), IsVal(true) {}

	/**
	 * @brief Constructor for a none range.
	 * @param _NoneVal The none value to initialize the range.
	 */
	Range(NoneType _NoneVal) :IsNone(true) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin, step, and end values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End) {}

	/**
	 * @brief Constructor for a range with none, step, and end values.
	 * @param _NoneVal The none value.
	 * @param _Step The step value.
	 * @param _End The end value.
	 */
	Range(NoneType _NoneVal, SizeType _Step, SizeType _End) :Begin(_Step > 0 ? 0 : -1), Step(_Step), End(_End) { UNUSED(_NoneVal); }

	/**
	 * @brief Constructor for a range with begin, step, and none values.
	 * @param _Begin The begining value.
	 * @param _Step The step value.
	 * @param _NoneVal The none value.
	 */
	Range(SizeType _Begin, SizeType _Step, NoneType _NoneVal) :Begin(_Begin), Step(_Step), End(_Step > 0 ? -1 : 0) { UNUSED(_NoneVal); }

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
	Range(SizeType _Begin, NoneType _NoneVal) :Begin(_Begin), End(-1) { UNUSED(_NoneVal); }

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

using SliceOptions = Vector<Range>; ///< Alias for vector of ranges

/**
 * @brief Multiply elements of a shape vector.
 * @param _Input The input shape vector.
 * @return The product of the elements.
 */
SizeType VectorMul(const Dimensions& _Input);

/**
 * @brief Multiply elements of a slice options vector.
 * @param _Input The input slice options vector.
 * @return The product of the elements.
 */
SizeType VectorMul(const SliceOptions& _Input);

/**
 * @brief Get the begining indices from slice options.
 * @param _Input The input slice options vector.
 * @return The begining indices as a shape type.
 */
Dimensions GetBeginIndices(const SliceOptions& _Input);

/**
 * @brief Check if all ranges in the vector are none.
 * @param _Input The input vector of ranges.
 * @return True if all ranges are none, false otherwise.
 */
bool RangeIsAllNone(const Vector<Range>& _Input);

/**
 * @brief Set the random seed.
 * @param _Seed The seed value.
 */
void SetRandomSeed(SizeType _Seed);

/**
 * @class Tensor
 * @brief Tensor with a specified value type and device.
 * @tparam _TensorType The value type of the tensor.
 * @tparam _MyDevice The device of the tensor.
 */
template<typename _TensorType = float32, Device _MyDevice = Device::CPU>
class Tensor : public Value
{
public:
	using InvokeFnType = void(*)(Tensor&);
	using ValueType = std::remove_reference_t<_TensorType>;
	using Pointer = std::shared_ptr<ValueType>;
	using RawPointer = ValueType*;
	using Reference = ValueType&;
	using ConstReference = const ValueType&;

protected:
	Allocator _MyAllocator = nullptr;
	Pointer _MyFirst = nullptr;
	RawPointer _MyLast = nullptr, _MyData = nullptr;

	Dimensions _MyShape;
	Dimensions _MyViewStep;
	Dimensions _MyViewLeft;
	Dimensions _MyViewStride;

	//SliceOptions _MyOpSlice;

	bool IsBroadCasted_ = false;

public:
	~Tensor() override = default;
	Tensor(const Tensor& Left) = default;
	Tensor(Tensor&& Right) noexcept = default;

	Tensor& operator=(const Tensor& _Left) = default;
	Tensor& operator=(Tensor&& _Right) noexcept = default;

	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 2D tensor, then tensor[0] will return the 1st row of the tensor.
	 * @param _Index The index of the element tensor.
	 * @return The element tensor.
	 */
	Tensor operator[](SizeType _Index) const
	{
		return GatherRef(_Index);
	}

	/**
	 * @brief Get a sliced tensor of the tensor.
	 * @param _SliceOptions The slice options of the tensor.
	 * @return The sliced tensor.
	 */
	Tensor operator[](const SliceOptions& _SliceOptions) const
	{
		return Slice(_SliceOptions);
	}

	/**
	 * @brief Get an element tensor of the tensor. for example, if the tensor is a 3D tensor, then tensor[{0, 0}] will return the 1st row of the 1st matrix of the tensor.
	 * @param _Indice 
	 * @return 
	 */
	Tensor operator[](const Dimensions& _Indice) const
	{
		Tensor Ret = CreateView();
		for (auto i : _Indice)
			Ret = Ret.GatherRef(i);
		return Ret;
	}

	//****************************************************Constructor****************************************************//

	/**
	 * @brief Create a new tensor with the specified shape.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param MyShape The shape of the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> New(const Dimensions& MyShape)
	{
		return Tensor<_Type, _Device>(MyShape);
	}

	/**
	 * @brief Create an empty new tensor.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> New()
	{
		return Tensor<_Type, _Device>();
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with ones.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> Ones(const Dimensions& _Shape)
	{
		Tensor<_Type, _Device> Ret(_Shape);
		Ret.Assign(_Type(1));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with zeros.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> Zeros(const Dimensions& _Shape)
	{
		Tensor<_Type, _Device> Ret(_Shape);
		Ret.Assign(_Type(0));
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with a constant value.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> ConstantOf(const Dimensions& _Shape, _Type _Val)
	{
		Tensor<_Type, _Device> Ret(_Shape);
		Ret.Assign(_Val);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _Shape The shape of the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> Rand(const Dimensions& _Shape)
	{
		Tensor<_Type, _Device> Ret(_Shape);
		Ret.RandFix();
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the specified shape, and fix the tensor with random values.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _Shape The shape of the tensor.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> Randn(const Dimensions& _Shape, double _Mean = 0., double _Sigma = 1.)
	{
		Tensor<_Type, _Device> Ret(_Shape);
		Ret.RandnFix(_Mean, _Sigma);
		return Ret;
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with ones.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> OnesLike(const Tensor& _ShapeReference)
	{
		return Ones<_Type, _Device>(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with zeros.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> ZerosLike(const Tensor& _ShapeReference)
	{
		return Zeros<_Type, _Device>(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with a constant value.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Val The constant value to fix the tensor.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> ConstantLike(const Tensor& _ShapeReference, _Type _Val)
	{
		return ConstantOf<_Type, _Device>(_ShapeReference.Shape(), _Val);
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> RandLike(const Tensor& _ShapeReference)
	{
		return Rand<_Type, _Device>(_ShapeReference.Shape());
	}

	/**
	 * @brief Create a new tensor with the same shape as the specified tensor, and fix the tensor with random values.
	 * @tparam _Type Value type of the tensor.
	 * @tparam _Device Device of the tensor.
	 * @param _ShapeReference The tensor to reference the shape.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 * @return The new tensor.
	 */
	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> RandnLike(const Tensor& _ShapeReference,  double _Mean = 0., double _Sigma = 1.)
	{
		return Randn<_Type, _Device>(_ShapeReference.Shape(), _Mean, _Sigma);
	}

	template<typename _Type = float32, Device _Device = Device::CPU>
	static Tensor<_Type, _Device> Arange(float64 _Begin, float64 _End, float64 _Step);

private:
	bool AllocateMemory(const Dimensions& MyShape, Allocator MyAlloc)
	{
		if (MyShape.Empty())
			return false;

		_MyAllocator = MyAlloc;
		const auto Size = VectorMul(MyShape);
		_MyFirst = Pointer(
			(RawPointer)MyAlloc->Allocate(Size * sizeof(ValueType)),
			[=](void* _Pointer)
			{
				_MyAllocator->Free(_Pointer);
			}
		);
		_MyData = _MyFirst.get();
		_MyLast = _MyData + Size;
		return true;
	}

	Tensor() = default;

	Tensor(const Dimensions& MyShape)
	{
		if(AllocateMemory(MyShape, GetMemoryProvider(_MyDevice)))
		{
			_MyShape = MyShape;
			_MyViewStep.Resize(_MyShape.Size());
			auto _Begin = _MyViewStep.ReversedBegin();
			auto _End = _MyViewStep.ReversedEnd();
			auto _Iter = _MyShape.ReversedBegin();
			*_Begin-- = 1;
			while (_Begin != _End) *_Begin-- = *_Iter--;
			_MyViewLeft = { _MyShape.Size(), 0ll, _MyShape.GetAllocator() };
			_MyViewStride = { _MyShape.Size(), 1ll, _MyShape.GetAllocator() };
		}
	}

public:

	//********************************************************Info********************************************************//

	/**
	 * @brief Get the alignment size of the value type.
	 * @return The alignment size of the value type.
	 */
	static DRAGONIANLIBCONSTEXPR SizeType GetAlignSize()
	{
		return alignof(ValueType);
	}

	/**
	 * @brief Get the device of the tensor.
	 * @return The device of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR Device GetDevice() const
	{
		return _MyAllocator->GetDevice();
	}

	/**
	 * @brief Get the allocator of the tensor.
	 * @return The allocator of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR Allocator GetAllocator() const
	{
		return _MyAllocator;
	}

	/**
	 * @brief Get the buffer of the tensor.
	 * @return The buffer of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Buffer()
	{
		return _MyFirst;
	}

	/**
	 * @brief Get the data pointer of the tensor.
	 * @return The data pointer of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Data() const
	{
		return _MyData;
	}

	/**
	 * @brief Get the data pointer of the tensor with the specified indices.
	 * @param _Indices The indices of the tensor.
	 * @return The data pointer of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Data(const Dimensions& _Indices) const
	{
		SizeType Index = 0;
		for (size_t i = 0; i < _Indices.Size(); ++i)
		{
			const SizeType Idx = CalcIndex(_Indices[i], _MyShape[i]);
			Index += ((Idx * _MyViewStride[i]) + _MyViewLeft[i]) * _MyViewStep[i];
		}
		return _MyData + Index;
	}

	/**
	 * @brief Get a val of the tensor with the specified indices.
	 * @param Index The indices.
	 * @return The val.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Get(SizeType Index) const
	{
		return *(_MyFirst.get() + Index);
	}

	/**
	 * @brief Get a val of the tensor with the specified indices.
	 * @param _Indices The indices.
	 * @return The val.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Item(const Dimensions& _Indices) const
	{
		return *Data(_Indices);
	}

	/**
	 * @brief Get the first val of the tensor.
	 * @return The val.
	 */
	DRAGONIANLIBCONSTEXPR decltype(auto) Item() const
	{
		return *(Data());
	}

	//******************************************************Operator******************************************************//

	void Assign(ValueType _Value) const
	{
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssign(
			_MyData,
			GetShapeInfo(),
			_Value,
			!IsBroadCasted() && IsContinuous()
		);
	}

	void Assign(const ValueType* _Buffer, SizeType _Count) const
	{
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssign(
			_MyData,
			GetShapeInfo(),
			_Buffer,
			_Count,
			!IsBroadCasted() && IsContinuous()
		);
	}

	void Assign(const Tensor& _Val) const
	{
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssign(
			_MyData,
			GetShapeInfo(),
			_Val.Data(),
			_Val.GetShapeInfo(),
			!IsBroadCasted() && !_Val.IsBroadCasted() && IsContinuous() && _Val.IsContinuous()
		);
	}

	/**
	 * @brief Assign the tensor with ones.
	 */
	DRAGONIANLIBCONSTEXPR void FixOnes() const
	{
		Assign(ValueType(1));
	}

	/**
	 * @brief Assign the tensor with zeros.
	 */
	DRAGONIANLIBCONSTEXPR void FixZeros() const
	{
		Assign(ValueType(0));
	}

	/**
	 * @brief Assign the tensor with a constant value.
	 * @param _Val The constant value.
	 */
	DRAGONIANLIBCONSTEXPR void Fix(ValueType _Val) const
	{
		Assign(_Val);
	}

	/**
	 * @brief Assign the tensor with random values.
	 */
	void RandFix() const
	{
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRand(
			_MyData,
			GetShapeInfo(),
			!IsBroadCasted() && IsContinuous()
		);
	}

	/**
	 * @brief Assign the tensor with random values.
	 * @param _Mean The mean value of the random values.
	 * @param _Sigma The sigma value of the random values.
	 */
	void RandnFix(double _Mean = 0., double _Sigma = 1.) const
	{
		Operators::OperatorsBase<_TensorType, _MyDevice>::ImplAssignRandn(
			_MyData,
			GetShapeInfo(),
			_Mean,
			_Sigma,
			!IsBroadCasted() && IsContinuous()
		);
	}

	Tensor operator+(const Tensor& _Right) const;
	Tensor operator-(const Tensor& _Right) const;
	Tensor operator*(const Tensor& _Right) const;
	Tensor operator/(const Tensor& _Right) const;
	Tensor operator+(ValueType _Right) const;
	Tensor operator-(ValueType _Right) const;
	Tensor operator*(ValueType _Right) const;
	Tensor operator/(ValueType _Right) const;

	Tensor& operator+=(const Tensor& _Right);
	Tensor& operator-=(const Tensor& _Right);
	Tensor& operator*=(const Tensor& _Right);
	Tensor& operator/=(const Tensor& _Right);
	Tensor& operator+=(ValueType _Right);
	Tensor& operator-=(ValueType _Right);
	Tensor& operator*=(ValueType _Right);
	Tensor& operator/=(ValueType _Right);

	Tensor operator!=(const Tensor& _Right) const;
	Tensor operator==(const Tensor& _Right) const;
	Tensor operator<(const Tensor& _Right) const;
	Tensor operator>(const Tensor& _Right) const;
	Tensor operator<=(const Tensor& _Right) const;
	Tensor operator>=(const Tensor& _Right) const;
	Tensor operator!=(ValueType _Right) const;
	Tensor operator==(ValueType _Right) const;
	Tensor operator<(ValueType _Right) const;
	Tensor operator>(ValueType _Right) const;
	Tensor operator<=(ValueType _Right) const;
	Tensor operator>=(ValueType _Right) const;

	//*********************************************************Info*********************************************************//

	/**
	 * @brief Get the shape info of the tensor.
	 * @param CurrentRank The current rank of the tensor.
	 * @return The shape info of the tensor.
	 */
	Operators::TensorShapeInfo GetShapeInfo(SizeType CurrentRank = INT64_MAX) const
	{
		if (CurrentRank == INT64_MAX)
			CurrentRank = Rank();

		if (CurrentRank > 6)
			DragonianLibThrow("The Rank Of The Tensor Is Too High! In General, Axis Which Greater Than 6 Is A Batch Axis, You Can Use Invoke() Or Write A Loop.");
		Operators::TensorShapeInfo Ret;
		SizeType i = 0;
		SizeType Count = 6 - CurrentRank;
		while (i < Count)
		{
			Ret.Shape[i] = 1;
			Ret.ViewStep[i] = 1;
			Ret.ViewLeft[i] = 0;
			Ret.ViewStride[i] = 1;
			++i;
		}
		for (; i < 6; ++i)
		{
			const auto CurIndex = i - Count;
			Ret.Shape[i] = _MyShape[CurIndex];
			Ret.ViewStep[i] = _MyViewStep[CurIndex];
			Ret.ViewLeft[i] = _MyViewLeft[CurIndex];
			Ret.ViewStride[i] = _MyViewStride[CurIndex];
		}
		return Ret;
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR const Dimensions& Shape() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index 
	 * @return 
	 */
	DRAGONIANLIBCONSTEXPR SizeType Shape(SizeType _Index) const
	{
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the shape of the tensor.
	 * @return The shape of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR const Dimensions& Size() const
	{
		return _MyShape;
	}

	/**
	 * @brief Get the shape of the specified axis of the tensor.
	 * @param _Index
	 * @return
	 */
	DRAGONIANLIBCONSTEXPR SizeType Size(SizeType _Index) const
	{
		return _MyShape[_Index];
	}

	/**
	 * @brief Get the rank of the tensor.
	 * @return The rank of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR SizeType Rank() const
	{
		return _MyShape.Size();
	}

	/**
	 * @brief Get the strides of the tensor.
	 * @return The strides of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR const Dimensions& ViewStrides() const
	{
		return _MyViewStride;
	}

	/**
	 * @brief Get the steps of the tensor.
	 * @return The steps of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR const Dimensions& ViewSteps() const
	{
		return _MyViewStep;
	}

	/**
	 * @brief Get the left indices of the tensor.
	 * @return The left indices of the tensor.
	 */
	DRAGONIANLIBCONSTEXPR const Dimensions& ViewLeft() const
	{
		return _MyViewLeft;
	}

	/**
	 * @brief Get the default slice vector of the tensor.
	 * @return The default slice vector of the tensor.
	 */
	SliceOptions GetDefaultSliceVector() const
	{
		SliceOptions Ret;
		Ret.Reserve(_MyShape.Size());
		for (auto i : _MyShape)
			Ret.EmplaceBack(0, i);
		return Ret;
	}

	/**
	 * @brief Get the continuous access order of the tensor.
	 * @return The continuous access order of the tensor.
	 */
	Dimensions CalcContinuousAccessOrder() const
	{
		const auto Dims = Rank();
		if (Dims == 1)
			return Dimensions(6, 0, _MyShape.GetAllocator());
		Vector<std::pair<SizeType, SizeType>> Ret;
		Ret.Reserve(Dims);
		for (SizeType i = 0; i < Dims; ++i)
			Ret.EmplaceBack(_MyViewStep[i], i);
		std::ranges::sort(Ret);
		std::ranges::reverse(Ret);
		Dimensions Rtn;
		for (const auto& i : Ret | std::views::values)
			Rtn.EmplaceBack(i);
		return Rtn;
	}

	//********************************************************Check********************************************************//

	/**
	 * @brief Check if the tensor is enabled.
	 * @return True if the tensor is enabled, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsEnabled() const
	{
		return _MyData != nullptr;
	}

	/**
	 * @brief Check if the tensor is scalar.
	 * @return True if the tensor is scalar, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsScalar() const
	{
		return _MyShape.Size() == 1 && _MyShape[0] == 1;
	}

	/**
	 * @brief Check if the tensor is vector.
	 * @return True if the tensor is vector, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsVector() const
	{
		return _MyShape.Size() == 1;
	}

	/**
	 * @brief Check if the tensor is continuous in the specified range.
	 * @param _Begin start axis
	 * @param _End end axis
	 * @return True if the tensor is continuous, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsContinuous(SizeType _Begin = 0, SizeType _End = INT64_MAX) const
	{
		if (_End == INT64_MAX)
			_End = Rank();

		_Begin = CalcIndex(_Begin, Rank());
		_End = CalcRange(_End, Rank());

		for (size_t i = _Begin; i < _End; ++i)
			if (_MyViewStride[i] != 1 || _MyViewLeft[i] != 0 || (Rank() > 1 && _MyViewStep[i - 1] / _MyViewStep[i] != _MyViewStep[i]))
				return false;

		return true;
	}

	/**
	 * @brief Check if the tensor is view.
	 * @return True if the tensor is view, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsView() const
	{
		return _MyData != _MyFirst.get();
	}

	/**
	 * @brief Check if the tensor is broadcasted.
	 * @return True if the tensor is broadcasted, false otherwise.
	 */
	DRAGONIANLIBCONSTEXPR bool IsBroadCasted() const
	{
		return IsBroadCasted_;
	}

private:

	DRAGONIANLIBCONSTEXPR void ThrowOnNotEnabled() const
	{
		if (!IsEnabled())
			DragonianLibFatalError;
	}

public:
	//*******************************************************Iterator*******************************************************//

	/**
	 * @brief Add 1 to the indices of a loop iterator.
	 * @param _Indices The indices of the loop iterator.
	 */
	DRAGONIANLIBCONSTEXPR void IteratorAdd(Dimensions& _Indices) const
	{
		auto Val = _Indices.Data() + _Indices.Size() - 1;
		const auto ShapePtr = _MyShape.Data();
		for (size_t i = _Indices.Size() - 1; ; --i)
		{
			const auto Ret = *Val + 1;
			if (Ret < *(ShapePtr + i))
			{
				*Val = Ret;
				return;
			}
			if (i == 0)
				return;
			*Val = 0;
			--Val;
		}
	}

	/**
	 * @brief Sub 1 to the indices of a loop iterator.
	 * @param _Indices The indices of the loop iterator.
	 */
	DRAGONIANLIBCONSTEXPR void IteratorSub(Dimensions& _Indices) const
	{
		auto Val = _Indices.Data() + _Indices.Size() - 1;
		const auto ShapePtr = _MyShape.Data();

		for (size_t i = _Indices.Size() - 1; ; --i)
		{
			const auto Ret = *Val - 1;
			if (Ret >= 0)
			{
				*Val = Ret;
				return;
			}
			if (i == 0)
				return;
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
	static DRAGONIANLIBCONSTEXPR SizeType CalcIndex(SizeType _Index, SizeType _Max)
	{
		if (_Index < 0)
			_Index += _Max;
		if (_Index >= _Max || _Index < 0)
			DragonianLibThrow("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Transform the range index which is negative to the positive range index and check if it is out of range.
	 * @param _Index The index to transform.
	 * @param _Max The max index.
	 * @return The transformed index.
	 */
	static DRAGONIANLIBCONSTEXPR SizeType CalcRange(SizeType _Index, SizeType _Max)
	{
		if (_Index < 0)
			_Index += _Max + 1;
		if (_Index > _Max || _Index < -1)
			DragonianLibThrow("Index Out Of Range!");
		return _Index;
	}

	/**
	 * @brief Calculate the ceil of the division of two numbers.
	 * @param _Left The left number.
	 * @param _Right The right number.
	 * @return The ceil of the division.
	 */
	static DRAGONIANLIBCONSTEXPR SizeType Ceil(SizeType _Left, SizeType _Right)
	{
		auto Mul = _Left / _Right;
		if (_Left > (Mul * _Right))
			++Mul;
		return Mul;
	}

	//*********************************************************View*********************************************************//

	/**
	 * @brief Create a view of the tensor, view means the tensor has the same data but different shape, stride, and range, and the data is shared(has no copy).
	 * @return The view tensor.
	 */
	Tensor CreateView() const
	{
		return *this;
	}

	/**
	 * @brief Slice the tensor, the order of the axes is ([0, 1, ... , N_DIMS - 1]).
	 * @param _SliceOptions A [[begin, step, end]/null, ...] array of all sliced axes, null means no slice.
	 * @return A sliced tensor(view).
	 */
	Tensor Slice(const SliceOptions& _SliceOptions) const
	{
		ThrowOnNotEnabled();
		if (IsBroadCasted())
			DragonianLibThrow("Broad Casted Could Not Be Sliced!");
		if (_MyShape.Empty() || _SliceOptions.Size() > _MyShape.Size())
			DragonianLibThrow("Axis Out Of Range!");

		Tensor Ret = CreateView();
		for (size_t i = 0; i < _SliceOptions.Size(); ++i)
		{
			if (_SliceOptions[i].IsNone)
				continue;
			const auto SliceBeginPos = CalcIndex(_SliceOptions[i].Begin, _MyShape[i]);
			auto SliceEndPos = _SliceOptions[i].End;
			if (SliceEndPos > _MyShape[i] || SliceEndPos < -(_MyShape[i] + 1))
				DragonianLibThrow("Index Out Of Range!");
			if (SliceEndPos == -(_MyShape[i] + 1))
				SliceEndPos = -1;
			else if (SliceEndPos < 0)
				SliceEndPos += _MyShape[i] + 1;
			const auto SliceLength = SliceEndPos - SliceBeginPos;
			if (SliceLength == 0)
				DragonianLibThrow("Slice Length Must > 0");
			if (SliceLength > 0 && _SliceOptions[i].Step < 0 ||
				SliceLength < 0 && _SliceOptions[i].Step > 0)
				DragonianLibThrow("Step And (SliceEnd - SliceBegin) Should Have The Same Sign!");
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
	Tensor ReversedSlice(const SliceOptions& _SliceOptions) const
	{
		Vector<Range> TempRange = _SliceOptions, NewRange;
		TempRange.Resize(_MyShape.Size(), None);
		for (size_t i = TempRange.Size() - 1; i < TempRange.Size(); --i)
		{
			if (TempRange[i].IsNone)
				NewRange.EmplaceBack(None);
			else
				NewRange.EmplaceBack(TempRange[i].Begin, TempRange[i].Step, TempRange[i].End);
		}
		return Slice(NewRange);
	}

	/**
	 * @brief Permute the order of axes of a tensor, the order of original axes is ([0, 1, ... , N_DIMS - 1]). for example, we have a tensor with [N, H, C] shape, we can permute it to [N, C, H] shape with Permute([0, 2, 1])
	 * @param _PremuteOrder The new order of axes.
	 * @return A permuted tensor(view).
	 */
	Tensor Permute(const Dimensions& _PremuteOrder) const
	{
		ThrowOnNotEnabled();
		if (_MyShape.Empty() || _PremuteOrder.Size() != _MyShape.Size())
			DragonianLibThrow("N_DIMS MisMatch!");
		Tensor Ret = CreateView();
		auto TransposedDims = _PremuteOrder;
		std::ranges::sort(TransposedDims);
		if (TransposedDims[0] != 0)
			DragonianLibThrow("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");
		for (size_t i = 1; i < TransposedDims.Size(); ++i)
			if (TransposedDims[i] != TransposedDims[i - 1] + 1)
				DragonianLibThrow("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");

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
	 * @brief Swap the last axis with the specified axis. for example, we have a tensor with [N, H, W, C] shape, we can swap the last axis with the second axis with SwapLastDim(1) to get a tensor with [N, C, W, H] shape.
	 * @param _Dim The specified axis.
	 * @return A swapped tensor(view).
	 */
	Tensor SwapLastDim(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		const auto AxisCount = (SizeType)_MyShape.Size();
		_Dim = CalcIndex(_Dim, AxisCount);
		Tensor Ret = CreateView();
		if (_Dim == AxisCount - 1)
			return Ret;
		Ret._MyShape.Back() = _MyShape[_Dim];
		Ret._MyViewStep.Back() = _MyViewStep[_Dim];
		Ret._MyViewLeft.Back() = _MyViewLeft[_Dim];
		Ret._MyViewStride.Back() = _MyViewStride[_Dim];
		Ret._MyShape[_Dim] = _MyShape.Back();
		Ret._MyViewStep[_Dim] = _MyViewStep.Back();
		Ret._MyViewLeft[_Dim] = _MyViewLeft.Back();
		Ret._MyViewStride[_Dim] = _MyViewStride.Back();
		return Ret;
	}

	/**
	 * @brief Unsqueeze the tensor, add a new axis at the specified position. for example, we have a tensor with [N, C, H] shape, we can unsqueeze it at the 1st axis with UnSqueeze(1) to get a tensor with [N, 1, C, H] shape.
	 * @param _Dim The specified position.
	 * @return An unsqueezed tensor(view).
	 */
	Tensor UnSqueeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		Tensor Ret = CreateView();
		_Dim = CalcIndex(_Dim, SizeType(Ret._MyShape.Size() + 1));
		Ret._MyShape.Insert(Ret._MyShape.begin() + _Dim, 1);
		if (_Dim == SizeType(Ret._MyViewStep.Size()))
			Ret._MyViewStep.Insert(Ret._MyViewStep.begin() + _Dim, 1);
		else
			Ret._MyViewStep.Insert(Ret._MyViewStep.begin() + _Dim, *(Ret._MyViewStep.begin() + _Dim));
		Ret._MyViewLeft.Insert(Ret._MyViewLeft.begin() + _Dim, 0);
		Ret._MyViewStride.Insert(Ret._MyViewStride.begin() + _Dim, 1);
		return Ret;
	}

	/**
	 * @brief Squeeze the tensor, remove the axis with size 1 at the specified position. for example, we have a tensor with [N, 1, C, H] shape, we can squeeze it at the 1st axis with Squeeze(1) to get a tensor with [N, C, H] shape.
	 * @param _Dim The specified position.
	 * @return A squeezed tensor(view).
	 */
	Tensor Squeeze(SizeType _Dim) const
	{
		ThrowOnNotEnabled();
		Tensor Ret = CreateView();
		_Dim = CalcIndex(_Dim, SizeType(Ret._MyShape.Size()));
		if (Ret._MyShape[_Dim] != 1)
			DragonianLibThrow("The Dim Must Be 1!");

		if (Ret._MyViewLeft[_Dim])
			_MyData += _MyViewLeft[_Dim] * _MyViewStep[_Dim];
		Ret._MyShape.Erase(Ret._MyShape.begin() + _Dim);
		Ret._MyViewStep.Erase(Ret._MyViewStep.begin() + _Dim);
		Ret._MyViewLeft.Erase(Ret._MyViewLeft.begin() + _Dim);
		Ret._MyViewStride.Erase(Ret._MyViewStride.begin() + _Dim);
		return Ret;
	}

	/**
	 * @brief Squeeze the tensor, remove all axes with size 1. for example, we have a tensor with [N, 1, C, 1, H] shape, we can squeeze it with Squeeze() to get a tensor with [N, C, H] shape.
	 * @return A squeezed tensor(view).
	 */
	Tensor Squeeze() const
	{
		ThrowOnNotEnabled();
		Tensor Ret = CreateView();
		for (size_t i = 0; i < Ret._MyShape.Size();)
		{
			if (Ret._MyShape[i] == 1)
			{
				if (Ret._MyViewLeft[i])
					_MyData += _MyViewLeft[i] * _MyViewStep[i];
				Ret._MyShape.Erase(Ret._MyShape.begin() + i);
				Ret._MyViewStep.Erase(Ret._MyViewStep.begin() + i);
				Ret._MyViewLeft.Erase(Ret._MyViewLeft.begin() + i);
				Ret._MyViewStride.Erase(Ret._MyViewStride.begin() + i);
			}
			else
				++i;
		}
		return Ret;
	}

	/**
	 * @brief View the tensor with the specified shape. for example, we have a tensor with [N, C, H, W] shape, we can view it with View([N, -1]) to get a tensor with [N, C * H * W] shape.
	 * @param _ViewShape The specified shape.
	 * @return A viewed tensor(view).
	 */
	Tensor View(const Dimensions& _ViewShape)
	{
		if (!IsContinuous())
			DragonianLibThrow("View Should Be Continuous!");
		if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), -1) > 1)
			DragonianLibThrow("Count Of Dynamic Axis Should <= 1!");
		for (const auto i : _ViewShape)
			if (i <= 0 && i != -1)
				DragonianLibThrow("Count Of Size Should > 0 Or = -1 (Dynamic Axis)!");
		Tensor Ret = CreateView();
		const auto SrcSize = VectorMul(Ret._MyShape);
		const auto DstSize = VectorMul(_ViewShape);
		if ((DstSize < 0 && (SrcSize % abs(DstSize)) != 0) || (DstSize > 0 && (SrcSize != DstSize)))
			DragonianLibThrow("Size MisMatch!");
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
		Ret._MyViewLeft = { Ret._MyShape.Size(), 0ll, Ret._MyShape.GetAllocator() };
		Ret._MyViewStride = { Ret._MyShape.Size(), 1ll, Ret._MyShape.GetAllocator() };
		return Ret;
	}

	Tensor Clone() const;
	Tensor& MakeContinuous();

	//********************************************************Operation********************************************************//

	static void Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn);

	void Invoke(SizeType InvokedDim, InvokeFnType _Fn);

	Tensor Gather(
		const Tensor& _Indices,
		SizeType _Axis = 0
	) const;

	Tensor Gather(
		const Tensor& _Indices
	) const
	{
		return Gather(_Indices, 0);
	}

	Tensor Cast() const;

	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		ValueType _Val
	);

	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		ValueType _Val
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

protected:

	static std::pair<Tensor, Tensor> BroadCast(const Tensor& _A, const Tensor& _B)
	{
		std::pair Ret{ _A.CreateView(), _B.CreateView() };
		Tensor& First = Ret.first;
		Tensor& Second = Ret.second;
		const auto Dims = std::max(First._MyShape.Size(), Second._MyShape.Size());
		std::ranges::reverse(First._MyShape);
		std::ranges::reverse(Second._MyShape);
		std::ranges::reverse(First._MyViewStep);
		std::ranges::reverse(Second._MyViewStep);
		std::ranges::reverse(First._MyViewLeft);
		std::ranges::reverse(Second._MyViewLeft);
		std::ranges::reverse(First._MyViewStride);
		std::ranges::reverse(Second._MyViewStride);
		for (size_t i = 0; i < Dims; ++i)
		{
			auto XSize = 1ll, YSize = 1ll;
			if (i < First._MyShape.Size())
				XSize = First._MyShape[i];
			else
			{
				First._MyShape.EmplaceBack(1);
				First._MyViewStep.EmplaceBack(1);
				First._MyViewLeft.EmplaceBack(0);
				First._MyViewStride.EmplaceBack(0);
				First.IsBroadCasted_ = true;
			}
			if (i < Second._MyShape.Size())
				YSize = Second._MyShape[i];
			else
			{
				Second._MyShape.EmplaceBack(1);
				Second._MyViewStep.EmplaceBack(1);
				Second._MyViewLeft.EmplaceBack(0);
				Second._MyViewStride.EmplaceBack(0);
				Second.IsBroadCasted_ = true;
			}
			if (XSize == YSize)
				continue;
			if (XSize == 1)
			{
				First._MyShape[i] = YSize;
				First._MyViewStride[i] = 0;
				First.IsBroadCasted_ = true;
			}
			else if (YSize == 1)
			{
				Second._MyShape[i] = XSize;
				Second._MyViewStride[i] = 0;
				Second.IsBroadCasted_ = true;
			}
			else
				DragonianLibThrow("TensorA & TensorB Can Not Be BroadCast!");
		}
		std::ranges::reverse(First._MyShape);
		std::ranges::reverse(Second._MyShape);
		std::ranges::reverse(First._MyViewStep);
		std::ranges::reverse(Second._MyViewStep);
		std::ranges::reverse(First._MyViewLeft);
		std::ranges::reverse(Second._MyViewLeft);
		std::ranges::reverse(First._MyViewStride);
		std::ranges::reverse(Second._MyViewStride);
		return Ret;
	}

	Tensor BroadCast(const Tensor& _Other) const
	{
		decltype(auto) Bd = BroadCast(*this, _Other);
		if (Bd.first.IsBroadCasted())
			DragonianLibThrow("TensorA & TensorB Can Not  Be BroadCast At The Same Time In This Operator!");
		return std::move(Bd.second);
	}

	DRAGONIANLIBCONSTEXPR Tensor GatherRef(SizeType _Index) const
	{
		const auto Idx = CalcIndex(_Index, _MyShape.Front());
		Tensor Ret;

		Ret._MyShape = { _MyShape.begin() + 1,_MyShape.end(), _MyShape.GetAllocator() };
		Ret._MyViewStep = { _MyViewStep.begin() + 1,_MyViewStep.end(), _MyShape.GetAllocator() };
		Ret._MyViewLeft = { _MyViewLeft.begin() + 1,_MyViewLeft.end(), _MyShape.GetAllocator() };
		Ret._MyViewStride = { _MyViewStride.begin() + 1,_MyViewStride.end(), _MyShape.GetAllocator() };

		auto Index = _MyViewLeft.Front() + (Idx * _MyViewStride.Front());
		Index = ((Index * _MyViewStride.Front()) + _MyViewLeft.Front()) * _MyViewStep.Front();
		Ret._MyData = _MyData + Index;
		Ret._MyFirst = _MyFirst;
		Ret._MyLast = _MyLast;
		Ret._MyAllocator = _MyAllocator;
		Ret.IsBroadCasted_ = IsBroadCasted_;

		if (Ret._MyShape.Empty())
		{
			Ret._MyShape.EmplaceBack(1);
			Ret._MyViewStep.EmplaceBack(1);
			Ret._MyViewLeft.EmplaceBack(0);
			Ret._MyViewStride.EmplaceBack(1);
		}

		return Ret;
	}

public:
	static Tensor Pow(const Tensor& _InputA, const Tensor& _InputB);
	static Tensor Pow(const Tensor& _InputA, float64 _Val);
	Tensor Pow(const Tensor& _InputB) const;
	Tensor Pow(float64 _Val) const;
	Tensor& Pow_(const Tensor& _InputB);
	Tensor& Pow_(float64 _Val);

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

DragonianLibSpaceEnd