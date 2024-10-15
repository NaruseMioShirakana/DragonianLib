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
#include "Tensor/TensorBase.h"
#include "Util/ThreadPool.h"
#include "Tensor/Macro.h"
#include "MyTemplateLibrary/Vector.h"

DragonianLibSpaceBegin

using ShapeType = Vector<SizeType>; ///< Alias for vector of size types
using ShapeIterator = ShapeType::iterator; ///< Alias for iterator of shape type
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
	Nearest2P, ///< Nearest neighbor interpolation for 2P
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
SizeType VectorMul(const ShapeType& _Input);

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
ShapeType GetBeginIndices(const SliceOptions& _Input);

/**
 * @brief Check if all ranges in the vector are none.
 * @param _Input The input vector of ranges.
 * @return True if all ranges are none, false otherwise.
 */
bool RangeIsAllNone(const Vector<Range>& _Input);

/**
 * @class Tensor
 * @brief Class representing a tensor.
 */
class Tensor : public TensorBase
{
public:
	using InvokeFnType = void(*)(Tensor&); ///< Alias for invoke function type

	Tensor() = delete; ///< Deleted default constructor
	~Tensor() override; ///< Destructor
	Tensor(const Tensor& _Left); ///< Copy constructor
	Tensor(Tensor&& _Right) noexcept; ///< Move constructor

	/**
	 * @brief Constructor for a tensor with a data type and device.
	 * @param _DType The data type of the tensor.
	 * @param _Device The device of the tensor.
	 */
	Tensor(TensorType _DType, Device _Device) :TensorBase(_DType), Device_(GetMemoryProvider(_Device)) {}

	/**
	 * @brief Constructor for a tensor with a shape, data type, and device (same as torch.empty()).
	 * @param _Shape The shape of the tensor.
	 * @param _DType The data type of the tensor.
	 * @param _Device The device of the tensor.
	 */
	Tensor(const ShapeType& _Shape, TensorType _DType, Device _Device);

	/**
	 * @brief Create a tensor from a vector of float32 values.
	 *
	 * @param _Array The input vector of float32 values.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor FloatTensor(const Vector<float32>& _Array, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor from a vector of int64 values.
	 *
	 * @param _Array The input vector of int64 values.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor LongTensor(const Vector<int64>& _Array, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with ones.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Type The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Ones(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with zeros.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Type The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Zeros(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with a constant value.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value.
	 * @param _Type The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor ConstantOf(const ShapeType& _Shape, double _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with a constant value.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Val The constant value.
	 * @param _Type The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor ConstantOf(const ShapeType& _Shape, int64 _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with random values.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Type The type of the tensor.
	 * @param _Seed The seed for the random number generator.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Rand(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor and fill it with random values.
	 *
	 * @param _Shape The shape of the tensor.
	 * @param _Type The type of the tensor.
	 * @param _Seed The seed for the random number generator.
	 * @param _Mean The mean of the random values.
	 * @param _Sigma The standard deviation of the random values.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Randn(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with ones.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with ones.
	 */
	static Tensor OnesLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with zeros.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with zeros.
	 */
	static Tensor ZerosLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with a constant value.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _Val The constant value to fill the new tensor with.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with the constant value.
	 */
	static Tensor ConstantLike(const Tensor& _Shape, double _Val, ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with a constant value.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _Val The constant value to fill the new tensor with.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with the constant value.
	 */
	static Tensor ConstantLike(const Tensor& _Shape, int64 _Val, ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with random values.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _Seed The seed for the random number generator.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with random values.
	 */
	static Tensor RandLike(const Tensor& _Shape, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Creates a tensor with the same shape as the given tensor, filled with random values.
	 *
	 * @param _Shape The tensor whose shape will be used to create the new tensor.
	 * @param _Seed The seed for the random number generator.
	 * @param _Mean The mean of the random values.
	 * @param _Sigma The standard deviation of the random values.
	 * @param _ThreadPool Optional thread pool for parallel operations.
	 *
	 * @return A new tensor with the same shape as the given tensor, filled with random values.
	 */
	static Tensor RandnLike(const Tensor& _Shape, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Create a tensor with a range of values.
	 *
	 * @param _Begin The begining value.
	 * @param _End The end value.
	 * @param _Step The step value.
	 * @param _Dtype The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Arange(float64 _Begin, float64 _End, float64 _Step, TensorType _Dtype = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor with a range of values.
	 *
	 * @param _Begin The begining value.
	 * @param _End The end value.
	 * @param _Step The step value.
	 * @param _Dtype The type of the tensor.
	 * @param _ThreadPool The thread pool to use.
	 * @param _Device The device to use.
	 *
	 * @return The created tensor.
	 */
	static Tensor Arange(int64 _Begin, int64 _End, int64 _Step, TensorType _Dtype = TensorType::Int64, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);

	/**
	 * @brief Create a tensor from a vector (will not be copied and the vector will lose ownership of the data).
	 *
	 * @tparam _ValueType The type of the vector.
	 *
	 * @param _Vector The input vector.
	 * @param _Shape The shape of the tensor.
	 *
	 * @return The created tensor.
	 */
	template<typename _ValueType>
	Tensor(DragonianLibSTL::Vector<_ValueType>& _Vector, const ShapeType& _Shape)
	{
		if ((size_t)VectorMul(_Shape) != _Vector.Size())
			DragonianLibThrow("Size MisMatch!");

		if constexpr (std::is_same_v<_ValueType, int8>)
			DType_ = TensorType::Int8;
		else if constexpr (std::is_same_v<_ValueType, int16>)
			DType_ = TensorType::Int16;
		else if constexpr (std::is_same_v<_ValueType, int32>)
			DType_ = TensorType::Int32;
		else if constexpr (std::is_same_v<_ValueType, int64>)
			DType_ = TensorType::Int64;
		else if constexpr (std::is_same_v<_ValueType, float8>)
			DType_ = TensorType::Float8;
		else if constexpr (std::is_same_v<_ValueType, float16>)
			DType_ = TensorType::Float16;
		else if constexpr (std::is_same_v<_ValueType, float32>)
			DType_ = TensorType::Float32;
		else if constexpr (std::is_same_v<_ValueType, float64>)
			DType_ = TensorType::Float64;
		else
			DragonianLibNotImplementedError;

		Device_ = _Vector.GetAllocator();
		AlignSize_ = DType2Size(DType_);
		ShapeBack_ = _Shape;
		StepFront_.clear();
		StepBack_ = { _Shape.begin() + 1,_Shape.end(), ShapeType::allocator_type() };
		StepBack_.emplace_back(AlignSize_);
		std::ranges::reverse(StepBack_);
		for (size_t i = 1; i < StepBack_.size(); ++i)
			StepBack_[i] *= StepBack_[i - 1];
		std::ranges::reverse(StepBack_);
		SliceBegin_ = { _Shape.size(),0ll, ShapeType::allocator_type() };
		DimStride_ = { _Shape.size(),1ll, ShapeType::allocator_type() };
		CurIndices_.clear();

		ViewParent_ = nullptr;
		DataPtr_ = (byte*)_Vector.Release().first;
		ViewChild_.clear();
	}

	/**
	 * @brief Create a vector view from the tensor (the vector will not have ownership of the data).
	 *
	 * @tparam _ValueType
	 *
	 * @return The created vector view.
	 */
	template<typename _ValueType>
	DragonianLibSTL::Vector<_ValueType> VectorView()
	{
		if (sizeof(_ValueType) != AlignSize_)
			DragonianLibThrow("Type MisMatch!");
		if (IsView())
			DragonianLibThrow("Tensor View Could Not Have Vector View!");
		std::lock_guard LockRel(RelMx_);
		if (!DataPtr_)
			return {};
		auto Ptr = (_ValueType*)DataPtr_;
		return { &Ptr, (size_t)VectorMul(ShapeBack_), Device_, false };
	}

	/**
	 * @brief Set the thread count for the tensor operations.
	 *
	 * @param _Count The thread count.
	 */
	static void SetThreadCount(SizeType _Count);

	/**
	 * @brief Enable the time logger for the tensor operations.
	 *
	 * @param _Enabled True to enable, false to disable.
	 */
	static void EnableTimeLogger(bool _Enabled);

	/**
	 * @brief Get the alignment size.
	 *
	 * @return The alignment size.
	 */
	SizeType GetAlignSize() const
	{
		return AlignSize_;
	}

	/**
	 * @brief Get the mutex for resource release.
	 *
	 * @return Reference to the mutex.
	 */
	std::mutex& GetRelMx() const
	{
		return RelMx_;
	}

	/**
	 * @brief Get the device.
	 *
	 * @return The device.
	 */
	Device GetDevice() const
	{
		return Device_->GetDevice();
	}

	/**
	 * @brief Get the allocator.
	 *
	 * @return The allocator.
	 */
	Allocator GetAllocator() const
	{
		return Device_;
	}

	/**
	 * @brief Get the element at the specified index from raw data.
	 *
	 * @tparam _Ty The type of the element.
	 *
	 * @param Index The index of the element.
	 *
	 * @return Reference to the element.
	 */
	template<typename _Ty>
	_Ty& Get(SizeType Index)
	{
		return *((_Ty*)DataPtr_ + Index);
	}

protected:
	byte* DataPtr_ = nullptr; ///< Pointer to the data
	Tensor* ViewParent_ = nullptr; ///< Pointer to the view parent, view means the tensor has no ownership of the data but has the different attributes (like shape, strides, etc.), for better performance.

	ShapeType ShapeBack_; ///< Shape of the tensor
	ShapeType StepFront_, StepBack_; ///< Steps of the tensor
	ShapeType SliceBegin_; ///< Begining indices of the tensor
	ShapeType DimStride_; ///< Strides of the tensor
	ShapeType CurIndices_; ///< Current indices of the tensor
	int64_t AlignSize_ = 4; ///< Alignment size of the tensor
	bool IsBroadCasted_ = false; ///< Flag indicating if the tensor is broadcasted
	std::deque<Tensor*> ViewChild_; ///< Child views of the tensor
	mutable std::mutex ViewMx_, RelMx_; ///< Mutex for view and resource release
	SliceOptions OpSlice; ///< Slice options for operations
	
public:

	/**
	 * @brief Remind the operator not to use the thread pool.
	 *
	 * @return Reference to the current Tensor object.
	 */
	Tensor& DisableThreadPool()
	{
		UseThreadPool_ = false;
		return *this;
	}

	/**
	 * @brief Remind the operator to use the thread pool.
	 *
	 * @return Reference to the current Tensor object.
	 */
	Tensor& PlUseThreadPool()
	{
		UseThreadPool_ = true;
		return *this;
	}

	/**
	 * @brief Get an element using indices.
	 *
	 * @tparam Ref The type of the element.
	 *
	 * @param _Indices The indices to access the element.
	 *
	 * @return Reference to the element.
	 */
	template <typename Ref>
	decltype(auto) Item(const ShapeType& _Indices)
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	/**
	 * @brief Get the first element.
	 *
	 * @tparam Ref The type of the element.
	 *
	 * @return Reference to the first element.
	 */
	template <typename Ref>
	decltype(auto) Item()
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(GetPtr());
	}

	/**
	 * @brief Get an element using indices (const version).
	 *
	 * @tparam Ref The type of the element.
	 *
	 * @param _Indices The indices to access the element.
	 *
	 * @return Const reference to the element.
	 */
	template <typename Ref>
	decltype(auto) Item(const ShapeType& _Indices) const
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	/**
	 * @brief Get the first element (const version).
	 *
	 * @tparam Ref The type of the element.
	 *
	 * @return Const reference to the first element.
	 */
	template <typename Ref>
	decltype(auto) Item() const
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(GetPtr());
	}

	/**
	 * @brief Fill the entire Tensor with data of type _Type pointed to by _Val.
	 *
	 * @param _Val Pointer to the data.
	 * @param _Type The type of the data.
	 * @param _ThreadPool Optional thread pool for parallel execution.
	 */
	void Assign(const void* _Val, TensorType _Type, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Assign the entire Tensor using a Buffer, in row-major order, until the Buffer is exhausted or the entire Tensor is filled.
	 *
	 * @param _Buffer Pointer to the buffer.
	 * @param _BufferSize Size of the buffer.
	 * @param _ThreadPool Optional thread pool for parallel execution.
	 */
	void Assign(const void* _Buffer, SizeType _BufferSize, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with the value _Val.
	 *
	 * @param _Val The value to fill the Tensor with.
	 * @param _ThreadPool Optional thread pool for parallel execution.
	 */
	void Assign(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with the value _Val.
	 *
	 * @param _Val The value to fill the Tensor with.
	 * @param _ThreadPool Optional thread pool for parallel execution.
	 */
	void Assign(float64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Assign the entire Tensor using another Tensor, the input Shape must be the same or broadcastable.
	 *
	 * @param _Val The Tensor to assign from.
	 * @param _ThreadPool Optional thread pool for parallel execution.
	 */
	void Assign(const Tensor& _Val, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Replace the current Tensor with a View of another Tensor (no copy); if the input is a View, the source of the other View cannot be the current Tensor (use Clone function to copy the Tensor).
	 *
	 * @param _Left The Tensor to assign from.
	 * @return Reference to the assigned Tensor.
	 */
	Tensor& operator=(const Tensor& _Left);

	/**
	 * @brief Move assignment.
	 *
	 * @param _Right The Tensor to move from.
	 * @return Reference to the assigned Tensor.
	 */
	Tensor& operator=(Tensor&& _Right) noexcept;

	/**
	 * @brief Fill the entire Tensor with the value _Val.
	 *
	 * @param _Val The value to fill the Tensor with.
	 * @return Reference to the assigned Tensor.
	 */
	Tensor& operator=(int64 _Val);

	/**
	 * @brief Fill the entire Tensor with the value _Val.
	 *
	 * @param _Val The value to fill the Tensor with.
	 * @return Reference to the assigned Tensor.
	 */
	Tensor& operator=(float64 _Val);

	/**
	 * @brief Add two Tensors element-wise.
	 *
	 * @param _Right The Tensor to add.
	 * @return A new Tensor containing the result of the addition.
	 */
	Tensor operator+(const Tensor& _Right) const;

	/**
	 * @brief Subtract one Tensor from another element-wise.
	 *
	 * @param _Right The Tensor to subtract.
	 * @return A new Tensor containing the result of the subtraction.
	 */
	Tensor operator-(const Tensor& _Right) const;

	/**
	 * @brief Multiply two Tensors element-wise.
	 *
	 * @param _Right The Tensor to multiply.
	 * @return A new Tensor containing the result of the multiplication.
	 */
	Tensor operator*(const Tensor& _Right) const;

	/**
	 * @brief Divide one Tensor by another element-wise.
	 *
	 * @param _Right The Tensor to divide by.
	 * @return A new Tensor containing the result of the division.
	 */
	Tensor operator/(const Tensor& _Right) const;

	/**
	 * @brief Add a scalar value to each element of the Tensor.
	 *
	 * @param _Right The scalar value to add.
	 * @return A new Tensor containing the result of the addition.
	 */
	Tensor operator+(int64 _Right) const;

	/**
	 * @brief Subtract a scalar value from each element of the Tensor.
	 *
	 * @param _Right The scalar value to subtract.
	 * @return A new Tensor containing the result of the subtraction.
	 */
	Tensor operator-(int64 _Right) const;

	/**
	 * @brief Multiply each element of the Tensor by a scalar value.
	 *
	 * @param _Right The scalar value to multiply by.
	 * @return A new Tensor containing the result of the multiplication.
	 */
	Tensor operator*(int64 _Right) const;

	/**
	 * @brief Divide each element of the Tensor by a scalar value.
	 *
	 * @param _Right The scalar value to divide by.
	 * @return A new Tensor containing the result of the division.
	 */
	Tensor operator/(int64 _Right) const;

	/**
	 * @brief Add a scalar value to each element of the Tensor.
	 *
	 * @param _Right The scalar value to add.
	 * @return A new Tensor containing the result of the addition.
	 */
	Tensor operator+(float64 _Right) const;

	/**
	 * @brief Subtract a scalar value from each element of the Tensor.
	 *
	 * @param _Right The scalar value to subtract.
	 * @return A new Tensor containing the result of the subtraction.
	 */
	Tensor operator-(float64 _Right) const;

	/**
	 * @brief Multiply each element of the Tensor by a scalar value.
	 *
	 * @param _Right The scalar value to multiply by.
	 * @return A new Tensor containing the result of the multiplication.
	 */
	Tensor operator*(float64 _Right) const;

	/**
	 * @brief Divide each element of the Tensor by a scalar value.
	 *
	 * @param _Right The scalar value to divide by.
	 * @return A new Tensor containing the result of the division.
	 */
	Tensor operator/(float64 _Right) const;


	/**
	 * @brief Add another Tensor to this Tensor element-wise and assign the result to this Tensor.
	 *
	 * @param _Right The Tensor to add.
	 * @return A reference to this Tensor after the addition.
	 */
	Tensor& operator+=(const Tensor& _Right);

	/**
	 * @brief Subtract another Tensor from this Tensor element-wise and assign the result to this Tensor.
	 *
	 * @param _Right The Tensor to subtract.
	 * @return A reference to this Tensor after the subtraction.
	 */
	Tensor& operator-=(const Tensor& _Right);

	/**
	 * @brief Multiply this Tensor by another Tensor element-wise and assign the result to this Tensor.
	 *
	 * @param _Right The Tensor to multiply by.
	 * @return A reference to this Tensor after the multiplication.
	 */
	Tensor& operator*=(const Tensor& _Right);

	/**
	 * @brief Divide this Tensor by another Tensor element-wise and assign the result to this Tensor.
	 *
	 * @param _Right The Tensor to divide by.
	 * @return A reference to this Tensor after the division.
	 */
	Tensor& operator/=(const Tensor& _Right);

	/**
	 * @brief Add a scalar value to each element of this Tensor and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to add.
	 * @return A reference to this Tensor after the addition.
	 */
	Tensor& operator+=(int64 _Right);

	/**
	 * @brief Subtract a scalar value from each element of this Tensor and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to subtract.
	 * @return A reference to this Tensor after the subtraction.
	 */
	Tensor& operator-=(int64 _Right);

	/**
	 * @brief Multiply each element of this Tensor by a scalar value and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to multiply by.
	 * @return A reference to this Tensor after the multiplication.
	 */
	Tensor& operator*=(int64 _Right);

	/**
	 * @brief Divide each element of this Tensor by a scalar value and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to divide by.
	 * @return A reference to this Tensor after the division.
	 */
	Tensor& operator/=(int64 _Right);

	/**
	 * @brief Add a scalar value to each element of this Tensor and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to add.
	 * @return A reference to this Tensor after the addition.
	 */
	Tensor& operator+=(float64 _Right);

	/**
	 * @brief Subtract a scalar value from each element of this Tensor and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to subtract.
	 * @return A reference to this Tensor after the subtraction.
	 */
	Tensor& operator-=(float64 _Right);

	/**
	 * @brief Multiply each element of this Tensor by a scalar value and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to multiply by.
	 * @return A reference to this Tensor after the multiplication.
	 */
	Tensor& operator*=(float64 _Right);

	/**
	 * @brief Divide each element of this Tensor by a scalar value and assign the result to this Tensor.
	 *
	 * @param _Right The scalar value to divide by.
	 * @return A reference to this Tensor after the division.
	 */
	Tensor& operator/=(float64 _Right);


	/**
	 * @brief Access a specific element of the Tensor by index.
	 *
	 * @param _Index The index of the element to access.
	 * @return A Tensor representing the element at the specified index.
	 */
	Tensor operator[](SizeType _Index) const;

	/**
	 * @brief Access a slice of the Tensor using slice options.
	 *
	 * @param _SliceOptions The options defining the slice.
	 * @return A Tensor representing the slice defined by the slice options.
	 */
	Tensor operator[](const SliceOptions& _SliceOptions) const;

	/**
	 * @brief Access elements of the Tensor using a shape type as indices.
	 *
	 * @param _Indice The shape type defining the indices.
	 * @return A Tensor representing the elements at the specified indices.
	 */
	Tensor operator[](const ShapeType& _Indice) const;

	/**
	 * @brief Compare this Tensor with another Tensor for inequality.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the inequality comparison.
	 */
	Tensor operator!=(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with another Tensor for equality.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the equality comparison.
	 */
	Tensor operator==(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with another Tensor to check if it is less than the other.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the less-than comparison.
	 */
	Tensor operator<(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with another Tensor to check if it is greater than the other.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the greater-than comparison.
	 */
	Tensor operator>(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with another Tensor to check if it is less than or equal to the other.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the less-than-or-equal-to comparison.
	 */
	Tensor operator<=(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with another Tensor to check if it is greater than or equal to the other.
	 *
	 * @param _Right The Tensor to compare with.
	 * @return A Tensor representing the result of the greater-than-or-equal-to comparison.
	 */
	Tensor operator>=(const Tensor& _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value for inequality.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the inequality comparison.
	 */
	Tensor operator!=(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value for equality.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the equality comparison.
	 */
	Tensor operator==(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is less than the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the less-than comparison.
	 */
	Tensor operator<(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is greater than the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the greater-than comparison.
	 */
	Tensor operator>(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is less than or equal to the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the less-than-or-equal-to comparison.
	 */
	Tensor operator<=(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is greater than or equal to the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the greater-than-or-equal-to comparison.
	 */
	Tensor operator>=(float64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value for inequality.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the inequality comparison.
	 */
	Tensor operator!=(int64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value for equality.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the equality comparison.
	 */
	Tensor operator==(int64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is less than the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the less-than comparison.
	 */
	Tensor operator<(int64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is greater than the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the greater-than comparison.
	 */
	Tensor operator>(int64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is less than or equal to the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the less-than-or-equal-to comparison.
	 */
	Tensor operator<=(int64 _Right) const;

	/**
	 * @brief Compare this Tensor with a scalar value to check if it is greater than or equal to the value.
	 *
	 * @param _Right The scalar value to compare with.
	 * @return A Tensor representing the result of the greater-than-or-equal-to comparison.
	 */
	Tensor operator>=(int64 _Right) const;

private:
	
	/**
	 * @brief Free the resources associated with this Tensor.
	 */
	void Free();

	/**
	 * @brief Clear all child views of this Tensor.
	 */
	void ClearViewChilds();

	/**
	 * @brief Throw an exception if this Tensor is not enabled.
	 *
	 * @throws std::exception if the Tensor is not enabled.
	 */
	void ThrowOnNotEnabled() const;

	/**
	 * @brief Remove the pointer to this Tensor from its parent view.
	 */
	void RemoveSelfViewPtr();

	/**
	 * @brief Check if this Tensor has a specific child Tensor.
	 *
	 * @param _Child The child Tensor to check for.
	 * @return True if the child Tensor exists, false otherwise.
	 */
	bool HasChild(const Tensor* _Child) const;

	/**
	 * @brief Assign a 1D array to this Tensor.
	 *
	 * @param _Val Pointer to the 1D array to assign.
	 */
	void Assign1D(const void* _Val) const;

	/**
	 * @brief Recalculate the view information for this Tensor.
	 */
	void ReCalcViewInfo();


public:
	/**
	 * @brief Increment the index of the LoopIterator by 1.
	 *
	 * @param _Indices The indices to increment.
	 */
	void IteratorAdd(ShapeType& _Indices) const;

	/**
	 * @brief Decrement the index of the LoopIterator by 1.
	 *
	 * @param _Indices The indices to decrement.
	 */
	void IteratorSub(ShapeType& _Indices) const;

	/**
	 * @brief Calculate the index based on the given maximum value.
	 *
	 * @param _Index The index to calculate.
	 * @param _Max The maximum value.
	 *
	 * @return The calculated index.
	 */
	static SizeType CalcIndex(SizeType _Index, SizeType _Max);

	/**
	 * @brief Calculate the range based on the given maximum value.
	 *
	 * @param _Index The index to calculate.
	 * @param _Max The maximum value.
	 *
	 * @return The calculated range.
	 */
	static SizeType CalcRange(SizeType _Index, SizeType _Max);

	/**
	 * @brief Calculate the ceiling of two int64 values.
	 *
	 * @param _Left The left value.
	 * @param _Right The right value.
	 *
	 * @return The ceiling value.
	 */
	static SizeType Ceil(SizeType _Left, SizeType _Right);

	/**
	 * @brief Check if the current Tensor is enabled.
	 *
	 * @return True if the Tensor is enabled, false otherwise.
	 */
	bool IsEnabled() const;

	/**
	 * @brief Check if the current Tensor is a scalar.
	 *
	 * @return True if the Tensor is a scalar, false otherwise.
	 */
	bool IsScalar() const;

	/**
	 * @brief Check if the current Tensor has the features of a view-type Tensor.
	 *
	 * @return True if the Tensor has view features, false otherwise.
	 */
	bool HasViewedFeature() const;

	/**
	 * @brief Check if the index order of the current Tensor is strictly memory contiguous.
	 *
	 * @param _Dim The dimension to check (default is 0).
	 *
	 * @return True if the Tensor is contiguous, false otherwise.
	 */
	bool IsContinuous(SizeType _Dim = 0) const;

	/**
	 * @brief Check if the index order of the current Tensor is strictly memory contiguous.
	 *
	 * @param _SlicePos The slice options to use.
	 * @param _Dim The dimension to check (default is 0).
	 *
	 * @return 
	 */
	bool IsContinuous(const SliceOptions& _SlicePos, SizeType _Dim = 0) const;

	/**
	 * @brief Check if the current Tensor can become memory contiguous by permuting last two axes.
	 *
	 * @return True if the Tensor can be permuted to be contiguous, false otherwise.
	 */
	bool IsTranSposedContinuous() const;

	/**
	 * @brief Check if the current Tensor has a view source.
	 *
	 * @return True if the Tensor has a view source, false otherwise.
	 */
	bool IsView() const;

	/**
	 * @brief Clone the current Tensor, returning a new Tensor with independent memory.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The cloned Tensor.
	 */
	Tensor Clone(ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Create a view of the current Tensor. The view does not have independent memory and only has its own properties. The current Tensor is the view source of this view.
	 *
	 * @return The created view.
	 */
	Tensor CreateView() const;

	/**
	 * @brief Create a view of the current Tensor, slice it, and return the view.
	 *
	 * @param _SliceOptions The slice options to use.
	 *
	 * @return The sliced view.
	 */
	Tensor Slice(const SliceOptions& _SliceOptions) const;

	/**
	 * @brief Create a view of the current Tensor, reverse slice it, and return the view.
	 *
	 * @param _SliceOptions The slice options to use.
	 *
	 * @return The reversed sliced view.
	 */
	Tensor ReversedSlice(const SliceOptions& _SliceOptions) const;

	/**
	 * @brief Create a view of the current Tensor, change its axis order, and return the view.
	 *
	 * @param _DPremute The new axis order.
	 *
	 * @return The permuted view.
	 */
	Tensor Permute(const ShapeType& _DPremute) const;

	/**
	 * @brief Create a view of the current Tensor, swap its _Dim axis with the last axis, and return the view.
	 *
	 * @param _Dim The axis to swap with the last axis.
	 *
	 * @return The swapped view.
	 */
	Tensor SwapLastDim(SizeType _Dim) const;

	/**
	 * @brief Get the default slice vector.
	 *
	 * @return The default slice vector.
	 */
	SliceOptions GetDefaultSliceVector() const;

	/**
	 * @brief Invoke a function on a specified axis of a Tensor.
	 *
	 * @param _Tensor The Tensor to invoke the function on.
	 * @param InvokedDim The axis to invoke the function on.
	 * @param _Fn The function to invoke.
	 */
	static void Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn);

	/**
	 * @brief Invoke a function on a specified axis of the current Tensor.
	 *
	 * @param InvokedDim The axis to invoke the function on.
	 * @param _Fn The function to invoke.
	 */
	void Invoke(SizeType InvokedDim, InvokeFnType _Fn);

	/**
	 * @brief Get the shape of the current Tensor.
	 *
	 * @return The shape of the Tensor.
	 */
	const ShapeType& Shape() const;

	/**
	 * @brief Get the shape of a specified axis of the current Tensor.
	 *
	 * @param _Index The axis index.
	 *
	 * @return The shape of the specified axis.
	 */
	SizeType Shape(SizeType _Index) const;

	/**
	 * @brief Get the size of the current Tensor.
	 *
	 * @return The size of the Tensor.
	 */
	const ShapeType& Size() const;

	/**
	 * @brief Get the size of a specified axis of the current Tensor.
	 *
	 * @param _Index The axis index.
	 *
	 * @return The size of the specified axis.
	 */
	SizeType Size(SizeType _Index) const;

	/**
	 * @brief Get the stride information of all axes of the current Tensor.
	 *
	 * @return The stride information.
	 */
	const ShapeType& Strides() const;

	/**
	 * @brief Get the step information of all axes of the current Tensor.
	 *
	 * @return The step information.
	 */
	const ShapeType& StepsBack() const;

	/**
	 * @brief Get the slice start positions of all axes of the current Tensor.
	 *
	 * @return The slice start positions.
	 */
	const ShapeType& SliceBegins() const;

	/**
	 * @brief Fill the entire Tensor with 1.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void FixOnes(ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with 0.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void FixZeros(ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with a specified value.
	 *
	 * @param _Val The value to fill.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void Fix(double _Val, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with a specified int64 value.
	 *
	 * @param _Val The int64 value to fill.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void Fix(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with random numbers.
	 *
	 * @param _Seed The seed for the random number generator (default is 114514).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void RandFix(uint64 _Seed = 114514, ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with random numbers.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void RandFix(ThreadPool* _ThreadPool) const { RandFix(114514, _ThreadPool); }

	/**
	 * @brief Fill the entire Tensor with random numbers generated from a normal distribution.
	 *
	 * @param _Seed The seed for the random number generator (default is 114514).
	 * @param _Mean The mean of the normal distribution (default is 0.0).
	 * @param _Sigma The standard deviation of the normal distribution (default is 1.0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void RandnFix(uint64 _Seed = 114514, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr) const;

	/**
	 * @brief Fill the entire Tensor with random numbers generated from a normal distribution.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 */
	void RandnFix(ThreadPool* _ThreadPool) const { RandnFix(114514, 0., 1., _ThreadPool); }

	/**
	 * @brief Get the starting address of the buffer of the current Tensor (if the Tensor is a view, return the view source's buffer address).
	 *
	 * @return The buffer address.
	 */
	byte* Buffer() const;

	/**
	 * @brief Get the address of the current Tensor's axis.
	 *
	 * @return The axis address.
	 */
	byte* Data() const;

	/**
	 * @brief Get the address of the data at the specified indices of the current Tensor.
	 *
	 * @param _Indices The indices to get the data address for.
	 *
	 * @return The data address.
	 */
	byte* Data(const ShapeType& _Indices) const;

	/**
	 * @brief Create a view of the current Tensor and treat it as a Tensor with the specified view shape, returning the view.
	 *
	 * @param _ViewShape The view shape to use.
	 *
	 * @return The created view.
	 */
	Tensor View(const ShapeType& _ViewShape) const;

	/**
	 * @brief Make the current Tensor memory contiguous, returning a reference to the current Tensor.
	 *
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return A reference to the current Tensor.
	 */
	Tensor& Continuous(ThreadPool* _ThreadPool = nullptr);

	/**
	 * @brief Create a view of the current Tensor, insert an axis at the specified position, and return the view.
	 *
	 * @param Dim The position to insert the axis.
	 *
	 * @return The created view.
	 */
	Tensor UnSqueeze(SizeType Dim) const;

	/**
	 * @brief Create a view of the current Tensor, and if the size of the specified axis is 1, delete the axis and return the view.
	 *
	 * @param Dim The axis to delete if its size is 1.
	 *
	 * @return The created view.
	 */
	Tensor Squeeze(SizeType Dim) const;

	/**
	 * @brief Create a view of the current Tensor, delete all axes with size 1, and return the view.
	 *
	 * @return The created view.
	 */
	Tensor Squeeze() const;

	/**
	 * @brief Create views of two Tensors, broadcast them to have the same shape, and return the views.
	 *
	 * @param _A The first Tensor.
	 * @param _B The second Tensor.
	 *
	 * @return A pair of the broadcasted views.
	 */
	static std::pair<Tensor, Tensor> BroadCast(const Tensor& _A, const Tensor& _B);

	/**
	 * @brief Create a view of the input Tensor, broadcast it to have the same shape as the current Tensor, and return the view.
	 *
	 * @param _Other The input Tensor.
	 *
	 * @return The broadcasted view.
	 */
	Tensor BroadCast(const Tensor& _Other) const;

	/**
	 * @brief Check if the current Tensor is a broadcasted Tensor.
	 *
	 * @return True if the Tensor is broadcasted, false otherwise.
	 */
	bool IsBroadCasted() const;

	/**
	 * @brief Get the number of axes of the current Tensor.
	 *
	 * @return The number of axes.
	 */
	SizeType DimCount() const;

	/**
	 * @brief Check if the current Tensor is a vector.
	 *
	 * @return True if the Tensor is a vector, false otherwise.
	 */
	bool IsVector() const;

	/**
	 * @brief Get the starting address of the current Tensor's data area.
	 *
	 * @return A pointer to the starting address of the data area.
	 */
	byte* GetPtr() const;

	/**
	 * @brief Get the axis order that minimizes the traversal cost of the Tensor.
	 *
	 * @return The axis order with the minimum traversal cost.
	 */
	ShapeType CalcContinuous() const;

	bool IsTransposed(size_t _Size) const;

	/**
	 * @brief Get the Tensor at the specified indices along Axis (creates a new Tensor).
	 *
	 * @param _Indices The indices Tensor.
	 * @param _Axis The axis along which to gather (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The gathered Tensor.
	 */
	Tensor Gather(
		const Tensor& _Indices,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	) const;

	/**
	 * @brief Get the Tensor at the specified indices along Axis[0] (creates a new Tensor).
	 *
	 * @param _Indices The indices Tensor.
	 * @param _ThreadPool The thread pool to use.
	 *
	 * @return The gathered Tensor.
	 */
	Tensor Gather(
		const Tensor& _Indices,
		ThreadPool* _ThreadPool
	) const
	{
		return Gather(_Indices, 0, _ThreadPool);
	}

	/**
	 * @brief Convert the Tensor to a different type (creates a new Tensor).
	 *
	 * @param _Dtype The target data type.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The converted Tensor.
	 */
	Tensor Cast(
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	) const;

	/**
	 * @brief Apply padding to the input Tensor (forward order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Type The padding type.
	 * @param _ValueType The value type for padding.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use.
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	/**
	 * @brief Apply padding to the input Tensor (reverse order, i.e., Torch order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Type The padding type.
	 * @param _ValueType The value type for padding.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use.
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	/**
	 * @brief Apply padding to the input Tensor (forward order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Type The padding type (default is PaddingType::Zero).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Apply padding to the input Tensor (reverse order, i.e., Torch order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Type The padding type (default is PaddingType::Zero).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Apply padding to the input Tensor (forward order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Apply padding to the input Tensor (reverse order, i.e., Torch order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Apply padding to the input Tensor (forward order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Apply padding to the input Tensor (reverse order, i.e., Torch order), returning the padded Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Pad The padding ranges.
	 * @param _Val The padding value.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The padded Tensor.
	 */
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Repeat the input Tensor (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Repeat The repeat ranges, for example, { {2, 3}, {1, 2} } means repeat the first axis 2 times and the second axis 3 times.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The repeated Tensor.
	 */
	static Tensor Repeat(
		const Tensor& _Input,
		const Vector<std::pair<SizeType, SizeType>>& _Repeat,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Stack the input Tensors along a new axis (creates a new Tensor). The input Tensors must have the same shape.
	 *
	 * @param _Inputs The input Tensors.
	 * @param _Dim The axis along which to stack (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The stacked Tensor.
	 */
	static Tensor Stack(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Concatenate the input Tensors along an axis (creates a new Tensor). The input Tensors must have the same shape except for the specified axis.
	 *
	 * @param _Inputs The input Tensors.
	 * @param _Dim The axis along which to concatenate (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The concatenated Tensor.
	 */
	static Tensor Cat(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Get the Tensor at the specified indices along Axis[0] (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Indices The indices Tensor.
	 * @param _Axis The axis along which to gather (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The gathered Tensor.
	 */
	static Tensor Gather(
		const Tensor& _Input,
		const Tensor& _Indices,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Convert the input Tensor to a different type (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Dtype The target data type.
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The converted Tensor.
	 */
	static Tensor Cast(
		const Tensor& _Input,
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Compute the sum of the input Tensor along an axis (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Axis The axis along which to sum (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The summed Tensor.
	 */
	static Tensor Sum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Compute the cumulative sum of the input Tensor along an axis (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Axis The axis along which to compute the cumulative sum (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The cumulative summed Tensor.
	 */
	static Tensor CumSum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Compute the difference of the input Tensor along an axis (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Axis The axis along which to compute the difference (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The differenced Tensor.
	 */
	static Tensor Diff(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	/**
	 * @brief Compute the cumulative product of the input Tensor along an axis (creates a new Tensor).
	 *
	 * @param _Input The input Tensor.
	 * @param _Axis The axis along which to compute the cumulative product (default is 0).
	 * @param _ThreadPool The thread pool to use (default is nullptr).
	 *
	 * @return The cumulative product Tensor.
	 */
	static Tensor CumProd(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);


protected:
	Tensor GatherRef(SizeType _Index) const;

public:
	static Tensor Pow(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr);
	static Tensor Pow(const Tensor& _InputA, float64 _Val, ThreadPool* _ThreadPool = nullptr);
	Tensor Pow(const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr) const;
	Tensor Pow(float64 _Val, ThreadPool* _ThreadPool = nullptr) const;
	Tensor& Pow_(const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr);
	Tensor& Pow_(float64 _Val, ThreadPool* _ThreadPool = nullptr);

	DragonianLibTensorFnDef(Abs);
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
	DragonianLibTensorFnDef(Round);

private:
	bool UseThreadPool_ = true;
	Allocator Device_;
};

DragonianLibSpaceEnd