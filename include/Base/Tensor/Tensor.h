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
using ShapeType = Vector<SizeType>;
using ShapeIterator = ShapeType::iterator;
static inline double DoubleZero = 0.;
struct Range
{
	SizeType Begin = 0;
	SizeType Step = 1;
	SizeType End = 0;
	bool IsVal = false;
	bool IsNone = false;

	Range(SizeType _Val) :Begin(_Val), Step(_Val), End(_Val), IsVal(true) {}
	Range(NoneType _NoneVal) :IsNone(true) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End) {}
	Range(NoneType _NoneVal, SizeType _Step, SizeType _End) :Begin(_Step > 0 ? 0 : -1), Step(_Step), End(_End) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _Step, NoneType _NoneVal) :Begin(_Begin), Step(_Step), End(_Step > 0 ? -1 : 0) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _End) :Begin(_Begin), End(_End) {}
	Range(NoneType _NoneVal, SizeType _End) :End(_End) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, NoneType _NoneVal) :Begin(_Begin), End(-1) { UNUSED(_NoneVal); }
	void Reverse() { std::swap(Begin, End); Step = -Step; }
	bool operator==(const NoneType& _NoneVal) const { UNUSED(_NoneVal); return IsNone; }
};

enum class PaddingType
{
	Zero,
	Constant,
	Reflect,
	Cicular,
	Replicate
};

enum class InterpolateType
{
	Nearest1D,
	Nearest2D,
	Nearest2P,
	Linear,
	Bilinear,
	Bicubic,
	Trilinear,
	Area,
};

using SliceOptions = Vector<Range>;

SizeType VectorMul(const ShapeType& _Input);

SizeType VectorMul(const SliceOptions& _Input);

ShapeType GetBeginIndices(const SliceOptions& _Input);

bool RangeIsAllNone(const Vector<Range>& _Input);

class Tensor : public TensorBase
{
public:
	using InvokeFnType = void(*)(Tensor&);

	Tensor() = delete;
	~Tensor() override;
	Tensor(TensorType _DType, Device _Device) :TensorBase(_DType), Device_(GetMemoryProvider(_Device)) {}
	Tensor(const ShapeType& _Shape, TensorType _DType, Device _Device);
	Tensor(const Tensor& _Left);
	Tensor(Tensor&& _Right) noexcept;
	static Tensor FloatTensor(const Vector<float32>& _Array, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor LongTensor(const Vector<int64>& _Array, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor Ones(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor Zeros(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor ConstantOf(const ShapeType& _Shape, double _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor ConstantOf(const ShapeType& _Shape, int64 _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor Rand(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor Randn(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor OnesLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);
	static Tensor ZerosLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantLike(const Tensor& _Shape, double _Val, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantLike(const Tensor& _Shape, int64 _Val, ThreadPool* _ThreadPool = nullptr);
	static Tensor RandLike(const Tensor& _Shape, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr);
	static Tensor RandnLike(const Tensor& _Shape, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr);
	static Tensor Arange(float64 _Begin, float64 _End, float64 _Step, TensorType _Dtype = TensorType::Float32, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	static Tensor Arange(int64 _Begin, int64 _End, int64 _Step, TensorType _Dtype = TensorType::Int64, ThreadPool* _ThreadPool = nullptr, Device _Device = Device::CPU);
	//using _ValueType = float;
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

	static void SetThreadCount(SizeType _Count);
	static void EnableTimeLogger(bool _Enabled);
	SizeType GetAlignSize() const
	{
		return AlignSize_;
	}
	std::mutex& GetRelMx() const
	{
		return RelMx_;
	}
	Device GetDevice() const
	{
		return Device_->GetDevice();
	}
	Allocator GetAllocator() const
	{
		return Device_;
	}
	template<typename _Ty>
	_Ty& Get(SizeType Index)
	{
		return *((_Ty*)DataPtr_ + Index);
	}

protected:
	byte* DataPtr_ = nullptr;
	Tensor* ViewParent_ = nullptr;

	ShapeType ShapeBack_;
	ShapeType StepFront_, StepBack_;
	ShapeType SliceBegin_;
	ShapeType DimStride_;
	ShapeType CurIndices_;
	int64_t AlignSize_ = 4;
	bool IsBroadCasted_ = false;
	std::deque<Tensor*> ViewChild_;
	mutable std::mutex ViewMx_, RelMx_;
	SliceOptions OpSlice;
	
public:
	//提醒运算符不要使用线程池
	Tensor& DisableThreadPool()
	{
		UseThreadPool_ = false;
		return *this;
	}

	//提醒运算符使用线程池
	Tensor& PlUseThreadPool()
	{
		UseThreadPool_ = true;
		return *this;
	}

	//使用Indice获取一个数据
	template <typename Ref>
	Ref& Item(const ShapeType& _Indices)
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	//获取当前第一个数据
	template <typename Ref>
	Ref& Item()
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(GetPtr());
	}

	//使用Indice获取一个数据
	template <typename Ref>
	const Ref& Item(const ShapeType& _Indices) const
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	//获取当前第一个数据
	template <typename Ref>
	const Ref& Item() const
	{
		if (sizeof(Ref) != AlignSize_)
			DragonianLibThrow("TypeError!");
		return *(Ref*)(GetPtr());
	}

	//使用_Val指向的_Type类型数据填充整个Tensor
	void Assign(const void* _Val, TensorType _Type, ThreadPool* _ThreadPool = nullptr) const;

	//使用一个Buffer对整个Tensor赋值，按照行优先索引顺序，直到Buffer用完或整个Tensor都被赋值完毕
	void Assign(const void* _Buffer, SizeType _BufferSize, ThreadPool* _ThreadPool = nullptr) const;

	//使用_Val填充整个Tensor
	void Assign(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//使用_Val填充整个Tensor
	void Assign(float64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//使用一个Tensor对整个Tensor赋值，输入的Shape必须相同或可以广播
	void Assign(const Tensor& _Val, ThreadPool* _ThreadPool = nullptr) const;

	//将当前Tensor替换维另一个Tensor的View（不会拷贝）；如果输入为View，则另一个View的View源不可为当前Tensor（如果想拷贝Tensor，请使用Clone函数）
	Tensor& operator=(const Tensor& _Left);

	//移动赋值
	Tensor& operator=(Tensor&& _Right) noexcept;

	//使用_Val填充整个Tensor
	Tensor& operator=(int64 _Val);

	//使用_Val填充整个Tensor
	Tensor& operator=(float64 _Val);

	Tensor operator+(const Tensor& _Right) const;
	Tensor operator-(const Tensor& _Right) const;
	Tensor operator*(const Tensor& _Right) const;
	Tensor operator/(const Tensor& _Right) const;
	Tensor operator+(int64 _Right) const;
	Tensor operator-(int64 _Right) const;
	Tensor operator*(int64 _Right) const;
	Tensor operator/(int64 _Right) const;
	Tensor operator+(float64 _Right) const;
	Tensor operator-(float64 _Right) const;
	Tensor operator*(float64 _Right) const;
	Tensor operator/(float64 _Right) const;

	Tensor& operator+=(const Tensor& _Right);
	Tensor& operator-=(const Tensor& _Right);
	Tensor& operator*=(const Tensor& _Right);
	Tensor& operator/=(const Tensor& _Right);
	Tensor& operator+=(int64 _Right);
	Tensor& operator-=(int64 _Right);
	Tensor& operator*=(int64 _Right);
	Tensor& operator/=(int64 _Right);
	Tensor& operator+=(float64 _Right);
	Tensor& operator-=(float64 _Right);
	Tensor& operator*=(float64 _Right);
	Tensor& operator/=(float64 _Right);

	Tensor operator[](SizeType _Index) const;
	Tensor operator[](const SliceOptions& _SliceOptions) const;
	Tensor operator[](const ShapeType& _Indice) const;

	Tensor operator!=(const Tensor& _Right) const;
	Tensor operator==(const Tensor& _Right) const;
	Tensor operator<(const Tensor& _Right) const;
	Tensor operator>(const Tensor& _Right) const;
	Tensor operator<=(const Tensor& _Right) const;
	Tensor operator>=(const Tensor& _Right) const;
	Tensor operator!=(float64 _Right) const;
	Tensor operator==(float64 _Right) const;
	Tensor operator<(float64 _Right) const;
	Tensor operator>(float64 _Right) const;
	Tensor operator<=(float64 _Right) const;
	Tensor operator>=(float64 _Right) const;
	Tensor operator!=(int64 _Right) const;
	Tensor operator==(int64 _Right) const;
	Tensor operator<(int64 _Right) const;
	Tensor operator>(int64 _Right) const;
	Tensor operator<=(int64 _Right) const;
	Tensor operator>=(int64 _Right) const;

private:
	void Free();
	void ClearViewChilds();
	void ThrowOnNotEnabled() const;
	void RemoveSelfViewPtr();
	bool HasChild(const Tensor* _Child) const;
	void Assign1D(const void* _Val) const;
	void ReCalcViewInfo();

public:
	//使LoopIterator的索引 + 1
	void IteratorAdd(ShapeType& _Indices) const;

	//使LoopIterator的索引 - 1
	void IteratorSub(ShapeType& _Indices) const;

	//换算Index
	static SizeType CalcIndex(SizeType _Index, SizeType _Max);

	//换算Range
	static SizeType CalcRange(SizeType _Index, SizeType _Max);

	//Ceil两个Int64数据
	static SizeType Ceil(SizeType _Left, SizeType _Right);

	//当前Tensor是否“不”存在问题
	bool IsEnabled() const;

	//当前Tensor是否为标量
	bool IsScalar() const;

	//当前Tensor是否具有View型Tensor的特征
	bool HasViewedFeature() const;

	//当前Tensor的索引顺序是否严格内存连续
	bool IsContinuous(SizeType _Dim = 0) const;
	bool IsContinuous(const SliceOptions& _SlicePos, SizeType _Dim = 0) const;

	//当前Tensor是否可以通过Permute变得索引顺序内存连续
	bool IsTranSposedContinuous() const;

	//当前Tensor是否具有View源
	bool IsView() const;

	//克隆当前Tensor，返回一个具有独立内存的新Tensor（新建Tensor）
	Tensor Clone(ThreadPool* _ThreadPool = nullptr) const;

	//创建一个当前Tensor的View，View不具有独立内存，只具有自身属性，当前Tensor就是该View的View源
	Tensor CreateView() const;

	//创建一个当前Tensor的View，对其进行切片，返回该View
	Tensor Slice(const SliceOptions& _SliceOptions) const;

	//创建一个当前Tensor的View，对其进行切片，返回该View
	Tensor ReversedSlice(const SliceOptions& _SliceOptions) const;

	//创建一个当前Tensor的View，改变其轴排列顺序，返回该View
	Tensor Permute(const ShapeType& _DPremute) const;

	//创建一个当前Tensor的View，将其_Dim轴与Last轴互换排列顺序，返回该View
	Tensor SwapLastDim(SizeType _Dim) const;

	SliceOptions GetDefaultSliceVector() const;

	//在一个Tensor的指定轴调用函数
	static void Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn);

	//在当前Tensor的指定轴调用函数
	void Invoke(SizeType InvokedDim, InvokeFnType _Fn);

	//获取Shape
	const ShapeType& Shape() const;

	//获取指定轴的Shape
	SizeType Shape(SizeType _Index) const;

	//获取Shape
	const ShapeType& Size() const;

	//获取指定轴的Shape
	SizeType Size(SizeType _Index) const;

	//获取所有轴的Stride信息
	const ShapeType& Strides() const;

	//获取所有轴的Step信息
	const ShapeType& StepsBack() const;

	//获取所有轴的切片起始位置
	const ShapeType& SliceBegins() const;

	//使用 1 填充整个Tensor
	void FixOnes(ThreadPool* _ThreadPool = nullptr) const;

	//使用 0 填充整个Tensor
	void FixZeros(ThreadPool* _ThreadPool = nullptr) const;

	//使用_Val填充整个Tensor
	void Fix(double _Val, ThreadPool* _ThreadPool = nullptr) const;

	//使用_Val填充整个Tensor
	void Fix(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//使用随机数填充整个Tensor
	void RandFix(uint64 _Seed = 114514, ThreadPool* _ThreadPool = nullptr) const;
	void RandFix(ThreadPool* _ThreadPool) const { RandFix(114514, _ThreadPool); }

	//使用正态生成的随机数填充整个Tensor
	void RandnFix(uint64 _Seed = 114514, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr) const;
	void RandnFix(ThreadPool* _ThreadPool) const { RandnFix(114514, 0., 1., _ThreadPool); }

	//获取当前Tensor的缓冲区首地址（如果Tensor为View则返回View源的）
	byte* Buffer() const;

	//获取当前Tensor的轴地址
	byte* Data() const;

	//获取当前Tensor指定Indice位置数据的地址
	byte* Data(const ShapeType& _Indices) const;

	//创建一个当前Tensor的View，并将其直接看做Shape为ViewShape的Tensor，返回该View
	Tensor View(const ShapeType& _ViewShape) const;

	//使当前Tensor内存连续，返回当前Tensor的左值引用
	Tensor& Continuous(ThreadPool* _ThreadPool = nullptr);

	//创建一个当前Tensor的View，在其指定位置插入一个轴，返回该View
	Tensor UnSqueeze(SizeType Dim) const;

	//创建一个当前Tensor的View，若其指定轴的Size为 1，则删除该轴，返回该View
	Tensor Squeeze(SizeType Dim) const;

	//创建一个当前Tensor的View，将所有Size为 1 的轴删除，返回该View
	Tensor Squeeze() const;

	//创建两个Tensor的View，并将它们广播为两个Shape一致的View，返回这两个View
	static std::pair<Tensor, Tensor> BroadCast(const Tensor& _A, const Tensor& _B);

	//创建输入Tensor的View，将其广播为与当前Tensor的Shape一致的View，返回该View
	Tensor BroadCast(const Tensor& _Other) const;

	//当前Tensor是否为被广播的Tensor
	bool IsBroadCasted() const;

	//获取当前Tensor的轴数
	SizeType DimCount() const;

	//判断当前Tensor是否为向量
	bool IsVector() const;

	//获取当前Tensor的数据区首地址
	byte* GetPtr() const;

	//获取遍历该张量开销最小的轴序
	ShapeType CalcContinuous() const;

	bool IsTransposed(size_t _Size) const;

	//沿着Axis[0]，获取当前Tensor中Indices处的Tensor（新建Tensor）
	Tensor Gather(
		const Tensor& _Indices,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	) const;
	Tensor Gather(
		const Tensor& _Indices,
		ThreadPool* _ThreadPool
	) const
	{
		return Gather(_Indices, 0, _ThreadPool);
	}

	//转换Tensor的类型（新建Tensor）
	Tensor Cast(
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	) const;

	//对输入的Tensor进行Padding（顺序为正向），返回Padding后的Tensor（新建Tensor）
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	//对输入的Tensor进行Padding（顺序为反向，即Torch的顺序），返回Padding后的Tensor（新建Tensor）
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	//对输入的Tensor进行Padding（顺序为正向），返回Padding后的Tensor（新建Tensor）
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行Padding（顺序为反向，即Torch的顺序），返回Padding后的Tensor（新建Tensor）
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行Padding（顺序为正向），返回Padding后的Tensor（新建Tensor）
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行Padding（顺序为反向，即Torch的顺序），返回Padding后的Tensor（新建Tensor）
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行Padding（顺序为正向），返回Padding后的Tensor（新建Tensor）
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行Padding（顺序为反向，即Torch的顺序），返回Padding后的Tensor（新建Tensor）
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//对输入的Tensor进行重复操作（新建Tensor）
	static Tensor Repeat(
		const Tensor& _Input,
		const Vector<std::pair<SizeType, SizeType>>& _Repeat,
		ThreadPool* _ThreadPool = nullptr
	);

	//以输入Tensor的Shape为模板，在_Dim处插入一个新的Shape，并沿着_Dim轴进行拷贝（新建Tensor），必须保证输入Tensor的Shape完全相同
	static Tensor Stack(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//以输入Tensor的Shape为模板，将所有Tensor在_Dim处合并（新建Tensor），必须保证输入Tensor的Shape在除了_Dim处都完全相同
	static Tensor Cat(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//沿着Axis[0]，获取Input中Indices处的Tensor（新建Tensor）
	static Tensor Gather(
		const Tensor& _Input,
		const Tensor& _Indices,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//转换Tensor的类型（新建Tensor）
	static Tensor Cast(
		const Tensor& _Input,
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor Sum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor CumSum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor Diff(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

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