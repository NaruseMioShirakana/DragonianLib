#pragma once
#include <deque>
#include <ranges>
#include <random>
#include "Tensor/TensorBase.h"
#include "Util/ThreadPool.h"
#include "Util/Avx256.h"

LibSvcBegin
using ShapeType = Vector<SizeType>;
using ShapeIterator = ShapeType::iterator;

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

using SliceOptions = Vector<Range>;

SizeType VectorMul(const ShapeType& _Input);

bool RangeIsAllNone(const Vector<Range>& _Input);

class Tensor : public TensorBase
{
public:
	Tensor() = delete;
	~Tensor() override;
	Tensor(TensorType _DType) :TensorBase(_DType) {}
	Tensor(const ShapeType& _Shape, TensorType _DType);
	Tensor(const Tensor& _Left);
	Tensor(Tensor&& _Right) noexcept;
	using InvokeFnType = void(*)(Tensor&);
	static void SetThreadCount(SizeType _Count);

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
	mutable std::mutex ViewMx_;

public:
	//使用Indice获取一个数据
	template <typename Ref>
	Ref& Item(const ShapeType& _Indices)
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	//获取当前第一个数据
	template <typename Ref>
	Ref& Item()
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
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

	Tensor operator[](SizeType _Index) const;

	Tensor operator[](const SliceOptions& _SliceOptions) const;

	Tensor operator[](const ShapeType& _Indice) const;

private:
	void Free();
	void ClearViewChilds();
	void ThrowOnNotEnabled() const;
	void RemoveSelfViewPtr();
	bool HasChild(const Tensor* _Child) const;
	void Assign1D(const void* _Val) const;
	void ReCalcViewInfo();
	Tensor GatherRef(SizeType _Index) const;

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
	bool IsContinuous() const;

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

	//使用正态生成的随机数填充整个Tensor
	void RandnFix(uint64 _Seed = 114514, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr) const;

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

	//沿着Axis[0]，获取当前Tensor中Indices处的Tensor（新建Tensor）
	Tensor Gather(
		const Tensor& _Indices,
		ThreadPool* _ThreadPool = nullptr
	) const;

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
		ThreadPool* _ThreadPool = nullptr
	);

	//转换Tensor的类型（新建Tensor）
	static Tensor Cast(
		const Tensor& _Input,
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	);
};

template<typename _1Ty, typename _2Ty>
void CastFrom(const Tensor& _InputA, const Tensor& _InputB, const SizeType CurDims)
{
	_1Ty* DataPtr1 = (_1Ty*)_InputA.Data();
	const _2Ty* DataPtr2 = (_2Ty*)_InputB.Data();

	if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
	{
		DataPtr1 = (_1Ty*)_InputA.GetPtr();
		DataPtr2 = (_2Ty*)_InputB.GetPtr();
		const size_t BufferSize = VectorMul(_InputA.Shape());
		const auto DataEnd = DataPtr1 + BufferSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = (_1Ty)(*(DataPtr2++));
		return;
	}

	auto Steps1 = _InputA.StepsBack();
	for (auto& i : Steps1)
		i /= sizeof(_1Ty);
	auto Steps2 = _InputB.StepsBack();
	for (auto& i : Steps2)
		i /= sizeof(_2Ty);
	const SizeType* __restrict ShapePtr = _InputA.Shape().data();
	const SizeType* __restrict StepPtr1 = Steps1.data();
	const SizeType* __restrict StepPtr2 = Steps2.data();
	const SizeType* __restrict BeginsPtr1 = _InputA.SliceBegins().data();
	const SizeType* __restrict BeginsPtr2 = _InputB.SliceBegins().data();
	const SizeType* __restrict StridesPtr1 = _InputA.Strides().data();
	const SizeType* __restrict StridesPtr2 = _InputB.Strides().data();

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_1Ty)DataPtr2[Index2];
			}
		);

		return;
	}

	auto Cont = _InputA.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_1Ty)DataPtr2[IndexAxis4B];
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_1Ty)DataPtr2[IndexAxis3B];
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_1Ty)DataPtr2[IndexAxis2B];
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_1Ty)DataPtr2[IndexAxis1B];
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_1Ty)DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]];
		}
	}
}
LibSvcEnd