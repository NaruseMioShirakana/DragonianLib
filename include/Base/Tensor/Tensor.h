#pragma once
#include <deque>
#include <ranges>

#include "Tensor/TensorBase.h"

LibSvcBegin
using ShapeType = std::vector<SizeType>;
using ShapeIterator = ShapeType::iterator;
using SliceOptions = std::vector<std::vector<SizeType>>;

SizeType VectorMul(const ShapeType& _Input);

class Tensor : public TensorBase
{
public:
	Tensor();
	~Tensor() override;
	Tensor(const ShapeType& _Shape, TensorType _DType = TensorType::Float32);
	Tensor(const Tensor& _Left);
	Tensor(Tensor&& _Right) noexcept;
	using InvokeFnType = void(*)(Tensor&);

protected:
	byte* DataPtr_ = nullptr;
	Tensor* ViewParent_ = nullptr;

	ShapeType ShapeFront_, ShapeBack_;
	ShapeType StepFront_, StepBack_;
	ShapeType SliceBegin_;
	ShapeType DimStride_;
	ShapeType CurIndices_;
	int64_t AlignSize_ = 4;

	std::deque<Tensor*> ViewChild_;
	mutable std::mutex ViewMx_;

public:
	template <typename Ref>
	Ref& Item(const ShapeType& _Indices)
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}
	template <typename Ref>
	Ref& Item()
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(Data());
	}
	void Assign(const void* _Val, TensorType _Type) const;
	void Assign(const void* _Buffer, SizeType _BufferSize) const;
	void Assign(int64 _Val) const;
	void Assign(float64 _Val) const;
	Tensor& operator=(const Tensor& _Left);
	Tensor& operator=(Tensor&& _Right) noexcept;
	Tensor& operator=(int64 _Val);
	Tensor& operator=(float64 _Val);
	Tensor operator[](SizeType _Index) const;

private:
	void Free();
	void ClearViewChilds();
	void ThrowOnNotEnabled() const;
	void RemoveSelfViewPtr();
	bool HasChild(const Tensor* _Child) const;
	template <typename Ref>
	void Assign1D(const Ref& _Value) const
	{
		SizeType CurIndex = 0;
		for (size_t i = 0; i < CurIndices_.size(); ++i)
			CurIndex += CurIndices_[i] * StepFront_[i];
		for (SizeType i = 0; i < Size(0); ++i)
			*(Ref*)(DataPtr_ + CurIndex + (((i * DimStride_[0]) + SliceBegin_[0]) * StepBack_[0])) = _Value;
	}
	void Assign1D(const void* _Val) const;
	void CalcInfo();

public:
	void IteratorAdd(ShapeType& _Indices) const;
	void IteratorSub(ShapeType& _Indices) const;
	static SizeType CalcIndex(SizeType _Index, SizeType _Max);
	static SizeType Ceil(SizeType _Left, SizeType _Right);
	static SizeType CalcRange(SizeType _Index, SizeType _Max);
	bool IsEnabled() const;
	bool IsScalar() const;
	bool HasViewedFeature() const;
	bool IsContinuous() const;
	bool IsView() const;
	Tensor Clone() const;
	Tensor CreateView() const;
	Tensor Slice(const SliceOptions& _SliceOptions) const;
	Tensor Permute(const ShapeType& _DPremute) const;
	static void Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn);
	void Invoke(SizeType InvokedDim, InvokeFnType _Fn);
	const ShapeType& Shape() const;
	SizeType Shape(SizeType _Index) const;
	const ShapeType& Size() const;
	SizeType Size(SizeType _Index) const;
	const ShapeType& Strides() const;
	const ShapeType& StepsBack() const;
	const ShapeType& StepsFront() const;
	const ShapeType& CurIndices() const;
	const ShapeType& SliceBegins() const;
	void FixOnes() const;
	void FixZeros() const;
	void Fix(double _Val) const;
	void Fix(int64 _Val) const;
	void RandFix(int _Seed) const;
	void RandnFix(int _Seed, double _Mean, double _Sigma) const;
	byte* Buffer() const;
	byte* Data() const;
	byte* Data(const ShapeType& _Indices) const;
	Tensor View(const ShapeType& _ViewShape) const;
	Tensor& Continuous();
	Tensor UnSqueeze(SizeType Dim) const;
	Tensor Squeeze(SizeType Dim) const;
	Tensor Squeeze() const;
};
LibSvcEnd