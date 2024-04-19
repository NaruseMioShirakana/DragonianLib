#pragma once
#include <deque>
#include <ranges>

#include "Tensor/TensorBase.h"

LibSvcBegin
using ShapeType = std::vector<SizeType>;
using ShapeIterator = ShapeType::iterator;
using SliceOptions = std::vector<std::vector<SizeType>>;

SizeType VectorMul(const ShapeType& _Input);

class TensorIterator
{
public:
	using lpbyte = byte*;
	TensorIterator() = delete;
	TensorIterator(
		const lpbyte& _Ptr,
		const ShapeType& _Shape,
		const ShapeType& _Step,
		const ShapeType& _SliceBegin,
		const ShapeType& _DimStride,
		const ShapeType& _CurIndices,
		SizeType _Align
	);

private:
	const lpbyte& Ptr_;
	const ShapeType& Shape_;
	const ShapeType& Step_;
	const ShapeType& SliceBegin_;
	const ShapeType& DimStride_;
	const ShapeType& CurIndices_;
	SizeType AlignSize_ = 4;
	SizeType Pos_ = 0;
};

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
	template <typename Ref>
	void Assign(const Ref& _Value)
	{
		ThrowOnNotEnabled();
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		if (ShapeBack_.size() == 1)
			Assign1D(_Value);
		else
			for (SizeType i = 0; i < Size(0); ++i)
				operator[](i).Assign(_Value);
	}
	void Assign(const void* _Buffer, SizeType _BufferSize) const;
	Tensor& operator=(const Tensor& _Left);
	Tensor& operator=(Tensor&& _Right) noexcept;
	Tensor operator[](SizeType _Index);

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
	void FixOnes();
	void FixZeros();
	void RandFix(int _Seed);
	void RandnFix(int _Seed, double _Mean, double _Sigma);
	byte* Buffer() const;
	byte* Data() const;
	byte* Data(const ShapeType& _Indices) const;
	Tensor View(const ShapeType& _ViewShape);
	Tensor& Continuous();
};
LibSvcEnd