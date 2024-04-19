#include "Tensor/Tensor.h"

#include <random>

#include "Tensor/Int8Tensor.h"
#include "Tensor/Int16Tensor.h"
#include "Tensor/Int32Tensor.h"
#include "Tensor/Int64Tensor.h"
#include "Tensor/Float16Tensor.h"
#include "Tensor/Float32Tensor.h"
#include "Tensor/Float64Tensor.h"
#include "Tensor/Complex32Tensor.h"

LibSvcBegin

SizeType VectorMul(const ShapeType& _Input)
{
	SizeType All = 1;
	for (const auto i : _Input)
		All *= i;
	return All;
}

//Construct
Tensor::Tensor(): TensorBase(TensorType::Float32)
{
	
}

Tensor::~Tensor()
{
	Free();
}

Tensor::Tensor(const ShapeType& _Shape, TensorType _DType) : TensorBase(_DType)
{
	AlignSize_ = DType2Size(DType_);
	ShapeFront_.clear();
	ShapeBack_ = _Shape;
	StepFront_.clear();
	StepBack_ = { _Shape.begin() + 1,_Shape.end() };
	StepBack_.emplace_back(AlignSize_);
	std::ranges::reverse(StepBack_);
	for (size_t i = 1; i < StepBack_.size(); ++i)
		StepBack_[i] *= StepBack_[i - 1];
	std::ranges::reverse(StepBack_);
	SliceBegin_ = { _Shape.size(),0, ShapeType::allocator_type() };
	DimStride_ = { _Shape.size(),1, ShapeType::allocator_type() };
	CurIndices_.clear();

	ViewParent_ = nullptr;
	DataPtr_ = (byte*)LIBSVC_MALLOC(VectorMul(ShapeBack_) * AlignSize_);
	ViewChild_.clear();
}

Tensor::Tensor(const Tensor& _Left) : TensorBase(_Left.DType_)
{
	//TODO
	_Left.ThrowOnNotEnabled();
	ShapeFront_ = _Left.ShapeFront_;
	ShapeBack_ = _Left.ShapeBack_;
	StepFront_ = _Left.StepFront_;
	StepBack_ = _Left.StepBack_;
	AlignSize_ = _Left.AlignSize_;
	SliceBegin_ = _Left.SliceBegin_;
	DimStride_ = _Left.DimStride_;

	if(!_Left.IsView())
	{
		ViewParent_ = nullptr;
		DataPtr_ = (byte*)LIBSVC_MALLOC(VectorMul(ShapeBack_) * AlignSize_);
		ViewChild_.clear();
	}
	else
	{
		ViewParent_ = _Left.ViewParent_;
		DataPtr_ = _Left.DataPtr_;
		ViewParent_->ViewChild_.emplace_back(this);
	}
}

Tensor::Tensor(Tensor&& _Right) noexcept : TensorBase(_Right.DType_)
{
	ShapeFront_ = std::move(_Right.ShapeFront_);
	ShapeBack_ = std::move(_Right.ShapeBack_);
	StepFront_ = std::move(_Right.StepFront_);
	StepBack_ = std::move(_Right.StepBack_);
	SliceBegin_ = std::move(_Right.SliceBegin_);
	DimStride_ = std::move(_Right.DimStride_);
	CurIndices_ = std::move(_Right.CurIndices_);
	AlignSize_ = _Right.AlignSize_;

	if (!_Right.IsView())
	{
		ViewParent_ = nullptr;
		DataPtr_ = _Right.DataPtr_;
		std::lock_guard Lock2(_Right.ViewMx_);
		ViewChild_ = std::move(_Right.ViewChild_);
		RemoveSelfViewPtr();
		std::lock_guard Lock1(ViewMx_);
		for (const auto i : ViewChild_)
			i->ViewParent_ = this;
		_Right.DataPtr_ = nullptr;
		_Right.ViewParent_ = nullptr;
	}
	else
	{
		ViewParent_ = _Right.ViewParent_;
		if (!ViewParent_->HasChild(this))
		{
			std::lock_guard Lock(ViewParent_->ViewMx_);
			ViewParent_->ViewChild_.emplace_back(this);
		}
		DataPtr_ = _Right.DataPtr_;
		ViewChild_.clear();
	}
}

//Private

void Tensor::Free()
{
	if (DataPtr_ && !IsView())
	{
		ClearViewChilds();
		LIBSVC_FREE(DataPtr_);
	}
	else if(IsView())
	{
		if(ViewParent_->ViewParent_)
			LibSvcThrow("View Parent Can Not Have View Parent, Please Report This Bug!");
		std::lock_guard Lock(ViewParent_->ViewMx_);
		auto& Views = ViewParent_->ViewChild_;
		const auto& Iter = std::ranges::find(Views.begin(), Views.end(), this);
		if (Iter != Views.end())
			Views.erase(Iter);
		else
			LibSvcThrow("View Not In Parent's Child List, Please Report This Bug!");
	}
	DataPtr_ = nullptr;
	ViewParent_ = nullptr;
}

void Tensor::ClearViewChilds()
{
	std::lock_guard Lock(ViewMx_);
	for (const auto& Iter : ViewChild_)
	{
		if(Iter == this)
			continue;
		Iter->ViewParent_ = nullptr;
		Iter->DataPtr_ = nullptr;
	}
	ViewChild_.clear();
}

void Tensor::ThrowOnNotEnabled() const
{
	if (AlignSize_ != DType2Size(DType_))
		LibSvcThrow("AlignSize MisMatch!");
	if (ShapeBack_.empty() || StepBack_.empty())
		LibSvcThrow("Axis Out Of Range!");
	if (!DataPtr_)
		LibSvcThrow("NullPointer Error!");
#ifdef LIBSVC_DEBUG
	if ((ShapeFront_.size() != StepFront_.size()) || (CurIndices_.size() != StepFront_.size()) ||
		(DimStride_.size() != ShapeBack_.size()) || (DimStride_.size() != StepBack_.size()))
	{
		LibSvcThrow("Indices Error!");
	}
#endif
	if (!IsView() && HasViewedFeature())
		LibSvcThrow("Src Tensor Should Not Has Viewed Feature, Please Report This Bug!");

	if (IsView())
	{
		std::lock_guard Lock(ViewParent_->ViewMx_);
		auto& Views = ViewParent_->ViewChild_;
		const auto& Iter = std::ranges::find(Views.begin(), Views.end(), this);
		if (Iter == Views.end())
			LibSvcThrow("View Not In Parent's Child List, Please Report This Bug!");
	}
}

void Tensor::RemoveSelfViewPtr()
{
	std::lock_guard Lock(ViewMx_);
	const auto& Iter = std::ranges::find(ViewChild_.begin(), ViewChild_.end(), this);
	if (Iter != ViewChild_.end())
		ViewChild_.erase(Iter);
}

bool Tensor::HasChild(const Tensor* _Child) const
{
	std::lock_guard Lock(ViewMx_);
	const auto& Iter = std::ranges::find(ViewChild_.begin(), ViewChild_.end(), _Child);
	if (Iter != ViewChild_.end())
		return true;
	return false;
}

//Operator=

Tensor& Tensor::operator=(const Tensor& _Left)
{
	if (&_Left == this)
		return *this;
	if (_Left.ViewParent_ && _Left.ViewParent_ == this)
		LibSvcThrow("Assign To Parent Is Not Allowed!");
	_Left.ThrowOnNotEnabled();
	return *this = _Left.CreateView();
}

Tensor& Tensor::operator=(Tensor&& _Right) noexcept
{
	if (_Right.ViewParent_ && _Right.ViewParent_ == this)
		LibSvcThrow("Assign To Parent Is Not Allowed!");
	Free();
	DType_ = _Right.DType_;
	ShapeFront_ = std::move(_Right.ShapeFront_);
	ShapeBack_ = std::move(_Right.ShapeBack_);
	StepFront_ = std::move(_Right.StepFront_);
	StepBack_ = std::move(_Right.StepBack_);
	SliceBegin_ = std::move(_Right.SliceBegin_);
	DimStride_ = std::move(_Right.DimStride_);
	CurIndices_ = std::move(_Right.CurIndices_);
	AlignSize_ = _Right.AlignSize_;

	if (!_Right.IsView())
	{
		ViewParent_ = nullptr;
		DataPtr_ = _Right.DataPtr_;
		std::lock_guard Lock2(_Right.ViewMx_);
		ViewChild_ = std::move(_Right.ViewChild_);
		RemoveSelfViewPtr();
		std::lock_guard Lock1(ViewMx_);
		for (const auto i : ViewChild_)
			i->ViewParent_ = this;
		_Right.DataPtr_ = nullptr;
		_Right.ViewParent_ = nullptr;
	}
	else
	{
		ViewParent_ = _Right.ViewParent_;
		if (!ViewParent_->HasChild(this))
		{
			std::lock_guard Lock(ViewParent_->ViewMx_);
			ViewParent_->ViewChild_.emplace_back(this);
		}
		DataPtr_ = _Right.DataPtr_;
		ViewChild_.clear();
	}

	return *this;
}

Tensor Tensor::operator[](SizeType _Index)
{
	ThrowOnNotEnabled();
	const auto Idx = CalcIndex(_Index, ShapeBack_.front());
	Tensor Ret;
	Ret.DType_ = DType_;

	Ret.ShapeFront_ = ShapeFront_;
	Ret.ShapeFront_.emplace_back(ShapeBack_.front());

	Ret.ShapeBack_ = { ShapeBack_.begin() + 1,ShapeBack_.end() };

	Ret.StepFront_ = StepFront_;
	Ret.StepFront_.emplace_back(StepBack_.front());

	Ret.StepBack_ = { StepBack_.begin() + 1,StepBack_.end() };
	Ret.SliceBegin_ = { SliceBegin_.begin() + 1,SliceBegin_.end() };
	Ret.DimStride_ = { DimStride_.begin() + 1,DimStride_.end() };

	Ret.CurIndices_ = CurIndices_;
	Ret.CurIndices_.emplace_back(SliceBegin_.front() + (Idx * DimStride_.front()));

	Ret.AlignSize_ = AlignSize_;
	Ret.DataPtr_ = DataPtr_;

	if (Ret.ShapeBack_.empty())
	{
		Ret.ShapeBack_.emplace_back(1);
		Ret.StepBack_.emplace_back(1);
		Ret.SliceBegin_.emplace_back(0);
		Ret.DimStride_.emplace_back(1);
	}

	if (IsView())
		Ret.ViewParent_ = ViewParent_;
	else
		Ret.ViewParent_ = (Tensor*)((size_t)(this));

	std::lock_guard Lock(Ret.ViewParent_->ViewMx_);
	Ret.ViewParent_->ViewChild_.emplace_back(&Ret);

	return Ret;
}

void Tensor::Assign(const void* _Buffer, SizeType _BufferSize) const
{
	ThrowOnNotEnabled();
	const auto DataSize = VectorMul(ShapeBack_);
	const auto SrcSize = (_BufferSize / AlignSize_);
	const auto CurSize = std::min(DataSize, SrcSize);
	if(IsContinuous())
	{
		memcpy(Data(), _Buffer, CurSize * AlignSize_);
		return;
	}
	ShapeType Indice(ShapeBack_.size(), 0);
	const byte* buf = (const byte*)_Buffer;
	for (SizeType i = 0; i < CurSize; ++i)
	{
		memcpy(Data(Indice), buf, AlignSize_);
		IteratorAdd(Indice);
		buf += AlignSize_;
	}
}

//Public

void Tensor::IteratorAdd(ShapeType& _Indices) const
{
	auto Val = _Indices.data() + _Indices.size() - 1;

	for (size_t i = _Indices.size() - 1; ; --i)
	{
		const auto Ret = *Val + 1;
		if (Ret < ShapeBack_[i])
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

void Tensor::IteratorSub(ShapeType& _Indices) const
{
	auto Val = _Indices.data() + _Indices.size() - 1;

	for (size_t i = _Indices.size() - 1; ; --i)
	{
		const auto Ret = *Val - 1;
		if (Ret >= 0)
		{
			*Val = Ret;
			return;
		}
		if (i == 0)
			return;
		*Val = (ShapeBack_[i] - 1);
		--Val;
	}
}

SizeType Tensor::CalcIndex(SizeType _Index, SizeType _Max)
{
	if (_Index < 0)
		_Index += _Max;
	if (_Index >= _Max || _Index < 0)
		LibSvcThrow("Index Out Of Range!");
	return _Index;
}

SizeType Tensor::CalcRange(SizeType _Index, SizeType _Max)
{
	if (_Index < 0)
		_Index += _Max + 1;
	if (_Index > _Max || _Index < 0)
		LibSvcThrow("Index Out Of Range!");
	return _Index;
}

SizeType Tensor::Ceil(SizeType _Left, SizeType _Right)
{
	auto Mul = _Left / _Right;
	if (_Left > (Mul * _Right))
		++Mul;
	return Mul;
}

bool Tensor::IsEnabled() const
{
	if (AlignSize_ != DType2Size(DType_))
		return false;
	if (ShapeBack_.empty() || StepBack_.empty())
		return false;
	if (!DataPtr_)
		return false;

	if ((ShapeFront_.size() != StepFront_.size()) || (CurIndices_.size() != StepFront_.size()) ||
		(DimStride_.size() != ShapeBack_.size()) || (DimStride_.size() != StepBack_.size()))
	{
		return false;
	}

	if (!IsView() && HasViewedFeature())
		return false;

	if (IsView())
	{
		std::lock_guard Lock(ViewParent_->ViewMx_);
		auto& Views = ViewParent_->ViewChild_;
		const auto& Iter = std::ranges::find(Views.begin(), Views.end(), this);
		if (Iter == Views.end())
			LibSvcThrow("View Not In Parent's Child List, Please Report This Bug!");
	}
	return true;
}

bool Tensor::IsScalar() const
{
	if (ShapeBack_.size() != 1)
		return false;
	if (ShapeBack_.back() == 1)
		return true;
	return false;
}

bool Tensor::HasViewedFeature() const
{
	if (!CurIndices_.empty())
		return true;
	if (!(StepFront_.empty() && ShapeFront_.empty()))
		return true;
	return !IsContinuous();
}

bool Tensor::IsContinuous() const
{
	for (const auto i : DimStride_)
		if (i != 1)
			return false;
	for (const auto i : SliceBegin_)
		if (i != 0)
			return false;
	for (size_t i = 1; i < StepBack_.size(); ++i)
		if (StepBack_[i] > StepBack_[i - 1])
			return false;
	return true;
}

bool Tensor::IsView() const
{
	return ViewParent_;
}

Tensor Tensor::Clone() const
{
	ThrowOnNotEnabled();
	return {}; //TODO
}

Tensor Tensor::CreateView() const
{
	ThrowOnNotEnabled();
	Tensor Ret;
	Ret.DType_ = DType_;
	Ret.ShapeFront_ = ShapeFront_;
	Ret.ShapeBack_ = ShapeBack_;
	Ret.StepFront_ = StepFront_;
	Ret.StepBack_ = StepBack_;
	Ret.SliceBegin_ = SliceBegin_;
	Ret.DimStride_ = DimStride_;
	Ret.CurIndices_ = CurIndices_;
	Ret.AlignSize_ = AlignSize_;

	Ret.DataPtr_ = DataPtr_;
	
	if(IsView())
		Ret.ViewParent_ = ViewParent_;
	else
		Ret.ViewParent_ = (Tensor*)((size_t)(this));

	std::lock_guard Lock(Ret.ViewParent_->ViewMx_);
	Ret.ViewParent_->ViewChild_.emplace_back(&Ret);

	return Ret;
}

Tensor Tensor::Slice(const SliceOptions& _SliceOptions) const
{
	ThrowOnNotEnabled();
	if (ShapeBack_.empty() || _SliceOptions.size() > ShapeBack_.size())
		LibSvcThrow("Axis Out Of Range!");
	Tensor Ret = CreateView();
	for (size_t i = 0; i < _SliceOptions.size(); ++i)
	{
		//TODO
		Ret.SliceBegin_[i] = CalcIndex(_SliceOptions[i][0], ShapeBack_[i]);
		auto SliceEndPos = _SliceOptions[i][1];
		if (SliceEndPos > ShapeBack_[i] || SliceEndPos < -(ShapeBack_[i] + 1))
			LibSvcThrow("Indew Out Of Range!");
		if (SliceEndPos == -(ShapeBack_[i] + 1))
			SliceEndPos = -1;
		else if (SliceEndPos < 0)
			SliceEndPos += ShapeBack_[i] + 1;

		auto SliceLength = SliceEndPos - Ret.SliceBegin_[i];
		if (SliceLength == 0)
		{
			Ret.ShapeBack_[i] = 1;
			continue;
		}
		if (SliceLength > 0 && _SliceOptions[i][2] < 0 ||
			SliceLength < 0 && _SliceOptions[i][2] > 0)
			LibSvcThrow("Step & (SliceEnd - SliceBegin) Should Have The Same Sign!");
		if(abs(SliceLength) < abs(_SliceOptions[i][2]))
			LibSvcThrow("abs(Step) Should < abs(SliceEnd - SliceBegin)!");
		Ret.DimStride_[i] *= _SliceOptions[i][2];
		if (SliceLength < 0)
			--SliceLength;
		Ret.ShapeBack_[i] = Ceil(abs(SliceLength), abs(_SliceOptions[i][2]));
	}
	return Ret;
}

Tensor Tensor::Permute(const ShapeType& _DPremute) const
{
	ThrowOnNotEnabled();
	if (ShapeBack_.empty() || _DPremute.size() > ShapeBack_.size())
		LibSvcThrow("Axis Out Of Range!");
	Tensor Ret = CreateView();
	auto TransposedDims = _DPremute;
	std::ranges::sort(TransposedDims.begin(), TransposedDims.end());
	if (TransposedDims[0] != 0)
		LibSvcThrow("DPremute Must Have [0, 1, ... , N_DIMS]!");
	for (size_t i = 1; i < TransposedDims.size(); ++i)
		if (TransposedDims[i] != TransposedDims[i - 1] + 1)
			LibSvcThrow("DPremute Must Have [0, 1, ... , N_DIMS]!");

	for (size_t i = 0; i < _DPremute.size(); ++i)
	{
		Ret.ShapeBack_[i] = ShapeBack_[_DPremute[i]];
		Ret.StepBack_[i] = StepBack_[_DPremute[i]];
		Ret.SliceBegin_[i] = SliceBegin_[_DPremute[i]];
		Ret.DimStride_[i] = DimStride_[_DPremute[i]];
	}

	return Ret;
}

void Tensor::Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn)
{
	if (!_Fn)
		return;
	const auto Dims = SizeType(_Tensor.ShapeBack_.size());
	InvokedDim = CalcIndex(InvokedDim, Dims);
	if(SizeType(_Tensor.ShapeBack_.size()) <= InvokedDim + 1)
	{
		_Fn(_Tensor);
	}
	else
	{
		for (SizeType i = 0; i < _Tensor.ShapeBack_.front(); ++i)
		{
			auto UserData = _Tensor[i];
			Invoke(UserData, InvokedDim, _Fn);
		}
	}
}

void Tensor::Invoke(SizeType InvokedDim, InvokeFnType _Fn)
{
	ThrowOnNotEnabled();
	Invoke(*this, InvokedDim, _Fn);
}

const ShapeType& Tensor::Shape() const
{
	ThrowOnNotEnabled();
	return ShapeBack_;
}

SizeType Tensor::Shape(SizeType _Index) const
{
	ThrowOnNotEnabled();
	return ShapeBack_[_Index];
}

const ShapeType& Tensor::Size() const
{
	ThrowOnNotEnabled();
	return ShapeBack_;
}

SizeType Tensor::Size(SizeType _Index) const
{
	ThrowOnNotEnabled();
	return ShapeBack_[_Index];
}

byte* Tensor::Data() const
{
	ThrowOnNotEnabled();
	SizeType Index = 0;
	for (size_t i = 0; i < CurIndices_.size(); ++i)
		Index += CurIndices_[i] * StepFront_[i];
	return DataPtr_ + Index;
}

byte* Tensor::Data(const ShapeType& _Indices) const
{
	ThrowOnNotEnabled();
	if (_Indices.size() != ShapeBack_.size())
		LibSvcThrow("Axis Out Of Range!");
	SizeType Index = 0;
	for (size_t i = 0; i < CurIndices_.size(); ++i)
		Index += CurIndices_[i] * StepFront_[i];
	for (size_t i = 0; i < _Indices.size(); ++i)
	{
		const SizeType Idx = CalcIndex(_Indices[i], ShapeBack_[i]);
		Index += ((Idx * DimStride_[i]) + SliceBegin_[i]) * StepBack_[i];
	}
	return DataPtr_ + Index;
}

byte* Tensor::Buffer() const
{
	return DataPtr_;
}

void Tensor::FixOnes()
{

}

void Tensor::FixZeros()
{

}

void Tensor::RandFix(int _Seed)
{


}

void Tensor::RandnFix(int _Seed, double _Mean, double _Sigma)
{

}

Tensor Tensor::View(const ShapeType& _ViewShape)
{
	//TODO
	if (!IsContinuous())
		LibSvcThrow("View Should Be Continuous!");
	if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), -1) > 1)
		LibSvcThrow("Count Of Dynamic Axis Should <= 1!");
	Tensor Ret = CreateView();
	if(VectorMul(Ret.ShapeBack_) % VectorMul(_ViewShape))
		LibSvcThrow("Size MisMatch!");



	return Ret;
}

Tensor& Tensor::Continuous()
{
	ThrowOnNotEnabled();
	if (!IsView() && !IsContinuous())
		LibSvcThrow("Src Should Be Continuous, Please Report This Bug!");
	if (!IsView() || IsContinuous())
		return *this;

	const auto DataSize = VectorMul(ShapeBack_);
	const auto BufSize = DataSize * AlignSize_;
	DataPtr_ = (byte*)LIBSVC_MALLOC(BufSize);

	if (ViewParent_->ViewParent_)
		LibSvcThrow("View Parent Can Not Have View Parent, Please Report This Bug!");
	std::lock_guard Lock(ViewParent_->ViewMx_);
	auto& Views = ViewParent_->ViewChild_;
	const auto& Iter = std::ranges::find(Views.begin(), Views.end(), this);
	if (Iter != Views.end())
		Views.erase(Iter);
	else
		LibSvcThrow("View Not In Parent's Child List, Please Report This Bug!");

	auto It = DataPtr_;
	ShapeType Indice(ViewParent_->ShapeBack_.size(), 0);
	for (SizeType i = 0; i < DataSize; ++i)
	{
		memcpy(It, ViewParent_->Data(Indice), AlignSize_);
		ViewParent_->IteratorAdd(Indice);
		It += AlignSize_;
	}

	ShapeFront_.clear();
	StepFront_.clear();
	StepBack_ = { ShapeBack_.begin() + 1,ShapeBack_.end() };
	StepBack_.emplace_back(AlignSize_);
	std::ranges::reverse(StepBack_);
	for (size_t i = 1; i < StepBack_.size(); ++i)
		StepBack_[i] *= StepBack_[i - 1];
	std::ranges::reverse(StepBack_);
	SliceBegin_ = { ShapeBack_.size(),0, ShapeType::allocator_type() };
	DimStride_ = { ShapeBack_.size(),1, ShapeType::allocator_type() };
	CurIndices_.clear();

	ViewParent_ = nullptr;
	ViewChild_.clear();
	ViewParent_ = nullptr;
	return *this;
}

LibSvcEnd