#include <random>

#include "Tensor/Tensor.h"
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

bool RangeIsAllNone(const Vector<Range>& _Input)
{
	for (const auto& i : _Input)
		if (!i.IsNone)
			return false;
	return true;
}

//Construct
Tensor::~Tensor()
{
	Free();
}

Tensor::Tensor(const ShapeType& _Shape, TensorType _DType) : TensorBase(_DType)
{
	for (const auto i : _Shape)
		if (i <= 0)
			LibSvcThrow("Shape Must > 0");
	AlignSize_ = DType2Size(DType_);
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
	_Left.ThrowOnNotEnabled();
	*this = _Left.Clone();
}

Tensor::Tensor(Tensor&& _Right) noexcept : TensorBase(_Right.DType_)
{
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
#ifdef LIBSVC_DEBUG
	if (AlignSize_ != DType2Size(DType_))
		LibSvcThrow("AlignSize MisMatch!");
	if (ShapeBack_.empty() || StepBack_.empty())
		LibSvcThrow("Axis Out Of Range!");
	if (!DataPtr_)
		LibSvcThrow("NullPointer Error!");

	if ((CurIndices_.size() != StepFront_.size()) ||
		(DimStride_.size() != ShapeBack_.size()) ||
		(DimStride_.size() != StepBack_.size()))
	{
		LibSvcThrow("Indices Error!");
	}

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
#endif
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

void Tensor::ReCalcViewInfo()
{
	StepBack_ = { ShapeBack_.begin() + 1,ShapeBack_.end() };
	StepBack_.emplace_back(AlignSize_);
	std::ranges::reverse(StepBack_);
	for (size_t i = 1; i < StepBack_.size(); ++i)
		StepBack_[i] *= StepBack_[i - 1];
	std::ranges::reverse(StepBack_);

	if (SliceBegin_.front())
	{
		CurIndices_.emplace_back(SliceBegin_.front());
		StepFront_.emplace_back(StepBack_.front());
	}
	SliceBegin_ = { ShapeBack_.size(),0, ShapeType::allocator_type() };
	DimStride_ = { ShapeBack_.size(),1, ShapeType::allocator_type() };
}

void Tensor::Assign1D(const void* _Val) const
{
	SizeType CurIndex = 0;
	for (size_t i = 0; i < CurIndices_.size(); ++i)
		CurIndex += CurIndices_[i] * StepFront_[i];
	for (SizeType i = 0; i < Size(0); ++i)
		memcpy(DataPtr_ + CurIndex + (((i * DimStride_[0]) + SliceBegin_[0]) * StepBack_[0]), _Val, AlignSize_);
}

Tensor Tensor::GatherRef(SizeType _Index) const
{
	const auto Idx = CalcIndex(_Index, ShapeBack_.front());
	Tensor Ret(DType_);
	Ret.DType_ = DType_;

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

	Ret.IsBroadCasted_ = IsBroadCasted_;

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

Tensor& Tensor::operator=(const Tensor& _Left)
{
	if (&_Left == this)
		return *this;
	if (_Left.ViewParent_ && _Left.ViewParent_ == this)
		LibSvcThrow("Assign To Parent Is Not Allowed!");
	_Left.ThrowOnNotEnabled();
	Free();
	return *this = _Left.CreateView();
}

Tensor& Tensor::operator=(Tensor&& _Right) noexcept
{
	if (&_Right == this)
		return *this;
	if (_Right.ViewParent_ && _Right.ViewParent_ == this)
		LibSvcThrow("Assign To Parent Is Not Allowed!");
	Free();
	DType_ = _Right.DType_;
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

Tensor& Tensor::operator=(float64 _Val)
{
	Fix(_Val);
	return *this;
}

Tensor& Tensor::operator=(int64 _Val)
{
	Fix(_Val);
	return *this;
}

Tensor Tensor::operator[](SizeType _Index) const
{
	ThrowOnNotEnabled();
	return GatherRef(_Index);
}

Tensor Tensor::operator[](const SliceOptions& _SliceOptions) const
{
	return Slice(_SliceOptions);
}

void Tensor::Assign(const void* _Buffer, SizeType _BufferSize, ThreadPool* _ThreadPool) const
{
	ThrowOnNotEnabled();
	LibSvcOperatorNoRetrun(AssignBuffer, *this, _Buffer, (const byte*)_Buffer + _BufferSize, _ThreadPool);
}

void Tensor::Assign(const void* _Val, TensorType _Type, ThreadPool* _ThreadPool) const
{
	ThrowOnNotEnabled();
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, _Type, _ThreadPool);
}

void Tensor::Assign(float64 _Val, ThreadPool* _ThreadPool) const
{
	Fix(_Val, _ThreadPool);
}

void Tensor::Assign(int64 _Val, ThreadPool* _ThreadPool) const
{
	Fix(_Val, _ThreadPool);
}

void Tensor::Assign(const Tensor& _Val, ThreadPool* _ThreadPool) const
{
	LibSvcOperatorNoRetrun(AssignTensor, *this, _Val, _ThreadPool);
}

//Public

void Tensor::IteratorAdd(ShapeType& _Indices) const
{
	auto Val = _Indices.data() + _Indices.size() - 1;
	const auto ShapePtr = ShapeBack_.data();

	for (size_t i = _Indices.size() - 1; ; --i)
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

void Tensor::IteratorSub(ShapeType& _Indices) const
{
	auto Val = _Indices.data() + _Indices.size() - 1;
	const auto ShapePtr = ShapeBack_.data();

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
		*Val = (*(ShapePtr + i) - 1);
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

	if ((CurIndices_.size() != StepFront_.size()) || (DimStride_.size() != ShapeBack_.size()) ||
		(DimStride_.size() != StepBack_.size()))
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
	if (!StepFront_.empty())
		return true;
	return !IsContinuous();
}

bool Tensor::IsContinuous() const
{
	for (const auto i : DimStride_)
		if (i != 1)
			return false;
	for (size_t i = 1; i < SliceBegin_.size(); ++i)
		if (SliceBegin_[i] != 0)
			return false;
	if (StepBack_.back() != AlignSize_)
		return false;
	for (size_t i = 1; i < StepBack_.size(); ++i)
		if (StepBack_[i - 1] / ShapeBack_[i] != StepBack_[i])
			return false;
	return true;
}

bool Tensor::IsTranSposedContinuous() const
{
	//const auto Dims = DimCount();
	for (const auto i : DimStride_)
		if (i != 1)
			return false;
	for (size_t i = 1; i < SliceBegin_.size(); ++i)
		if (SliceBegin_[i] != 0)
			return false;
	const auto MaxStep = std::ranges::max(StepBack_);
	if (SliceBegin_[0] != 0 && StepBack_[0] != MaxStep)
		return false;
	SizeType DimsMul = AlignSize_;
	for (size_t i = 0; i < StepBack_.size(); ++i)
		if (StepBack_[i] != MaxStep)
			DimsMul *= ShapeBack_[i];
	if (DimsMul != MaxStep)
		return false;
	return true;
}

bool Tensor::IsView() const
{
	return ViewParent_;
}

Tensor Tensor::Clone(ThreadPool* _ThreadPool) const
{
	ThrowOnNotEnabled();
	Tensor Ret(ShapeBack_, DType_);
	Ret.Assign(*this, _ThreadPool);
	return Ret;
}

Tensor Tensor::CreateView() const
{
	ThrowOnNotEnabled();
	Tensor Ret(DType_);
	Ret.DType_ = DType_;
	Ret.ShapeBack_ = ShapeBack_;
	Ret.StepFront_ = StepFront_;
	Ret.StepBack_ = StepBack_;
	Ret.SliceBegin_ = SliceBegin_;
	Ret.DimStride_ = DimStride_;
	Ret.CurIndices_ = CurIndices_;
	Ret.AlignSize_ = AlignSize_;
	Ret.IsBroadCasted_ = IsBroadCasted_;
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
	if (IsBroadCasted())
		LibSvcThrow("Broad Casted Could Not Be Sliced!");
	if (ShapeBack_.empty() || _SliceOptions.size() > ShapeBack_.size())
		LibSvcThrow("Axis Out Of Range!");
	Tensor Ret = CreateView();
	for (size_t i = 0; i < _SliceOptions.size(); ++i)
	{
		if (_SliceOptions[i].IsNone)
			continue;
		const auto SliceBeginPos = CalcIndex(_SliceOptions[i].Begin, ShapeBack_[i]);
		auto SliceEndPos = _SliceOptions[i].End;
		if (SliceEndPos > ShapeBack_[i] || SliceEndPos < -(ShapeBack_[i] + 1))
			LibSvcThrow("Index Out Of Range!");
		if (SliceEndPos == -(ShapeBack_[i] + 1))
			SliceEndPos = -1;
		else if (SliceEndPos < 0)
			SliceEndPos += ShapeBack_[i] + 1;

		const auto SliceLength = SliceEndPos - SliceBeginPos;
		if (SliceLength == 0)
			LibSvcThrow("Slice Length Must > 0");
		if (SliceLength > 0 && _SliceOptions[i].Step < 0 ||
			SliceLength < 0 && _SliceOptions[i].Step > 0)
			LibSvcThrow("Step & (SliceEnd - SliceBegin) Should Have The Same Sign!");
		Ret.SliceBegin_[i] += SliceBeginPos * Ret.DimStride_[i];
		Ret.ShapeBack_[i] = Ceil(abs(SliceLength), abs(_SliceOptions[i].Step));
		Ret.DimStride_[i] *= _SliceOptions[i].Step;
	}
	return Ret;
}

Tensor Tensor::ReversedSlice(const SliceOptions& _SliceOptions) const
{
	Vector<Range> TempRange = _SliceOptions, NewRange;
	TempRange.resize(ShapeBack_.size(), None);
	for (size_t i = TempRange.size() - 1; i < TempRange.size(); --i)
	{
		if (TempRange[i].IsNone)
			NewRange.emplace_back(None);
		else
			NewRange.emplace_back(TempRange[i].Begin, TempRange[i].Step, TempRange[i].End);
	}
	return Slice(NewRange);
}

Tensor Tensor::Permute(const ShapeType& _DPremute) const
{
	ThrowOnNotEnabled();
	if (ShapeBack_.empty() || _DPremute.size() != ShapeBack_.size())
		LibSvcThrow("N_DIMS MisMatch!");
	Tensor Ret = CreateView();
	auto TransposedDims = _DPremute;
	std::ranges::sort(TransposedDims);
	if (TransposedDims[0] != 0)
		LibSvcThrow("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");
	for (size_t i = 1; i < TransposedDims.size(); ++i)
		if (TransposedDims[i] != TransposedDims[i - 1] + 1)
			LibSvcThrow("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");

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
	auto IndicPtr = CurIndices_.data();
	auto StepPtr = StepFront_.data();
	const auto AllSize = CurIndices_.size();
	for (size_t i = 0; i < AllSize; ++i)
		Index += *(IndicPtr++) * *(StepPtr++);
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

void Tensor::FixOnes(ThreadPool* _ThreadPool) const
{
	float _Val = 1.f;
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float32, _ThreadPool);
}

void Tensor::FixZeros(ThreadPool* _ThreadPool) const
{
	float _Val = 0.f;
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float32, _ThreadPool);
}

void Tensor::Fix(double _Val, ThreadPool* _ThreadPool) const
{
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float64, _ThreadPool);
}

void Tensor::Fix(int64 _Val, ThreadPool* _ThreadPool) const
{
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Int64, _ThreadPool);
}

void Tensor::RandFix(uint64 _Seed, ThreadPool* _ThreadPool) const
{
	LibSvcOperatorNoRetrun(FixWithRandom, *this, _Seed, 0., 11451.41919810, _ThreadPool);
}

void Tensor::RandnFix(uint64 _Seed, double _Mean, double _Sigma, ThreadPool* _ThreadPool) const
{
	LibSvcOperatorNoRetrun(FixWithRandom, *this, _Seed, _Mean, _Sigma, _ThreadPool);
}

Tensor Tensor::View(const ShapeType& _ViewShape) const
{
	if (!IsContinuous())
		LibSvcThrow("View Should Be Continuous!");
	if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), -1) > 1)
		LibSvcThrow("Count Of Dynamic Axis Should <= 1!");
	for (const auto i : _ViewShape)
		if (i <= 0 && i != -1)
			LibSvcThrow("Count Of Size Should > 0 Or = -1 (Dynamic Axis)!");
	Tensor Ret = CreateView();
	const auto SrcSize = VectorMul(Ret.ShapeBack_);
	const auto DstSize = VectorMul(_ViewShape);
	if ((DstSize < 0 && (SrcSize % abs(DstSize)) != 0) || (DstSize > 0 && (SrcSize != DstSize)))
		LibSvcThrow("Size MisMatch!");
	const auto DynamicAxes = SrcSize / DstSize;
	Ret.ShapeBack_ = _ViewShape;
	for (auto& i : Ret.ShapeBack_)
		if (i == -1)
		{
			i = DynamicAxes;
			break;
		}
	Ret.ReCalcViewInfo();
	return Ret;
}

Tensor& Tensor::Continuous(ThreadPool* _ThreadPool)
{
	ThrowOnNotEnabled();
	if (!IsView() && !IsContinuous())
		LibSvcThrow("Src Should Be Continuous, Please Report This Bug!");
	if (!IsView() || IsContinuous())
		return *this;
	return *this = Clone(_ThreadPool);
}

const ShapeType& Tensor::SliceBegins() const
{
	return SliceBegin_;
}

const ShapeType& Tensor::StepsBack() const
{
	return StepBack_;
}

const ShapeType& Tensor::Strides() const
{
	return DimStride_;
}

Tensor Tensor::UnSqueeze(SizeType Dim) const
{
	auto Ret = CreateView();
	Dim = CalcIndex(Dim, SizeType(Ret.ShapeBack_.size() + 1));
	Ret.ShapeBack_.insert(Ret.ShapeBack_.begin() + Dim, 1);
	if (Dim == SizeType(Ret.ShapeBack_.size()))
		Ret.StepBack_.insert(Ret.StepBack_.begin() + Dim, DType2Size(Ret.DType()));
	else
		Ret.StepBack_.insert(Ret.StepBack_.begin() + Dim, *(Ret.StepBack_.begin() + Dim));
	Ret.SliceBegin_.insert(Ret.SliceBegin_.begin() + Dim, 0);
	Ret.DimStride_.insert(Ret.DimStride_.begin() + Dim, 1);
	return Ret;
}

Tensor Tensor::Squeeze(SizeType Dim) const
{
	auto Ret = CreateView();
	Dim = CalcIndex(Dim, SizeType(Ret.ShapeBack_.size()));
	if (Ret.ShapeBack_[Dim] != 1)
		return Ret;
	if (Ret.SliceBegin_[Dim])
	{
		Ret.CurIndices_.emplace_back(Ret.SliceBegin_[Dim]);
		Ret.StepFront_.emplace_back(Ret.StepBack_[Dim]);
	}
	Ret.ShapeBack_.erase(Ret.ShapeBack_.begin() + Dim);
	Ret.StepBack_.erase(Ret.StepBack_.begin() + Dim);
	Ret.SliceBegin_.erase(Ret.SliceBegin_.begin() + Dim);
	Ret.DimStride_.erase(Ret.DimStride_.begin() + Dim);
	return Ret;
}

Tensor Tensor::Squeeze() const
{
	auto Ret = CreateView();
	auto Iter = std::ranges::find(Ret.ShapeBack_, 1);
	while (Iter != Ret.ShapeBack_.end())
	{
		const auto Idx = Iter - Ret.ShapeBack_.begin();
		if (Ret.SliceBegin_[Idx])
		{
			Ret.CurIndices_.emplace_back(Ret.SliceBegin_[Idx]);
			Ret.StepFront_.emplace_back(Ret.StepBack_[Idx]);
		}
		Ret.ShapeBack_.erase(Iter);
		Ret.StepBack_.erase(Ret.StepBack_.begin() + Idx);
		Ret.SliceBegin_.erase(Ret.SliceBegin_.begin() + Idx);
		Ret.DimStride_.erase(Ret.DimStride_.begin() + Idx);
		Iter = std::ranges::find(Ret.ShapeBack_, 1);
	}
	return Ret;
}

std::pair<Tensor, Tensor> Tensor::BroadCast(const Tensor& _A, const Tensor& _B)
{
	std::pair Ret{ _A.CreateView(), _B.CreateView() };
	auto& First = Ret.first;
	auto& Second = Ret.second;
	const auto Dims = std::max(First.ShapeBack_.size(), Second.ShapeBack_.size());
	std::ranges::reverse(First.ShapeBack_);
	std::ranges::reverse(Second.ShapeBack_);
	std::ranges::reverse(First.StepBack_);
	std::ranges::reverse(Second.StepBack_);
	std::ranges::reverse(First.SliceBegin_);
	std::ranges::reverse(Second.SliceBegin_);
	std::ranges::reverse(First.DimStride_);
	std::ranges::reverse(Second.DimStride_);
	for (size_t i = 0; i < Dims; ++i)
	{
		auto XSize = 1ll, YSize = 1ll;
		if (i < First.ShapeBack_.size())
			XSize = First.ShapeBack_[i];
		else
		{
			First.ShapeBack_.emplace_back(1);
			First.StepBack_.emplace_back(First.StepBack_.back());
			First.SliceBegin_.emplace_back(0);
			First.DimStride_.emplace_back(0);
			First.IsBroadCasted_ = true;
		}
		if (i < Second.ShapeBack_.size())
			YSize = Second.ShapeBack_[i];
		else
		{
			Second.ShapeBack_.emplace_back(1);
			Second.StepBack_.emplace_back(First.StepBack_.back());
			Second.SliceBegin_.emplace_back(0);
			Second.DimStride_.emplace_back(0);
			Second.IsBroadCasted_ = true;
		}
		if (XSize == YSize)
			continue;
		if (XSize == 1)
		{
			First.ShapeBack_[i] = YSize;
			First.DimStride_[i] = 0;
			First.IsBroadCasted_ = true;
		}
		else if (YSize == 1)
		{
			Second.ShapeBack_[i] = XSize;
			Second.DimStride_[i] = 0;
			Second.IsBroadCasted_ = true;
		}
		else
			LibSvcThrow("TensorA & TensorB Can Not Be BroadCast!");
	}
	std::ranges::reverse(First.ShapeBack_);
	std::ranges::reverse(Second.ShapeBack_);
	std::ranges::reverse(First.StepBack_);
	std::ranges::reverse(Second.StepBack_);
	std::ranges::reverse(First.SliceBegin_);
	std::ranges::reverse(Second.SliceBegin_);
	std::ranges::reverse(First.DimStride_);
	std::ranges::reverse(Second.DimStride_);
	return Ret;
}

Tensor Tensor::BroadCast(const Tensor& _Other) const
{
	auto Bd = BroadCast(*this, _Other);
	if (Bd.first.IsBroadCasted())
		LibSvcThrow("TensorA & TensorB Can Not Be BroadCast In This Operator!");
	return std::move(Bd.second);
}

bool Tensor::IsBroadCasted() const
{
	return IsBroadCasted_;
}

SizeType Tensor::DimCount() const
{
	return (SizeType)ShapeBack_.size();
}

bool Tensor::IsVector() const
{
	return DimCount() == 1;
}

byte* Tensor::GetPtr() const
{
	return Data(ShapeType(DimCount(), 0));
}

ShapeType Tensor::CalcContinuous() const
{
	const auto Dims = DimCount();
	if (Dims == 1)
		return { 0, 0, 0, 0, 0 };
	Vector<std::pair<SizeType, SizeType>> Ret;
	Ret.reserve(Dims);
	for (SizeType i = 0; i < Dims; ++i)
		Ret.emplace_back(StepBack_[i], i);
	std::ranges::sort(Ret);
	std::ranges::reverse(Ret);
	ShapeType Rtn;
	for (const auto& i : Ret | std::views::values)
		Rtn.emplace_back(i);
	return Rtn;
}

Tensor Tensor::Padding(const Tensor& _Input, const Vector<Range>& _Pad, PaddingType _Type, TensorType _ValueType, lpvoid _Val, ThreadPool* _ThreadPool)
{
	_Input.ThrowOnNotEnabled();

	if (_Pad.size() > _Input.Shape().size())
		LibSvcThrow("Dim Out Of Range");
	auto Shape = _Input.Shape();
	const auto InputShapePtr = _Input.Shape().data();
	Vector<Range> SliceSrc, SliceDst;

	for (size_t i = 0; i < _Pad.size(); ++i)
	{
		if (_Pad[i].Begin < 0 || _Pad[i].End < 0)
			LibSvcThrow("Use Slice Instead");

		if (_Pad[i].IsNone || (_Pad[i].Begin == 0 && _Pad[i].End == 0))
		{
			SliceSrc.emplace_back(None);
			SliceDst.emplace_back(None);
			continue;
		}

		Shape[i] += (_Pad[i].Begin + _Pad[i].End);
		if (Shape[i] <= 0)
			LibSvcThrow("Incorrect Pad Size!");

		SliceSrc.emplace_back(None);
		SliceDst.emplace_back(_Pad[i].Begin, InputShapePtr[i] + _Pad[i].Begin);
	}

	Tensor Ret(Shape, _Input.DType());

	if (_Type == PaddingType::Zero)
	{
		Ret.FixZeros(_ThreadPool);
		Ret.Slice(SliceDst).Assign(_Input.Slice(SliceSrc), _ThreadPool);
		return Ret;
	}
	if (_Type == PaddingType::Constant)
	{
		Ret.Assign(_Val, _ValueType, _ThreadPool);
		Ret.Slice(SliceDst).Assign(_Input.Slice(SliceSrc), _ThreadPool);
		return Ret;
	}

	Ret.Slice(SliceDst).Assign(_Input.Slice(SliceSrc), _ThreadPool);

	if (_Type == PaddingType::Zero)
	{
		for (size_t i = _Pad.size() - 1; i < _Pad.size(); --i)
		{
			Vector<Range> RngFront, RngBack;
			for (size_t j = 0; j < _Pad.size(); ++j)
			{
				if (i == j && !_Pad[i].IsNone)
				{
					if (_Pad[i].Begin <= 0)
						RngFront.emplace_back(None);
					else
						RngFront.emplace_back(0, _Pad[i].Begin);
					if (_Pad[i].End <= 0)
						RngBack.emplace_back(None);
					else
						RngBack.emplace_back(InputShapePtr[i] + _Pad[i].Begin, -1);
				}
				else
				{
					RngFront.emplace_back(None);
					RngBack.emplace_back(None);
				}
			}
			if (!RangeIsAllNone(RngFront))
				Ret.Slice(RngFront).FixZeros(_ThreadPool);
			if (!RangeIsAllNone(RngBack))
				Ret.Slice(RngBack).FixZeros(_ThreadPool);
		}
	}
	

	return Ret;
}

Tensor Tensor::Pad(const Tensor& _Input, const Vector<Range>& _Pad, PaddingType _Type, TensorType _ValueType, lpvoid _Val, ThreadPool* _ThreadPool)
{
	Vector<Range> TempRange = _Pad, NewRange;
	TempRange.resize(_Input.ShapeBack_.size(), None);
	for (size_t i = TempRange.size() - 1; i < TempRange.size(); --i)
	{
		if (TempRange[i].IsNone)
			NewRange.emplace_back(None);
		else
			NewRange.emplace_back(TempRange[i].Begin, TempRange[i].End);
	}
	return Padding(_Input, NewRange, _Type, _ValueType, _Val, _ThreadPool);
}

Tensor Tensor::Repeat(const Tensor& _Input, const ShapeType& _Dims, const ShapeType& _Count)
{
	_Input.ThrowOnNotEnabled();
	if (_Dims.size() != _Count.size())
		LibSvcThrow("Size Of Dims And Count MisMatch!");
	const auto& ShapeInp = _Input.Shape();
	auto Shape = _Input.Shape();
	const auto TensorAxis = (SizeType)Shape.size();
	const auto DimCount = (SizeType)_Dims.size();
	
	for (SizeType i = 0; i < DimCount; ++i)
	{
		if (_Dims[i] < TensorAxis)
			Shape[_Dims[i]] *= _Count[i];
		else
			LibSvcThrow("Axis Out Of Range!");
	}

	Vector<Range> Slices
	for (SizeType i = 0; i < TensorAxis; ++i)
	{
		
	}
}

LibSvcEnd