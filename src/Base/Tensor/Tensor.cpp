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

void Tensor::CalcInfo()
{
	StepBack_ = { ShapeBack_.begin() + 1,ShapeBack_.end() };
	StepBack_.emplace_back(AlignSize_);
	std::ranges::reverse(StepBack_);
	for (size_t i = 1; i < StepBack_.size(); ++i)
		StepBack_[i] *= StepBack_[i - 1];
	std::ranges::reverse(StepBack_);

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

//Operator=

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
	LibSvcOperatorNoRetrun(AssignBuffer, *this, _Buffer, (byte*)_Buffer + _BufferSize);
}

void Tensor::Assign(const void* _Val, TensorType _Type) const
{
	ThrowOnNotEnabled();
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, _Type);
}

void Tensor::Assign(float64 _Val) const
{
	Fix(_Val);
}

void Tensor::Assign(int64 _Val) const
{
	Fix(_Val);
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
	auto TotalStep = StepFront_;
	TotalStep.insert(TotalStep.end(), StepBack_.begin(), StepBack_.end());
	for (size_t i = 1; i < TotalStep.size(); ++i)
		if (TotalStep[i] > TotalStep[i - 1])
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

	Tensor Ret;

	Ret.DType_ = DType_;
	Ret.AlignSize_ = AlignSize_;

	Ret.ShapeFront_.clear();
	Ret.ShapeBack_ = ShapeBack_;

	Ret.StepFront_.clear();
	Ret.StepBack_ = { Ret.ShapeBack_.begin() + 1,Ret.ShapeBack_.end() };
	Ret.StepBack_.emplace_back(Ret.AlignSize_);
	std::ranges::reverse(Ret.StepBack_);
	for (size_t i = 1; i < Ret.StepBack_.size(); ++i)
		Ret.StepBack_[i] *= Ret.StepBack_[i - 1];
	std::ranges::reverse(Ret.StepBack_);

	Ret.SliceBegin_ = { Ret.ShapeBack_.size(),0, ShapeType::allocator_type() };
	Ret.DimStride_ = { Ret.ShapeBack_.size(),1, ShapeType::allocator_type() };
	Ret.CurIndices_.clear();

	const auto DataSize = VectorMul(Ret.ShapeBack_);
	const auto BufSize = DataSize * Ret.AlignSize_;
	Ret.DataPtr_ = (byte*)LIBSVC_MALLOC(BufSize);

	auto It = Ret.DataPtr_;
	ShapeType Indice(Ret.ShapeBack_.size(), 0);
	for (SizeType i = 0; i < DataSize; ++i)
	{
		memcpy(It, Data(Indice), Ret.AlignSize_);
		IteratorAdd(Indice);
		It += Ret.AlignSize_;
	}

	Ret.ViewParent_ = nullptr;
	Ret.ViewChild_.clear();
	
	return Ret;
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

void Tensor::FixOnes() const
{
	float _Val = 1.f;
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float32);
}

void Tensor::FixZeros() const
{
	if (!IsContinuous())
	{
		float _Val = 0.f;
		LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float32);
	}
	else
		memset(DataPtr_, 0, VectorMul(ShapeBack_) * AlignSize_);
}

void Tensor::Fix(double _Val) const
{
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Float64);
}

void Tensor::Fix(int64 _Val) const
{
	LibSvcOperatorNoRetrun(AssignValue, *this, &_Val, TensorType::Int64);
}

void Tensor::RandFix(int _Seed) const
{


}

void Tensor::RandnFix(int _Seed, double _Mean, double _Sigma) const
{

}

Tensor Tensor::View(const ShapeType& _ViewShape) const
{
	//TODO
	if (!IsContinuous())
		LibSvcThrow("View Should Be Continuous!");
	if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), -1) > 1)
		LibSvcThrow("Count Of Dynamic Axis Should <= 1!");
	if (std::ranges::count(_ViewShape.begin(), _ViewShape.end(), 0))
		LibSvcThrow("Count Of Size Should > 0 Or = -1 (Dynamic Axis)!");
	Tensor Ret = CreateView();
	const auto SrcSize = VectorMul(Ret.ShapeBack_);
	const auto DstSize = abs(VectorMul(_ViewShape));
	if ((SrcSize % DstSize) != 0)
		LibSvcThrow("Size MisMatch!");
	const auto DynamicAxes = SrcSize % DstSize;
	Ret.ShapeBack_ = _ViewShape;
	for (auto& i : Ret.ShapeBack_)
		if (i == -1)
		{
			i = DynamicAxes;
			break;
		}
	Ret.CalcInfo();
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
	const auto NewData = (byte*)LIBSVC_MALLOC(BufSize);

	{
		if (ViewParent_->ViewParent_)
			LibSvcThrow("View Parent Can Not Have View Parent, Please Report This Bug!");
		std::lock_guard Lock(ViewParent_->ViewMx_);
		auto& Views = ViewParent_->ViewChild_;
		const auto& Iter = std::ranges::find(Views.begin(), Views.end(), this);
		if (Iter != Views.end())
			Views.erase(Iter);
		else
			LibSvcThrow("View Not In Parent's Child List, Please Report This Bug!");
	}

	auto It = NewData;
	ShapeType Indice(ShapeBack_.size(), 0);
	for (SizeType i = 0; i < DataSize; ++i)
	{
		memcpy(It, Data(Indice), AlignSize_);
		IteratorAdd(Indice);
		It += AlignSize_;
	}

	DataPtr_ = NewData;

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
	return *this;
}

const ShapeType& Tensor::CurIndices() const
{
	return CurIndices_;
}

const ShapeType& Tensor::SliceBegins() const
{
	return SliceBegin_;
}

const ShapeType& Tensor::StepsBack() const
{
	return StepBack_;
}

const ShapeType& Tensor::StepsFront() const
{
	return StepFront_;
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
		Ret.StepBack_.insert(Ret.StepBack_.begin() + Dim, *(Ret.StepBack_.begin() + Dim + 1));
	Ret.SliceBegin_.insert(Ret.SliceBegin_.begin() + Dim, 0);
	Ret.DimStride_.insert(Ret.DimStride_.begin() + Dim, 1);
	return Ret;
}

Tensor Tensor::Squeeze(SizeType Dim) const
{
	auto Ret = CreateView();
	Dim = CalcIndex(Dim, SizeType(Ret.ShapeBack_.size()));
	if (Ret.ShapeBack_[Dim] != 1 || Ret.SliceBegin_[Dim] != 0)
		return Ret;
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
		if(*(Ret.SliceBegin_.begin() + Idx) != 0)
			continue;
		Ret.ShapeBack_.erase(Iter);
		Ret.StepBack_.erase(Ret.StepBack_.begin() + Idx);
		Ret.SliceBegin_.erase(Ret.SliceBegin_.begin() + Idx);
		Ret.DimStride_.erase(Ret.DimStride_.begin() + Idx);
		Iter = std::ranges::find(Ret.ShapeBack_, 1);
	}
	return Ret;
}

LibSvcEnd