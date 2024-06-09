#pragma once

#define LibSvcTensorFnImpl(_Name) \
Tensor Tensor::_Name(const Tensor& _Input, ThreadPool* _ThreadPool) \
{ \
	_Input.ThrowOnNotEnabled(); \
	Tensor Ret(_Input.DType_, _Input.Device_->GetDevice()); \
	LibSvcOperator(_Input.DType_, Ret, _Name, _Input, _ThreadPool); \
	return Ret; \
} \
Tensor Tensor::_Name(ThreadPool* _ThreadPool) const \
{ \
	return _Name(*this, _ThreadPool); \
} \
Tensor& Tensor::_Name##_(ThreadPool* _ThreadPool) \
{ \
	ThrowOnNotEnabled(); \
	LibSvcOperatorNoRetrun(_Name##Inplace, *this, _ThreadPool); \
	return *this; \
}

#define LibSvcFloatTensorFnImpl(_Name) \
Tensor Tensor::_Name(const Tensor& _Input, ThreadPool* _ThreadPool) \
{ \
	_Input.ThrowOnNotEnabled(); \
	Tensor Ret(_Input.DType_, _Input.GetDevice()); \
	if (_Input.DType_ == TensorType::Float64) \
		Ret = ::libsvc::Float64::_Name(_Input, _ThreadPool); \
	else if (_Input.DType_ == TensorType::Float32) \
		Ret = ::libsvc::Float32::_Name(_Input, _ThreadPool); \
	else \
		return _Input.CreateView(); \
	return Ret; \
} \
Tensor Tensor::_Name(ThreadPool* _ThreadPool) const \
{ \
	return _Name(*this, _ThreadPool); \
} \
Tensor& Tensor::_Name##_(ThreadPool* _ThreadPool) \
{ \
	ThrowOnNotEnabled(); \
	if (DType_ == TensorType::Float64) \
		::libsvc::Float64::_Name##Inplace(*this, _ThreadPool); \
	else if (DType_ == TensorType::Float32) \
		::libsvc::Float32::_Name##Inplace(*this, _ThreadPool); \
	return *this; \
}

#define LibSvcTensorFnDef(_Name) \
static Tensor _Name(const Tensor& _Input, ThreadPool* _ThreadPool = nullptr); \
Tensor _Name(ThreadPool* _ThreadPool = nullptr) const; \
Tensor& _Name##_(ThreadPool* _ThreadPool = nullptr)

#define LibSvcMonoOperatorFunctionDef(_Name) \
Tensor _Name(const Tensor& _Input, ThreadPool* _ThreadPool); \
void _Name##Inplace(const Tensor& _Input, ThreadPool* _ThreadPool)

#define LibSvcMonoOperatorFunctionImpl(_Name, _OpFn) \
Tensor _Name(const Tensor& _Input, ThreadPool* _ThreadPool) \
{ \
	Tensor Ret(_Input.Shape(), _Input.DType(), _Input.GetDevice()); \
	const auto InputSqueeze = _Input.Squeeze(); \
	const auto Result = Ret.Squeeze(); \
	const auto CurDims = (SizeType)InputSqueeze.Shape().size(); \
	const auto& InputShape = InputSqueeze.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputSqueeze.Slice(ThreadSlices), \
						CurDims \
					); \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputSqueeze, CurDims); \
	} \
	else \
		_OpFn(Result, InputSqueeze, CurDims); \
	return Ret; \
}

#define LibSvcMonoOperatorInplaceFunctionImpl(_Name, _OpFn) \
void _Name##Inplace(const Tensor& _Input, ThreadPool* _ThreadPool) \
{ \
	const auto InputSqueeze = _Input.Squeeze(); \
	const auto& Result = InputSqueeze; \
	const auto CurDims = (SizeType)InputSqueeze.Shape().size(); \
	const auto& InputShape = InputSqueeze.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputSqueeze.Slice(ThreadSlices), \
						CurDims \
					); \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputSqueeze, CurDims); \
	} \
	else \
		_OpFn(Result, InputSqueeze, CurDims); \
}

#define LibSvcMultiOperatorFunctionImpl(_Name, _OpFn) \
Tensor _Name(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool) \
{ \
	if (_A.DType() != _B.DType()) \
		LibSvcThrow("Type MisMatch!"); \
	if (_A.GetDevice() != _B.GetDevice()) \
		LibSvcThrow("Device MisMatch!"); \
 \
	const auto BroadCast = Tensor::BroadCast(_A, _B); \
	Tensor Ret(BroadCast.first.Shape(), BroadCast.first.DType(), _A.GetDevice()); \
	const auto InputA = BroadCast.first.Squeeze(); \
	const auto InputB = BroadCast.second.Squeeze(); \
	const auto Result = Ret.Squeeze(); \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						InputB.Slice(ThreadSlices), \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, InputB, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, InputB, CurDims); \
 \
	return Ret; \
}

#define LibSvcMultiOperatorScalarFunctionImpl(_Name, _OpFn) \
Tensor _Name(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool) \
{ \
	const auto _Value = CastFrom(_ValType, _Val); \
 \
	Tensor Ret(_A.Shape(), _A.DType(), _A.GetDevice()); \
	const auto InputA = _A.Squeeze(); \
	const auto Result = Ret.Squeeze(); \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						_Value, \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, _Value, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, _Value, CurDims); \
 \
	return Ret; \
}

#define LibSvcMultiOperatorInplaceFunctionImpl(_Name, _OpFn) \
void _Name(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool) \
{ \
	if (_A.DType() != _B.DType()) \
		LibSvcThrow("Type MisMatch!"); \
 \
	const auto InputA = _A.Squeeze(); \
	const auto InputB = _A.BroadCast(_B).Squeeze(); \
	const auto& Result = InputA; \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						InputB.Slice(ThreadSlices), \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, InputB, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, InputB, CurDims); \
}

#define LibSvcMultiOperatorScalarInplaceFunctionImpl(_Name, _OpFn) \
void _Name(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool) \
{ \
	const auto _Value = CastFrom(_ValType, _Val); \
 \
	const auto InputA = _A.Squeeze(); \
	const auto Result = InputA.CreateView(); \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						_Value, \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, _Value, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, _Value, CurDims); \
}

#define LibSvcCompareOperatorFunctionImpl(_Name, _OpFn) \
Tensor _Name(const Tensor& _A, const Tensor& _B, ThreadPool* _ThreadPool) \
{ \
	if (_A.DType() != _B.DType()) \
		LibSvcThrow("Type MisMatch!"); \
	if (_A.GetDevice() != _B.GetDevice()) \
		LibSvcThrow("Device MisMatch!"); \
 \
	const auto BroadCast = Tensor::BroadCast(_A, _B); \
	Tensor Ret(BroadCast.first.Shape(), TensorType::Boolean, _A.GetDevice()); \
	const auto InputA = BroadCast.first.Squeeze(); \
	const auto InputB = BroadCast.second.Squeeze(); \
	const auto Result = Ret.Squeeze(); \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						InputB.Slice(ThreadSlices), \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, InputB, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, InputB, CurDims); \
 \
	return Ret; \
}

#define LibSvcCompareOperatorScalarFunctionImpl(_Name, _OpFn) \
Tensor _Name(const Tensor& _A, cpvoid _Val, TensorType _ValType, ThreadPool* _ThreadPool) \
{ \
	const auto _Value = CastFrom(_ValType, _Val); \
 \
	Tensor Ret(_A.Shape(), TensorType::Boolean, _A.GetDevice()); \
	const auto InputA = _A.Squeeze(); \
	const auto Result = Ret.Squeeze(); \
 \
	const auto CurDims = (SizeType)InputA.Shape().size(); \
	const auto& InputShape = InputA.Shape(); \
	const auto TotalSize = VectorMul(InputShape); \
 \
	if (_ThreadPool && TotalSize > LIBSVC_CONT_THRESHOLD_MIN_SIZE) \
	{ \
		const auto NWorkers = _ThreadPool->GetThreadCount(); \
		const auto SqueezedDims = (SizeType)InputShape.size(); \
 \
		Vector<Range> Slices; \
		for (SizeType i = 0; i < SqueezedDims; ++i) \
		{ \
			if (InputShape[i] < NWorkers) \
				Slices.emplace_back(None); \
			else \
			{ \
				const auto Step = Tensor::Ceil(InputShape[i], NWorkers); \
				for (SizeType j = 0; ; j += Step) \
				{ \
					const auto End = std::min(j + Step, InputShape[i]); \
					if (j >= End) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
					auto ThreadSlices = Slices; \
					ThreadSlices.emplace_back(j, End); \
 \
					_ThreadPool->Commit( \
						_OpFn, \
						Result.Slice(ThreadSlices), \
						InputA.Slice(ThreadSlices), \
						_Value, \
						CurDims \
					); \
 \
					if (End == InputShape[i]) \
					{ \
						_ThreadPool->Join(); \
						return Ret; \
					} \
				} \
			} \
		} \
		_OpFn(Result, InputA, _Value, CurDims); \
	} \
	else \
		_OpFn(Result, InputA, _Value, CurDims); \
 \
	return Ret; \
}
