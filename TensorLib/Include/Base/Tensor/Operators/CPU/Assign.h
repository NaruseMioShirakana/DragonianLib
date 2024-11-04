#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Random_Unfold = 8;
constexpr int64_t _D_Dragonian_Lib_Operator_Assign_Randn_Unfold = 8;

template<typename _TypeDest, typename _TypeSrc>
void AssignTensorCont(
	_TypeDest* _Dest,
	const _TypeSrc* _Src,
	SizeType DestSize,
	void*
)
{
	if constexpr (_Impl_Dragonian_Lib_Constexpr_Is_Same_Type_v<_TypeDest, _TypeSrc>)
	{
		DestSize *= sizeof(_TypeDest);
		Vectorized<_TypeDest>::DragonianLibMemCpy(_Dest, _Src, DestSize);
	}
	else
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				_Dest[i] = (_TypeDest)_Src[i];
				++i;
			}
		}
		while (i < DestSize)
		{
			_Dest[i] = (_TypeDest)_Src[i];
			++i;
		}
	}
}

template<typename _TypeDest, typename _TypeSrc>
void AssignTensor(
	_TypeDest* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	const _TypeSrc* _Src,
	std::shared_ptr<OperatorParameter> _SrcInfoOld,
	void*
)
{
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	const OperatorParameter& _SrcInfo = *_SrcInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();

	const auto Func = [&](int64_t _IndexA, int64_t _IndexB) { _Dest[_IndexA] = (_TypeDest)_Src[_IndexB]; };
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();
	const SizeType* __restrict SrcViewStep = _SrcInfo.ViewStep.Data();
	const SizeType* __restrict SrcViewLeft = _SrcInfo.ViewLeft.Data();
	const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data();

	if (ViewRank == 1)
		DoubleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 2)
		DoubleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 3)
		DoubleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 4)
		DoubleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 5)
		DoubleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 6)
		DoubleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 7)
		DoubleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 8)
		DoubleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 9)
		DoubleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else if (ViewRank == 10)
		DoubleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStep, ViewLeft, ViewStride,
			SrcViewStep, SrcViewLeft, SrcViewStride,
			Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
template <typename _TypeSrc>
void OperatorsBase<_Type, Device::CPU>::ImplCast(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _TypeSrc* _Src,
	const OperatorParameter& _SrcInfo,
	bool Continuous
)
{
	ImplMultiThreadDouble<_Type, _TypeSrc, void*>(
		_Dest,
		_DestInfo,
		_Src,
		_SrcInfo,
		nullptr,
		Continuous,
		AssignTensor<_Type, _TypeSrc>,
		AssignTensorCont<_Type, _TypeSrc>
	);
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignTensor(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src,
	const OperatorParameter& _SrcInfo,
	bool Continuous
)
{
	ImplMultiThreadDouble<_Type, _Type, void*>(
		_Dest,
		_DestInfo,
		_Src,
		_SrcInfo,
		nullptr,
		Continuous,
		AssignTensor<_Type, _Type>,
		AssignTensorCont<_Type, _Type>
	);
}

template<typename _Type>
struct _Struct_Buffer
{
	const _Type* _DestBegin;
	const _Type* _Src;
	SizeType _Count;
};

template<typename _Type>
void AssignBufferCont(
	_Type* _Dest,
	SizeType DestSize,
	_Struct_Buffer<_Type> _Value
)
{
	const auto Index = _Dest - _Value._DestBegin;
	if (Index >= _Value._Count) return;
	DestSize = std::min(DestSize, _Value._Count - Index) * sizeof(_Type);
	Vectorized<_Type>::DragonianLibMemCpy(_Dest, _Value._Src, DestSize);
}

template<typename _Type>
void AssignBuffer(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	_Struct_Buffer<_Type> _Value
)
{
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();

	const auto End = _Value._Src + _Value._Count;
	const _Type* _Src;
	if(std::ranges::count(_DestInfo.Begin, 0) == ViewRank)
		_Src = _Value._Src;
	else if (std::ranges::count(_DestInfo.Begin, 0) == ViewRank - 1)
	{
		SizeType TaskIndex = 1;
		for (SizeType i = 0; i < ViewRank; ++i)
			if (_DestInfo.Begin[i] != 0)
				TaskIndex *= _DestInfo.Begin[i];
			else
				TaskIndex *= _DestInfo.Shape[i];
		_Src = _Value._Src + TaskIndex;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
	if (_Src >= End) return;
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();

	if (ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_S_0(
			_DestInfo,
			0,
			F, Begin[0], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis0FA] = *(_Src++);
			}
		);
	else if (ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_S_1(
			_DestInfo,
			0,
			F, Begin[0], Begin[1], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis1FA] = *(_Src++);
			}
		);
	else if (ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_S_2(
			_DestInfo,
			0,
			F, Begin[0], Begin[1], Begin[2], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis2FA] = *(_Src++);
			}
		);
	else if (ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_S_3(
			_DestInfo,
			0,
			F, Begin[0], Begin[1], Begin[2], Begin[3], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis3FA] = *(_Src++);
			}
		);
	else if (ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_S_4(
			_DestInfo,
			0,
			F, Begin[0], Begin[1], Begin[2], Begin[3], Begin[4], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis4FA] = *(_Src++);
			}
		);
	else if (ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_S_5(
			_DestInfo,
			0,
			F, Begin[0], Begin[1], Begin[2], Begin[3], Begin[4], Begin[5], 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis5FA] = *(_Src++);
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignBuffer(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	const _Type* _Src,
	SizeType _Count,
	bool Continuous
)
{
	_Struct_Buffer<_Type> _Value = { _Dest, _Src, _Count };
	ImplMultiThreadSingle<_Type, _Struct_Buffer<_Type>>(
		_Dest,
		_DestInfo,
		_Value,
		Continuous,
		AssignBuffer<_Type>,
		AssignBufferCont<_Type>
	);
}

template<typename _Type>
void AssignScalarCont(
	_Type* _Dest,
	SizeType DestSize,
	_Type _Value
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;

	auto _VectorizedValue1 = Vectorized<_Type>(_Value);
	auto _VectorizedValue2 = Vectorized<_Type>(_Value);

	SizeType i = 0;
	for (; i < DestSize - LoopStride; i += LoopStride)
	{
		_VectorizedValue1.Store(_Dest + i);
		_VectorizedValue2.Store(_Dest + i + Stride);
	}

	for (; i < DestSize; ++i)
		_Dest[i] = _Value;
}

template<typename _Type>
void AssignScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	_Type _Value
)
{
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();

	const auto Func = [&](int64_t _Index) {_Dest[_Index] = _Value; };
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	if (ViewRank == 1)
		SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 2)
		SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 3)
		SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 4)
		SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 5)
		SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 6)
		SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 7)
		SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 8)
		SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 9)
		SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 10)
		SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignScalar(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	_Type _Value,
	bool Continuous
)
{
	ImplMultiThreadSingle<_Type, _Type>(
		_Dest,
		_DestInfo,
		_Value,
		Continuous,
		AssignScalar<_Type>,
		AssignScalarCont<_Type>
	);
}

template<typename RandomType>
struct _Impl_Normal_Distribution
{
	_Impl_Normal_Distribution(double _Mean, double _Sigma, void* const _Begin)
		: Normal(std::make_shared<std::normal_distribution<RandomType>>(RandomType(_Mean), RandomType(_Sigma))),
		RandomEngine(std::make_shared<std::mt19937_64>(_Impl_Global_Seed)), Begin(_Begin) { }
	_D_Dragonian_Lib_Constexpr_Force_Inline RandomType operator()()
	{
		return Normal->operator()(*RandomEngine);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void Seed(const void* const Ptr) const
	{
		RandomEngine->seed((const char* const)Ptr - (const char* const)Begin + _Impl_Global_Seed);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void Seed(SizeType Seed) const
	{
		RandomEngine->seed(Seed + _Impl_Global_Seed);
	}
	std::shared_ptr<std::normal_distribution<RandomType>> Normal;
	std::shared_ptr<std::mt19937_64> RandomEngine;
	void* Begin = nullptr;
};

template<typename _Type, typename _GetRandomDeviceFn>
void AssignRandnCont(
	_Type* _Dest,
	SizeType DestSize,
	_GetRandomDeviceFn _EngineFn
)
{
	auto NormalEngine = _EngineFn();
	NormalEngine.Seed(_Dest);
	SizeType i = 0;
	for (; i < DestSize - 8; i += 8)
	{
		_Dest[i] = (_Type)NormalEngine();
		_Dest[i + 1] = (_Type)NormalEngine();
		_Dest[i + 2] = (_Type)NormalEngine();
		_Dest[i + 3] = (_Type)NormalEngine();
		_Dest[i + 4] = (_Type)NormalEngine();
		_Dest[i + 5] = (_Type)NormalEngine();
		_Dest[i + 6] = (_Type)NormalEngine();
		_Dest[i + 7] = (_Type)NormalEngine();
	}
	for (; i < DestSize; ++i)
		_Dest[i] = (_Type)NormalEngine();
}

template<typename _Type, typename _GetRandomDeviceFn>
void AssignRandn(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	_GetRandomDeviceFn _EngineFn
)
{
	//using RandomType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<sizeof(_Type) == sizeof(double), double, float>;
	auto NormalEngine = _EngineFn();
	
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	SizeType _Seed = 0;
	for (auto i : _DestInfo.Begin) _Seed += i;
	NormalEngine.Seed(_Seed);

	const auto Func = [&](int64_t _Index) { _Dest[_Index] = (_Type)NormalEngine(); };
	if (ViewRank == 1)
		SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 2)
		SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 3)
		SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 4)
		SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 5)
		SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 6)
		SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 7)
		SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 8)
		SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 9)
		SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 10)
		SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRandn(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	double _Mean,
	double _Sigma,
	bool Continuous
)
{
	using RandomType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<sizeof(_Type) == sizeof(double), double, float>;
	auto _ThreadData = std::make_shared<std::atomic_int64_t>(0);
	auto _GetEngineFn = [=] { return _Impl_Normal_Distribution<RandomType>(_Mean, _Sigma, _Dest); };

	ImplMultiThreadSingle<_Type, decltype(_GetEngineFn)>(
		_Dest,
		_DestInfo,
		_GetEngineFn,
		Continuous,
		AssignRandn<_Type, decltype(_GetEngineFn)>,
		AssignRandnCont<_Type, decltype(_GetEngineFn)>
	);
}

template<typename _Type>
_D_Dragonian_Lib_Force_Inline _Type _Impl_Generate_Random_Value(std::mt19937_64& RandomEngine)
{
	if constexpr (std::is_same_v<_Type, double>)
		return RandomDoubleDistribution(RandomEngine);
	else if constexpr (std::is_same_v<_Type, float>)
		return RandomFloatDistribution(RandomEngine);
	else if constexpr (std::is_same_v<_Type, int64_t>)
		return RandomInt64Distribution(RandomEngine);
	else if constexpr (std::is_same_v<_Type, int32_t>)
		return RandomInt32Distribution(RandomEngine);
	else if constexpr (std::is_same_v<_Type, int16_t>)
		return RandomInt16Distribution(RandomEngine);
	else if constexpr (std::is_same_v<_Type, int8_t>)
		return (uint8_t)RandomInt16Distribution(RandomEngine);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type, typename _SeedFnType>
void AssignRandomCont(
	_Type* _Dest,
	SizeType DestSize,
	_SeedFnType _SeedFn
)
{
	std::mt19937_64 RandomEngine(_SeedFn());
	SizeType i = 0;
	for (; i < DestSize - 8; i += 8)
	{
		_Dest[i] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 1] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 2] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 3] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 4] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 5] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 6] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
		_Dest[i + 7] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
	}
	for (;i < DestSize; ++i)
		_Dest[i] = _Impl_Generate_Random_Value<_Type>(RandomEngine);
}

template<typename _Type, typename _SeedFnType>
void AssignRandom(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter> _DestInfoOld,
	_SeedFnType _SeedFn
)
{
	std::mt19937_64 RandomEngine(_SeedFn());
	const OperatorParameter& _DestInfo = *_DestInfoOld;
	SizeType ViewRank = _DestInfo.GetRank();

	const auto Func = [&](int64_t _Index) { _Dest[_Index] = _Impl_Generate_Random_Value<_Type>(RandomEngine); };
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStep = _DestInfo.ViewStep.Data();
	const SizeType* __restrict ViewLeft = _DestInfo.ViewLeft.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	if (ViewRank == 1)
		SingleTensorLoop<1, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 2)
		SingleTensorLoop<2, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 3)
		SingleTensorLoop<3, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 4)
		SingleTensorLoop<4, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 5)
		SingleTensorLoop<5, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 6)
		SingleTensorLoop<6, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 7)
		SingleTensorLoop<7, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 8)
		SingleTensorLoop<8, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 9)
		SingleTensorLoop<9, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else if (ViewRank == 10)
		SingleTensorLoop<10, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
			0, Shape, Begin, ViewStep, ViewLeft, ViewStride, Func
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRand(
	_Type* _Dest,
	const OperatorParameter& _DestInfo,
	bool Continuous
)
{
	auto _ThreadData = std::make_shared<std::atomic_int64_t>(0);
	auto _SeedFn = [=] { return _Impl_Global_Seed + _ThreadData->fetch_add(1); };
	ImplMultiThreadSingle<_Type, decltype(_SeedFn)>(
		_Dest,
		_DestInfo,
		_SeedFn,
		Continuous,
		AssignRandom<_Type, decltype(_SeedFn)>,
		AssignRandomCont<_Type, decltype(_SeedFn)>
	);
}

_D_Dragonian_Lib_Operator_Space_End