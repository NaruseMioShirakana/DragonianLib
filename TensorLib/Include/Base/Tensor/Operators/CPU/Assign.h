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
	if constexpr (TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc> && std::is_trivially_copy_assignable_v<_TypeSrc>)
	{
		DestSize *= sizeof(_TypeDest);
		Vectorized<_TypeDest>::DragonianLibMemCpy(_Dest, _Src, DestSize);
	}
	else if constexpr (
		(TypeTraits::CouldBeConvertedFromValue<_TypeDest, _TypeSrc> && std::is_move_assignable_v<_TypeDest>) ||
		(TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc> && std::is_copy_assignable_v<_TypeDest>)
		)
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				if constexpr (TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc>)
					_Dest[i] = _Src[i];
				else
					_Dest[i] = _TypeDest(_Src[i]);
				++i;
			}
		}
		while (i < DestSize)
		{
			if constexpr (TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc>)
				_Dest[i] = _Src[i];
			else
				_Dest[i] = _TypeDest(_Src[i]);
			++i;
		}
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _TypeDest, typename _TypeSrc, size_t _NRank>
void AssignTensor(
	_TypeDest* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const _TypeSrc* _Src,
	std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoOld,
	void*
)
{
	if constexpr (
		(TypeTraits::CouldBeConvertedFromValue<_TypeDest, _TypeSrc> && std::is_move_assignable_v<_TypeDest>) ||
		(TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc> && std::is_copy_assignable_v<_TypeDest>)
		)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		const OperatorParameter<_NRank>& _SrcInfo = *_SrcInfoOld;

		const auto Func = [&](int64_t _IndexA, int64_t _IndexB)
			{
				if constexpr (TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc>)
					_Dest[_IndexA] = _Src[_IndexB];
				else
					_Dest[_IndexA] = _TypeDest(_Src[_IndexB]);
			};
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();
		const SizeType* __restrict SrcViewStride = _SrcInfo.ViewStride.Data();

		DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 0,
			Shape, Begin,
			ViewStride, SrcViewStride,
			Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template <typename _Type>
template <typename _TypeSrc, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCast(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _TypeSrc* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	if constexpr (TypeTraits::CouldBeConvertedFromValue<_Type, _TypeSrc> && std::is_move_assignable_v<_Type>)
	{
		ImplMultiThreadDouble(
			_Dest,
			_DestInfo,
			_Src,
			_SrcInfo,
			nullptr,
			Continuous,
			AssignTensor<_Type, _TypeSrc, _NRank>,
			AssignTensorCont<_Type, _TypeSrc>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplAssignTensor(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		ImplMultiThreadDouble(
			_Dest,
			_DestInfo,
			_Src,
			_SrcInfo,
			nullptr,
			Continuous,
			AssignTensor<_Type, _Type, _NRank>,
			AssignTensorCont<_Type, _Type>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
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
	DestSize = std::min(DestSize, _Value._Count - Index);
	const auto _SrcPtr = _Value._Src + Index;
	if constexpr (std::is_trivially_copy_assignable_v<_Type>)
		Vectorized<_Type>::DragonianLibMemCpy(_Dest, _SrcPtr, DestSize * sizeof(_Type));
	else if constexpr (std::is_copy_assignable_v<_Type>)
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				_Dest[i] = _SrcPtr[i];
				++i;
			}
		}
		while (i < DestSize)
		{
			_Dest[i] = _SrcPtr[i];
			++i;
		}
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type, size_t _NRank>
void AssignBuffer(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	_Struct_Buffer<_Type> _Value
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;

		const auto End = _Value._Src + _Value._Count;
		const _Type* _Src;
		if (std::ranges::count(_DestInfo.Begin, 0) == _NRank)
			_Src = _Value._Src;
		else if (std::ranges::count(_DestInfo.Begin, 0) == _NRank - 1)
		{
			SizeType TaskIndex = 1;
			for (SizeType i = 0; i < _NRank; ++i)
				if (_DestInfo.Begin[i] != 0)
					TaskIndex *= _DestInfo.Begin[i];
				else
					TaskIndex *= _DestInfo.Shape[i];
			_Src = _Value._Src + TaskIndex;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
		if (_Src >= End) return;

		const auto Func = [&](int64_t _Index)
			{
				if (_Src >= End) return;
				_Dest[_Index] = *(_Src++);
			};

		SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0, 
			_DestInfo.Shape.Data(), _DestInfo.Begin.Data(),
			_DestInfo.ViewStride.Data(),
			Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void MoveBufferCont(
	_Type* _Dest,
	SizeType DestSize,
	_Struct_Buffer<_Type> _Value
)
{
	const auto Index = _Dest - _Value._DestBegin;
	if (Index >= _Value._Count) return;
	DestSize = std::min(DestSize, _Value._Count - Index);
	const auto _SrcPtr = _Value._Src + Index;
	if constexpr (std::is_trivially_copy_assignable_v<_Type>)
		Vectorized<_Type>::DragonianLibMemCpy(_Dest, _SrcPtr, DestSize * sizeof(_Type));
	else if constexpr (std::is_move_assignable_v<_Type>)
	{
		int64_t i = 0;
		while (i < DestSize - _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold)
		{
			for (int64_t j = 0; j < _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold; ++j)
			{
				_Dest[i] = std::move(_SrcPtr[i]);
				++i;
			}
		}
		while (i < DestSize)
		{
			_Dest[i] = std::move(_SrcPtr[i]);
			++i;
		}
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type, size_t _NRank>
void MoveBuffer(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	_Struct_Buffer<_Type> _Value
)
{
	if constexpr (std::is_move_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;

		const auto End = _Value._Src + _Value._Count;
		const _Type* _Src;
		if (std::ranges::count(_DestInfo.Begin, 0) == _NRank)
			_Src = _Value._Src;
		else if (std::ranges::count(_DestInfo.Begin, 0) == _NRank - 1)
		{
			SizeType TaskIndex = 1;
			for (SizeType i = 0; i < _NRank; ++i)
				if (_DestInfo.Begin[i] != 0)
					TaskIndex *= _DestInfo.Begin[i];
				else
					TaskIndex *= _DestInfo.Shape[i];
			_Src = _Value._Src + TaskIndex;
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
		if (_Src >= End) return;

		const auto Func = [&](int64_t _Index)
			{
				if (_Src >= End) return;
				_Dest[_Index] = std::move(*(_Src++));
			};

		SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
			0,
			_DestInfo.Shape.Data(), _DestInfo.Begin.Data(),
			_DestInfo.ViewStride.Data(),
			Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplMoveBuffer(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	SizeType _Count,
	bool Continuous
)
{
	if constexpr (std::is_move_assignable_v<_Type>)
	{
		_Struct_Buffer<_Type> _Value = { _Dest, _Src, _Count };
		ImplMultiThreadSingle<_Type, _Struct_Buffer<_Type>>(
			_Dest,
			_DestInfo,
			_Value,
			Continuous,
			MoveBuffer<_Type, _NRank>,
			MoveBufferCont<_Type>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplAssignBuffer(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	SizeType _Count,
	bool Continuous
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		_Struct_Buffer<_Type> _Value = { _Dest, _Src, _Count };
		ImplMultiThreadSingle<_Type, _Struct_Buffer<_Type>>(
			_Dest,
			_DestInfo,
			_Value,
			Continuous,
			AssignBuffer<_Type, _NRank>,
			AssignBufferCont<_Type>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignScalarCont(
	_Type* _Dest,
	SizeType DestSize,
	std::shared_ptr<_Type> _ValPtr
)
{
	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;
	const auto& _Value = *_ValPtr;

	SizeType i = 0;

	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
		{
			auto _VectorizedValue1 = Vectorized<_Type>(_Value);
			auto _VectorizedValue2 = Vectorized<_Type>(_Value);
			for (; i < DestSize - LoopStride; i += LoopStride)
			{
				_VectorizedValue1.Store(_Dest + i);
				_VectorizedValue2.Store(_Dest + i + Stride);
			}
		}
		else
			for (; i < DestSize - OpThroughput; i += OpThroughput)
				for (int64_t j = 0; j < OpThroughput; ++j)
					_Dest[i + j] = _Value;

		for (; i < DestSize; ++i)
			_Dest[i] = _Value;
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type, size_t _NRank>
void AssignScalar(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	std::shared_ptr<_Type> _ValPtr
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
		const auto& _Value = *_ValPtr;

		const auto Func = [&](int64_t _Index) { _Dest[_Index] = _Value; };
		const SizeType* __restrict Shape = _DestInfo.Shape.Data();
		const SizeType* __restrict Begin = _DestInfo.Begin.Data();
		const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

		SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Scalar_Unfold>(
			0, Shape, Begin, ViewStride, Func
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplAssignScalar(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type& _Value,
	bool Continuous
)
{
	if constexpr (std::is_constructible_v<_Type>)
	{
		ImplMultiThreadSingle(
			_Dest,
			_DestInfo,
			std::make_shared<_Type>(_Value),
			Continuous,
			AssignScalar<_Type, _NRank>,
			AssignScalarCont<_Type>
		);
	}
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void AssignRandnCont(
	_Type* _Dest,
	SizeType DestSize,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings._Mean, Settings._Sigma);

	SizeType i = 0;
	if constexpr (TypeTraits::IsComplexValue<_Type>)
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 1] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 2] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 3] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 4] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 5] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 6] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			_Dest[i + 7] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
		}
		for (; i < DestSize; ++i)
			_Dest[i] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
	}
	else
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 1] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 2] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 3] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 4] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 5] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 6] = (_Type)NormalDistribution(RandomDevice);
			_Dest[i + 7] = (_Type)NormalDistribution(RandomDevice);
		}
		for (; i < DestSize; ++i)
			_Dest[i] = (_Type)NormalDistribution(RandomDevice);
	}
}

template<typename _Type, size_t _NRank>
void AssignRandn(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings._Mean, Settings._Sigma);

	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;
	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	const auto Func = [&](int64_t _Index)
		{
			if constexpr (TypeTraits::IsComplexValue<_Type>)
				_Dest[_Index] = _Type(NormalDistribution(RandomDevice), NormalDistribution(RandomDevice));
			else
				_Dest[_Index] = (_Type)NormalDistribution(RandomDevice);
		};

	SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Randn_Unfold>(
		0, Shape, Begin, ViewStride, Func
	);
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRandn(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	double _Mean,
	double _Sigma,
	bool Continuous
)
{
	if constexpr (!TypeTraits::IsArithmeticValue<_Type>)
		_D_Dragonian_Lib_Not_Implemented_Error;
	else
	{
		using RandomType = _Impl_Dragonian_Lib_Random_Type<_Type>;
		using RandomNormalType = _Impl_Dragonian_Lib_Random_Normal_Type<_Type>;
		ImplMultiThreadSingle<_Type>(
			_Dest,
			_DestInfo,
			RandomSettings<_Type>{ (RandomType)0, (RandomType)0, (RandomNormalType)_Mean, (RandomNormalType)_Sigma},
			Continuous,
			AssignRandn<_Type, _NRank>,
			AssignRandnCont<_Type>
		);
	}
}

template<typename _Type>
void AssignRandomCont(
	_Type* _Dest,
	SizeType DestSize,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings._Min, Settings._Max);

	SizeType i = 0;
	if constexpr (TypeTraits::IsComplexValue<_Type>)
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 1] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 2] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 3] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 4] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 5] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 6] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			_Dest[i + 7] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
		}
		for (; i < DestSize; ++i)
			_Dest[i] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
	}
	else
	{
		for (; i < DestSize - 8; i += 8)
		{
			_Dest[i] = (_Type)Distribution(RandomDevice);
			_Dest[i + 1] = (_Type)Distribution(RandomDevice);
			_Dest[i + 2] = (_Type)Distribution(RandomDevice);
			_Dest[i + 3] = (_Type)Distribution(RandomDevice);
			_Dest[i + 4] = (_Type)Distribution(RandomDevice);
			_Dest[i + 5] = (_Type)Distribution(RandomDevice);
			_Dest[i + 6] = (_Type)Distribution(RandomDevice);
			_Dest[i + 7] = (_Type)Distribution(RandomDevice);
		}
		for (; i < DestSize; ++i)
			_Dest[i] = (_Type)Distribution(RandomDevice);
	}
}

template<typename _Type, size_t _NRank>
void AssignRandom(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	RandomSettings<_Type> Settings
)
{
	auto RandomDevice = _Valdef_My_Thread_Pool.GetRandomEngine(Settings._ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings._Min, Settings._Max);

	const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;

	const SizeType* __restrict Shape = _DestInfo.Shape.Data();
	const SizeType* __restrict Begin = _DestInfo.Begin.Data();
	const SizeType* __restrict ViewStride = _DestInfo.ViewStride.Data();

	const auto Func = [&](int64_t _Index)
		{
			if constexpr (TypeTraits::IsComplexValue<_Type>)
				_Dest[_Index] = _Type(Distribution(RandomDevice), Distribution(RandomDevice));
			else
				_Dest[_Index] = (_Type)Distribution(RandomDevice);
		};

	SingleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Random_Unfold>(
		0, Shape, Begin, ViewStride, Func
	);
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRand(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type& _Min,
	const _Type& _Max,
	bool Continuous
)
{
	if constexpr (!TypeTraits::IsArithmeticValue<_Type>)
		_D_Dragonian_Lib_Not_Implemented_Error;
	else
	{
		using RandomType = _Impl_Dragonian_Lib_Random_Type<_Type>;
		using RandomNormalType = _Impl_Dragonian_Lib_Random_Normal_Type<_Type>;

		if constexpr (TypeTraits::IsComplexValue<_Type>)
			ImplMultiThreadSingle<_Type>(
				_Dest,
				_DestInfo,
				RandomSettings<_Type>{(RandomType)_Min.real(), (RandomType)_Max.real(), (RandomNormalType)0., (RandomNormalType)0.},
				Continuous,
				AssignRandom<_Type, _NRank>,
				AssignRandomCont<_Type>
			);
		else
			ImplMultiThreadSingle<_Type>(
				_Dest,
				_DestInfo,
				RandomSettings<_Type>{(RandomType)_Min, (RandomType)_Max, (RandomNormalType)0., (RandomNormalType)0.},
				Continuous,
				AssignRandom<_Type, _NRank>,
				AssignRandomCont<_Type>
			);
	}
}

template <typename _Type>
struct ArangeParams
{
	_Type _Start;
	_Type _Step;
	const _Type* _DestBegin;
};

template<typename _Type>
void ArangeImpCont(
	_Type* _Dest,
	SizeType DestSize,
	ArangeParams<_Type> _Value
)
{
	const auto Index = _Dest - _Value._DestBegin;
	_Dest[0] = _Value._Start + _Type(Index) * _Value._Step;

	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;

	int64_t i = 0;
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		if (DestSize >= LoopStride)
		{
			for (int64_t j = 1; j < LoopStride; ++j)
				_Dest[j] = _Dest[j - 1] + _Value._Step;

			Vectorized<_Type> _VectorizedValue1(_Dest);
			Vectorized<_Type> _VectorizedValue2(_Dest + Stride);
			Vectorized<_Type> _VectorizedStep1(_Value._Step * static_cast<_Type>(LoopStride));
			Vectorized<_Type> _VectorizedStep2(_Value._Step * static_cast<_Type>(LoopStride));
			for (; i < DestSize - LoopStride; i += LoopStride)
			{
				_VectorizedValue1.Store(_Dest + i);
				_VectorizedValue2.Store(_Dest + i + Stride);
				_VectorizedValue1 += _VectorizedStep1;
				_VectorizedValue2 += _VectorizedStep2;
			}
		}
		else
			++i;
	}
	else
	{
		if (DestSize >= LoopStride)
		{
			for (int64_t j = 1; j < LoopStride; ++j)
				_Dest[j] = _Dest[j - 1] + _Value._Step;
			for (; i < DestSize - LoopStride; i += LoopStride)
				for (int64_t j = 0; j < LoopStride; ++j)
					_Dest[i + j] = _Dest[i + j - 1] + _Value._Step;
		}
		else
			++i;
	}
	for (; i < DestSize; ++i)
		_Dest[i] = _Dest[i - 1] + _Value._Step;
}

template<typename _Type, size_t _NRank>
void ArangeImp(
	_Type* _Dest,
	std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	ArangeParams<_Type> Settings
)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplArange(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type& _Start,
	const _Type& _Step,
	bool Continuous
)
{
	if constexpr (!TypeTraits::IsArithmeticValue<_Type>)
		_D_Dragonian_Lib_Not_Implemented_Error;
	else
	{
		if (!Continuous)
			_D_Dragonian_Lib_Not_Implemented_Error;
		ArangeParams<_Type> _Value = { _Start, _Step, _Dest };
		ImplMultiThreadSingle<_Type, ArangeParams<_Type>>(
			_Dest,
			_DestInfo,
			_Value,
			Continuous,
			ArangeImp<_Type, _NRank>,
			ArangeImpCont<_Type>
		);
	}
}

_D_Dragonian_Lib_Operator_Space_End