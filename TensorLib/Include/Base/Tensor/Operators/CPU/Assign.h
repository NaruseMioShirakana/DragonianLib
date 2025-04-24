/**
 * @file Assign.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Assign operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

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
	const std::shared_ptr<int>&
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
	std::shared_ptr<OperatorParameter<_NRank>> _IDestInfoOld,
	const _TypeSrc* _Src,
	std::shared_ptr<OperatorParameter<_NRank>> _ISrcInfoOld,
	const std::shared_ptr<int>&
)
{
	if constexpr (
		(TypeTraits::CouldBeConvertedFromValue<_TypeDest, _TypeSrc> && std::is_move_assignable_v<_TypeDest>) ||
		(TypeTraits::IsSameTypeValue<_TypeDest, _TypeSrc> && std::is_copy_assignable_v<_TypeDest>)
		)
	{
		auto _DestInfoOld = std::move(_IDestInfoOld);
		auto _SrcInfoOld = std::move(_ISrcInfoOld);

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
		ImplMultiThreadCaller<2, _NRank, 0, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			_Src,
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
			nullptr,
			nullptr,
			std::make_shared<int>(0),
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
		ImplMultiThreadCaller<2, _NRank, 0, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			_Src,
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
			nullptr,
			nullptr,
			std::make_shared<int>(0),
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
	const std::shared_ptr<_Struct_Buffer<_Type>> _Value
)
{
	const auto Index = _Dest - _Value->_DestBegin;
	if (Index >= _Value->_Count) return;
	DestSize = std::min(DestSize, _Value->_Count - Index);
	const auto _SrcPtr = _Value->_Src + Index;
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
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const std::shared_ptr<_Struct_Buffer<_Type>> _Value
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;

		const auto End = _Value->_Src + _Value->_Count;
		const _Type* _Src;
		if (std::ranges::count(_DestInfo.Begin, 0) == _NRank)
			_Src = _Value->_Src;
		else if (std::ranges::count(_DestInfo.Begin, 0) == _NRank - 1)
		{
			SizeType TaskIndex = 1;
			for (SizeType i = 0; i < _NRank; ++i)
				if (_DestInfo.Begin[i] != 0)
					TaskIndex *= _DestInfo.Begin[i];
				else
					TaskIndex *= _DestInfo.Shape[i];
			_Src = _Value->_Src + TaskIndex;
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
	const std::shared_ptr<_Struct_Buffer<_Type>> _Value
)
{
	const auto Index = _Dest - _Value->_DestBegin;
	if (Index >= _Value->_Count) return;
	DestSize = std::min(DestSize, _Value->_Count - Index);
	const auto _SrcPtr = _Value->_Src + Index;
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
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const std::shared_ptr<_Struct_Buffer<_Type>> _Value
)
{
	if constexpr (std::is_move_assignable_v<_Type>)
	{
		const OperatorParameter<_NRank>& _DestInfo = *_DestInfoOld;

		const auto End = _Value->_Src + _Value->_Count;
		const _Type* _Src;
		if (std::ranges::count(_DestInfo.Begin, 0) == _NRank)
			_Src = _Value->_Src;
		else if (std::ranges::count(_DestInfo.Begin, 0) == _NRank - 1)
		{
			SizeType TaskIndex = 1;
			for (SizeType i = 0; i < _NRank; ++i)
				if (_DestInfo.Begin[i] != 0)
					TaskIndex *= _DestInfo.Begin[i];
				else
					TaskIndex *= _DestInfo.Shape[i];
			_Src = _Value->_Src + TaskIndex;
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
		ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			std::make_shared<_Struct_Buffer<_Type>>(_Dest, _Src, _Count),
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
		ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			std::make_shared<_Struct_Buffer<_Type>>(_Dest, _Src, _Count),
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
	std::shared_ptr<_Type> _IValPtr
)
{
	const auto _ValPtr = std::move(_IValPtr);
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
	std::shared_ptr<OperatorParameter<_NRank>> _IDestInfoOld,
	std::shared_ptr<_Type> _IValPtr
)
{
	if constexpr (std::is_copy_assignable_v<_Type>)
	{
		auto _DestInfoOld = std::move(_IDestInfoOld);
		auto _ValPtr = std::move(_IValPtr);

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
		ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
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
	const std::shared_ptr<RandomSettings<_Type>> Settings
)
{
	auto RandomDevice = GetThreadPool().GetRandomEngine(Settings->_ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings->_Mean, Settings->_Sigma);

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
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const std::shared_ptr<RandomSettings<_Type>> Settings
)
{
	auto RandomDevice = GetThreadPool().GetRandomEngine(Settings->_ThreadId);
	_Impl_Dragonian_Lib_Normal_Distribution_Type<_Type> NormalDistribution(Settings->_Mean, Settings->_Sigma);

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
		ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			std::make_shared<RandomSettings<_Type>>((RandomType)0, (RandomType)0, (RandomNormalType)_Mean, (RandomNormalType)_Sigma),
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
	const std::shared_ptr<RandomSettings<_Type>> Settings
)
{
	auto RandomDevice = GetThreadPool().GetRandomEngine(Settings->_ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings->_Min, Settings->_Max);

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
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoOld,
	const std::shared_ptr<RandomSettings<_Type>> Settings
)
{
	auto RandomDevice = GetThreadPool().GetRandomEngine(Settings->_ThreadId);
	using RandomDistributionType = _Impl_Dragonian_Lib_Random_Distribution_Type<_Type>;
	RandomDistributionType Distribution(Settings->_Min, Settings->_Max);

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
			ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
				_Dest,
				std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
				nullptr,
				nullptr,
				nullptr,
				nullptr,
				std::make_shared<RandomSettings<_Type>>((RandomType)_Min.real(), (RandomType)_Max.real(), (RandomNormalType)0., (RandomNormalType)0.),
				Continuous,
				AssignRandom<_Type, _NRank>,
				AssignRandomCont<_Type>
			);
		else
			ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
				_Dest,
				std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
				nullptr,
				nullptr,
				nullptr,
				nullptr,
				std::make_shared<RandomSettings<_Type>>((RandomType)_Min, (RandomType)_Max, (RandomNormalType)0., (RandomNormalType)0.),
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
	std::shared_ptr<ArangeParams<_Type>> _IValue
)
{
	auto _Value = std::move(_IValue);

	const auto Index = _Dest - _Value->_DestBegin;
	_Dest[0] = _Value->_Start + _Type(Index) * _Value->_Step;

	constexpr int64_t OpThroughput = 2;
	constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
	constexpr int64_t LoopStride = OpThroughput * Stride;

	int64_t i = 0;
	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		if (DestSize >= LoopStride)
		{
			for (int64_t j = 1; j < LoopStride; ++j)
				_Dest[j] = _Dest[j - 1] + _Value->_Step;

			Vectorized<_Type> _VectorizedValue1(_Dest);
			Vectorized<_Type> _VectorizedValue2(_Dest + Stride);
			Vectorized<_Type> _VectorizedStep1(_Value->_Step * static_cast<_Type>(LoopStride));
			Vectorized<_Type> _VectorizedStep2(_Value->_Step * static_cast<_Type>(LoopStride));
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
				_Dest[j] = _Dest[j - 1] + _Value->_Step;
			for (; i < DestSize - LoopStride; i += LoopStride)
				for (int64_t j = 0; j < LoopStride; ++j)
					_Dest[i + j] = _Dest[i + j - 1] + _Value->_Step;
		}
		else
			++i;
	}
	for (; i < DestSize; ++i)
		_Dest[i] = _Dest[i - 1] + _Value->_Step;
}

template<typename _Type, size_t _NRank>
void ArangeImp(
	_Type*,
	const std::shared_ptr<OperatorParameter<_NRank>>&,
	const std::shared_ptr<ArangeParams<_Type>>&
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
		ImplMultiThreadCaller<1, _NRank, 0, _Type, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			nullptr,
			nullptr,
			nullptr,
			nullptr,
			std::make_shared<ArangeParams<_Type>>(_Start, _Step, _Dest),
			Continuous,
			ArangeImp<_Type, _NRank>,
			ArangeImpCont<_Type>
		);
	}
}

template<int64_t LoopCount, int64_t LoopUnfold, int64_t _Dim, typename _Fn>
_D_Dragonian_Lib_Constexpr_Force_Inline void GatherLoop(
	int64_t Value1, int64_t Value2, int64_t Value3,
	const int64_t* __restrict Shape, const int64_t* __restrict LoopBegin,
	const int64_t* __restrict Stride1, const int64_t* __restrict Stride2, const int64_t* __restrict Stride3,
	_Fn _Func
) requires (IsCallableValue<_Fn>)
{
	if constexpr (LoopCount - 1)
		for (int64_t i = *LoopBegin; i < *Shape; ++i)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = _Dim == 0 ? Value2 : Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			GatherLoop<LoopCount - 1, LoopUnfold, _Dim - 1>(
				Val1, Val2, Val3,
				Shape + 1, LoopBegin + 1,
				Stride1 + 1, Stride2 + 1, Stride3 + 1,
				_Func
			);
		}
	else
	{
		int64_t i = *LoopBegin;
		while (i < *Shape - LoopUnfold)
		{
			for (int64_t j = 0; j < LoopUnfold; ++j)
			{
				const auto Val1 = Value1 + i * *Stride1;
				const auto Val2 = _Dim == 0 ? Value2 : Value2 + i * *Stride2;
				const auto Val3 = Value3 + i * *Stride3;
				_Func(Val1, Val2, Val3);
				++i;
			}
		}
		while (i < *Shape)
		{
			const auto Val1 = Value1 + i * *Stride1;
			const auto Val2 = _Dim == 0 ? Value2 : Value2 + i * *Stride2;
			const auto Val3 = Value3 + i * *Stride3;
			_Func(Val1, Val2, Val3);
			++i;
		}
	}
}

template<typename _Type, typename _IndexType, size_t _NRank, size_t _Dim>
void GatherOp(
	_Type* _Dest,
	const std::shared_ptr<OperatorParameter<_NRank>> _DestInfo,
	const _Type* _Src,
	const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfo,
	const _IndexType* _Index,
	const std::shared_ptr<OperatorParameter<_NRank>> _IndexInfo,
	const std::shared_ptr<int>&
)
{
	const auto Func = [&](int64_t _IndexDst, int64_t _IndexSrc, int64_t _IndexIdx)
		{
			const auto Index = CalcIndexOp((SizeType)_Index[_IndexIdx], _SrcInfo->Shape[_Dim]) *
				_SrcInfo->ViewStride[_Dim];
			_Dest[_IndexDst] = _Src[_IndexSrc + Index];
		};
	GatherLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold, _Dim>(
		0, 0, 0,
		_DestInfo->Shape.Data(), _DestInfo->Begin.Data(),
		_DestInfo->ViewStride.Data(), _SrcInfo->ViewStride.Data(), _IndexInfo->ViewStride.Data(),
		Func
	);
}

template<typename _Type>
template<typename _IndexType, size_t _NRank, size_t _Dim>
void OperatorsBase<_Type, Device::CPU>::ImplGather(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const _IndexType* _Index,
	const OperatorParameter<_NRank>& _IndexInfo
)
{
	ImplMultiThreadCaller<3, _NRank, 0>(
		_Dest,
		std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
		_Src,
		std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
		_Index,
		std::make_shared<OperatorParameter<_NRank>>(_IndexInfo),
		std::make_shared<int>(0),
		false,
		GatherOp<_Type, _IndexType, _NRank, _Dim>,
		nullptr
	);
}

template<typename _Type>
template<typename _MaskType, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplMaskedAssign(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const _MaskType* _Mask,
	const OperatorParameter<_NRank>& _MaskInfo,
	bool Continuous
)
{
	using OPP = std::shared_ptr<OperatorParameter<_NRank>>;

	const auto Func = [=](int64_t _IndexA, int64_t _IndexB, int64_t _IndexC)
		{
			if (_Mask[_IndexC])
				_Dest[_IndexA] = _Src[_IndexB];
		};

	auto Fn = [Func](_Type*, const OPP _MyDestInfoPointer, const _Type*, const OPP _MySrcInfoPointer, const _MaskType*, const OPP _MyMaskInfoPointer, const std::shared_ptr<int>&)
		{

			const auto& _MyDestInfo = *_MyDestInfoPointer;
			const auto& _MySrcInfo = *_MySrcInfoPointer;
			const auto& _MyMaskInfo = *_MyMaskInfoPointer;

			const SizeType* __restrict Shape = _MyDestInfo.Shape.Data();
			const SizeType* __restrict Begin = _MyDestInfo.Begin.Data();
			const SizeType* __restrict ViewStride = _MyDestInfo.ViewStride.Data();
			const SizeType* __restrict SrcViewStride = _MySrcInfo.ViewStride.Data();
			const SizeType* __restrict MaskViewStride = _MyMaskInfo.ViewStride.Data();

			TripleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0, 0,
				Shape, Begin,
				ViewStride, SrcViewStride, MaskViewStride,
				Func
			);
		};

	auto ContFn = [](_Type* _MyDest, const _Type* _MySrc, const _MaskType* _MyMask, SizeType _Count, const std::shared_ptr<int>&)
		{
			for (SizeType i = 0; i < _Count; ++i)
				if (_MyMask[i])
					_MyDest[i] = _MySrc[i];
		};

	ImplMultiThreadCaller<3, _NRank, 0>(
		_Dest,
		std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
		_Src,
		std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
		_Mask,
		std::make_shared<OperatorParameter<_NRank>>(_MaskInfo),
		std::make_shared<int>(0),
		Continuous,
		Fn,
		ContFn
	);
}

template<typename _Type>
template<typename _MaskType, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplMaskedAssignScalar(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _MaskType* _Mask,
	const OperatorParameter<_NRank>& _MaskInfo,
	const _Type& _Value,
	bool Continuous
)
{
	using OPP = std::shared_ptr<OperatorParameter<_NRank>>;

	const auto Func = [=](int64_t _IndexA, int64_t _IndexC)
		{
			if (_Mask[_IndexC])
				_Dest[_IndexA] = _Value;
		};

	auto Fn = [Func](_Type*, const OPP _MyDestInfoPointer, const _MaskType*, const OPP _MyMaskInfoPointer, const std::shared_ptr<int>&)
		{
			const auto& _MyDestInfo = *_MyDestInfoPointer;
			const auto& _MyMaskInfo = *_MyMaskInfoPointer;

			const SizeType* __restrict Shape = _MyDestInfo.Shape.Data();
			const SizeType* __restrict Begin = _MyDestInfo.Begin.Data();
			const SizeType* __restrict ViewStride = _MyDestInfo.ViewStride.Data();
			const SizeType* __restrict MaskViewStride = _MyMaskInfo.ViewStride.Data();

			DoubleTensorLoop<_NRank, _D_Dragonian_Lib_Operator_Assign_Tensor_Unfold>(
				0, 0,
				Shape, Begin,
				ViewStride, MaskViewStride,
				Func
			);
		};

	auto ContFn = [=](_Type* _MyDest, const _MaskType* _MyMask, SizeType _Count, const std::shared_ptr<int>&)
		{
			for (SizeType i = 0; i < _Count; ++i)
				if (_MyMask[i])
					_MyDest[i] = _Value;
		};

	ImplMultiThreadCaller<2, _NRank, 0, _Type>(
		_Dest,
		std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
		_Mask,
		std::make_shared<OperatorParameter<_NRank>>(_MaskInfo),
		nullptr,
		nullptr,
		std::make_shared<int>(0),
		Continuous,
		Fn,
		ContFn
	);
}

_D_Dragonian_Lib_Operator_Space_End