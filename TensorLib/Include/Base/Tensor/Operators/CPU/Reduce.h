/**
 * @file Reduce.h
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
 * @brief Reduce operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

template <
	Int64 Throughput, typename _Type, size_t _NRank,
	typename _FunctionTypePre, typename _FunctionTypeMid, typename _FunctionTypeEnd,
	typename _FunctionTypePreVec, typename _FunctionTypeMidVec, typename = std::enable_if_t<IsCallableValue<_FunctionTypeMid>>
> _D_Dragonian_Lib_Force_Inline void ImplReduceOperators(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous,
	_Type ReduceInitValue,
	_FunctionTypePre ReducePreOperator,
	_FunctionTypePreVec ReducePreOperatorVec,
	_FunctionTypeMid ReduceMidOperator,
	_FunctionTypeMidVec ReduceMidOperatorVec,
	_FunctionTypeEnd ReducePostOperator
)
{
	constexpr size_t _ReduceDim = _NRank - 1;
	const auto _SrcShape = _SrcInfo.Shape[_ReduceDim];
	const auto _SrcStride = _SrcInfo.ViewStride[_ReduceDim];

	auto RedeceFn = [=](int64_t _IndexA, int64_t _IndexB)
		{
			const auto _SrcBegin = _Src + _IndexB;
			_Type Val = ReduceInitValue;
			for (SizeType i = 0; i < _SrcShape; ++i)
			{
				auto ValueTemp = *(_SrcBegin + i * _SrcStride);
				if constexpr (IsCallableValue<decltype(ReducePreOperator)>)
					ValueTemp = ReducePreOperator(ValueTemp);
				Val = ReduceMidOperator(Val, ValueTemp);
			}
			if constexpr (IsCallableValue<decltype(ReducePostOperator)>)
				Val = ReducePostOperator(Val);
			*(_Dest + _IndexA) = Val;
		};

	auto LoopFn = [=](_Type*, const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoNew, const _Type*, const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoNew, const std::shared_ptr<int>&)
		{
			DoubleTensorLoop<_ReduceDim, 8>(
				0, 0,
				_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
				_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
				RedeceFn
			);
		};

	if (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ContRedeceFn = [=](_Type* _DestBegin, const _Type* _SrcBegin, SizeType BatchCount, const std::shared_ptr<int>&)
		{
			const auto _DestEnd = _DestBegin + BatchCount;
			while (_DestBegin < _DestEnd)
			{
				*_DestBegin++ = ReduceFunction<Throughput>(
					_SrcBegin, _SrcShape, ReduceInitValue,
					ReducePreOperator, ReducePreOperatorVec,
					ReduceMidOperator, ReduceMidOperatorVec,
					ReducePostOperator
				);
				_SrcBegin += _SrcShape;
			}
		};

		ImplMultiThreadCaller<2, _NRank, 1, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			_Src,
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
			nullptr,
			nullptr,
			std::make_shared<int>(0),
			Continuous,
			LoopFn,
			ContRedeceFn
		);
	}
	else
	{
		auto ContRedeceFn = [=](_Type* _DestBegin, const _Type* _SrcBegin, SizeType BatchCount, const std::shared_ptr<int>&)
			{
				for (SizeType i = 0; i < BatchCount; ++i)
				{
					_Type Val = ReduceInitValue;
					for (SizeType j = 0; j < _SrcShape; ++j)
					{
						auto ValueTemp = *_SrcBegin++;
						if constexpr (IsCallableValue<decltype(ReducePreOperator)>)
							ValueTemp = ReducePreOperator(ValueTemp);
						Val = ReduceMidOperator(Val, ValueTemp);
					}
					if constexpr (IsCallableValue<decltype(ReducePostOperator)>)
						Val = ReducePostOperator(Val);
					*_DestBegin++ = Val;
				}
			};

		ImplMultiThreadCaller<2, _NRank, 1, _Type>(
			_Dest,
			std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
			_Src,
			std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
			nullptr,
			nullptr,
			std::make_shared<int>(0),
			Continuous,
			LoopFn,
			ContRedeceFn
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceSumUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	auto ReducePostOperator = 0;

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A + _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceProdUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A * _B; };
	auto ReducePostOperator = 0;

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A * _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceMaxUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = std::numeric_limits<_Type>::lowest();
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A > _B ? _A : _B; };
	auto ReducePostOperator = 0;

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A.Max(_B); };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceMinUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = std::numeric_limits<_Type>::max();
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A < _B ? _A : _B; };
	auto ReducePostOperator = 0;

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A.Min(_B); };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceMeanUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	const auto MeanSize = static_cast<_Type>(_SrcInfo.Shape[_NRank - 1]);

	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	auto ReducePostOperator = [=](const _Type& _A) { return _A / MeanSize; };

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A + _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceLpScalar(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const _Type& _P,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = [=](const _Type& _A) { return pow(abs(_A), _P); };
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	auto ReducePostOperator = [=](const _Type& _A) { return pow(_A, _Type(1) / _P); };

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		const Vectorized<_Type> PVec(_P);
		auto ReducePreOperatorVec = [=](const Vectorized<_Type>& _A) { return _A.Abs().Pow(PVec); };
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A + _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceLogSumUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = 0;
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	auto ReducePostOperator = [=](const _Type& _A) { return log(_A); };

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = 0;
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A + _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceLogSumExpUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr _Type ReduceInitValue = 0;
	auto ReducePreOperator = [](const _Type& _A) { return exp(_A); };
	auto ReduceMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	auto ReducePostOperator = [](const _Type& _A) { return log(_A); };

	if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
	{
		auto ReducePreOperatorVec = [](const Vectorized<_Type>& _A) { return _A.Exp(); };
		auto ReduceMidOperatorVec = [](const Vectorized<_Type>& _A, const Vectorized<_Type>& _B) { return _A + _B; };
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, ReducePreOperatorVec, ReduceMidOperator, ReduceMidOperatorVec, ReducePostOperator
		);
	}
	else
	{
		ImplReduceOperators<2>(
			_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
			ReduceInitValue, ReducePreOperator, 0, ReduceMidOperator, 0, ReducePostOperator
		);
	}
}

template <typename _Type>
template <typename _ResultType, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceArgMaxUnary(
	_ResultType* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr size_t _ReduceDim = _NRank - 1;
	const auto _SrcShape = _SrcInfo.Shape[_ReduceDim];
	const auto _SrcStride = _SrcInfo.ViewStride[_ReduceDim];

	auto RedeceFn = [=](int64_t _IndexA, int64_t _IndexB)
		{
			const auto _SrcBegin = _Src + _IndexB;
			_Type Val = *(_SrcBegin);
			SizeType Arg = 0;
			for (SizeType i = 0; i < _SrcShape; ++i)
			{
				auto ValueTemp = *(_SrcBegin + i * _SrcStride);
				if (ValueTemp > Val)
				{
					Val = ValueTemp;
					Arg = i;
				}
			}
			*(_Dest + _IndexA) = _ResultType(Arg);
		};

	auto LoopFn = [=](_Type*, const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoNew, const _Type*, const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoNew, const std::shared_ptr<int>&)
		{
			DoubleTensorLoop<_ReduceDim, 8>(
				0, 0,
				_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
				_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
				RedeceFn
			);
		};

	ImplMultiThreadCaller<2, _NRank, 1, _Type>(
		_Dest,
		std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
		_Src,
		std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
		nullptr,
		nullptr,
		std::make_shared<int>(0),
		Continuous,
		LoopFn,
		0
	);
}

template <typename _Type>
template <typename _ResultType, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplReduceArgMinUnary(
	_ResultType* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr size_t _ReduceDim = _NRank - 1;
	const auto _SrcShape = _SrcInfo.Shape[_ReduceDim];
	const auto _SrcStride = _SrcInfo.ViewStride[_ReduceDim];
	auto RedeceFn = [=](int64_t _IndexA, int64_t _IndexB)
		{
			const auto _SrcBegin = _Src + _IndexB;
			_Type Val = *(_SrcBegin);
			SizeType Arg = 0;
			for (SizeType i = 0; i < _SrcShape; ++i)
			{
				auto ValueTemp = *(_SrcBegin + i * _SrcStride);
				if (ValueTemp < Val)
				{
					Val = ValueTemp;
					Arg = i;
				}
			}
			*(_Dest + _IndexA) = _ResultType(Arg);
		};
	auto LoopFn = [=](_Type*, const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoNew, const _Type*, const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoNew, const std::shared_ptr<int>&)
		{
			DoubleTensorLoop<_ReduceDim, 8>(
				0, 0,
				_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
				_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
				RedeceFn
			);
		};
	ImplMultiThreadCaller<2, _NRank, 1, _Type>(
		_Dest,
		std::make_shared<OperatorParameter<_NRank>>(_DestInfo),
		_Src,
		std::make_shared<OperatorParameter<_NRank>>(_SrcInfo),
		nullptr,
		nullptr,
		std::make_shared<int>(0),
		Continuous,
		LoopFn,
		0
	);
}

_D_Dragonian_Lib_Operator_Space_End