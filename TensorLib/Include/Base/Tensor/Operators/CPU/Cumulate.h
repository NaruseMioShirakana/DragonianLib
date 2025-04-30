/**
 * @file Cumulate.h
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
 * @brief Cumulate operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/CPU/CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

template <
	typename _Type, size_t _NRank,
	typename _FunctionTypeMid
> _D_Dragonian_Lib_Force_Inline void ImplCumulateOperators(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous,
	_FunctionTypeMid CumulateMidOperator
) requires (IsCallableValue<_FunctionTypeMid>)
{
	constexpr size_t _CumulateDim = _NRank - 1;
	const auto _SrcShape = _SrcInfo.Shape[_CumulateDim];
	const auto _DestShape = _DestInfo.Shape[_CumulateDim];
	const auto _SrcStride = _SrcInfo.ViewStride[_CumulateDim];
	const auto _DestStride = _DestInfo.ViewStride[_CumulateDim];

	if (_SrcShape != _DestShape)
		_D_Dragonian_Lib_Throw_Exception("The shape of the source and destination tensors must be the same.");

	auto CumulateFn = [=](int64_t _IndexA, int64_t _IndexB)
		{
			const auto _SrcBegin = _Src + _IndexB;
			auto _DestBegin = _Dest + _IndexA;
			_Type Val = *_SrcBegin;
			*_DestBegin = Val;
			for (SizeType i = 1; i < _SrcShape; ++i)
				*(_DestBegin + i * _DestStride) = (Val = CumulateMidOperator(Val, *(_SrcBegin + i * _SrcStride)));
		};

	auto LoopFn = [=](_Type*, const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoNew, const _Type*, const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoNew, const std::shared_ptr<int>&)
		{
			DoubleTensorLoop<_CumulateDim, 8>(
				0, 0,
				_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
				_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
				CumulateFn
			);
		};

	auto ContCumulateFn = [=](_Type* _DestBegin, const _Type* _SrcBegin, SizeType BatchCount, const std::shared_ptr<int>&)
		{
			for (SizeType i = 0; i < BatchCount; ++i)
			{
				const auto _MSrcBegin = _SrcBegin + i * _SrcShape;
				const auto _MDestBegin = _DestBegin + i * _DestShape;
				_Type Val = *_MSrcBegin;
				*_MDestBegin = Val;
				for (SizeType j = 1; j < _SrcShape; ++j)
					_MDestBegin[j] = (Val = CumulateMidOperator(Val, _MSrcBegin[j]));
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
		ContCumulateFn
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumSumUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A + _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumSubUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A - _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumProdUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A * _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumDivUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A / _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumMaxUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A > _B ? _A : _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplCumMinUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	auto CumulateMidOperator = [](const _Type& _A, const _Type& _B) { return _A < _B ? _A : _B; };
	ImplCumulateOperators(
		_Dest, _DestInfo, _Src, _SrcInfo, Continuous,
		CumulateMidOperator
	);
}

template <typename _Type>
template <size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplDiffUnary(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	bool Continuous
)
{
	constexpr size_t _CumulateDim = _NRank - 1;
	const auto _SrcShape = _SrcInfo.Shape[_CumulateDim];
	const auto _DestShape = _DestInfo.Shape[_CumulateDim];
	const auto _SrcStride = _SrcInfo.ViewStride[_CumulateDim];
	const auto _DestStride = _DestInfo.ViewStride[_CumulateDim];

	if (_SrcShape != _DestShape + 1)
		_D_Dragonian_Lib_Throw_Exception("The shape of the source must be one more than the shape of the destination.");

	auto CumulateFn = [=](int64_t _IndexA, int64_t _IndexB)
		{
			const auto _SrcBegin = _Src + _IndexB;
			auto _DestBegin = _Dest + _IndexA;
			for (SizeType i = 1; i < _SrcShape; ++i)
				_DestBegin[(i - 1) * _DestStride] = _SrcBegin[i * _SrcStride] - _SrcBegin[(i - 1) * _SrcStride];
		};

	auto LoopFn = [=](_Type*, std::shared_ptr<OperatorParameter<_NRank>> _IDestInfoNew, const _Type*, std::shared_ptr<OperatorParameter<_NRank>> _ISrcInfoNew, const std::shared_ptr<int>&)
		{
			auto _DestInfoNew = std::move(_IDestInfoNew);
			auto _SrcInfoNew = std::move(_ISrcInfoNew);

			DoubleTensorLoop<_CumulateDim, 8>(
				0, 0,
				_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
				_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
				CumulateFn
			);
		};

	auto ContCumulateFn = [=](_Type* _DestBegin, const _Type* _SrcBegin, SizeType BatchCount, const std::shared_ptr<int>&)
		{
			for (SizeType i = 0; i < BatchCount; ++i)
			{
				const auto _MSrcBegin = _SrcBegin + i * _SrcShape;
				const auto _MDestBegin = _DestBegin + i * _DestShape;
				for (SizeType j = 1; j < _SrcShape; ++j)
					_MDestBegin[j - 1] = _MSrcBegin[j] - _MSrcBegin[j - 1];
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
		ContCumulateFn
	);
}

_D_Dragonian_Lib_Operator_Space_End