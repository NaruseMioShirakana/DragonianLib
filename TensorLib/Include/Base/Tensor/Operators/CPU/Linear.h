/**
 * @file Linear.h
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
 * @brief Linear operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Created <
 */

#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

namespace Linear
{

	constexpr Int CblasNoTrans = 111;
	constexpr Int CblasTrans = 112;
	constexpr Int CblasConjTrans = 113;
	constexpr Int CblasConjNoTrans = 114;

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Float32* InData, const Float32* WeightData, Float32* OutData,
		const Float32* BiasData, bool BroadcastBias,
		Float32 Alpha, Float32 Beta, Float32 AlphaB
	);

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Float64* InData, const Float64* WeightData, Float64* OutData,
		const Float64* BiasData, bool BroadcastBias,
		Float64 Alpha, Float64 Beta, Float64 AlphaB
	);

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Complex32* InData, const Complex32* WeightData, Complex32* OutData,
		const Complex32* BiasData, bool BroadcastBias,
		Complex32 Alpha, Complex32 Beta, Complex32 AlphaB
	);

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Complex64* InData, const Complex64* WeightData, Complex64* OutData,
		const Complex64* BiasData, bool BroadcastBias,
		Complex64 Alpha, Complex64 Beta, Complex64 AlphaB
	);
}

template <typename _Type>
template<size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::MatMul(
	_Type* _OutFeature, const OperatorParameter<_NRank>& _OutFeatureInfo,
	const _Type* _InFeature, const OperatorParameter<_NRank>& _InFeatureInfo,
	const _Type* _Weight, const OperatorParameter<_NRank>& _WeightInfo,
	const _Type* _Bias, std::shared_ptr<OperatorParameter<_NRank>> _BiasInfo,
	_Type Alpha, _Type AlphaBias,
	bool _Conj
)
{
	constexpr size_t BatchDim = _NRank - 2;
	Int64 InShape[3]{ 1, _InFeatureInfo.Shape[BatchDim], _InFeatureInfo.Shape[BatchDim + 1] };
	Int64 WeightShape[3]{ 1, _WeightInfo.Shape[BatchDim], _WeightInfo.Shape[BatchDim + 1] };
	Int64 OutShape[3]{ 1, _OutFeatureInfo.Shape[BatchDim], _OutFeatureInfo.Shape[BatchDim + 1] };

	for (size_t i = 0; i < BatchDim; ++i)
	{
		InShape[0] *= _InFeatureInfo.Shape[i];
		WeightShape[0] *= _WeightInfo.Shape[i];
		OutShape[0] *= _OutFeatureInfo.Shape[i];
	}
	Int64 InStride = InShape[0] == 1 ? 0 : InShape[1] * InShape[2];
	Int64 WeightStride = WeightShape[0] == 1 ? 0 : WeightShape[1] * WeightShape[2];

	Int64 BiasShape[3]{ 1, 1, 1 };
	if (_BiasInfo && _Bias)
	{
		BiasShape[1] = _BiasInfo->Shape[BatchDim];
		BiasShape[2] = _BiasInfo->Shape[BatchDim + 1];
		for (size_t i = 0; i < BatchDim; ++i)
			BiasShape[0] *= _BiasInfo->Shape[i];
	}
	Int64 BiasStride = BiasShape[0] == 1 ? 0 : BiasShape[1] * BiasShape[2];
	bool BroadcastBias = BiasShape[1] == 1;

	if (InShape[0] != OutShape[0] && InStride)
		_D_Dragonian_Lib_Throw_Exception("Batch shape of input mismatch! expected: " + std::to_string(OutShape[0]) + ", got: " + std::to_string(InShape[0]));
	if (WeightShape[0] != OutShape[0] && WeightStride)
		_D_Dragonian_Lib_Throw_Exception("Batch shape of weight mismatch! expected: " + std::to_string(OutShape[0]) + ", got: " + std::to_string(WeightShape[0]));
	if (BiasShape[0] != OutShape[0] && BiasStride)
		_D_Dragonian_Lib_Throw_Exception("Batch shape of bias mismatch! expected: " + std::to_string(OutShape[0]) + ", got: " + std::to_string(BiasShape[0]));

	//Out[Batch, CommonDim, OutFeature] = In[Batch, CommonDim, InFeature] * Weight[Batch, InFeature, OutFeature]
	Int CommonDim = Int(OutShape[1]);
	Int OutFeature = Int(OutShape[2]);
	Int InFeature = Int(InShape[1]) == CommonDim ? Int(InShape[2]) : Int(InShape[1]);

	Int TransA, TransB;
	if (InShape[1] == CommonDim && InShape[2] == InFeature)
		TransA = Linear::CblasNoTrans;
	else if (InShape[1] == InFeature && InShape[2] == CommonDim)
		TransA = Linear::CblasTrans;
	else
		_D_Dragonian_Lib_Throw_Exception("InFeature shape mismatch!");
	if (WeightShape[1] == InFeature && WeightShape[2] == OutFeature)
		TransB = Linear::CblasNoTrans;
	else if (WeightShape[1] == OutFeature && WeightShape[2] == InFeature)
		TransB = Linear::CblasTrans;
	else
		_D_Dragonian_Lib_Throw_Exception("OutFeature shape mismatch!");

	if constexpr (IsComplexValue<_Type>)
	{
		if (_Conj)
		{
			TransA = TransA == Linear::CblasNoTrans ? Linear::CblasConjNoTrans : Linear::CblasConjTrans;
			TransB = TransB == Linear::CblasNoTrans ? Linear::CblasConjNoTrans : Linear::CblasConjTrans;
		}
	}

	auto& ThreadPool = GetThreadPool();

	TemplateLibrary::Array<std::shared_ptr<void>, 5> _DataPointer{
	_OutFeatureInfo.Data, _InFeatureInfo.Data, _WeightInfo.Data,
		_BiasInfo ? _BiasInfo->Data : std::shared_ptr<void>(nullptr), nullptr
	};

	for (Int64 i = 0; i < OutShape[0]; ++i)
	{
		const auto InData = _InFeature + i * InStride;
		const auto WeightData = _Weight + i * WeightStride;
		const auto OutData = _OutFeature + i * CommonDim * OutFeature;
		const auto BiasData = _Bias ? _Bias + i * BiasStride : nullptr;
		std::shared_future<void> Future = ThreadPool.Commit([=]()
			{
				Linear::Gemm(
					TransA, TransB,
					InFeature, OutFeature, CommonDim,
					InData, WeightData, OutData,
					BiasData, BroadcastBias,
					Alpha, _Type(0), AlphaBias
				);
			});
		_OutFeatureInfo.ResultDependency->emplace_back(Future, _DataPointer);
		_InFeatureInfo.ArgumentDependency->emplace_back(Future, _DataPointer);
		_WeightInfo.ArgumentDependency->emplace_back(Future, _DataPointer);
		if (_BiasInfo)
			_BiasInfo->ArgumentDependency->emplace_back(Future, _DataPointer);
	}
}

_D_Dragonian_Lib_Operator_Space_End