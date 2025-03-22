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
	template <typename _Type>
	void Gemm_x_AT_Cont(
		_Type* y,
		const _Type* x, const _Type* A, const _Type* b,
		SizeType N, SizeType O, SizeType I,
		bool yIsTran, bool bIsTran
	)
	{

	}

	template <typename _Type>
	void Gemm_xT_A_Cont(
		_Type* y,
		const _Type* x, const _Type* A, const _Type* b,
		SizeType N, SizeType O, SizeType I,
		bool yIsTran, bool bIsTran
	)
	{

	}

	template <typename _Type>
	void Gemm_x_A_Cont(
		_Type* y,
		const _Type* x, const _Type* A, const _Type* b,
		SizeType N, SizeType O, SizeType I,
		bool yIsTran, bool bIsTran
	)
	{

	}

	template <typename _Type>
	void Gemm_xT_AT_Cont(
		_Type* y,
		const _Type* x, const _Type* A, const _Type* b,
		SizeType N, SizeType O, SizeType I,
		bool yIsTran, bool bIsTran
	)
	{

	}


	template <InterpolateMode _Mode, typename _Type, size_t _NRank>
	_D_Dragonian_Lib_Force_Inline void MatMul(
		_Type * _OutFeature,
		const OperatorParameter<_NRank>&_OutFeatureInfo,
		const _Type * _InFeature,
		const OperatorParameter<_NRank>&_InFeatureInfo,
		const _Type * _Weight,
		const OperatorParameter<_NRank>&_WeightInfo,
		const _Type * _Bias,
		const OperatorParameter<_NRank>&_BiasInfo,
		bool _IsContiguous,
		bool _NotSliced
	)
	{
		// x[B, N, I] * A[B, I, O] + b[B, N, O] = y[B, N, O]

		const auto y = _OutFeature;
		const auto yShape = _OutFeatureInfo.Shape;
		const auto yStride = _OutFeatureInfo.ViewStride;
		const auto N = yShape[_NRank - 2];
		const auto O = yShape[_NRank - 1];
		bool yIsTran = yStride != 1;

		const auto x = _InFeature;
		const auto& xShape = _InFeatureInfo.Shape;
		const auto& xStride = _InFeatureInfo.ViewStride;
		const auto I = xShape[_NRank - 1];
		bool xIsTran = xStride != 1;

		const auto A = _Weight;
		const auto& AShape = _WeightInfo.Shape;
		const auto& AStride = _WeightInfo.ViewStride;
		bool AIsTran = AStride != 1;

		const auto b = _Bias;
		const auto& bShape = _BiasInfo.Shape;
		const auto& bStride = _BiasInfo.ViewStride;
		bool bIsTran = bStride != 1;

		const bool yIsCont = yStride[_NRank - 2] == O && yStride[_NRank - 2] == 1 || yStride[_NRank - 2] == N && yStride[_NRank - 2] == 1;
		const bool xIsCont = xStride[_NRank - 2] == I && xStride[_NRank - 2] == 1 || xStride[_NRank - 2] == N && xStride[_NRank - 2] == 1;
		const bool AIsCont = AStride[_NRank - 2] == O && AStride[_NRank - 2] == 1 || AStride[_NRank - 2] == I && AStride[_NRank - 2] == 1;
		const bool bIsCont = bStride[_NRank - 2] == O && bStride[_NRank - 2] == 1 || bStride[_NRank - 2] == N && bStride[_NRank - 2] == 1;
		const bool IsCont = yIsCont && xIsCont && AIsCont && bIsCont && _NotSliced;
	}
}



_D_Dragonian_Lib_Operator_Space_End