/**
 * @file Interpolate.h
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
 * @brief Interpolation operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

namespace Interpolate
{
	template <typename _Type>
	decltype(auto) InterpolateNearest(
		const _Type& _A,
		const _Type& _B,
		double _Weight
	)
	{
		return _Weight < 0.5 ? _A : _B;
	}

	template <typename _Type>
	decltype(auto) InterpolateLinear(
		const _Type& _A,
		const _Type& _B,
		double _Weight
	)
	{
		return _Type(double(_A) + _Weight * double(_B - _A));
	}

	template <typename _Type, typename Fn>
	void LinearInterpolate1D(
		_Type* DB, const _Type* SB,
		const SizeType* DS, const SizeType* DST,
		const SizeType* SS, const SizeType* SST,
		const double* BG, const double* ST,
		Fn OP
	)
	{
		//Shape: [N0]  [N1]
		using STT = SizeType; using DBL = double;
		const auto DEI0 = DS[0] - 1; const auto SEI0 = SS[0] - 1;
		const auto DST0 = DST[0]; const auto SST0 = SST[0];
		const auto BG0 = BG[0]; const auto ST0 = ST[0];

		const auto N0 = DEI0 * DST0; const auto N1 = SEI0 * SST0;
		//DB[0] = SB[0];	DB[N0] = SB[N1];
		DB[0] = SB[0];		DB[N0] = SB[N1];
		for (STT i = 1; i < DEI0; ++i)
		{
			const DBL LSI = BG0 + DBL(i) * ST0; const STT LSIF = STT(LSI);
			const DBL LSWI = LSI - DBL(LSIF); const STT LSIC = std::min(LSIF + 1, SEI0);
			const auto ID = i * DST0; const bool AEI = LSIF == LSIC;
			const auto I0 = LSIF * SST0; const auto I1 = LSIC * SST0;
			if (AEI) 
				DB[ID] = SB[I0];
			else 
				DB[ID] = OP(SB[I0], SB[I1], LSWI);
		}
	}

	template <typename _Type, typename Fn>
	void LinearInterpolate2D(
		_Type* DB, const _Type* SB,
		const SizeType* DS, const SizeType* DST,
		const SizeType* SS, const SizeType* SST,
		const double* BG, const double* ST,
		Fn OP
	)
	{
		//Shape: [N0, M0]  [N1, M1]
		using STT = SizeType; using DBL = double;
		const auto DEI0 = DS[0] - 1; const auto SEI0 = SS[0] - 1;
		const auto DEI1 = DS[1] - 1; const auto SEI1 = SS[1] - 1;
		const auto DST0 = DST[0]; const auto SST0 = SST[0];
		const auto DST1 = DST[1]; const auto SST1 = SST[1];
		const auto BG0 = BG[0]; const auto ST0 = ST[0];
		const auto BG1 = BG[1]; const auto ST1 = ST[1];

		const auto N0 = DEI0 * DST0; const auto N1 = SEI0 * SST0;
		const auto M0 = DEI1 * DST1; const auto M1 = SEI1 * SST1;
		//DB[0][0] = SB[0][0];	DB[N0][0] = SB[N1][0];	DB[0][M0] = SB[0][M1];	DB[N0][M0] = SB[N1][M1];
		DB[0] = SB[0];			DB[N0] = SB[N1];		DB[M0] = SB[M1];		DB[N0 + M0] = SB[N1 + M1];
		for (STT i = 1; i < DEI0; ++i)
		{
			const DBL LSI = BG0 + DBL(i) * ST0; const STT LSIF = STT(LSI);
			const DBL LSWI = LSI - DBL(LSIF); const STT LSIC = std::min(LSIF + 1, SEI0);
			const auto ID = i * DST0; const bool AEI = LSIF == LSIC;
			const auto I0 = LSIF * SST0; const auto I1 = LSIC * SST0;
			for (STT j = 1; j < DEI1; ++j)
			{
				const DBL LSJ = BG1 + DBL(j) * ST1; const STT LSJF = STT(LSJ);
				const DBL LSWJ = LSJ - DBL(LSJF); const STT LSJC = std::min(LSJF + 1, SEI1);
				const auto JD = j * DST1; const bool AEJ = LSJF == LSJC;
				const auto J0 = LSJF * SST1; const auto J1 = LSJC * SST1;
				if (AEI && AEJ) 
					DB[ID + JD] = SB[I0 + J0];
				else if (AEI) 
					DB[ID + JD] = OP(SB[I0 + J0], SB[I0 + J1], LSWJ);
				else if (AEJ) 
					DB[ID + JD] = OP(SB[I0 + J0], SB[I1 + J0], LSWI);
				else 
					DB[ID + JD] = OP(OP(SB[I0 + J0], SB[I0 + J1], LSWJ), OP(SB[I1 + J0], SB[I1 + J1], LSWJ), LSWI);
			}
		}
	}

	template <typename _Type, typename Fn>
	void LinearInterpolate3D(
		_Type* DB, const _Type* SB,
		const SizeType* DS, const SizeType* DST,
		const SizeType* SS, const SizeType* SST,
		const double* BG, const double* ST,
		Fn OP
	)
	{
		//Shape [N0, M0, L0]  [N1, M1, L1]
		using STT = SizeType; using DBL = double;
		const auto DEI0 = DS[0] - 1; const auto SEI0 = SS[0] - 1;
		const auto DEI1 = DS[1] - 1; const auto SEI1 = SS[1] - 1;
		const auto DEI2 = DS[2] - 1; const auto SEI2 = SS[2] - 1;
		const auto DST0 = DST[0]; const auto SST0 = SST[0];
		const auto DST1 = DST[1]; const auto SST1 = SST[1];
		const auto DST2 = DST[2]; const auto SST2 = SST[2];
		const auto BG0 = BG[0]; const auto ST0 = ST[0];
		const auto BG1 = BG[1]; const auto ST1 = ST[1];
		const auto BG2 = BG[2]; const auto ST2 = ST[2];

		
		const auto N0 = DEI0 * DST0; const auto N1 = SEI0 * SST0;
		const auto M0 = DEI1 * DST1; const auto M1 = SEI1 * SST1;
		const auto L0 = DEI2 * DST2; const auto L1 = SEI2 * SST2;
		//DB[0][0][0] = SB[0][0][0];		DB[N0][0][0] = SB[N1][0][0];	DB[0][M0][0] = SB[0][M1][0];	DB[0][0][L0] = SB[0][0][L1];
		DB[0] = SB[0];						DB[N0] = SB[N1];				DB[M0] = SB[M1];				DB[L0] = SB[L1];
		//DB[N0][M0][0] = SB[N1][M1][0];	DB[N0][0][L0] = SB[N1][0][L1];	DB[0][M0][L0] = SB[0][M1][L1];	DB[N0][M0][L0] = SB[N1][M1][L1];
		DB[N0 + M0] = SB[N1 + M1];			DB[N0 + L0] = SB[N1 + L1];		DB[M0 + L0] = SB[M1 + L1];		DB[N0 + M0 + L0] = SB[N1 + M1 + L1];
		for (STT i = 1; i < DEI0; ++i)
		{
			const DBL LSI = BG0 + DBL(i) * ST0; const STT LSIF = STT(LSI);
			const DBL LSWI = LSI - DBL(LSIF); const STT LSIC = std::min(LSIF + 1, SEI0);
			const auto ID = i * DST0; const bool AEI = LSIF == LSIC;
			const auto I0 = LSIF * SST0; const auto I1 = LSIC * SST0;
			for (STT j = 1; j < DEI1; ++j)
			{
				const DBL LSJ = BG1 + DBL(j) * ST1; const STT LSJF = STT(LSJ);
				const DBL LSWJ = LSJ - DBL(LSJF); const STT LSJC = std::min(LSJF + 1, SEI1);
				const auto JD = j * DST1; const bool AEJ = LSJF == LSJC;
				const auto J0 = LSJF * SST1; const auto J1 = LSJC * SST1;
				for (STT k = 1; k < DEI2; ++k)
				{
					const DBL LSK = BG2 + DBL(k) * ST2; const STT LSKF = STT(LSK);
					const DBL LSWK = LSK - DBL(LSKF); const STT LSKC = std::min(LSKF + 1, SEI2);
					const auto KD = k * DST2; const bool AEK = LSKF == LSKC;
					const auto K0 = LSKF * SST2; const auto K1 = LSKC * SST2;
					if (AEI && AEJ && AEK) 
						DB[ID + JD + KD] = SB[I0 + J0 + K0];
					else if (AEI && AEJ) 
						DB[ID + JD + KD] = OP(SB[I0 + J0 + K0], SB[I0 + J0 + K1], LSWK);
					else if (AEI && AEK) 
						DB[ID + JD + KD] = OP(SB[I0 + J0 + K0], SB[I0 + J1 + K0], LSWJ);
					else if (AEJ && AEK) 
						DB[ID + JD + KD] = OP(SB[I0 + J0 + K0], SB[I1 + J0 + K0], LSWI);
					else if (AEI) 
						DB[ID + JD + KD] = OP(OP(SB[I0 + J0 + K0], SB[I0 + J0 + K1], LSWK), OP(SB[I0 + J1 + K0], SB[I0 + J1 + K1], LSWK), LSWJ);
					else if (AEJ) 
						DB[ID + JD + KD] = OP(OP(SB[I0 + J0 + K0], SB[I0 + J0 + K1], LSWK), OP(SB[I1 + J0 + K0], SB[I1 + J0 + K1], LSWK), LSWI);
					else if (AEK) 
						DB[ID + JD + KD] = OP(OP(SB[I0 + J0 + K0], SB[I0 + J1 + K0], LSWJ), OP(SB[I0 + J0 + K1], SB[I0 + J1 + K1], LSWJ), LSWK);
					else 
						DB[ID + JD + KD] = OP(OP(OP(SB[I0 + J0 + K0], SB[I0 + J0 + K1], LSWK), OP(SB[I0 + J1 + K0], SB[I0 + J1 + K1], LSWK), LSWJ), OP(OP(SB[I1 + J0 + K0], SB[I1 + J0 + K1], LSWK), OP(SB[I1 + J1 + K0], SB[I1 + J1 + K1], LSWK), LSWJ), LSWI);
				}
			}
		}
	}

	template <size_t _Dim, typename _Type, typename Fn>
	void LinearInterpolateND(
		_Type* DB, const _Type* SB,
		const SizeType* DS, const SizeType* DST,
		const SizeType* SS, const SizeType* SST,
		const double* BG, const double* ST,
		Fn OP
	)
	{
		if constexpr (_Dim == 1)
			LinearInterpolate1D(DB, SB, DS, DST, SS, SST, BG, ST, OP);
		else if constexpr (_Dim == 2)
			LinearInterpolate2D(DB, SB, DS, DST, SS, SST, BG, ST, OP);
		else if constexpr (_Dim == 3)
			LinearInterpolate3D(DB, SB, DS, DST, SS, SST, BG, ST, OP);
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid interpolation dimension");
	}

	template <typename _Type>
	void BicubicInterpolate(
		_Type* DB, const _Type* SB,
		const SizeType* DS, const SizeType* DST,
		const SizeType* SS, const SizeType* SST,
		const double* ST, bool ACB
	)
	{
		//Shape [N0, M0]  [N1, M1]
		using STT = SizeType; using DBL = double;
		using STTA = TemplateLibrary::Array<STT, 4>; using DBLA = TemplateLibrary::Array<DBL, 4>;

		DBLA WXS; DBLA WYS; STTA IXS; STTA IYS;
		const auto DEI0 = DS[0] - 1; const auto SEI0 = SS[0] - 1;
		const auto DEI1 = DS[1] - 1; const auto SEI1 = SS[1] - 1;
		const auto DST0 = DST[0]; const auto SST0 = SST[0];
		const auto DST1 = DST[1]; const auto SST1 = SST[1];
		const auto ST0 = ST[0]; const auto ST1 = ST[1];

		auto WFN = [](double Offset)
			{
				const double ABS = abs(Offset);
				if (ABS <= 1.)
					return 1.5 * ABS * ABS * ABS - 2.5 * Offset * Offset + 1.;
				if (ABS <= 2.)
					return -0.5 * ABS * ABS * ABS + 2.5 * Offset * Offset - 4. * ABS + 2.;
				return 0.;
			};

		auto CALW = [=](double CIndex, STTA& Index, DBLA& Weight, STT MaxIndex, STT Stride)
			{
				Index[0] = STT(round(CIndex)) - 2;
				Index[1] = Index[0] + 1;
				Index[2] = Index[0] + 2;
				Index[3] = Index[0] + 3;
				for (int k = 0; k < 4; ++k)
				{
					Weight[k] = WFN(CIndex - DBL(Index[k]));
					if (Index[k] < 0)
						Index[k] = -Index[k];
					else if (Index[k] > MaxIndex)
						Index[k] = (MaxIndex << 1) - Index[k];
					Index[k] *= Stride;
				}
			};

		auto CALV = [](const STTA& IndexX, const STTA& IndexY, const DBLA& WeightX, const DBLA& WeightY, const _Type* Begin)
			{
				_Type Result = 0;
				for (int i = 0; i < 4; ++i)
					for (int j = 0; j < 4; ++j)
						Result += _Type(DBL(Begin[IndexX[i] + IndexY[j]]) * WeightX[i] * WeightY[j]);
				return Result;
			};

		for (STT i = 0; i < DS[0]; ++i)
		{
			CALW(DBL(i) * ST0, IXS, WXS, SEI0, SST0);
			const auto ID = i * DST0;
			for (STT j = 0; j < DS[1]; ++j)
			{
				CALW(DBL(j) * ST1, IYS, WYS, SEI1, SST1);
				const auto JD = j * DST1;
				DB[ID + JD] = CALV(IXS, IYS, WXS, WYS, SB);
			}
		}
		if (ACB)
		{
			const auto N0 = DEI0 * DST0; const auto N1 = SEI0 * SST0;
			const auto M0 = DEI1 * DST1; const auto M1 = SEI1 * SST1;
			//DB[0][0] = SB[0][0];	DB[N0][0] = SB[N1][0];	DB[0][M0] = SB[0][M1];	DB[N0][M0] = SB[N1][M1];
			DB[0] = SB[0];			DB[N0] = SB[N1];		DB[M0] = SB[M1];		DB[N0 + M0] = SB[N1 + M1];
		}
	}

	template <InterpolateMode _Mode, typename _Type, size_t _NRank>
	_D_Dragonian_Lib_Force_Inline void ImplInterpolateOperators(
		_Type* _Dest,
		const OperatorParameter<_NRank>& _DestInfo,
		const _Type* _Src,
		const OperatorParameter<_NRank>& _SrcInfo,
		InterpolateParam<_Mode> _Param,
		bool Continuous
	)
	{
		using ParamType = InterpolateParam<_Mode>;
		constexpr size_t _InterpDim = _NRank - ParamType::_MyRank;
		typename ParamType::SizeTypeArrayT _DestShape, _SrcShape, _DestStride, _SrcStride;
		typename ParamType::DoubleArrayT _Step, _Begin;
		_DestShape.Assign(_DestInfo.Shape.Data() + _InterpDim);
		_SrcShape.Assign(_SrcInfo.Shape.Data() + _InterpDim);
		_DestStride.Assign(_DestInfo.ViewStride.Data() + _InterpDim);
		_SrcStride.Assign(_SrcInfo.ViewStride.Data() + _InterpDim);
		bool AlignCorners = _Param._MyAlignCorners;

		for (size_t i = 0; i < ParamType::_MyRank; ++i)
		{
			if (AlignCorners)
			{
				if constexpr (_Mode == InterpolateMode::Bicubic)
					_Step[i] = double(_SrcShape[i]) / double(_DestShape[i]);
				else
					_Step[i] = double(_SrcShape[i]) / double(_DestShape[i] + 1);
				_Begin[i] = 0.;
			}
			else
			{
				_Step[i] = double(_SrcShape[i]) / double(_DestShape[i]);
				_Begin[i] = 0.5 * (_Step[i] - 1);
			}
		}

		auto InterpolateOperator = [=](const _Type& _A, const _Type& _B, double _Weight)
			{
				if constexpr (_Mode == InterpolateMode::Nearest || _Mode == InterpolateMode::Nearest2D || _Mode == InterpolateMode::Nearest3D)
					return InterpolateNearest(_A, _B, _Weight);
				else if constexpr (_Mode == InterpolateMode::Linear || _Mode == InterpolateMode::Bilinear || _Mode == InterpolateMode::Trilinear)
					return InterpolateLinear(_A, _B, _Weight);
				else
					_D_Dragonian_Lib_Throw_Exception("Invalid interpolation mode");
			};

		auto InterpFn = [=](int64_t _IndexA, int64_t _IndexB)
			{
				const auto _SrcBegin = _Src + _IndexB;
				auto _DestBegin = _Dest + _IndexA;
				if constexpr (_Param._IsLinear)
					LinearInterpolateND<GetInterpolateModeRank<_Mode>>(
						_DestBegin, _SrcBegin,
						_DestShape.Data(), _DestStride.Data(),
						_SrcShape.Data(), _SrcStride.Data(),
						_Begin.Data(), _Step.Data(),
						InterpolateOperator
					);
				else if constexpr (_Mode == InterpolateMode::Bicubic)
					BicubicInterpolate(
						_DestBegin, _SrcBegin,
						_DestShape.Data(), _DestStride.Data(),
						_SrcShape.Data(), _SrcStride.Data(),
						_Step.Data(), AlignCorners
					);
				else
					_D_Dragonian_Lib_Throw_Exception("Invalid interpolation mode");
			};

		auto LoopFn = [=](_Type*, const std::shared_ptr<OperatorParameter<_NRank>> _DestInfoNew, const _Type*, const std::shared_ptr<OperatorParameter<_NRank>> _SrcInfoNew, const std::shared_ptr<int>&)
			{
				DoubleTensorLoop<_InterpDim, 1>(
					0, 0,
					_DestInfoNew->Shape.Data(), _DestInfoNew->Begin.Data(),
					_DestInfoNew->ViewStride.Data(), _SrcInfoNew->ViewStride.Data(),
					InterpFn
				);
			};

		ImplMultiThreadCaller<2, _NRank, ParamType::_MyRank, _Type>(
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
}

template <typename _Type>
template <InterpolateMode _Mode, size_t _NRank>
void OperatorsBase<_Type, Device::CPU>::ImplInterpolate(
	_Type* _Dest,
	const OperatorParameter<_NRank>& _DestInfo,
	const _Type* _Src,
	const OperatorParameter<_NRank>& _SrcInfo,
	const InterpolateParam<_Mode>& _Param,
	bool Continuous
)
{
	Interpolate::ImplInterpolateOperators<_Mode>(_Dest, _DestInfo, _Src, _SrcInfo, _Param, Continuous);
}

_D_Dragonian_Lib_Operator_Space_End