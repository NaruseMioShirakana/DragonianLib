#pragma once
#include "CPU.h"

_D_Dragonian_Lib_Operator_Space_Begin

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssign(
	_Type* _Dest,
	const TensorShapeInfo& _DestInfo,
	const _Type* _Src,
	const TensorShapeInfo& _SrcInfo,
	bool Continuous
)
{
	if (Continuous)
	{
		const auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5] * sizeof(_Type);
		Vectorized<_Type>::DragonianLibMemCpy(_Dest, _Src, DestSize);
		return;
	}

	const auto FrontDim = 6 - _DestInfo.ViewRank;

	if (_DestInfo.ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_D_0(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0,
			{
				_Dest[IndexAxis0FA] = _Src[IndexAxis0FA];
			}
		);
	else if(_DestInfo.ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_D_1(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0, 0,
			{
				_Dest[IndexAxis1FA] = _Src[IndexAxis1FA];
			}
		);
	else if (_DestInfo.ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_D_2(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis2FA] = _Src[IndexAxis2FA];
			}
		);
	else if (_DestInfo.ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_D_3(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis3FA] = _Src[IndexAxis3FA];
			}
		);
	else if (_DestInfo.ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_D_4(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis4FA] = _Src[IndexAxis4FA];
			}
		);
	else if (_DestInfo.ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_D_5(
			_DestInfo,
			_SrcInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis5FA] = _Src[IndexAxis5FA];
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssign(
	_Type* _Dest,
	const TensorShapeInfo& _DestInfo,
	const _Type* _Src,
	SizeType _Count,
	bool Continuous
)
{
	if (Continuous)
	{
		auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5] * sizeof(_Type);
		DestSize = std::min(DestSize, size_t(_Count) * sizeof(_Type));
		Vectorized<_Type>::DragonianLibMemCpy(_Dest, _Src, DestSize);
		return;
	}

	const auto End = _Src + _Count;
	const auto FrontDim = 6 - _DestInfo.ViewRank;

	if (_DestInfo.ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_S_0(
			_DestInfo, 
			FrontDim, 
			F, 0, 0, 
			{
				if (_Src >= End) return;
				_Dest[IndexAxis0FA] = *(_Src++);
			}
		);
	else if (_DestInfo.ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_S_1(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis1FA] = *(_Src++);
			}
		);
	else if (_DestInfo.ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_S_2(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis2FA] = *(_Src++);
			}
		);
	else if (_DestInfo.ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_S_3(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis3FA] = *(_Src++);
			}
		);
	else if (_DestInfo.ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_S_4(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis4FA] = *(_Src++);
			}
		);
	else if (_DestInfo.ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_S_5(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0,
			{
				if (_Src >= End) return;
				_Dest[IndexAxis5FA] = *(_Src++);
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssign(
	_Type* _Dest,
	const TensorShapeInfo& _DestInfo,
	_Type _Value,
	bool Continuous
)
{
	if (Continuous)
	{
		constexpr int64_t OpThroughput = 2;
		constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(_Type));
		constexpr int64_t LoopStride = OpThroughput * Stride;

		const auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5];

		const int64_t Remainder = DestSize % LoopStride;

		auto _VectorizedValue1 = Vectorized<_Type>(_Value);
		auto _VectorizedValue2 = Vectorized<_Type>(_Value);
		for (SizeType i = 0; i < DestSize - LoopStride; i += LoopStride)
		{
			_VectorizedValue1.Store(_Dest);
			_VectorizedValue2.Store(_Dest + Stride);
			_Dest += LoopStride;
		}

		for (SizeType i = 0; i < Remainder; ++i)
			_Dest[i] = _Value;

		return;
	}

	const auto FrontDim = 6 - _DestInfo.ViewRank;

	if (_DestInfo.ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_S_0(
			_DestInfo,
			FrontDim,
			F, 0, 0,
			{
				_Dest[IndexAxis0FA] = _Value;
			}
		);
	else if (_DestInfo.ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_S_1(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0,
			{
				_Dest[IndexAxis1FA] = _Value;
			}
		);
	else if (_DestInfo.ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_S_2(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0,
			{
				_Dest[IndexAxis2FA] = _Value;
			}
		);
	else if (_DestInfo.ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_S_3(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis3FA] = _Value;
			}
		);
	else if (_DestInfo.ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_S_4(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis4FA] = _Value;
			}
		);
	else if (_DestInfo.ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_S_5(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis5FA] = _Value;
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRandn(
	_Type* _Dest,
	const TensorShapeInfo& _DestInfo,
	double _Mean,
	double _Sigma,
	bool Continuous
)
{
	using RandomType = _Impl_Dragonian_Lib_Constexpr_Decltype_t<sizeof(_Type) == sizeof(double), double, float>;
	static std::uniform_real_distribution<RandomType> NormalPre(0.0, 1.0);
	std::normal_distribution<RandomType> Normal{ RandomType(_Mean), RandomType(_Sigma) };
	if (Continuous)
	{
		constexpr int64_t OpThroughput = 2;
		constexpr int64_t Stride = int64_t(sizeof(__m256) / sizeof(RandomType));
		constexpr int64_t LoopStride = OpThroughput * Stride;

		const auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5];
		const int64_t Front = DestSize / Stride * Stride;

		for (SizeType i = 0; i < DestSize - LoopStride; i += LoopStride)
		{
			for (SizeType j = 0; j < LoopStride; ++j)
				_Dest[i + j] = (_Type)Normal(RandomEngine);
		}

		for (SizeType i = Front; i < DestSize; ++i)
			_Dest[i] = (_Type)Normal(RandomEngine);

		return;

		//const int64_t Remainder = DestSize % Stride;

		RandomType RandomData[Stride * 2];
		const auto MeanVec = Vectorized<RandomType>(RandomType(_Mean));
		const auto SigmaVec = Vectorized<RandomType>(RandomType(_Sigma));
		const auto PiVec = Vectorized<RandomType>(RandomType(3.1415926535));
		const auto TwoVec = Vectorized<RandomType>(RandomType(2.0));
		const auto MinusTwoVec = Vectorized<RandomType>(RandomType(-2.0));
		for (SizeType i = 0; i < DestSize - Stride; i += Stride)
		{
			for (SizeType j = 0; j < Stride * 2; ++j)
				RandomData[j] = NormalPre(RandomEngine);

			const auto U1 = Vectorized<RandomType>(RandomData);
			const auto U2 = Vectorized<RandomType>(RandomData + Stride);

			const auto Result1 = (MinusTwoVec * U1.Log()).Sqrt() * (TwoVec * PiVec * U2).Cos() * SigmaVec + MeanVec;

			if constexpr (std::is_same_v<_Type, RandomType>)
				Result1.Store(_Dest + i);
			else
			{
				Result1.Store(RandomData);
				for (SizeType j = 0; j < Stride; ++j)
					_Dest[i + j] = (_Type)RandomData[j];
			}
		}

		for (SizeType i = Front; i < DestSize; ++i)
			_Dest[i] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
		
		return;
	}

	const auto FrontDim = 6 - _DestInfo.ViewRank;

	if (_DestInfo.ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_S_0(
			_DestInfo,
			FrontDim,
			F, 0, 0,
			{
				_Dest[IndexAxis0FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else if (_DestInfo.ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_S_1(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0,
			{
				_Dest[IndexAxis1FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else if (_DestInfo.ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_S_2(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0,
			{
				_Dest[IndexAxis2FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else if (_DestInfo.ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_S_3(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis3FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else if (_DestInfo.ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_S_4(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis4FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else if (_DestInfo.ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_S_5(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis5FA] = (_Type)(std::sqrt(-2.0 * std::log(NormalPre(RandomEngine))) * std::cos(2.0 * 3.1415926535 * NormalPre(RandomEngine)) * RandomType(_Sigma) + RandomType(_Mean));
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

template<typename _Type>
_D_Dragonian_Lib_Force_Inline _Type _Impl_Generate_Random_Value()
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

template<typename _Type>
void OperatorsBase<_Type, Device::CPU>::ImplAssignRand(
	_Type* _Dest,
	const TensorShapeInfo& _DestInfo,
	bool Continuous
)
{
	if (Continuous)
	{
		auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5];
		const int64_t Front = DestSize / 8 * 8;
		for (SizeType i = 0; i < DestSize - 8; i += 8)
		{
			_Dest[i] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 1] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 2] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 3] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 4] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 5] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 6] = _Impl_Generate_Random_Value<_Type>();
			_Dest[i + 7] = _Impl_Generate_Random_Value<_Type>();
		}
		for (SizeType i = Front; i < DestSize; ++i)
			_Dest[i] = _Impl_Generate_Random_Value<_Type>();
		return;
	}

	const auto FrontDim = 6 - _DestInfo.ViewRank;

	if (_DestInfo.ViewRank == 1)
		_D_Dragonian_Lib_Operator_Loop_S_0(
			_DestInfo,
			FrontDim,
			F, 0, 0,
			{
				_Dest[IndexAxis0FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else if (_DestInfo.ViewRank == 2)
		_D_Dragonian_Lib_Operator_Loop_S_1(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0,
			{
				_Dest[IndexAxis1FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else if (_DestInfo.ViewRank == 3)
		_D_Dragonian_Lib_Operator_Loop_S_2(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0,
			{
				_Dest[IndexAxis2FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else if (_DestInfo.ViewRank == 4)
		_D_Dragonian_Lib_Operator_Loop_S_3(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis3FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else if (_DestInfo.ViewRank == 5)
		_D_Dragonian_Lib_Operator_Loop_S_4(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis4FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else if (_DestInfo.ViewRank == 6)
		_D_Dragonian_Lib_Operator_Loop_S_5(
			_DestInfo,
			FrontDim,
			F, 0, 0, 0, 0, 0, 0, 0,
			{
				_Dest[IndexAxis5FA] = _Impl_Generate_Random_Value<_Type>();
			}
		);
	else
		_D_Dragonian_Lib_Not_Implemented_Error;
}

_D_Dragonian_Lib_Operator_Space_End