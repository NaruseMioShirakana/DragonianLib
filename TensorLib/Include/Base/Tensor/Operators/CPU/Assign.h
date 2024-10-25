#pragma once
#include "CPU.h"

DragonianLibOperatorSpaceBegin

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
		constexpr auto Stride = (256 * 8);
		const auto Remainder = DestSize % Stride;
		const auto Front = DestSize / Stride * Stride;
		for (SizeType i = 0; i < DestSize - Stride; i += Stride)
			Vectorized<_Type>::DragonianLibMemcpy256((__m256i*)(_Dest + i), (const __m256i*)(_Src + i));
		memcpy(_Dest + Front, _Src + Front, Remainder);
		return;
	}

	for (SizeType i = 0; i < _DestInfo.Shape[0]; ++i)
	{
		const auto IndexAxis0A = ((i * _DestInfo.ViewStride[0]) + _DestInfo.ViewLeft[0]) * _DestInfo.ViewStep[0];
		const auto IndexAxis0B = ((i * _SrcInfo.ViewStride[0]) + _SrcInfo.ViewLeft[0]) * _SrcInfo.ViewStep[0];
		for (SizeType j = 0; j < _DestInfo.Shape[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * _DestInfo.ViewStride[1]) + _DestInfo.ViewLeft[1]) * _DestInfo.ViewStep[1];
			const auto IndexAxis1B = IndexAxis0B +
				((j * _SrcInfo.ViewStride[1]) + _SrcInfo.ViewLeft[1]) * _SrcInfo.ViewStep[1];
			for (SizeType k = 0; k < _DestInfo.Shape[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * _DestInfo.ViewStride[2]) + _DestInfo.ViewLeft[2]) * _DestInfo.ViewStep[2];
				const auto IndexAxis2B = IndexAxis1B +
					((k * _SrcInfo.ViewStride[2]) + _SrcInfo.ViewLeft[2]) * _SrcInfo.ViewStep[2];
				for (SizeType l = 0; l < _DestInfo.Shape[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * _DestInfo.ViewStride[3]) + _DestInfo.ViewLeft[3]) * _DestInfo.ViewStep[3];
					const auto IndexAxis3B = IndexAxis2B +
						((l * _SrcInfo.ViewStride[3]) + _SrcInfo.ViewLeft[3]) * _SrcInfo.ViewStep[3];
					for (SizeType m = 0; m < _DestInfo.Shape[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * _DestInfo.ViewStride[4]) + _DestInfo.ViewLeft[4]) * _DestInfo.ViewStep[4];
						const auto IndexAxis4B = IndexAxis3B +
							((m * _SrcInfo.ViewStride[4]) + _SrcInfo.ViewLeft[4]) * _SrcInfo.ViewStep[4];
						for (SizeType n = 0; n < _DestInfo.Shape[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * _DestInfo.ViewStride[5]) + _DestInfo.ViewLeft[5]) * _DestInfo.ViewStep[5];
							const auto IndexAxis5B = IndexAxis4B +
								((n * _SrcInfo.ViewStride[5]) + _SrcInfo.ViewLeft[5]) * _SrcInfo.ViewStep[5];
							_Dest[IndexAxis5A] = _Src[IndexAxis5B];
						}
					}
				}
			}
		}
	}
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
		DestSize = std::min(DestSize, _Count * sizeof(_Type));
		constexpr auto Stride = (256 * 8);
		const auto Remainder = DestSize % Stride;
		const auto Front = DestSize / Stride * Stride;
		for (SizeType i = 0; i < DestSize - Stride; i += Stride)
			Vectorized<_Type>::DragonianLibMemcpy256((__m256i*)(_Dest + i), (const __m256i*)(_Src + i));
		memcpy(_Dest + Front, _Src + Front, Remainder);
		return;
	}

	const auto End = _Src + _Count;
	for (SizeType i = 0; i < _DestInfo.Shape[0]; ++i)
	{
		const auto IndexAxis0A = ((i * _DestInfo.ViewStride[0]) + _DestInfo.ViewLeft[0]) * _DestInfo.ViewStep[0];
		for (SizeType j = 0; j < _DestInfo.Shape[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * _DestInfo.ViewStride[1]) + _DestInfo.ViewLeft[1]) * _DestInfo.ViewStep[1];
			for (SizeType k = 0; k < _DestInfo.Shape[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * _DestInfo.ViewStride[2]) + _DestInfo.ViewLeft[2]) * _DestInfo.ViewStep[2];
				for (SizeType l = 0; l < _DestInfo.Shape[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * _DestInfo.ViewStride[3]) + _DestInfo.ViewLeft[3]) * _DestInfo.ViewStep[3];
					for (SizeType m = 0; m < _DestInfo.Shape[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * _DestInfo.ViewStride[4]) + _DestInfo.ViewLeft[4]) * _DestInfo.ViewStep[4];
						for (SizeType n = 0; n < _DestInfo.Shape[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * _DestInfo.ViewStride[5]) + _DestInfo.ViewLeft[5]) * _DestInfo.ViewStep[5];
							if (_Src >= End) return;
							_Dest[IndexAxis5A] = *(_Src++);
						}
					}
				}
			}
		}
	}
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
		const int64_t Remainder = _DestInfo.Shape[0] % LoopStride;
		const auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5];
		auto _VectorizedValue = Vectorized<_Type>(_Value);

		for (SizeType i = 0; i < DestSize - LoopStride; i += LoopStride)
		{
			_VectorizedValue.Store(_Dest);
			_VectorizedValue.Store(_Dest + Stride * 1);
			_Dest += LoopStride;
		}

		for (SizeType i = 0; i < Remainder; ++i)
			_Dest[i] = _Value;
		return;
	}

	for (SizeType i = 0; i < _DestInfo.Shape[0]; ++i)
	{
		const auto IndexAxis0A = ((i * _DestInfo.ViewStride[0]) + _DestInfo.ViewLeft[0]) * _DestInfo.ViewStep[0];
		for (SizeType j = 0; j < _DestInfo.Shape[1]; ++j)
		{
			const auto IndexAxis1A = IndexAxis0A +
				((j * _DestInfo.ViewStride[1]) + _DestInfo.ViewLeft[1]) * _DestInfo.ViewStep[1];
			for (SizeType k = 0; k < _DestInfo.Shape[2]; ++k)
			{
				const auto IndexAxis2A = IndexAxis1A +
					((k * _DestInfo.ViewStride[2]) + _DestInfo.ViewLeft[2]) * _DestInfo.ViewStep[2];
				for (SizeType l = 0; l < _DestInfo.Shape[3]; ++l)
				{
					const auto IndexAxis3A = IndexAxis2A +
						((l * _DestInfo.ViewStride[3]) + _DestInfo.ViewLeft[3]) * _DestInfo.ViewStep[3];
					for (SizeType m = 0; m < _DestInfo.Shape[4]; ++m)
					{
						const auto IndexAxis4A = IndexAxis3A +
							((m * _DestInfo.ViewStride[4]) + _DestInfo.ViewLeft[4]) * _DestInfo.ViewStep[4];
						for (SizeType n = 0; n < _DestInfo.Shape[5]; ++n)
						{
							const auto IndexAxis5A = IndexAxis4A +
								((n * _DestInfo.ViewStride[5]) + _DestInfo.ViewLeft[5]) * _DestInfo.ViewStep[5];
							_Dest[IndexAxis5A] = _Value;
						}
					}
				}
			}
		}
	}
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
	if (Continuous)
	{
		auto DestSize = _DestInfo.Shape[0] * _DestInfo.Shape[1] * _DestInfo.Shape[2] *
			_DestInfo.Shape[3] * _DestInfo.Shape[4] * _DestInfo.Shape[5];
		const int64_t Front = DestSize / 8 * 8;
		std::normal_distribution<double> Distribution(_Mean, _Sigma);
		for (SizeType i = 0; i < DestSize - 8; i += 8)
		{
			_Dest[i] = (_Type)Distribution(RandomEngine);
			_Dest[i + 1] = (_Type)Distribution(RandomEngine);
			_Dest[i + 2] = (_Type)Distribution(RandomEngine);
			_Dest[i + 3] = (_Type)Distribution(RandomEngine);
			_Dest[i + 4] = (_Type)Distribution(RandomEngine);
			_Dest[i + 5] = (_Type)Distribution(RandomEngine);
			_Dest[i + 6] = (_Type)Distribution(RandomEngine);
			_Dest[i + 7] = (_Type)Distribution(RandomEngine);
		}
		for (SizeType i = Front; i < DestSize; ++i)
			_Dest[i] = (_Type)Distribution(RandomEngine);
	}
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
			if constexpr (std::is_same_v<_Type, double>)
			{
				_Dest[i] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 1] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 2] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 3] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 4] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 5] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 6] = RandomDoubleDistribution(RandomEngine);
				_Dest[i + 7] = RandomDoubleDistribution(RandomEngine);
			}
			else if constexpr (std::is_same_v<_Type, float>)
			{
				_Dest[i] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 1] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 2] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 3] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 4] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 5] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 6] = RandomFloatDistribution(RandomEngine);
				_Dest[i + 7] = RandomFloatDistribution(RandomEngine);
			}
			else if constexpr (std::is_same_v<_Type, int64_t>)
			{
				_Dest[i] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 1] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 2] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 3] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 4] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 5] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 6] = RandomInt64Distribution(RandomEngine);
				_Dest[i + 7] = RandomInt64Distribution(RandomEngine);
			}
			else if constexpr (std::is_same_v<_Type, int32_t>)
			{
				_Dest[i] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 1] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 2] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 3] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 4] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 5] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 6] = RandomInt32Distribution(RandomEngine);
				_Dest[i + 7] = RandomInt32Distribution(RandomEngine);
			}
			else if constexpr (std::is_same_v<_Type, int16_t>)
			{
				_Dest[i] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 1] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 2] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 3] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 4] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 5] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 6] = RandomInt16Distribution(RandomEngine);
				_Dest[i + 7] = RandomInt16Distribution(RandomEngine);
			}
			else if constexpr (std::is_same_v<_Type, int8_t>)
			{
				_Dest[i] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 1] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 2] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 3] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 4] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 5] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 6] = (uint8_t)RandomInt16Distribution(RandomEngine);
				_Dest[i + 7] = (uint8_t)RandomInt16Distribution(RandomEngine);
			}
			else
				DragonianLibNotImplementedError;
		}
		for (SizeType i = Front; i < DestSize; ++i)
		{
			if constexpr (std::is_same_v<_Type, double>)
				_Dest[i] = RandomDoubleDistribution(RandomEngine);
			else if constexpr (std::is_same_v<_Type, float>)
				_Dest[i] = RandomFloatDistribution(RandomEngine);
			else if constexpr (std::is_same_v<_Type, int64_t>)
				_Dest[i] = RandomInt64Distribution(RandomEngine);
			else if constexpr (std::is_same_v<_Type, int32_t>)
				_Dest[i] = RandomInt32Distribution(RandomEngine);
			else if constexpr (std::is_same_v<_Type, int16_t>)
				_Dest[i] = RandomInt16Distribution(RandomEngine);
			else if constexpr (std::is_same_v<_Type, int8_t>)
				_Dest[i] = (uint8_t)RandomInt16Distribution(RandomEngine);
			else
				DragonianLibNotImplementedError;
		}
	}
}

DragonianLibOperatorSpaceEnd