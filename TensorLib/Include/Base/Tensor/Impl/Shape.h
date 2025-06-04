/**
 * @file Shape.h
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
 * @brief Shape ops for DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Impl/Grad/Shape.h"

_D_Dragonian_Lib_Space_Begin

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _Axis, size_t>
_D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::GatherAxis(
	SizeType _Index
) const requires (_NRank > 0 && _Axis < _NRank)
{
	ThrowOnNotEnabled();
	constexpr auto _MyRnk = _NRank - 1;
	Tensor<_TensorType, _MyRnk ? _MyRnk : 1, _MyDevice> Ret;
	Ret._MyFirst = _MyFirst;
	Ret._MyData = _MyData + CalcIndex(_Index, _MyShape[_Axis]) * _MyViewStride[_Axis];
	Ret._MyLast = _MyLast;
	Ret._MyFuturesAsResult = _MyFuturesAsResult;
	Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
	Ret._MyAllocator = _MyAllocator;
	Ret._IgnoreDep = _IgnoreDep;
	Ret._MyGraph = _MyGraph;
	Ret._MyFunction = _MyFunction;
	if constexpr (_NRank == 1)
	{
		Ret._MyShape[0] = 1;
		Ret._MyViewStride[0] = _MyViewStride.Back();
	}
	else
		for (size_t i = 0, j = 0; i < _NRank; ++i)
		{
			if (i == _Axis)
				continue;
			Ret._MyShape[j] = _MyShape[i];
			Ret._MyViewStride[j] = _MyViewStride[i];
			++j;
		}

	_D_Dragonian_Lib_Auto_Grad(Index, *this, _Index, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _TRank>
constexpr decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::ViewDimensions(
	const Dimensions<_TRank>& _Indice
) const requires (_NRank >= _TRank)
{
	constexpr auto RetRnk = _NRank - _TRank;
	Tensor<_TensorType, RetRnk ? RetRnk : 1, _MyDevice> Ret;
	Ret._MyFirst = _MyFirst;
	Ret._MyData = Data(_Indice);
	Ret._MyLast = _MyLast;
	Ret._MyFuturesAsResult = _MyFuturesAsResult;
	Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
	Ret._MyAllocator = _MyAllocator;
	Ret._IgnoreDep = _IgnoreDep;
	Ret._MyGraph = _MyGraph;
	Ret._MyFunction = _MyFunction;
	if constexpr (_NRank > _TRank)
	{
		Ret._MyShape.Assign(_MyShape.begin() + _TRank);
		Ret._MyViewStride.Assign(_MyViewStride.begin() + _TRank);
	}
	else
	{
		Ret._MyShape[0] = 1;
		Ret._MyViewStride[0] = _MyViewStride.Back();
	}
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _Type1, typename _Type2, size_t _Rank1, size_t _Rank2, Device _Device1, Device _Device2>
std::pair<
	Tensor<_Type1, _Rank1, _Device1>,
	Tensor<_Type2, _Rank1, _Device2>
> Tensor<_TensorType, _NRank, _MyDevice>::BroadCast(
	const Tensor<_Type1, _Rank1, _Device1>& _A,
	const Tensor<_Type2, _Rank2, _Device2>& _B,
	bool Inplace
) requires(_Rank1 >= _Rank2)
{
	constexpr auto CurrentRank = MaxOf(_Rank1, _Rank2);
	std::pair<
		Tensor<_Type1, CurrentRank, _Device1>,
		Tensor<_Type2, CurrentRank, _Device2>
	> Ret{ {},{} };

	auto& First = Ret.first;		auto& Second = Ret.second;
	First._MyShape.AssignConstant(1);					Second._MyShape.AssignConstant(1);
	First._MyViewStride.AssignConstant(0);				Second._MyViewStride.AssignConstant(0);
	First._MyFirst = _A._MyFirst;							Second._MyFirst = _B._MyFirst;
	First._MyLast = _A._MyLast;								Second._MyLast = _B._MyLast;
	First._MyFuturesAsResult = _A._MyFuturesAsResult;		Second._MyFuturesAsResult = _B._MyFuturesAsResult;
	First._MyFuturesAsArgument = _A._MyFuturesAsArgument;	Second._MyFuturesAsArgument = _B._MyFuturesAsArgument;
	First._MyData = _A._MyData;								Second._MyData = _B._MyData;
	First._MyAllocator = _A._MyAllocator;					Second._MyAllocator = _B._MyAllocator;
	First._IgnoreDep = _A._IgnoreDep;						Second._IgnoreDep = _B._IgnoreDep;
	First._MyGraph = _A._MyGraph;							Second._MyGraph = _B._MyGraph;
	First._MyFunction = _A._MyFunction;						Second._MyFunction = _B._MyFunction;

	for (size_t CurrentIndex = 0; CurrentIndex < CurrentRank; ++CurrentIndex)
	{
		//const auto i = CurrentRank - CurrentIndex - 1;
		const auto idx = CurrentRank - CurrentIndex - 1;
		auto XSize = 1ll, YSize = 1ll;
		if (CurrentIndex < _Rank1)
		{
			const auto i = _Rank1 - CurrentIndex - 1;
			First._MyShape[idx] = _A._MyShape[i];
			First._MyViewStride[idx] = _A._MyViewStride[i];
			XSize = _A._MyShape[i];
		}
		if (CurrentIndex < _Rank2)
		{
			const auto i = _Rank2 - CurrentIndex - 1;
			Second._MyShape[idx] = _B._MyShape[i];
			Second._MyViewStride[idx] = _B._MyViewStride[i];
			YSize = _B._MyShape[i];
		}
		if (XSize == YSize)
			continue;
		if (XSize == 1)
		{
			if (Inplace)
				_D_Dragonian_Lib_Throw_Exception(
					"Could Not Inplace Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
					"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
					"] From " + std::to_string(XSize) + " To " + std::to_string(YSize) + "!"
				);
			First._MyShape[idx] = YSize;					First._MyViewStride[idx] = 0;
		}
		else if (YSize == 1)
		{
			Second._MyShape[idx] = XSize;					Second._MyViewStride[idx] = 0;
		}
		else
			_D_Dragonian_Lib_Throw_Exception(
				"Could Not Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
				"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
				"] With Value [" + std::to_string(XSize) + ", " + std::to_string(YSize) + "]!"
			);
	}
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _Type1, typename _Type2, size_t _Rank1, size_t _Rank2, Device _Device1, Device _Device2>
std::pair<
	Tensor<_Type1, _Rank2, _Device1>,
	Tensor<_Type2, _Rank2, _Device2>
> Tensor<_TensorType, _NRank, _MyDevice>::BroadCast(
	const Tensor<_Type1, _Rank1, _Device1>& _A,
	const Tensor<_Type2, _Rank2, _Device2>& _B,
	bool Inplace
) requires (_Rank1 < _Rank2)
{
	constexpr auto CurrentRank = MaxOf(_Rank1, _Rank2);
	std::pair<
		Tensor<_Type1, CurrentRank, _Device1>,
		Tensor<_Type2, CurrentRank, _Device2>
	> Ret{ {},{} };

	auto& First = Ret.first;		auto& Second = Ret.second;
	First._MyShape.AssignConstant(1);					Second._MyShape.AssignConstant(1);
	First._MyViewStride.AssignConstant(0);				Second._MyViewStride.AssignConstant(0);
	First._MyFirst = _A._MyFirst;							Second._MyFirst = _B._MyFirst;
	First._MyLast = _A._MyLast;								Second._MyLast = _B._MyLast;
	First._MyFuturesAsResult = _A._MyFuturesAsResult;		Second._MyFuturesAsResult = _B._MyFuturesAsResult;
	First._MyFuturesAsArgument = _A._MyFuturesAsArgument;	Second._MyFuturesAsArgument = _B._MyFuturesAsArgument;
	First._MyData = _A._MyData;								Second._MyData = _B._MyData;
	First._MyAllocator = _A._MyAllocator;					Second._MyAllocator = _B._MyAllocator;
	First._IgnoreDep = _A._IgnoreDep;						Second._IgnoreDep = _B._IgnoreDep;
	First._MyGraph = _A._MyGraph;							Second._MyGraph = _B._MyGraph;
	First._MyFunction = _A._MyFunction;						Second._MyFunction = _B._MyFunction;

	for (size_t CurrentIndex = 0; CurrentIndex < CurrentRank; ++CurrentIndex)
	{
		//const auto i = CurrentRank - CurrentIndex - 1;
		const auto idx = CurrentRank - CurrentIndex - 1;
		auto XSize = 1ll, YSize = 1ll;
		if (CurrentIndex < _Rank1)
		{
			const auto i = _Rank1 - CurrentIndex - 1;
			First._MyShape[idx] = _A._MyShape[i];
			First._MyViewStride[idx] = _A._MyViewStride[i];
			XSize = _A._MyShape[i];
		}
		if (CurrentIndex < _Rank2)
		{
			const auto i = _Rank2 - CurrentIndex - 1;
			Second._MyShape[idx] = _B._MyShape[i];
			Second._MyViewStride[idx] = _B._MyViewStride[i];
			YSize = _B._MyShape[i];
		}
		if (XSize == YSize)
			continue;
		if (XSize == 1)
		{
			if (Inplace)
				_D_Dragonian_Lib_Throw_Exception(
					"Could Not Inplace Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
					"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
					"] From " + std::to_string(XSize) + " To " + std::to_string(YSize) + "!"
				);
			First._MyShape[idx] = YSize;					First._MyViewStride[idx] = 0;
		}
		else if (YSize == 1)
		{
			Second._MyShape[idx] = XSize;					Second._MyViewStride[idx] = 0;
		}
		else
			_D_Dragonian_Lib_Throw_Exception(
				"Could Not Broadcast Tensor[Shape: " + ToString(_A.Shape()) +
				"] And Tensor[Shape: " + ToString(_B.Shape()) + "] At Axis[" + std::to_string(idx) +
				"] With Value [" + std::to_string(XSize) + ", " + std::to_string(YSize) + "]!"
			);
	}
	return Ret;
}

_D_Dragonian_Lib_Space_End
