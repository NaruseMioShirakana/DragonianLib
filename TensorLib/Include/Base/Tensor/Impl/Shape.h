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

	_D_Dragonian_Lib_Auto_Grad(Index, *this, _Indice, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _SliceDim, typename _FirstType, typename... _ArgTypes>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::operator()(
	_FirstType _Index,
	_ArgTypes... _Args
	) const requires ((sizeof...(_ArgTypes) < _NRank) && TypeTraits::IsIntegerValue<_FirstType> && (_SliceDim < _NRank))
{
	if constexpr (TypeTraits::IsStringValue<_FirstType>)
		return operator() < _SliceDim > (Range(_Index), _Args...);
	else if constexpr (_SliceDim)
	{
		_Index = static_cast<_FirstType>(CalcIndex(static_cast<SizeType>(_Index), _MyShape[_SliceDim]));
		return operator() < _SliceDim > (Range{ static_cast<SizeType>(_Index), static_cast<SizeType>(_Index + 1) }, _Args...);
	}
	else if constexpr (_NRank == 1)
		return Get(static_cast<SizeType>(_Index));
	else if constexpr (sizeof...(_ArgTypes))
		return operator[](static_cast<SizeType>(_Index)).template operator() < 0 > (_Args...);
	else
		return operator[](static_cast<SizeType>(_Index));
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _SliceDim, typename ... _ArgTypes>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::operator()(
	Range _Range,
	_ArgTypes... _Args
	) const requires ((sizeof...(_ArgTypes) < _NRank) && (_SliceDim < _NRank))
{
	SliceOptions<_NRank> SliceOptions;
	SliceOptions[_SliceDim] = _Range;
	if constexpr (sizeof...(_ArgTypes))
		return Slice(SliceOptions).template operator() < _SliceDim + 1 > (_Args...);
	else
		return Slice(SliceOptions);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _Count>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Split(
	const Dimensions<_Count>& _Size,
	SizeType _Axis
) const
{
	ThrowOnNotEnabled();
	IDLArray<Tensor, _Count> Ret;
	_Axis = CalcIndex(_Axis, Rank());
	const auto _SizeTgr = _MyShape[_Axis];
	const auto _SizeSrc = _Size.Multiply();

	if (_SizeTgr != _SizeSrc)
		_D_Dragonian_Lib_Throw_Exception("The Size Of The Split Is Not Equal To The Source!");

	SliceOptions<_Count> SliceOptions;
	SliceOptions[_Axis] = { 0, 1, 0 };
	for (size_t i = 0; i < _Count; ++i)
	{
		SliceOptions[_Axis].End += _Size[i];
		Ret[i] = Slice(SliceOptions);
		SliceOptions[_Axis].Begin += _Size[i];
	}

	_D_Dragonian_Lib_Auto_Grad(Split, *this, _Size, _Axis, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Split(
	SizeType _Size,
	SizeType _Axis
) const
{
	ThrowOnNotEnabled();
	TemplateLibrary::Vector<Tensor> Ret;
	_Axis = CalcIndex(_Axis, Rank());

	if (_Size <= 0)
		_D_Dragonian_Lib_Throw_Exception("The Size Of The Split Is Not Valid!");

	auto Remain = _MyShape[_Axis];
	Ret.Reserve(Remain / _Size + 1);

	SliceOptions<_NRank> SliceOptions;
	SliceOptions[_Axis] = { 0, 1, 0 };
	while (Remain)
	{
		const auto CSize = std::min(_Size, Remain);
		SliceOptions[_Axis].End += CSize;
		Ret.EmplaceBack(Slice(SliceOptions));
		SliceOptions[_Axis].Begin += CSize;
		Remain -= CSize;
	}

	_D_Dragonian_Lib_Auto_Grad(Split, *this, _Size, _Axis, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _Rnk>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Slice(
	const SliceOptions<_Rnk>& _SliceOptions
) const requires (_Rnk <= _NRank && _Rnk > 0)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("Broad Casted Could Not Be Sliced!");

	Tensor Ret = View();
	for (size_t i = 0; i < _Rnk; ++i)
	{
		if (_SliceOptions[i].Begin == 0 &&
			_SliceOptions[i].Step == 1 &&
			(_SliceOptions[i].End == RangeEndPos || _SliceOptions[i].End == _MyShape[i]))
			continue;

		SizeType SliceBeginPos, SliceStep, SliceEndPos;

		if (_SliceOptions[i].Begin == _SliceOptions[i].Step && _SliceOptions[i].Begin == _SliceOptions[i].End)
		{
			SliceBeginPos = CalcIndex(_SliceOptions[i].Step, _MyShape[i], false);
			SliceStep = 1;
			SliceEndPos = SliceBeginPos + 1;
		}
		else
		{
			SliceStep = _SliceOptions[i].Step;
			if (SliceStep == 0)
				_D_Dragonian_Lib_Throw_Exception("SliceStep Should Not Be Zero!");
			SliceBeginPos = CalcIndex(_SliceOptions[i].Begin, _MyShape[i], false);
			SliceEndPos = CalcEndPos(_SliceOptions[i].End, _MyShape[i], false);
		}

		const auto SliceLength = SliceEndPos - SliceBeginPos;
		if (SliceLength == 0)
			_D_Dragonian_Lib_Throw_Exception("SliceLength could not be zero!");
		if (SliceLength / SliceStep < 0)
			_D_Dragonian_Lib_Throw_Exception("Shape could not be negative!");
		const auto SlicedShape = Ceil(abs(SliceLength), abs(SliceStep));

		Ret._MyData += SliceBeginPos * Ret._MyViewStride[i];
		Ret._MyShape[i] = SlicedShape;
		Ret._MyViewStride[i] *= SliceStep;
	}

	_D_Dragonian_Lib_Auto_Grad(Slice, *this, _SliceOptions, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _Rnk>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::ReversedSlice(
	const SliceOptions<_Rnk>& _SliceOptions
) const requires (_Rnk <= _NRank && _Rnk > 0)
{
	ThrowOnNotEnabled();
	if (IsBroadCasted())
		_D_Dragonian_Lib_Throw_Exception("Broad Casted Could Not Be Sliced!");

	Tensor Ret = View();
	for (size_t j = 0; j < _Rnk; ++j)
	{
		const auto i = _Rnk - 1 - j; // Reverse the index for reversed slice

		if (_SliceOptions[j].Begin == 0 &&
			_SliceOptions[j].Step == 1 &&
			(_SliceOptions[j].End == RangeEndPos || _SliceOptions[j].End == _MyShape[i]))
			continue;

		SizeType SliceBeginPos, SliceStep, SliceEndPos;

		if (_SliceOptions[j].Begin == _SliceOptions[j].Step && _SliceOptions[j].Begin == _SliceOptions[j].End)
		{
			SliceBeginPos = CalcIndex(_SliceOptions[j].Step, _MyShape[i], false);
			SliceStep = 1;
			SliceEndPos = SliceBeginPos + 1;
		}
		else
		{
			SliceStep = _SliceOptions[j].Step;
			if (SliceStep == 0)
				_D_Dragonian_Lib_Throw_Exception("SliceStep Should Not Be Zero!");
			SliceBeginPos = CalcIndex(_SliceOptions[j].Begin, _MyShape[i], false);
			SliceEndPos = CalcEndPos(_SliceOptions[j].End, _MyShape[i], false);
		}

		const auto SliceLength = SliceEndPos - SliceBeginPos;
		if (SliceLength == 0)
			_D_Dragonian_Lib_Throw_Exception("SliceLength could not be zero!");
		if (SliceLength / SliceStep < 0)
			_D_Dragonian_Lib_Throw_Exception("Shape could not be negative!");
		const auto SlicedShape = Ceil(abs(SliceLength), abs(SliceStep));

		Ret._MyData += SliceBeginPos * Ret._MyViewStride[i];
		Ret._MyShape[i] = SlicedShape;
		Ret._MyViewStride[i] *= SliceStep;
	}

	_D_Dragonian_Lib_Auto_Grad(ReverseSlice, *this, _SliceOptions, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Permute(
	const Dimensions<_NRank>& _PremuteOrder
) const
{
	ThrowOnNotEnabled();
	if (_MyShape.Empty() || _PremuteOrder.Size() != _MyShape.Size())
		_D_Dragonian_Lib_Throw_Exception("N_DIMS MisMatch!");
	Tensor Ret = View();
	Dimensions<_NRank> TransposedDims = _PremuteOrder;
	std::ranges::sort(TransposedDims);
	if (TransposedDims[0] != 0)
		_D_Dragonian_Lib_Throw_Exception("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");
	for (size_t i = 1; i < TransposedDims.Size(); ++i)
		if (TransposedDims[i] != TransposedDims[i - 1] + 1)
			_D_Dragonian_Lib_Throw_Exception("DPremute Must Have [0, 1, ... , N_DIMS - 1]!");

	for (size_t i = 0; i < _PremuteOrder.Size(); ++i)
	{
		Ret._MyShape[i] = _MyShape[_PremuteOrder[i]];
		Ret._MyViewStride[i] = _MyViewStride[_PremuteOrder[i]];
	}

	_D_Dragonian_Lib_Auto_Grad(Permute, *this, _PremuteOrder, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Transpose(
	SizeType _Axis1,
	SizeType _Axis2
) const
{
	ThrowOnNotEnabled();
	const auto AxisCount = (SizeType)_MyShape.Size();
	_Axis1 = CalcIndex(_Axis1, AxisCount);
	_Axis2 = CalcIndex(_Axis2, AxisCount);
	Tensor Ret = View();
	if (_Axis1 == _Axis2)
		return Ret;
	Ret._MyShape[_Axis2] = _MyShape[_Axis1];
	Ret._MyViewStride[_Axis2] = _MyViewStride[_Axis1];
	Ret._MyShape[_Axis1] = _MyShape[_Axis2];
	Ret._MyViewStride[_Axis1] = _MyViewStride[_Axis2];

	_D_Dragonian_Lib_Auto_Grad(Transpose, *this, _Axis1, _Axis2, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::AxisFromTo(
	SizeType _Begin,
	SizeType _End
) const
{
	ThrowOnNotEnabled();
	const auto AxisCount = (SizeType)_MyShape.Size();
	_Begin = CalcIndex(_Begin, AxisCount);
	_End = CalcIterator(_End, AxisCount);
	if (_Begin > _End)
		_D_Dragonian_Lib_Throw_Exception("Begin Should Not Be Greater Than End!");
	Tensor Ret = View();
	for (SizeType i = _Begin; i < _End - 1; ++i)
	{
		std::swap(Ret._MyShape[i], Ret._MyShape[i + 1]);
		std::swap(Ret._MyViewStride[i], Ret._MyViewStride[i + 1]);
	}
	_D_Dragonian_Lib_Auto_Grad(ShiftAxis, *this, _Axis1, _Axis2, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::UnSqueeze(
	SizeType _Dim
) const
{
	ThrowOnNotEnabled();
	Tensor<_TensorType, _NRank + 1, _MyDevice> Ret;
	_Dim = CalcIterator(_Dim, Rank());
	const auto _Value = _Dim == Rank() ? 1 : _MyViewStride[_Dim] * _MyShape[_Dim];
	Ret._MyShape = _MyShape.Insert(1, _Dim);
	Ret._MyViewStride = _MyViewStride.Insert(_Value, _Dim);
	Ret._MyFirst = _MyFirst;
	Ret._MyData = _MyData;
	Ret._MyLast = _MyLast;
	Ret._MyFuturesAsResult = _MyFuturesAsResult;
	Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
	Ret._MyAllocator = _MyAllocator;
	Ret._IgnoreDep = _IgnoreDep;
	Ret._MyGraph = _MyGraph;
	Ret._MyFunction = _MyFunction;

	_D_Dragonian_Lib_Auto_Grad(UnSqueeze, *this, _Dim, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Squeeze(
	SizeType _Dim
) const
{
	ThrowOnNotEnabled();
	if constexpr (_NRank == 1)
		return View();
	else
	{
		_Dim = CalcIndex(_Dim, SizeType(_MyShape.Size()));
		if (_MyShape[_Dim] != 1)
			_D_Dragonian_Lib_Throw_Exception("The Shape Of Dim Must Be 1!");
		Tensor<_TensorType, _NRank - 1, _MyDevice> Ret;
		Ret._MyShape = _MyShape.Erase(_Dim);
		Ret._MyViewStride = _MyViewStride.Erase(_Dim);
		Ret._MyFirst = _MyFirst;
		Ret._MyData = _MyData;
		Ret._MyLast = _MyLast;
		Ret._MyFuturesAsResult = _MyFuturesAsResult;
		Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
		Ret._MyAllocator = _MyAllocator;
		Ret._IgnoreDep = _IgnoreDep;
		Ret._MyGraph = _MyGraph;
		Ret._MyFunction = _MyFunction;

		_D_Dragonian_Lib_Auto_Grad(Squeeze, *this, _Dim, Ret);
		return Ret;
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <size_t _TRank>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::View(
	const Dimensions<_TRank>& _ViewShape
) const
{
	ThrowOnNotEnabled();

	if (!IsContiguous())
		_D_Dragonian_Lib_Throw_Exception("View Should Be Contiguous!");
	const auto DynamicCount = std::ranges::count(_ViewShape, -1);
	if (DynamicCount > 1)
		_D_Dragonian_Lib_Throw_Exception("Count Of Dynamic Axis Should Be <= 1!");
	bool HasDynamic = DynamicCount == 1;
	for (const auto i : _ViewShape)
		if (i <= 0 && i != -1)
			_D_Dragonian_Lib_Throw_Exception("Count Of Size Should Be Greater Than 0 Or Equal -1 (Dynamic Axis)!");

	const auto SrcSize = _MyShape.Multiply();
	const auto DstSize = std::abs(_ViewShape.Multiply());

	const auto Remainder = SrcSize % DstSize;
	const auto DynamicAxes = SrcSize / DstSize;

	if (Remainder || (!HasDynamic && DynamicAxes != 1))
		_D_Dragonian_Lib_Throw_Exception("Could Not View The Tensor With Size["
			+ std::to_string(SrcSize) + "] To Size[" + std::to_string(DstSize) + "]!");

	Tensor<_TensorType, _TRank, _MyDevice> Ret;
	Ret._MyShape = _ViewShape;
	if (DynamicAxes >= 1)
		*std::ranges::find(Ret._MyShape, -1) = DynamicAxes;
	auto _Begin = Ret._MyViewStride.ReversedBegin();
	const auto _End = Ret._MyViewStride.ReversedEnd();
	auto _Iter = Ret._MyShape.ReversedBegin();
	*_Begin-- = 1;
	while (_Begin != _End)
	{
		*_Begin = *(_Begin + 1) * *_Iter--;
		--_Begin;
	}

	Ret._MyFirst = _MyFirst;
	Ret._MyData = _MyData;
	Ret._MyLast = _MyLast;
	Ret._MyFuturesAsResult = _MyFuturesAsResult;
	Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
	Ret._MyAllocator = _MyAllocator;
	Ret._IgnoreDep = _IgnoreDep;
	Ret._MyGraph = _MyGraph;
	Ret._MyFunction = _MyFunction;

	_D_Dragonian_Lib_Auto_Grad(View, *this, _ViewShape, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
template <typename _Type>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::ViewAs() const requires (std::is_trivially_copy_assignable_v<_Type> && (bool(sizeof(_Type) % sizeof(ValueType)) || bool(sizeof(ValueType) % sizeof(_Type))))
{
	ThrowOnNotEnabled();
	if (!IsContiguous())
		_D_Dragonian_Lib_Throw_Exception("ViewAs Should Be Contiguous!");

	const auto TailShape = _MyShape.Back();
	const auto TailSize = size_t(TailShape) * sizeof(ValueType);
	if (TailSize % sizeof(_Type))
		_D_Dragonian_Lib_Throw_Exception("Could not view as this type!");
	const auto NewTailShape = SizeType(TailSize / sizeof(_Type));

	using RetType = Tensor<_Type, _NRank, _MyDevice>;
	RetType Ret;
	Ret._MyFirst = _MyFirst;
	Ret._MyLast = RetType::RawPointer(_MyLast);
	Ret._MyData = RetType::RawPointer(_MyData);
	Ret._MyShape = _MyShape;
	Ret._MyShape.Back() = NewTailShape;
	Ret._MyFuturesAsResult = _MyFuturesAsResult;
	Ret._MyFuturesAsArgument = _MyFuturesAsArgument;
	Ret._MyAllocator = _MyAllocator;
	Ret._IgnoreDep = _IgnoreDep;
	Ret._MyGraph = _MyGraph;
	Ret._MyFunction = _MyFunction;
	Ret.ConstructViewInfo(Ret._MyShape);

	_D_Dragonian_Lib_Auto_Grad(ViewAs, *this, Ret);
	return Ret;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
decltype(auto) Tensor<_TensorType, _NRank, _MyDevice>::Reverse(SizeType _Axis) const
{
	ThrowOnNotEnabled();
	_Axis = CalcIndex(_Axis, Rank());
	auto Ret = View();
	Ret._MyData += (_MyShape[_Axis] - 1) * _MyViewStride[_Axis];
	Ret._MyViewStride[_Axis] = -_MyViewStride[_Axis];
	_D_Dragonian_Lib_Auto_Grad(Reverse, *this, _Axis, Ret);
	return Ret;
}

_D_Dragonian_Lib_Space_End
