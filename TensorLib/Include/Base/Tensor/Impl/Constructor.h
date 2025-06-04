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
 * @brief Constructor for Tensor
 * @changes
 *  > 2025/6/4 NaruseMioShirakana Created <
 */

#pragma once 
#include "TensorLib/Include/Base/Tensor/Impl/Tensor.h"

_D_Dragonian_Lib_Space_Begin

template <typename _TensorType, size_t _NRank, Device _MyDevice>
_D_Dragonian_Lib_Constexpr_Force_Inline bool Tensor<_TensorType, _NRank, _MyDevice>::AllocateMemory(
	const Dimensions<_NRank>& MyShape,
	Allocator MyAlloc
)
{
	if (MyShape.Empty())
		return false;
	const auto Size = MyShape.Multiply();
	_MyFirst = Pointer(
		MyAlloc.allocate(std::max(Size * sizeof(ValueType), 256ull)),
		[MyAlloc, Size](void* _Pointer)
		{
			auto _DataPointer = static_cast<ValueType*>(_Pointer);
			TemplateLibrary::ImplDestroyRange(_DataPointer, _DataPointer + Size);
			MyAlloc.deallocate(_Pointer);
		}
	);
	_MyAllocator = MyAlloc;
	_MyData = (RawPointer)_MyFirst.get();
	_MyLast = _MyData + Size;
	return true;
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
_D_Dragonian_Lib_Constexpr_Force_Inline void Tensor<_TensorType, _NRank, _MyDevice>::ConstructViewInfo(
	const Dimensions<_NRank>& MyShape
)
{
	_MyShape = MyShape;
	auto _Begin = _MyViewStride.ReversedBegin();
	const auto _End = _MyViewStride.ReversedEnd();
	auto _Iter = _MyShape.ReversedBegin();
	*_Begin-- = 1;
	while (_Begin != _End)
	{
		*_Begin = *(_Begin + 1) * *_Iter--;
		--_Begin;
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
_D_Dragonian_Lib_Constexpr_Force_Inline void Tensor<_TensorType, _NRank, _MyDevice>::ReConstructViewInfo()
{
	auto _Begin = _MyViewStride.ReversedBegin();
	const auto _End = _MyViewStride.ReversedEnd();
	auto _Iter = _MyShape.ReversedBegin();
	*_Begin-- = 1;
	while (_Begin != _End)
	{
		*_Begin = *(_Begin + 1) * *_Iter--;
		--_Begin;
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
Tensor<_TensorType, _NRank, _MyDevice>::Tensor(
	const Dimensions<_NRank>& MyShape, Allocator Alloc
) requires (std::is_trivially_copy_assignable_v<ValueType> || std::is_default_constructible_v<ValueType>)
	: _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType), _IgnoreDep(std::make_shared<bool>(false))
{
	if (AllocateMemory(MyShape, Alloc))
	{
		ConstructViewInfo(MyShape);
		if constexpr (!std::is_trivially_copy_assignable_v<ValueType> && std::is_default_constructible_v<ValueType>)
		{
			auto IterData = _MyData;
			while (IterData != _MyLast)
				TemplateLibrary::ImplConstructAt(*IterData++);
		}
	}
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
Tensor<_TensorType, _NRank, _MyDevice>::Tensor(
	const Dimensions<_NRank>& MyShape,
	ValueType* Buffer,
	size_t BufferSize,
	Allocator Alloc
) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType), _IgnoreDep(std::make_shared<bool>(false))
{
	auto TSize = static_cast<size_t>(MyShape.Multiply());
	if (MyShape.Empty())
		return;
	if (BufferSize < TSize)
		_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
	if (BufferSize > TSize)
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
	_MyFirst = Pointer(
		Buffer,
		[Alloc](void* _Pointer) { Alloc.deallocate(_Pointer); }
	);
	_MyData = (RawPointer)_MyFirst.get();
	_MyLast = _MyData + BufferSize;
	_MyAllocator = Alloc;
	ConstructViewInfo(MyShape);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
Tensor<_TensorType, _NRank, _MyDevice>::Tensor(
	const Dimensions<_NRank>& MyShape,
	ValueType* Buffer,
	size_t BufferSize
) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType), _IgnoreDep(std::make_shared<bool>(false))
{
	auto TSize = static_cast<size_t>(MyShape.Multiply());
	if (MyShape.Empty())
		return;
	if (BufferSize < TSize)
		_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
	if (BufferSize > TSize)
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
	_MyFirst = Pointer(
		Buffer,
		[](void*) {}
	);
	_MyData = (RawPointer)_MyFirst.get();
	_MyLast = _MyData + BufferSize;
	_MyAllocator = Allocator();
	ConstructViewInfo(MyShape);
}

template <typename _TensorType, size_t _NRank, Device _MyDevice>
Tensor<_TensorType, _NRank, _MyDevice>::Tensor(
	const Dimensions<_NRank>& MyShape,
	const Pointer& Buffer,
	size_t BufferSize
) : _MyFuturesAsResult(new DependencyChainType), _MyFuturesAsArgument(new DependencyChainType), _IgnoreDep(std::make_shared<bool>(false))
{
	auto TSize = static_cast<size_t>(MyShape.Multiply());
	if (MyShape.Empty())
		return;
	if (BufferSize < TSize)
		_D_Dragonian_Lib_Throw_Exception("Buffer Size MisMatch!");
	if (BufferSize > TSize)
		_D_Dragonian_Lib_Namespace GetDefaultLogger()->Log(L"Buffer Size Is Greater Than Elememt Count, This Could Cause Undefined Behavior!", Logger::LogLevel::Warn);
	_MyFirst = Buffer;
	_MyData = (RawPointer)_MyFirst.get();
	_MyLast = _MyData + BufferSize;
	_MyAllocator = Allocator();
	ConstructViewInfo(MyShape);
}

_D_Dragonian_Lib_Space_End
