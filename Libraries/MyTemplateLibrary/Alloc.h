/**
 * FileName: Alloc.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/
#pragma once
#include "Util.h"

_D_Dragonian_Lib_Template_Library_Space_Begin

class BaseAllocator
{
public:
	friend class MemoryProvider;

	virtual ~BaseAllocator() {}
	BaseAllocator() {}
	BaseAllocator(const BaseAllocator&) = default;
	BaseAllocator(BaseAllocator&&) = default;
	BaseAllocator& operator=(const BaseAllocator&) = default;
	BaseAllocator& operator=(BaseAllocator&&) = default;

	static Device GetDevice() { return Device::CUSTOM; }
	virtual void* allocate(size_t _Size);
	virtual void deallocate(void* _Block);
};

class CPUAllocator
{
public:
	friend class MemoryProvider;
	~CPUAllocator() = default;
	CPUAllocator() = default;
	CPUAllocator(const CPUAllocator&) = default;
	CPUAllocator(CPUAllocator&&) = default;
	CPUAllocator& operator=(const CPUAllocator&) = default;
	CPUAllocator& operator=(CPUAllocator&&) = default;

	static void* allocate(size_t _Size);
	static void deallocate(void* _Block);
	static Device GetDevice() { return Device::CPU; }
};

class CudaAllocator
{
public:
	friend class MemoryProvider;
	~CudaAllocator() = default;
	CudaAllocator() = default;
	CudaAllocator(const CudaAllocator&) = default;
	CudaAllocator(CudaAllocator&&) = default;
	CudaAllocator& operator=(const CudaAllocator&) = default;
	CudaAllocator& operator=(CudaAllocator&&) = default;

	static void* allocate(size_t _Size);
	static void deallocate(void* _Block);
	static Device GetDevice() { return Device::CUDA; }
};

class RocmAllocator
{
public:
	friend class MemoryProvider;
	~RocmAllocator() = default;
	RocmAllocator() = default;
	RocmAllocator(const RocmAllocator&) = default;
	RocmAllocator(RocmAllocator&&) = default;
	RocmAllocator& operator=(const RocmAllocator&) = default;
	RocmAllocator& operator=(RocmAllocator&&) = default;

	static void* allocate(size_t _Size);
	static void deallocate(void* _Block);
	static Device GetDevice() { return Device::HIP; }
};

class DmlAllocator
{
public:
	friend class MemoryProvider;
	~DmlAllocator() = default;
	DmlAllocator() = default;
	DmlAllocator(const DmlAllocator&) = default;
	DmlAllocator(DmlAllocator&&) = default;
	DmlAllocator& operator=(const DmlAllocator&) = default;
	DmlAllocator& operator=(DmlAllocator&&) = default;

	static void* allocate(size_t _Size);
	static void deallocate(void* _Block);
	static Device GetDevice() { return Device::DIRECTX; }
};

class CustomAllocator
{
public:
	friend class MemoryProvider;
	~CustomAllocator() {}
	CustomAllocator(const CustomAllocator&) = default;
	CustomAllocator(CustomAllocator&&) = default;
	CustomAllocator& operator=(const CustomAllocator&) = default;
	CustomAllocator& operator=(CustomAllocator&&) = default;
	static Device GetDevice() { return Device::CUSTOM; }

protected:
	std::shared_ptr<BaseAllocator> _MyAlloc = nullptr;

public:
	CustomAllocator(const std::shared_ptr<BaseAllocator>& _Alloc) : _MyAlloc(_Alloc) {}
	CustomAllocator() = default;
	void* allocate(size_t _Size) const
	{
		if (!_MyAlloc)
			_D_Dragonian_Lib_Throw_Exception("Bad Alloc!");
		return _MyAlloc->allocate(_Size);
	}
	void deallocate(void* _Block) const
	{
		if (!_MyAlloc)
			_D_Dragonian_Lib_Throw_Exception("Bad Alloc!");
		_MyAlloc->deallocate(_Block);
	}
};

template <Device _Type>
struct GetAllocatorType__
{
	using Type = CustomAllocator;
};

template <>
struct GetAllocatorType__<Device::CPU>
{
	using Type = CPUAllocator;
};

template <>
struct GetAllocatorType__<Device::CUDA>
{
	using Type = CudaAllocator;
};

template <>
struct GetAllocatorType__<Device::HIP>
{
	using Type = RocmAllocator;
};

template <>
struct GetAllocatorType__<Device::DIRECTX>
{
	using Type = DmlAllocator;
};

template <Device _Type>
using GetAllocatorType = typename GetAllocatorType__<_Type>::Type;

_D_Dragonian_Lib_Template_Library_Space_End