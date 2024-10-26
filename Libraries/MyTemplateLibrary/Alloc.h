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
#define _D_Dragonian_Lib_Template_Library_Space_Begin namespace DragonianLibSTL {
#define _D_Dragonian_Lib_Template_Library_Space_End }

_D_Dragonian_Lib_Template_Library_Space_Begin

class BaseAllocator;
using Allocator = BaseAllocator*;

_D_Dragonian_Lib_Template_Library_Space_End

namespace DragonianLib
{
	static inline size_t NopID = size_t(-1);
	using DragonianLibSTL::Allocator;

	enum class Device
	{
		CPU = 0,
		CUDA,
		HIP,
		DIRECTX,
	};

	class MemoryProvider
	{
	public:
		~MemoryProvider();
		friend Allocator GetMemoryProvider(Device _Device);
		MemoryProvider(const MemoryProvider&) = delete;
		MemoryProvider(MemoryProvider&&) = delete;
		MemoryProvider& operator=(const MemoryProvider&) = delete;
		MemoryProvider& operator=(MemoryProvider&&) = delete;

	private:
		MemoryProvider();
	};

	Allocator GetMemoryProvider(Device _Device);
}

_D_Dragonian_Lib_Template_Library_Space_Begin
using DragonianLib::Device;
using DragonianLib::MemoryProvider;
using DragonianLib::GetMemoryProvider;
using DragonianLib::NopID;

class BaseAllocator
{
public:
	friend class MemoryProvider;
	virtual ~BaseAllocator() {}
	BaseAllocator(const BaseAllocator&) = default;
	BaseAllocator(BaseAllocator&&) = default;
	BaseAllocator& operator=(const BaseAllocator&) = default;
	BaseAllocator& operator=(BaseAllocator&&) = default;
	virtual unsigned char* Allocate(size_t _Size);
	virtual void Free(void* _Block);
	Device GetDevice() const;
protected:
	BaseAllocator(Device _Type) : Type_(_Type) {}
	Device Type_;
};

class CPUAllocator : public BaseAllocator
{
public:
	friend class MemoryProvider;
	~CPUAllocator() override {}
	CPUAllocator(const CPUAllocator&) = default;
	CPUAllocator(CPUAllocator&&) = default;
	CPUAllocator& operator=(const CPUAllocator&) = default;
	CPUAllocator& operator=(CPUAllocator&&) = default;
	unsigned char* Allocate(size_t _Size) override;
	void Free(void* _Block) override;
protected:
	CPUAllocator() : BaseAllocator(Device::CPU) {}
};

_D_Dragonian_Lib_Template_Library_Space_End