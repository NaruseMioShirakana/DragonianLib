#include "../Alloc.h"
#include <exception>
#include <memory>

namespace DragonianLib
{
	static std::shared_ptr<DragonianLibSTL::BaseAllocator> _Provider[8]
	{
		std::make_shared<DragonianLibSTL::CPUAllocator>(),
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr,
		nullptr
	};

	Allocator GetMemoryProvider(Device _Device)
	{
		return _Provider[size_t(_Device)].get();
	}
}

_D_Dragonian_Lib_Template_Library_Space_Begin

unsigned char* BaseAllocator::Allocate(size_t _Size)
{
	throw std::bad_alloc();
}

void BaseAllocator::Free(void* _Block)
{
	throw std::bad_alloc();
}

void* BaseAllocator::allocate(size_t _Size)
{
	throw std::bad_alloc();
}

void BaseAllocator::deallocate(void* _Block)
{
	throw std::bad_alloc();
}

Device BaseAllocator::GetDevice() const
{
	return Type_;
}

unsigned char* CPUAllocator::Allocate(size_t _Size)
{
#if DRAGONIANLIB_ALLOC_ALIG > 1
#if _MSC_VER
	return (unsigned char*)_aligned_malloc(_Size, DRAGONIANLIB_ALLOC_ALIG);
#else
	auto _Block = (unsigned char*)malloc(_Size + DRAGONIANLIB_ALLOC_ALIG * 2);
	if (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG == 0)
		return _Block;
	*(_Block++) = 1ui8;
	while (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG)
		*(_Block++) = 0ui8;
	return _Block;
#endif
#else
	return (unsigned char*)malloc(_Size);
#endif
}

void* CPUAllocator::allocate(size_t _Size)
{
#if DRAGONIANLIB_ALLOC_ALIG > 1
#if _MSC_VER
	return (unsigned char*)_aligned_malloc(_Size, DRAGONIANLIB_ALLOC_ALIG);
#else
	auto _Block = (unsigned char*)malloc(_Size + DRAGONIANLIB_ALLOC_ALIG * 2);
	if (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG == 0)
		return _Block;
	*(_Block++) = 1ui8;
	while (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG)
		*(_Block++) = 0ui8;
	return _Block;
#endif
#else
	return (unsigned char*)malloc(_Size);
#endif
}

void CPUAllocator::Free(void* _Block)
{
#if DRAGONIANLIB_ALLOC_ALIG > 1
#if _MSC_VER
	_aligned_free(_Block);
#else
	if (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG == 0)
	{
		free(_Block);
		return;
	}
	auto Block = (unsigned char*)_Block;
	--Block;
	while (*Block != 1ui8)
		--Block;
	free(Block);
#endif
#else
	free(_Block);
#endif
}

void CPUAllocator::deallocate(void* _Block)
{
#if DRAGONIANLIB_ALLOC_ALIG > 1
#if _MSC_VER
	_aligned_free(_Block);
#else
	if (size_t(_Block) % DRAGONIANLIB_ALLOC_ALIG == 0)
	{
		free(_Block);
		return;
	}
	auto Block = (unsigned char*)_Block;
	--Block;
	while (*Block != 1ui8)
		--Block;
	free(Block);
#endif
#else
	free(_Block);
#endif
}

_D_Dragonian_Lib_Template_Library_Space_End