#include "Alloc.h"
#include <exception>

namespace DragonianLib
{
	static Allocator _Provider[8];

	MemoryProvider::MemoryProvider()
	{
		_Provider[0] = new DragonianLibSTL::CPUAllocator;
		_Provider[1] = nullptr;
		_Provider[2] = nullptr;
		_Provider[3] = nullptr;
		_Provider[4] = nullptr;
		_Provider[5] = nullptr;
		_Provider[6] = nullptr;
		_Provider[7] = nullptr;
	}

	MemoryProvider::~MemoryProvider()
	{
		for (auto i : _Provider)
			delete i;
	}

	Allocator GetMemoryProvider(Device _Device)
	{
		static MemoryProvider _Instance;
		return _Provider[size_t(_Device)];
	}
}

DRAGONIANLIBSTLBEGIN

unsigned char* BaseAllocator::Allocate(size_t _Size)
{
	throw std::bad_alloc();
}

void BaseAllocator::Free(void* _Block)
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

void CPUAllocator::Free(void* _Block)
{
#if DRAGONIANLIB_ALLOC_ALIG > 1
#if _MSC_VER
	_aligned_free(_Block);
#else
	auto Block = (unsigned char*)_Block;
	--Block;
	if (*Block != 0ui8 && *Block != 1ui8)
		free(_Block);
	while (*Block != 1ui8)
		--Block;
	free(Block);
#endif
#else
	free(_Block);
#endif
}

DRAGONIANLIBSTLEND