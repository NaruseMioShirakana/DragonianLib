#include "Alloc.h"
#include <exception>

namespace libsvc
{
	MemoryProvider::MemoryProvider()
	{
		_Provider[0] = new libsvcstd::CPUAllocator;
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
		return _Instance._Provider[size_t(_Device)];
	}
}

LIBSVCSTLBEGIN

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
#if LIBSVC_ALLOC_ALIG > 1
#if _MSC_VER
	return (unsigned char*)_aligned_malloc(_Size, LIBSVC_ALLOC_ALIG);
#else
	auto _Block = (unsigned char*)malloc(_Size + LIBSVC_ALLOC_ALIG * 2);
	if (size_t(_Block) % LIBSVC_ALLOC_ALIG == 0)
		return _Block;
	*(_Block++) = 1ui8;
	while (size_t(_Block) % LIBSVC_ALLOC_ALIG)
		*(_Block++) = 0ui8;
	return _Block;
#endif
#else
	return (unsigned char*)malloc(_Size);
#endif
}

void CPUAllocator::Free(void* _Block)
{
#if LIBSVC_ALLOC_ALIG > 1
#if _MSC_VER
	_aligned_free(_Block);
#else
	auto Block = (unsigned char*)_Block;
	while (*Block != 1ui8)
		--Block;
	free(Block);
#endif
#else
	free(_Block);
#endif
}

LIBSVCSTLEND