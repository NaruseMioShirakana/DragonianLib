#include "../Alloc.h"
#include <exception>
#include <memory>
#ifdef DRAGONIANLIB_ENABLECUDA
#include <cuda_runtime.h>
#endif

_D_Dragonian_Lib_Template_Library_Space_Begin

void* BaseAllocator::allocate(size_t _Size)
{
	_D_Dragonian_Lib_Throw_Exception("Bad Alloc!");
}

void BaseAllocator::deallocate(void* _Block)
{
	_D_Dragonian_Lib_Throw_Exception("Bad Alloc!");
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

void* CudaAllocator::allocate(size_t _Size)
{
#ifdef DRAGONIANLIB_ENABLECUDA
	void* _Block = nullptr;
	if (cudaMalloc(&_Block, _Size))
		_D_Dragonian_Lib_CUDA_Error;
	return _Block;
#else
	_D_Dragonian_Lib_Throw_Exception("CUDA Not Enabled!");
#endif
}

void CudaAllocator::deallocate(void* _Block)
{
#ifdef DRAGONIANLIB_ENABLECUDA
	if (cudaFree(_Block))
		_D_Dragonian_Lib_CUDA_Error;
#endif
}

void* DmlAllocator::allocate(size_t _Size)
{
	_D_Dragonian_Lib_Throw_Exception("DirectX Memory Not Supported!");
}

void DmlAllocator::deallocate(void* _Block)
{
	_D_Dragonian_Lib_Throw_Exception("DirectX Memory Not Supported!");
}

void* RocmAllocator::allocate(size_t _Size)
{
	_D_Dragonian_Lib_Throw_Exception("ROCm Memory Not Supported!");
}

void RocmAllocator::deallocate(void* _Block)
{
	_D_Dragonian_Lib_Throw_Exception("ROCm Memory Not Supported!");
}

_D_Dragonian_Lib_Template_Library_Space_End