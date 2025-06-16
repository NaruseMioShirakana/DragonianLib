#include "cuda_runtime.h"
#include "kernel.h"

namespace DragonianLib
{
	namespace CudaProvider
	{
		void* cudaAllocate(size_t size)
		{
			void* block = nullptr;
			if (auto err = ::cudaMalloc(&block, size))
				fprintf(stderr, "%s\n", cudaGetErrorString(err));
			return block;
		}

		int cudaFree(void* block)
		{
			return ::cudaFree(block);
		}

		int host2Device(void* dst, const void* src, size_t size, stream_t stream)
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
		}

		int device2Host(void* dst, const void* src, size_t size, stream_t stream)
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
		}

		int device2Device(void* dst, const void* src, size_t size, stream_t stream)
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
		}

		stream_t createCudaStream()
		{
			cudaStream_t Ret;
			if (auto err = cudaStreamCreate(&Ret))
				fprintf(stderr, "%s\n", cudaGetErrorString(err));
			return stream_t(Ret);
		}

		int destoryCudaStream(stream_t stream)
		{
			return cudaStreamDestroy((cudaStream_t)stream);
		}

		int asyncCudaStream(stream_t stream)
		{
			return cudaStreamSynchronize((cudaStream_t)stream);
		}

		const char* getCudaError(int errorId)
		{
			return cudaGetErrorString(static_cast<cudaError_t>(errorId));
		}

		int getLastError()
		{
			return cudaGetLastError();
		}
	}
}
