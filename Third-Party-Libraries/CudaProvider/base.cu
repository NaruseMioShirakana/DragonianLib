#include <chrono>
#include <device_launch_parameters.h>

#include "base.h"

#include "cublas_v2.h"

namespace DragonianLib
{
    namespace CudaModules
    {
        class Timer  // NOLINT(cppcoreguidelines-special-member-functions)
        {
        public:
            Timer(std::string name, const handle_t* handle = nullptr) : Handle(handle), Name(std::move(name)), Start(std::chrono::high_resolution_clock::now()) {}
            ~Timer()
            {
                if (Handle && *Handle) CudaProvider::asyncCudaStream(getHandleStream(*Handle));
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - Start).count();
                printf("%s: %lld us\n", Name.c_str(), duration);
            }

        private:
            const handle_t* Handle;
            std::string Name;
            std::chrono::high_resolution_clock::time_point Start;
        };

        handle_t createHandle()
        {
            cublasHandle_t Handle;
            if (auto Ret = cublasCreate(&Handle))
                fprintf(stderr, "%s\n", cublasGetStatusString(Ret));
            cublasSetMathMode(Handle, CUBLAS_TF32_TENSOR_OP_MATH);
            return handle_t(Handle);
        }

        layerStatus_t destoryHandle(handle_t handle)
        {
            return static_cast<layerStatus_t>(cublasDestroy(cublasHandle_t(handle)));
        }

        const char* getErrorString(layerStatus_t errorId)
        {
            if (errorId == LAYER_STATUS_SIZE_MISMATCH)
                return "Input size mismatch!";
            return cublasGetStatusString(static_cast<cublasStatus_t>(errorId));
        }

        layerStatus_t setHandleStream(handle_t handle, stream_t stream)
        {
            return static_cast<layerStatus_t>(cublasSetStream((cublasHandle_t)handle, (cudaStream_t)stream));
        }

        stream_t getHandleStream(handle_t handle)
        {
            cudaStream_t stream;
            if (auto Ret = cublasGetStream((cublasHandle_t)handle, &stream))
                fprintf(stderr, "%s\n", cublasGetStatusString(Ret));
            return stream_t(stream);
        }

        Module::Module(Module* parent, const std::string& name)
        {
            if (parent)
            {
                if (parent->Name.empty())
                    Name = name;
                else
                    Name = parent->Name + '.' + name;
                parent->Children.emplace_back(this);
            }
            else
                Name = name;
        }

        void Module::LoadModel(DictType& dict)
        {
            for (auto layer : Children)
                layer->LoadModel(dict);
        }

        void Parameter::LoadModel(DictType& dict)
        {
            auto weight = dict.find(Name);
            if (weight != dict.end())
            {
                if (Strict)
                {
                    if (weight->second.N != TensorData.N ||
                        weight->second.C != TensorData.C ||
                        weight->second.H != TensorData.H ||
                        weight->second.W != TensorData.W)
                        throw std::runtime_error("Parameter " + Name + " shape mismatch in model dictionary.");
                }
                TensorData = std::move(weight->second);
            }
            else
                throw std::runtime_error("Parameter " + Name + " not found in model dictionary.");
        }

        static __global__ void broadcastAddKernel(float* A, const float* bias, unsigned YL, unsigned XL)
    	{
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float broadcastAddSharedBias[];

            if (threadIdx.y == 0 && x < XL)
                broadcastAddSharedBias[threadIdx.x] = bias[x];

            __syncthreads();

            if (y < YL && x < XL)
                A[y * XL + x] += broadcastAddSharedBias[threadIdx.x];
        }

        template <unsigned blockSizeX>
		static __global__ void layerReduceMeanKernel(
			const float* iFeat,
			float* oMean,
			unsigned featureSize
		)
		{
			//shape: [sampleCount, featureSize] -> [[gridDim.y, blockDim.y], [gridDim.x, blockDim.x]]
			//idx = sampleIdx * featureSize + featureIdx
			const unsigned by = blockIdx.y;
			const unsigned tx = threadIdx.x;

			const unsigned featureIdx = blockIdx.x * blockDim.x + tx;

			extern __shared__ float sharedReduceMeanData[];

			sharedReduceMeanData[tx] = 0.f;
			if (featureIdx >= featureSize)
				return;

			sharedReduceMeanData[tx] = iFeat[by * featureSize + featureIdx];

			__syncthreads();

			if constexpr (blockSizeX >= 1024)
			{
				if (tx < 512) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 512];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 512)
			{
				if (tx < 256) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 256];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 256)
			{
				if (tx < 128) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 128];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 128)
			{
				if (tx < 64) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 64];
				__syncthreads();
			}
			if (tx < 32)
			{
				if constexpr (blockSizeX >= 64) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 32];
				if constexpr (blockSizeX >= 32) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 16];
				if constexpr (blockSizeX >= 16) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 8];
				if constexpr (blockSizeX >= 8) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 4];
				if constexpr (blockSizeX >= 4) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 2];
				if constexpr (blockSizeX >= 2) sharedReduceMeanData[tx] += sharedReduceMeanData[tx + 1];
				if (tx == 0) atomicAdd(oMean + by, sharedReduceMeanData[0] / float(featureSize));
			}
		}

		template <unsigned blockSizeX>
		static __global__ void layerReduceVarKernel(
			const float* iFeat,
			const float* iMean,
			float* oVar,
			unsigned featureSize
		)
		{
			//shape: [sampleCount, featureSize] -> [[gridDim.y, blockDim.y], [gridDim.x, blockDim.x]]
			//idx = sampleIdx * featureSize + featureIdx
			const unsigned by = blockIdx.y;
			const unsigned tx = threadIdx.x;

			const unsigned featureIdx = blockIdx.x * blockDim.x + tx;

			extern __shared__ float sharedReduceVarData[];

			sharedReduceVarData[tx] = 0.f;
			if (featureIdx >= featureSize)
				return;

			{
				const float x = iFeat[by * featureSize + featureIdx] - iMean[by];
				sharedReduceVarData[tx] = x * x;
				__syncthreads();
			}

			if constexpr (blockSizeX >= 1024)
			{
				if (tx < 512) sharedReduceVarData[tx] += sharedReduceVarData[tx + 512];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 512)
			{
				if (tx < 256) sharedReduceVarData[tx] += sharedReduceVarData[tx + 256];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 256)
			{
				if (tx < 128) sharedReduceVarData[tx] += sharedReduceVarData[tx + 128];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 128)
			{
				if (tx < 64) sharedReduceVarData[tx] += sharedReduceVarData[tx + 64];
				__syncthreads();
			}
			if (tx < 32)
			{
				if constexpr (blockSizeX >= 64) sharedReduceVarData[tx] += sharedReduceVarData[tx + 32];
				if constexpr (blockSizeX >= 32) sharedReduceVarData[tx] += sharedReduceVarData[tx + 16];
				if constexpr (blockSizeX >= 16) sharedReduceVarData[tx] += sharedReduceVarData[tx + 8];
				if constexpr (blockSizeX >= 8) sharedReduceVarData[tx] += sharedReduceVarData[tx + 4];
				if constexpr (blockSizeX >= 4) sharedReduceVarData[tx] += sharedReduceVarData[tx + 2];
				if constexpr (blockSizeX >= 2) sharedReduceVarData[tx] += sharedReduceVarData[tx + 1];
				if (tx == 0) atomicAdd(oVar + by, sharedReduceVarData[0] / float(featureSize));
			}
		}

        static __global__ void implNormalizeKernel(
            float* ioFeat,
            const float* iMean,
            const float* iVar,
            unsigned featureSize,
            float eps
        )
        {
            const unsigned tx = threadIdx.x;
            unsigned x = blockIdx.x * blockDim.x + tx;

            if (x >= featureSize)
	            return;

            __shared__ float implNormalizeSharedMem[2];

            if (threadIdx.x == 0)
            {
                implNormalizeSharedMem[0] = iMean[blockIdx.y];
                implNormalizeSharedMem[1] = sqrtf(iVar[blockIdx.y] + eps);
            }

            __syncthreads();

            auto idx = blockIdx.y * featureSize + x;
            ioFeat[idx] = (ioFeat[idx] - implNormalizeSharedMem[0]) / implNormalizeSharedMem[1];
        }

		static void inplaceNorm(
			float* ioFeat,
			float* ioMean,
			float* ioVar,
			unsigned sampleCount,
			unsigned featureSize,
			float eps,
			cudaStream_t cudaStream
		)
		{
			{
                cudaMemsetAsync(ioMean, 0, sizeof(float) * sampleCount, cudaStream);
                cudaMemsetAsync(ioVar, 0, sizeof(float) * sampleCount, cudaStream);

				dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE);
				dim3 gridSize(
					(featureSize + blockSize.x - 1) / blockSize.x,
					sampleCount
				);

				constexpr auto sharedMemSize = DRAGONIANLIB_CUDA_BLOCK_SIZE * sizeof(float);

				layerReduceMeanKernel<DRAGONIANLIB_CUDA_BLOCK_SIZE><<<gridSize, blockSize, sharedMemSize, cudaStream>>>(
					ioFeat,
					ioMean,
					featureSize
					);

				layerReduceVarKernel<DRAGONIANLIB_CUDA_BLOCK_SIZE><<<gridSize, blockSize, sharedMemSize, cudaStream>>>(
					ioFeat,
					ioMean,
                    ioVar,
					featureSize
					);
			}

			dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE);
			dim3 gridSize(
				(featureSize + blockSize.x - 1) / blockSize.x,
				sampleCount
			);
			implNormalizeKernel<<<gridSize, blockSize, 8, cudaStream>>>(
				ioFeat, 
				ioMean,
                ioVar,
				featureSize,
				eps
				);
		}

        static __global__ void implAffineKernel(
            float* output,
            const float* weight,
            unsigned batchSize,
            unsigned numChannel
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float implAffineSharedMem[];

            if (y >= batchSize || x >= numChannel)
                return;

            if (threadIdx.y == 0)
                implAffineSharedMem[threadIdx.x] = weight[x];

            __syncthreads();

            output[y * numChannel + x] *= implAffineSharedMem[threadIdx.x];
        }

        static __global__ void implAffineBiasKernel(
            float* output,
            const float* weight,
            const float* bias,
            unsigned batchSize,
            unsigned numChannel
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float implAffineBiasSharedMem[];

            if (y >= batchSize || x >= numChannel)
                return;

            if (threadIdx.y == 0)
            {
                implAffineBiasSharedMem[threadIdx.x] = weight[x];
                implAffineBiasSharedMem[threadIdx.x + blockDim.x] = bias[x];
            }

            __syncthreads();

            (output[y * numChannel + x] *= implAffineBiasSharedMem[threadIdx.x]) +=
                implAffineBiasSharedMem[threadIdx.x + blockDim.x];
        }

        static __global__ void implAffineBias2DKernel(
            float* output,
            const float* weight,
            const float* bias,
            unsigned numChannel,
            unsigned featureSize
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float implAffineBias2DSharedMem[];

            if (y >= numChannel || x >= featureSize)
                return;

            if (threadIdx.x == 0)
            {
                implAffineBias2DSharedMem[threadIdx.y] = weight[y];
                implAffineBias2DSharedMem[threadIdx.y + blockDim.y] = bias[y];
            }

            __syncthreads();

            (output[(blockIdx.z * numChannel + y) * featureSize + x] *= implAffineBias2DSharedMem[threadIdx.y]) +=
                implAffineBias2DSharedMem[threadIdx.y + blockDim.y];
        }

        static __global__ void implBias2DKernel(
            float* output,
            const float* bias,
            unsigned numChannel,
            unsigned featureSize
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float implBias2DSharedMem[];

            if (y >= numChannel || x >= featureSize)
                return;

            if (threadIdx.x == 0)
	            implBias2DSharedMem[ty] = bias[y];

            __syncthreads();
            
        	output[(blockIdx.z * numChannel + y) * featureSize + x] += implBias2DSharedMem[ty];
        }

        Linear::Linear(
            Module* parent,
            const std::string& name,
            unsigned inFeatureDim,
            unsigned outFeatureDim,
            bool bias
        ) : Module(parent, name), InFeatureDim(inFeatureDim), OutFeatureDim(outFeatureDim), BiasEnabled(bias)
        {
            Weight = std::make_shared<Parameter>(
                this, "weight", Tensor<float>(OutFeatureDim, InFeatureDim)
            );
            if (BiasEnabled)
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<float>(OutFeatureDim)
                );
        }

        layerStatus_t Linear::Forward(
            const Tensor<float>& input,
            Tensor<float>& output
        ) const noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Linear " + Name, &output.Handle);
#endif

            unsigned inFeature = input.W;
            if (inFeature != InFeatureDim)
                return LAYER_STATUS_SIZE_MISMATCH;

            unsigned inputSize = input.N * input.C * input.H;
            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.H, OutFeatureDim);
            else if (input.Dim == 3)
                output.Resize(input.N, input.H, OutFeatureDim);
            else if (input.Dim == 2)
                output.Resize(input.H, OutFeatureDim);
            else
                output.Resize(OutFeatureDim);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            static constexpr float Alpha = 1.f;
            static constexpr float Beta = 0.f;

            if (auto Ret = cublasSgemm(
                cublasHandle_t(input.Handle), CUBLAS_OP_T, CUBLAS_OP_N,
                (int)OutFeatureDim, (int)inputSize, (int)InFeatureDim,
                &Alpha,
                Weight->GetTensor().Data, (int)InFeatureDim,
                input.Data, (int)InFeatureDim,
                &Beta,
                output.Data, (int)OutFeatureDim
            )) return static_cast<layerStatus_t>(Ret);

            if (BiasEnabled)
            {
                dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
                dim3 gridSize(
                    (OutFeatureDim + blockSize.x - 1) / blockSize.x,
                    (inputSize + blockSize.y - 1) / blockSize.y
                );
                unsigned sharedMemSize = blockSize.x * sizeof(float);
                broadcastAddKernel<<<gridSize, blockSize, sharedMemSize, cudaStream_t(getHandleStream(input.Handle))>>>
                    (output.Data, Bias->GetTensor().Data, inputSize, OutFeatureDim);
            }

            return LAYER_STATUS_SUCCESS;
        }

        LayerNorm1D::LayerNorm1D(
            Module* parent,
            const std::string& name,
            unsigned numChannels,
            float eps,
            bool affine,
            bool bias
        ) : Module(parent, name), NumChannels(numChannels), Epsilon(eps), BiasEnabled(bias), AffineEnabled(affine)
        {
            if (AffineEnabled)
            {
                Weight = std::make_shared<Parameter>(
                    this, "weight", Tensor<float>(numChannels)
                );
                if (BiasEnabled)
                    Bias = std::make_shared<Parameter>(
                        this, "bias", Tensor<float>(numChannels)
                    );
            }
        }

        layerStatus_t LayerNorm1D::Forward(
            // ReSharper disable once CppParameterMayBeConstPtrOrRef
            Tensor<float>& output,
            Tensor<float>& mean,
            Tensor<float>& var
        ) const noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("LayerNorm1D " + Name, &output.Handle);
#endif

            const auto featureDim = output.W;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            unsigned sampleCountNorm = output.N * output.C * output.H;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);
            if (mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(mean.Handle));
            mean.Handle = output.Handle;
            if (var.Handle) CudaProvider::asyncCudaStream(getHandleStream(var.Handle));
            var.Handle = output.Handle;

            auto Stream = cudaStream_t(getHandleStream(output.Handle));

            inplaceNorm(
                output.Data,
                mean.Data,
                var.Data,
                sampleCountNorm,
                NumChannels,
                Epsilon,
                Stream
            );

            if (AffineEnabled)
            {
                dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
                dim3 gridSize(
                    (NumChannels + blockSize.x - 1) / blockSize.x,
                    (sampleCountNorm + blockSize.y - 1) / blockSize.y
                );
                unsigned sharedMemSize = blockSize.x * 2ull * sizeof(float);
	            if (BiasEnabled)
                    implAffineBiasKernel<<<gridSize, blockSize, sharedMemSize, Stream>>>
						(output.Data, Weight->GetTensor().Data, Bias->GetTensor().Data, sampleCountNorm, NumChannels);
                else
                    implAffineKernel<<<gridSize, blockSize, sharedMemSize, Stream>>>
						(output.Data, Weight->GetTensor().Data, sampleCountNorm, NumChannels);
            }

            return LAYER_STATUS_SUCCESS;
        }

        GroupNorm1D::GroupNorm1D(
            Module* parent,
            const std::string& name,
            unsigned numGroups,
            unsigned numChannels,
            float eps,
            bool affine
        ) : Module(parent, name), NumGroups(numGroups), NumChannels(numChannels), Epsilon(eps), AffineEnabled(affine)
        {
            if (NumChannels % NumGroups)
                throw std::logic_error("NumChannels must be exactly divisible by NumGroups");

            if (AffineEnabled)
            {
                Weight = std::make_shared<Parameter>(
                    this, "weight", Tensor<float>(numChannels)
                );
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<float>(numChannels)
                );
            }
        }

        layerStatus_t GroupNorm1D::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output,
            Tensor<float>& mean,
            Tensor<float>& var
        ) const noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("GroupNorm1D " + Name, &output.Handle);
#endif

            unsigned featureDim = output.H;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            unsigned batchSize = output.N * output.C;
            unsigned featureSize = output.W;

            unsigned sampleCountNorm = batchSize * NumGroups;
            unsigned featureSizeNorm = output.H * output.W / NumGroups;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);
            if (mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(mean.Handle));
            mean.Handle = output.Handle;
            if (var.Handle) CudaProvider::asyncCudaStream(getHandleStream(var.Handle));
            var.Handle = output.Handle;

            auto Stream = cudaStream_t(getHandleStream(output.Handle));

            inplaceNorm(
                output.Data,
                mean.Data,
                var.Data,
                sampleCountNorm,
                featureSizeNorm,
                Epsilon,
                Stream
            );

            if (AffineEnabled)
            {
                dim3 blockSizeSp(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSizeSp(
                    (featureSize + blockSizeSp.x - 1) / blockSizeSp.x,
                    (NumChannels + blockSizeSp.y - 1) / blockSizeSp.y,
                    batchSize
                );
                unsigned sharedMemSize = blockSizeSp.y * 2ull * sizeof(float);
	            implAffineBias2DKernel<<<gridSizeSp, blockSizeSp, sharedMemSize, Stream>>>(
                    output.Data,
                    Weight->GetTensor().Data,
                    Bias->GetTensor().Data,
                    NumChannels,
                    featureSize
                    );
            }

            return LAYER_STATUS_SUCCESS;
        }

        static __global__ void leakyReLUKernel(
            const float* input,
            float* output,
            const float negativeSlope,
            const unsigned size)
    	{
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
                output[index] = input[index] > 0 ? input[index] : input[index] * negativeSlope;
        }

        layerStatus_t LeakyReLU::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output
        ) const noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("LeakyReLU", &output.Handle);
#endif

            auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            leakyReLUKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>
                (output.Data, output.Data, NegativeSlope, size);

            return LAYER_STATUS_SUCCESS;
        }

        Conv1D::Conv1D(
            Module* parent, const std::string& name,
            unsigned inputChannels, unsigned outputChannels, unsigned kernelSize,
            unsigned stride, unsigned padding, unsigned dilation, unsigned groups,
            bool bias
        ) : Module(parent, name), KernelSize(kernelSize), Stride(stride), Padding(padding), Dilation(dilation), Groups(groups), InputChannels(inputChannels), OutputChannels(outputChannels), BiasEnabled(bias)
        {
            Weight = std::make_shared<Parameter>(
                this, "weight", Tensor<float>(outputChannels, inputChannels / groups, kernelSize)
            );
            if (BiasEnabled)
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<float>(outputChannels)
                );
        }

        static __global__ void im2ColKernel(
            const float* input,
            float* output,
            unsigned groups,
            unsigned groupSize,
            unsigned inputLength,
            unsigned im2ColChannels,
            unsigned outputLength,
            unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned dilation
        )
        {
			//im2ColChannels = groupSize * kernelSize, groupSize = inputChannel / groups
            //[batchSize, groups, groupSize, inputLength] ->  [batchSize, groups, im2ColChannels, outputLength]
            const unsigned bg = blockIdx.z;
            const unsigned outCh = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned outPos = blockIdx.x * blockDim.x + threadIdx.x;
            if (outCh >= im2ColChannels || outPos >= outputLength)
				return;
            
            const unsigned batchIdx = bg / groups;
            const unsigned gIdx = bg % groups;

            const unsigned gcIdx = outCh / kernelSize;
            const unsigned kernelOffset = outCh % kernelSize;

			const int inPos = int(outPos * stride) - int(padding) + int(kernelOffset * dilation);
            const unsigned oPos = ((batchIdx * groups + gIdx) * im2ColChannels + outCh) * outputLength + outPos;
            const unsigned iPos = ((batchIdx * groups + gIdx) * groupSize + gcIdx) * inputLength + inPos;
            if (inPos >= 0 && inPos < int(inputLength))
                output[oPos] = input[iPos];
            else
                output[oPos] = 0.f;
        }

        layerStatus_t Conv1D::Forward(
            const Tensor<float>& input,
            Tensor<float>& output,
            Tensor<float>& col
        ) const noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Conv1D " + Name, &output.Handle);
#endif

            if (input.H != InputChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            const unsigned batchSize = input.N * input.C;
			const unsigned iGroupSize = InputChannels / Groups;
            const unsigned oGroupSize = OutputChannels / Groups;
            const unsigned inputLength = input.W;

            const unsigned outputLength = (inputLength + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1;

            output.Resize(batchSize, OutputChannels, outputLength);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            const unsigned im2ColChannels = iGroupSize * KernelSize;
            col.Resize(batchSize, Groups, outputLength, im2ColChannels);
            if (col.Handle) CudaProvider::asyncCudaStream(getHandleStream(col.Handle));
            col.Handle = input.Handle;

            cudaStream_t Stream = cudaStream_t(getHandleStream(input.Handle));

            static constexpr float Alpha = 1.f;
            static constexpr float Beta = 0.f;

            if (KernelSize == 1&&
                Stride == 1 &&
                Padding == 0 &&
                Dilation == 1 &&
                Groups == 1)
            {
				//[BatchSize, InputChannels, InputLength/OutputLength, 1]^T * [1, OutputChannels, InputChannels, 1]^T
                if (auto Ret = cublasSgemmStridedBatched(
                    cublasHandle_t(input.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)outputLength, (int)OutputChannels, (int)InputChannels,
                    &Alpha,
                    input.Data, (int)outputLength, (ptrdiff_t)InputChannels * outputLength,
                    Weight->GetTensor().Data, (int)InputChannels, 0,
                    &Beta,
					output.Data, (int)outputLength, (ptrdiff_t)OutputChannels * outputLength,
                    (int)batchSize
                )) return static_cast<layerStatus_t>(Ret);
            }
            else
            {
	            dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSize(
                    (outputLength + blockSize.x - 1) / blockSize.x,
                    (im2ColChannels + blockSize.y - 1) / blockSize.y,
                    batchSize * Groups
                );

	            im2ColKernel<<<gridSize, blockSize, 0, Stream>>>(
	                input.Data,
	                col.Data,
	                Groups,
                    iGroupSize,
	                inputLength,
	                im2ColChannels,
	                outputLength,
	                KernelSize,
	                Stride,
	                Padding,
	                Dilation
					);

				//[batchSize, Groups, Im2ColChannels, OutputLength]^T * [1, Groups, oGroupSize, Im2ColChannels]^T
				//[batchSize, InputChannels, OutputLength]^T * [1, OutputChannels, InputChannels, KernelSize]^T
				//[batchSize, Groups, oGroupSize, OutputLength]^T
                for (unsigned b = 0; b < batchSize; ++b)
                    if (auto Ret = cublasSgemmStridedBatched(
                        cublasHandle_t(col.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)outputLength, (int)oGroupSize, (int)im2ColChannels,
                        &Alpha,
                        col.Data + (ptrdiff_t)Groups * im2ColChannels * outputLength * b, (int)outputLength,
                        (ptrdiff_t)im2ColChannels * outputLength,
                        Weight->GetTensor().Data, (int)im2ColChannels, (ptrdiff_t)oGroupSize * im2ColChannels,
                        &Beta,
                        output.Data + (ptrdiff_t)OutputChannels * outputLength * b, (int)outputLength,
                        (ptrdiff_t)oGroupSize * outputLength,
                        (int)Groups
                    )) return static_cast<layerStatus_t>(Ret);
            }

            //const unsigned iGroupSize = InputChannels / Groups;
            //const unsigned oGroupSize = OutputChannels / Groups;
            //const unsigned im2ColChannels = iGroupSize * KernelSize;
            //[batchSize, Groups, outputLength, im2ColChannels] * [Groups, oGroupSize, im2ColChannels]
            //[batchSize, Groups, oGroupSize, outputLength] -> [batchSize, OutputChannels, outputLength]
            /*static constexpr float Alpha = 1.f;
            static constexpr float Beta = 0.f;
            for (unsigned b = 0; b < 1; ++b)
                for (unsigned g = 0; g < 1; ++g)
                {
                    const float* A = Weight->GetTensor().Data + (ptrdiff_t)g * oGroupSize * im2ColChannels;
                    const float* B = col.Data + ptrdiff_t(b * Groups + g) * outputLength * im2ColChannels;
                    float* C = output.Data + ptrdiff_t(b * Groups + g) * oGroupSize * outputLength;
                    if (auto Ret = cublasSgemm(
                        cublasHandle_t(input.Handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        (int)oGroupSize,
                        int(outputLength),
                        (int)im2ColChannels,
                        &Alpha,
                        A,
                        (int)im2ColChannels,
                        B,
                        (int)im2ColChannels,
                        &Beta,
                        C,
                        (int)oGroupSize
                    )) return static_cast<layerStatus_t>(Ret);
                }*/

            if (BiasEnabled)
            {
                dim3 blockSizeSp(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSizeSp(
                    (outputLength + blockSizeSp.x - 1) / blockSizeSp.x,
                    (OutputChannels + blockSizeSp.y - 1) / blockSizeSp.y,
                    batchSize
                );
                unsigned sharedMemSize = blockSizeSp.y * sizeof(float);
	            implBias2DKernel<<<gridSizeSp, blockSizeSp, sharedMemSize, Stream>>>(
                    output.Data,
                    Bias->GetTensor().Data,
                    OutputChannels,
                    outputLength
                    );
            }

            return LAYER_STATUS_SUCCESS;
        }

        static __global__ void implInplaceAddKernel(
            float* output,
            const float* input,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
				output[index] += input[index];
        }

        layerStatus_t AddTensor(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output,
            const Tensor<float>& input
        )
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Add", &output.Handle);
#endif

            if (output.Dim != input.Dim ||
                output.N != input.N ||
                output.C != input.C ||
                output.H != input.H ||
                output.W != input.W)
                return LAYER_STATUS_SIZE_MISMATCH;

            if (input.Handle) CudaProvider::asyncCudaStream(getHandleStream(input.Handle));

            const auto n = input.N * input.C * input.H * input.W;

			dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
			dim3 gridLength((n + blockLength.x - 1) / blockLength.x);

            implInplaceAddKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                input.Data,
                n
				);

            return LAYER_STATUS_SUCCESS;
        }

        static __device__ float sigmoid(float x)
        {
            return 1.0f / (1.0f + expf(-x));
        }

        static __global__ void implSigmoidKernel(
            const float* input,
            float* output,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
                output[index] = sigmoid(input[index]);
        }

        layerStatus_t SigmoidTensor(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output
        )
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Sigmoid", &output.Handle);
#endif

            const auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            implSigmoidKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                output.Data,
                size
                );

            return LAYER_STATUS_SUCCESS;
        }

        layerStatus_t Transpose::Forward(
            const Tensor<float>& input,
            Tensor<float>& output
        ) noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Transpose", &output.Handle);
#endif

            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.W, input.H);
            else if (input.Dim == 3)
                output.Resize(input.N, input.W, input.H);
            else if (input.Dim == 2)
                output.Resize(input.W, input.H);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            const auto BatchSize = input.N * input.C;
            const auto BatchStride = input.H * input.W;

            static constexpr float alpha = 1.f;
            static constexpr float beta = 0.f;

            for (unsigned b = 0; b < BatchSize; ++b)
                if (auto Ret = cublasSgeam(
                    cublasHandle_t(input.Handle),
                    CUBLAS_OP_T,
                    CUBLAS_OP_T,
                    static_cast<int>(input.H),
                    static_cast<int>(input.W),
                    &alpha,
                    input.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.W),
                    &beta,
                    input.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.W),
                    output.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.H)
                )) return static_cast<layerStatus_t>(Ret);

            return LAYER_STATUS_SUCCESS;
        }

        static __global__ void implGLUKernel(
            const float* input,
            float* output,
            unsigned half,
            unsigned featureSize
        )
        {
            //[batch, 2, half, featureSize] -> [batch, half, featureSize]
            const unsigned bz = blockIdx.z;
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            if (y < half && x < featureSize)
            {
                output[(bz * half + y) * featureSize + x] = 
                    input[((bz * 2 + 0) * half + y) * featureSize + x] * sigmoid(input[((bz * 2 + 1) * half + y) * featureSize + x]);
            }
        }

        layerStatus_t GLU::Forward(
            const Tensor<float>& input,
            Tensor<float>& output
        ) noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("GLU", &output.Handle);
#endif

            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.H / 2, input.W);
            else if (input.Dim == 3)
                output.Resize(input.N, input.H / 2, input.W);
            else if (input.Dim == 2)
                output.Resize(input.H / 2, input.W);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            const auto BatchSize = input.N * input.C;
            const auto Half = input.H / 2;
            const auto FeatureSize = input.W;
			

            dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
            dim3 gridSize(
                (FeatureSize + blockSize.x - 1) / blockSize.x,
                (Half + blockSize.y - 1) / blockSize.y,
                BatchSize
            );

            implGLUKernel<<<gridSize, blockSize, 0, cudaStream_t(getHandleStream(input.Handle))>>>(
                input.Data,
                output.Data,
                Half,
                FeatureSize
            );

            return LAYER_STATUS_SUCCESS;
        }

        static __global__ void implSiLUKernel(
            const float* input,
            float* output,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < size)
            {
                float x = input[index];
                output[index] = x / (1.0f + expf(-x));
            }
        }

        layerStatus_t SiLU::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output
        ) noexcept
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("SiLU", &output.Handle);
#endif

            const auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            implSiLUKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                output.Data,
                size
                );

            return LAYER_STATUS_SUCCESS;
        }
    }
}