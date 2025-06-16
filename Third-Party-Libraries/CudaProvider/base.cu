#include <device_launch_parameters.h>

#include "base.h"
#include "cublas_v2.h"



namespace DragonianLib
{
    namespace CudaModules
    {
        handle_t createHandle()
        {
            cublasHandle_t Handle;
            if (auto Ret = cublasCreate(&Handle))
                fprintf(stderr, "%s\n", cublasGetStatusString(Ret));
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

            output.Handle = input.Handle;

            static float Alpha = 1.f;
            static float Beta = 0.f;

            auto Ret = cublasSgemm(
                cublasHandle_t(input.Handle), CUBLAS_OP_T, CUBLAS_OP_N,
                (int)OutFeatureDim, (int)inputSize, (int)InFeatureDim,
                &Alpha,
                Weight->GetTensor().Data, (int)InFeatureDim,
                input.Data, (int)InFeatureDim,
                &Beta,
                output.Data, (int)OutFeatureDim
            );

            if (Ret)
                return static_cast<layerStatus_t>(Ret);

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

        static __global__ void implAffineKernel(
            const float* input,
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

            if (threadIdx.y == 0 && x < numChannel)
                implAffineSharedMem[threadIdx.x] = weight[x];

            __syncthreads();

            if (y < batchSize && x < numChannel)
            {
                auto idx = y * numChannel + x;
                output[idx] = input[idx] * implAffineSharedMem[threadIdx.x];
            }
        }

        static __global__ void implAffineBiasKernel(
            const float* input,
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

            if (threadIdx.y == 0 && x < numChannel)
            {
                implAffineBiasSharedMem[threadIdx.x] = weight[x];
                implAffineBiasSharedMem[threadIdx.x + blockDim.x] = bias[x];
            }

            __syncthreads();

            if (y < batchSize && x < numChannel)
            {
                auto idx = y * numChannel + x;
                output[idx] = (input[idx] * implAffineBiasSharedMem[threadIdx.x]) + implAffineBiasSharedMem[threadIdx.x + blockDim.x];
            }
        }

        static __global__ void implAffineBias2DKernel(
            const float* input,
            float* output,
            const float* weight,
            const float* bias,
            unsigned batchSize,
            unsigned numChannel,
            unsigned featureSize
        )
        {
            const unsigned tz = threadIdx.z;
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned z = blockIdx.z * blockDim.z + tz;
            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ float implAffineBias2DSharedMem[];

            if (threadIdx.x == 0 && threadIdx.z == 0 && y < numChannel)
            {
                implAffineBias2DSharedMem[threadIdx.y] = weight[y];
                implAffineBias2DSharedMem[threadIdx.y + blockDim.y] = bias[y];
            }

            __syncthreads();

            if (z < batchSize && y < numChannel && x < featureSize)
            {
                auto idx = z * featureSize * numChannel + y * featureSize + x;
                output[idx] = (input[idx] * implAffineBias2DSharedMem[threadIdx.y]) + implAffineBias2DSharedMem[threadIdx.y + blockDim.y];
            }
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
            const auto featureDim = output.W;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            unsigned sampleCountNorm = output.N * output.C * output.H;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);

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
                    implAffineBiasKernel<<<blockSize, gridSize, sharedMemSize, Stream>>>
						(output.Data, output.Data, Weight->GetTensor().Data, Bias->GetTensor().Data, sampleCountNorm, NumChannels);
                else
                    implAffineKernel<<<blockSize, gridSize, sharedMemSize, Stream>>>
						(output.Data, output.Data, Weight->GetTensor().Data, sampleCountNorm, NumChannels);
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
            unsigned featureDim = output.H;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            unsigned batchSize = output.N * output.C;
            unsigned featureSize = output.W;

            unsigned sampleCountNorm = batchSize * NumGroups;
            unsigned featureSizeNorm = output.H * output.W / NumGroups;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);

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
                const auto batchThreadCount = std::min(batchSize, 2u);
                dim3 blockSizeSp(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32 / batchThreadCount, 32, batchThreadCount);
                dim3 gridSizeSp(
                    (featureSize + blockSizeSp.x - 1) / blockSizeSp.x,
                    (NumChannels + blockSizeSp.y - 1) / blockSizeSp.y,
                    (batchSize + blockSizeSp.z - 1) / blockSizeSp.z
                );
                unsigned sharedMemSize = blockSizeSp.y * 2ull * sizeof(float);
	            implAffineBias2DKernel<<<blockSizeSp, gridSizeSp, sharedMemSize, Stream>>>
				   (output.Data, output.Data, Weight->GetTensor().Data, Bias->GetTensor().Data, batchSize, NumChannels, featureSize);
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
        ) : Module(parent, name), KernelSize(kernelSize), Stride(stride), Padding(padding), Dilation(dilation), Groups(groups), InputChannels(inputChannels), OutputChannels(outputChannels), biasEnabled(bias)
        {
            Weight = std::make_shared<Parameter>(
                this, "weight", Tensor<float>(outputChannels, inputChannels / groups, kernelSize)
            );
            if (biasEnabled)
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<float>(outputChannels)
                );
        }

        static __global__ void conv1DForwardKernelBias(
            const float* input,
            const float* weight,
            const float* bias,
            float* output,
            unsigned batchSize,
            unsigned inputChannels,
            unsigned inputLength,
            unsigned outputChannels,
            unsigned outputLength,
            unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned dilation,
            unsigned groups
        )
        {
            const unsigned batchIdx = blockIdx.z;
            const unsigned outChannelIdx = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned outIdx = blockIdx.x * blockDim.x + threadIdx.x;

            if (batchIdx >= batchSize || outChannelIdx >= outputChannels || outIdx >= outputLength)
                return;

            float sum = 0.0f;

            for (unsigned inputChannelIdx = 0; inputChannelIdx < inputChannels; ++inputChannelIdx)
            {
                for (unsigned kernelIdx = 0; kernelIdx < kernelSize; ++kernelIdx)
                {
                    int inIdx = int(outIdx * stride) - int(padding) + int(kernelIdx * dilation);
                    if (inIdx >= 0 && inIdx < int(inputLength))
                    {
                        
                        const unsigned weightIdx = (outChannelIdx * inputChannels + inputChannelIdx) * kernelSize + kernelIdx;
                        for (unsigned groupIdx = 0; groupIdx < groups; ++groupIdx)
                        {
                            const unsigned inputIdx = 
                                ((batchIdx * groups + groupIdx) * inputChannels + inputChannelIdx) * inputLength + inIdx;
                        	sum += input[inputIdx] * weight[weightIdx];
                        }
                    }
                }
            }

            extern __shared__ float conv1DSharedBias[];
            if (threadIdx.x == 0)
                conv1DSharedBias[threadIdx.y] = bias[outChannelIdx];
            __syncthreads();
            sum += conv1DSharedBias[threadIdx.y];

            const unsigned outputIdx = ((batchIdx * outputChannels + outChannelIdx) * outputLength) + outIdx;
            output[outputIdx] = sum;
        }

        static __global__ void conv1DForwardKernel(
            const float* input,
            const float* weight,
            float* output,
            unsigned batchSize,
            unsigned inputChannels,
            unsigned inputLength,
            unsigned outputChannels,
            unsigned outputLength,
            unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned dilation,
            unsigned groups
        )
        {
            const unsigned batchIdx = blockIdx.z;
            const unsigned outChannelIdx = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned outIdx = blockIdx.x * blockDim.x + threadIdx.x;

            if (batchIdx >= batchSize || outChannelIdx >= outputChannels || outIdx >= outputLength)
                return;

            float sum = 0.0f;

            
            for (unsigned inputChannelIdx = 0; inputChannelIdx < inputChannels; ++inputChannelIdx)
            {
                for (unsigned kernelIdx = 0; kernelIdx < kernelSize; ++kernelIdx)
                {
                    int inIdx = int(outIdx * stride) - int(padding) + int(kernelIdx * dilation);
                    if (inIdx >= 0 && inIdx < int(inputLength))
                    {
                        const unsigned inputIdx = (batchIdx * inputChannels + inputChannelIdx) * inputLength + inIdx;
                        const unsigned weightIdx = (outChannelIdx * inputChannels + inputChannelIdx) * kernelSize + kernelIdx;
                        sum += input[inputIdx] * weight[weightIdx];
                    }
                }
            }

            const unsigned outputIdx = ((batchIdx * outputChannels + outChannelIdx) * outputLength) + outIdx;
            output[outputIdx] = sum;
        }

        layerStatus_t Conv1D::Forward(
            const Tensor<float>& input,
            Tensor<float>& output
        ) const noexcept
        {
            if (input.H != InputChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            const unsigned batchDim = input.N * input.C;
            const unsigned inChannel = input.H / Groups;
            const unsigned inputLength = input.W;

            const unsigned outputLength = (inputLength + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1;

            output.Resize(batchDim, OutputChannels, outputLength);
            output.Handle = input.Handle;

            dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
            dim3 gridSize(
                (outputLength + blockSize.x - 1) / blockSize.x,
                (OutputChannels + blockSize.y - 1) / blockSize.y,
                batchDim
            );
            unsigned sharedMemSize = blockSize.y * sizeof(float);

            if (biasEnabled)
                conv1DForwardKernelBias<<<gridSize, blockSize, sharedMemSize, cudaStream_t(getHandleStream(input.Handle))>>>(
                    input.Data,
                    Weight->GetTensor().Data,
                    Bias->GetTensor().Data,
                    output.Data,
                    batchDim,
                    inChannel,
                    inputLength,
                    OutputChannels,
                    outputLength,
                    KernelSize,
                    Stride,
                    Padding,
                    Dilation,
                    Groups
                    );
            else
				conv1DForwardKernel<<<gridSize, blockSize, sharedMemSize, cudaStream_t(getHandleStream(input.Handle))>>>(
                    input.Data,
                    Weight->GetTensor().Data,
                    output.Data,
                    batchDim,
                    inChannel,
                    inputLength,
                    OutputChannels,
                    outputLength,
                    KernelSize,
                    Stride,
                    Padding,
                    Dilation,
                    Groups
                    );

            return LAYER_STATUS_SUCCESS;
        }

        layerStatus_t AddTensor(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<float>& output,
            const Tensor<float>& input
        )
        {
            output.Handle = input.Handle;
            const auto n = input.N * input.C * input.H * input.W;
            float alpha = 1.f;
            return static_cast<layerStatus_t>(cublasSaxpy(
                cublasHandle_t(output.Handle),
                static_cast<int>(n),
                &alpha,
                input.Data,
                1,
                output.Data,
                1
            ));
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
            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.W, input.H);
            else if (input.Dim == 3)
                output.Resize(input.N, input.W, input.H);
            else if (input.Dim == 2)
                output.Resize(input.W, input.H);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            const auto BatchSize = input.N * input.C;
            const auto BatchStride = input.H * input.W;
            output.Handle = input.Handle;
            float alpha = 1.f;
            float beta = 0.f;
            for (unsigned b = 0; b < BatchSize; ++b)
            {
                auto Ret = cublasSgeam(
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
                );
                if (Ret)
                    return static_cast<layerStatus_t>(Ret);
            }
            return LAYER_STATUS_SUCCESS;
        }

        static __global__ void implGLUKernel(
            const float* input,
            float* output,
            unsigned batchSize,
            unsigned featureSize
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            unsigned y = blockIdx.y * blockDim.y + ty;
            unsigned x = blockIdx.x * blockDim.x + tx;

            if (y < batchSize && x < featureSize)
            {
                const unsigned idx1 = y * featureSize + x;
                const unsigned idx2 = (y + batchSize) * featureSize + x;

                float linearPart = input[idx1];
                float gatePart = input[idx2];

                output[idx1] = linearPart * sigmoid(gatePart);
            }
        }

        layerStatus_t GLU::Forward(
            const Tensor<float>& input,
            Tensor<float>& output
        ) noexcept
        {
            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.H / 2, input.W);
            else if (input.Dim == 3)
                output.Resize(input.N, input.H / 2, input.W);
            else if (input.Dim == 2)
                output.Resize(input.H / 2, input.W);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            const auto BatchSize = input.N * input.C * input.H / 2;
            const auto FeatureSize = input.W;

            // Launch CUDA kernel for GLU computation
            dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
            dim3 gridSize(
                (FeatureSize + blockSize.x - 1) / blockSize.x,
                (BatchSize + blockSize.y - 1) / blockSize.y
            );

            implGLUKernel<<<gridSize, blockSize, 0, cudaStream_t(getHandleStream(input.Handle))>>>(
                input.Data,
                output.Data,
                BatchSize,
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