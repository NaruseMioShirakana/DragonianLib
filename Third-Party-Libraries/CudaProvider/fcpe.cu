#include <string>
#include <device_launch_parameters.h>

#include "fcpe.h"
#include "cuda_runtime.h"

namespace DragonianLib
{
	namespace CudaModules
	{
		namespace FCPE
		{
            ConformerConvModule::ConformerConvModule(
                Module* parent, const std::string& name,
                unsigned dimModel, unsigned expandFactor, unsigned kernelSize
            ) : Module(parent, name)
            {
                auto inner_dim = dimModel * expandFactor;

                net_0 = std::make_shared<LayerNorm1D>(
                    this,
					"net.0",
                    dimModel
                );
                net_2 = std::make_shared<Conv1D>(
                    this,
                    "net.2",
                    dimModel,
                    inner_dim * 2,
                    1
                );
                net_4_conv = std::make_shared<Conv1D>(
                    this,
                    "net.4.conv",
                    inner_dim,
                    inner_dim,
                    kernelSize,
                    1,
                    kernelSize / 2,
					1,
                    inner_dim
                );
                net_6 = std::make_shared<Conv1D>(
                    this,
                    "net.6",
                    inner_dim,
                    dimModel,
                    1
                );
            }

            layerStatus_t ConformerConvModule::Forward(
                Tensor<float>& output,
                Tensor<float>& mean,
                Tensor<float>& var,
                Tensor<float>& cache
            ) const
            {
                if (auto Ret = net_0->Forward(output, mean, var)) return Ret;

                if (auto Ret = net_1.Forward(output, cache)) return Ret;

                if (auto Ret = net_2->Forward(cache, output)) return Ret;

                if (auto Ret = net_3.Forward(output, cache)) return Ret;

                if (auto Ret = net_4_conv->Forward(cache, output)) return Ret;

                if (auto Ret = net_5.Forward(output)) return Ret;

                if (auto Ret = net_6->Forward(output, cache)) return Ret;

                return net_7.Forward(cache, output);
            }

            CFNEncoderLayer::CFNEncoderLayer(
                Module* parent, const std::string& name,
                unsigned dimModel, unsigned numHeads,
                bool useNorm, bool convOnly
            ) : Module(parent, name)
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                conformer = std::make_shared<ConformerConvModule>(
                    this,
                    "conformer",
                    dimModel
                );
                norm = std::make_shared<LayerNorm1D>(
                    this,
                    "norm",
                    dimModel
                );
            }

            layerStatus_t CFNEncoderLayer::Forward(
                Tensor<float>& output,
                Tensor<float>& mean,
                Tensor<float>& var,
                Tensor<float>& res,
                Tensor<float>& cache
            ) const
            {
                res.Copy(output);

                if (auto Ret = conformer->Forward(
                    output, mean, var, cache
                )) return Ret;

                return AddTensor(output, res);
            }

            ConformerNaiveEncoder::ConformerNaiveEncoder(
                Module* parent, const std::string& name,
                unsigned numLayers, unsigned numHeads, unsigned dimModel,
                bool useNorm, bool convOnly
            ) : Module(parent, name)
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                for (unsigned i = 0; i < numLayers; ++i)
                    encoder_layers.emplace_back(
                        std::make_shared<CFNEncoderLayer>(
                            this,
                            "encoder_layers." + std::to_string(i),
                            dimModel,
                            numHeads,
                            useNorm,
                            convOnly
                        )
                    );
            }

            layerStatus_t ConformerNaiveEncoder::Forward(
                Tensor<float>& output,
                Tensor<float>& mean,
                Tensor<float>& var,
                Tensor<float>& res,
                Tensor<float>& cache
            ) const
            {
                for (const auto& layer : encoder_layers)
                    if (auto Ret = layer->Forward(
                        output,
                        mean,
                        var,
                        res,
                        cache
                    )) return Ret;
                return LAYER_STATUS_SUCCESS;
            }

			Model::Model(
                unsigned inputChannels, unsigned outputDims, unsigned hiddenDims,
                unsigned numLayers, unsigned numHeads,
                float f0Max, float f0Min,
                bool useFaNorm, bool convOnly,
                bool useHarmonicEmb
            ) : Module(nullptr, "")
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                if (useHarmonicEmb)
                    throw std::overflow_error("not impl yet!");

                input_stack_0 = std::make_shared<Conv1D>(
                    this,
                    "input_stack.0",
                    inputChannels,
                    hiddenDims,
                    3,
                    1,
                    1
                );
                input_stack_1 = std::make_shared<GroupNorm1D>(
                    this,
                    "input_stack.1",
                    4,
                    hiddenDims
                );
                input_stack_3 = std::make_shared<Conv1D>(
                    this,
                    "input_stack.3",
                    hiddenDims,
                    hiddenDims,
                    3,
                    1,
                    1
                );

                net = std::make_shared<ConformerNaiveEncoder>(
                    this,
					"net",
                    numLayers,
                    numHeads,
                    hiddenDims,
                    useFaNorm,
                    convOnly
                );

                norm = std::make_shared<LayerNorm1D>(
                    this,
                    "norm",
                    hiddenDims
                );

                output_proj = std::make_shared<Linear>(
                    this,
                    "output_proj",
                    hiddenDims,
                    outputDims
                );
            }

            layerStatus_t Model::Forward(
                const Tensor<float>& input,
                Tensor<float>& output,
                CacheTensors& caches
            ) const
            {
                if (auto Ret = input_stack_0->Forward(
                    input,
                    caches.input_stack_out1
                )) return Ret;

                if (auto Ret = input_stack_1->Forward(
                    caches.input_stack_out1,
                    caches.mean,
                    caches.var
                )) return Ret;

                if (auto Ret = input_stack_2.Forward(
                    caches.input_stack_out1
                )) return Ret;

                if (auto Ret = input_stack_3->Forward(
                    caches.input_stack_out1,
                    caches.input_stack_out2
                )) return Ret;

                if (auto Ret = Transpose::Forward(
                    caches.input_stack_out2,
                    caches.input_stack_out1
                )) return Ret;

                if (auto Ret = net->Forward(
                    caches.input_stack_out1,
                    caches.mean,
                    caches.var,
                    caches.res,
                    caches.cache
                )) return Ret;

                if (auto Ret = norm->Forward(
                    caches.input_stack_out1,
                    caches.mean,
                    caches.var
                )) return Ret;

                if (auto Ret = output_proj->Forward(
                    caches.input_stack_out1,
                    output
                )) return Ret;

                return SigmoidTensor(output);
            }

		}
	}
}