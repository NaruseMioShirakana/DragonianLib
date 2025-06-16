#pragma once
#include "base.h"


namespace DragonianLib
{
    namespace CudaModules
    {
        namespace FCPE
        {
            class ConformerConvModule : public Module
            {
            public:
                ConformerConvModule(
                    Module* parent,
                    const std::string& name,
                    unsigned dimModel,
                    unsigned expandFactor = 2,
                    unsigned kernelSize = 31
                );

                layerStatus_t Forward(
                    Tensor<float>& output,
                    Tensor<float>& mean,
                    Tensor<float>& var,
                    Tensor<float>& cache
                ) const;

            private:
                std::shared_ptr<LayerNorm1D> net_0;
                Transpose net_1;
                std::shared_ptr<Conv1D> net_2;
                GLU net_3;
                std::shared_ptr<Conv1D> net_4_conv;
                SiLU net_5;
                std::shared_ptr<Conv1D> net_6;
                Transpose net_7;
            };

            class CFNEncoderLayer : public Module
            {
            public:
                CFNEncoderLayer(
                    Module* parent,
                    const std::string& name,
                    unsigned dimModel,
                    unsigned numHeads,
                    bool useNorm = false,
                    bool convOnly = false
                );

                layerStatus_t Forward(
                    Tensor<float>& output,
                    Tensor<float>& mean,
                    Tensor<float>& var,
                    Tensor<float>& res,
                    Tensor<float>& cache
                ) const;

            private:
                std::shared_ptr<ConformerConvModule> conformer;
                mutable Tensor<float> conformerOut;
                std::shared_ptr<LayerNorm1D> norm;
            };


            class ConformerNaiveEncoder : public Module
            {
            public:
                ConformerNaiveEncoder(
                    Module* parent,
                    const std::string& name,
                    unsigned numLayers,
                    unsigned numHeads,
                    unsigned dimModel,
                    bool useNorm = false,
                    bool convOnly = false
                );

                layerStatus_t Forward(
                    Tensor<float>& output,
                    Tensor<float>& mean,
                    Tensor<float>& var,
                    Tensor<float>& res,
                    Tensor<float>& cache
                ) const;

            private:
                std::vector<std::shared_ptr<CFNEncoderLayer>> encoder_layers;
            };

			class Model : public Module
            {
            public:
                struct CacheTensors
                {
                    Tensor<float> input_stack_out1;
                    Tensor<float> input_stack_out2;
                    Tensor<float> mean;
                    Tensor<float> var;
                    Tensor<float> res;
                    Tensor<float> cache;
                };

                Model(
                    unsigned inputChannels,
                    unsigned outputDims,
                    unsigned hiddenDims = 512,
                    unsigned numLayers = 6,
                    unsigned numHeads = 8,
                    float f0Max = 1975.5f,
                    float f0Min = 32.70f,
                    bool useFaNorm = false,
                    bool convOnly = true,
                    bool useHarmonicEmb = false
                );

                layerStatus_t Forward(
                    const Tensor<float>& input,
                    Tensor<float>& output,
                    CacheTensors& caches
                ) const;

            private:
                std::shared_ptr<Conv1D> input_stack_0;
                std::shared_ptr<GroupNorm1D> input_stack_1;
                LeakyReLU input_stack_2;
                std::shared_ptr<Conv1D> input_stack_3;
                std::shared_ptr<ConformerNaiveEncoder> net;
                std::shared_ptr<LayerNorm1D> norm;
                std::shared_ptr<Linear> output_proj;
            };
        }
    }
}
