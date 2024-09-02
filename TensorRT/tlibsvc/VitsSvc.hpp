#pragma once
#include "SvcBase.hpp"
#include "Cluster/ClusterManager.hpp"

namespace tlibsvc {

    class VitsSvc
    {
    public:
        using ioBuffer_t = InferenceDeviceBuffer[2];
        VitsSvc(
            const VitsSvcConfig& _Hps,
            const ProgressCallback& _ProgressCallback
        );
        ~VitsSvc();
        VitsSvc(const VitsSvc&) = delete;
        VitsSvc(VitsSvc&&) = delete;
        VitsSvc& operator=(const VitsSvc&) = delete;
        VitsSvc& operator=(VitsSvc&&) = delete;

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(
            const SingleSlice& _Slice,
            const InferenceParams& _Params,
            ioBuffer_t& _IOBuffer
        ) const;

    private:
        std::unique_ptr<TrtModel> VitsSvcModel;
        std::shared_ptr<TrtModel> HubertModel;

        int64_t MySamplingRate, HopSize, HiddenUnitKDims, SpeakerCount, ClusterCenterSize;
        bool EnableVolume, EnableCharaMix, EnableCluster;
        std::wstring VitsSvcVersion;
        ProgressCallback ProgressFn;
        DragonianLib::ClusterWrp Cluster;

    	TensorXData SoVits4Preprocess(
            const DragonianLibSTL::Vector<float>& HiddenUnit,
            const DragonianLibSTL::Vector<float>& F0,
            const DragonianLibSTL::Vector<float>& Volume,
            const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
            const InferenceParams& Params,
            int64_t AudioSize
        ) const;

        TensorXData RVCTensorPreprocess(
            const DragonianLibSTL::Vector<float>& HiddenUnit,
            const DragonianLibSTL::Vector<float>& F0,
            const DragonianLibSTL::Vector<float>& Volume,
            const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
            const InferenceParams& Params,
            int64_t AudioSize
        ) const;

        TensorXData Preprocess(
            const DragonianLibSTL::Vector<float>& HiddenUnit,
            const DragonianLibSTL::Vector<float>& F0,
            const DragonianLibSTL::Vector<float>& Volume,
            const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
            const InferenceParams& Params,
            int64_t AudioSize
        ) const;

        static inline DragonianLibSTL::Vector<DynaShapeSlice> VitsSvcDynaSetting{
        {"c", nvinfer1::Dims3(1, 20, 256), nvinfer1::Dims3(1, 2500, 256), nvinfer1::Dims3(1, 5000, 256)},
        {"f0", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"mel2ph", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"uv", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"noise", nvinfer1::Dims3(1, 192, 20), nvinfer1::Dims3(1, 192, 2500), nvinfer1::Dims3(1, 192, 5000)},
        {"sid", nvinfer1::Dims2(20, 1), nvinfer1::Dims2(2500, 1), nvinfer1::Dims2(5000, 1) },
        {"vol", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"phone", nvinfer1::Dims3(1, 20, 256), nvinfer1::Dims3(1, 2500, 256), nvinfer1::Dims3(1, 5000, 256)},
        {"pitch", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"pitchf", nvinfer1::Dims2(1, 20), nvinfer1::Dims2(1, 2500), nvinfer1::Dims2(1, 5000) },
        {"rnd", nvinfer1::Dims3(1, 192, 20), nvinfer1::Dims3(1, 192, 2500), nvinfer1::Dims3(1, 192, 5000) }
        };
    };

}
