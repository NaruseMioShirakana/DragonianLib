#pragma once
#include "DiffSvc.hpp"

namespace tlibsvc {

    class VitsSvc
    {
    public:

        VitsSvc(const Hparams& _Hps, const ProgressCallback& _ProgressCallback, unsigned DeviceID_ = 0);
        ~VitsSvc();
        VitsSvc(const VitsSvc&) = delete;
        VitsSvc(VitsSvc&&) = delete;
        VitsSvc& operator=(const VitsSvc&) = delete;
        VitsSvc& operator=(VitsSvc&&) = delete;

        void Destory();

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(
            const SingleSlice& _Slice,
            const InferenceParams& _Params,
            size_t& _Process
        ) const;

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> InferPCMData(
            const DragonianLibSTL::Vector<int16_t>& _PCMData,
            long _SrcSamplingRate,
            const InferenceParams& _Params
        ) const;

        [[nodiscard]] std::vector<Ort::Value> MelExtractor(const float* PCMAudioBegin, const float* PCMAudioEnd) const = delete;

    private:
        Ort::Session* VitsSvcModel = nullptr;
        std::wstring VitsSvcVersion = L"SoVits4.0";
    };

}
