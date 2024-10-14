#pragma once
#include <map>
#include "SvcBase.hpp"
/*namespace tlibsvc {
    class ReflowSvc
    {
    public:
        ReflowSvc(const Hparams& _Hps, const ProgressCallback& _ProgressCallback, unsigned DeviceID_ = 0);
        ~ReflowSvc();
        ReflowSvc(const ReflowSvc&) = delete;
        ReflowSvc(ReflowSvc&&) = delete;
        ReflowSvc& operator=(const ReflowSvc&) = delete;
        ReflowSvc& operator=(ReflowSvc&&) = delete;

        void Destory();

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> SliceInference(const SingleSlice& _Slice, const InferenceParams& _Params, size_t& _Process) const;

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> InferPCMData(const DragonianLibSTL::Vector<int16_t>& _PCMData, long _SrcSamplingRate, const InferenceParams& _Params) const;

        [[nodiscard]] DragonianLibSTL::Vector<int16_t> ShallowDiffusionInference(
            DragonianLibSTL::Vector<float>& _16KAudioHubert,
            const InferenceParams& _Params,
            std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
            const DragonianLibSTL::Vector<float>& _SrcF0,
            const DragonianLibSTL::Vector<float>& _SrcVolume,
            const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
            size_t& Process,
            int64_t SrcSize
        ) const;

        [[nodiscard]] int64_t GetMaxStep() const
        {
            return MaxStep;
        }

        [[nodiscard]] const std::wstring& GetReflowSvcVer() const
        {
            return ReflowSvcVersion;
        }

        [[nodiscard]] int64_t GetMelBins() const
        {
            return melBins;
        }

        void NormMel(DragonianLibSTL::Vector<float>& MelSpec) const;

    private:
        Ort::Session* encoder = nullptr;
        Ort::Session* velocity = nullptr;
        Ort::Session* after = nullptr;

        int64_t melBins = 128;
        int64_t MaxStep = 100;
        float SpecMin = -12;
        float SpecMax = 2;
        float Scale = 1000.f;
        bool VaeMode = true;

        std::wstring ReflowSvcVersion = L"DiffusionSvc";

        const std::vector<const char*> nsfInput = { "c", "f0" };
        const std::vector<const char*> nsfOutput = { "audio" };
        const std::vector<const char*> afterInput = { "x" };
        const std::vector<const char*> afterOutput = { "mel_out" };
        const std::vector<const char*> OutputNamesEncoder = { "x", "cond", "f0_pred" };
    };

}*/

