#pragma once
#include "SuperResolution.hpp"

namespace DragonianLib
{
    namespace LibSuperResolution
    {
        class MoeSR : public SuperResolution
        {
        public:

            MoeSR(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
            ~MoeSR() override;

            DragonianLib::Image& Infer(DragonianLib::Image& _Image, int64_t _BatchSize) const override;
        private:
            void Destory();
            MoeSR(const MoeSR&) = delete;
            MoeSR(MoeSR&&) = delete;
            MoeSR& operator=(const MoeSR&) = delete;
            MoeSR& operator=(MoeSR&&) = delete;

            Ort::Session* Model = nullptr;
            long ScaleFactor = 2;
        };
    }
}