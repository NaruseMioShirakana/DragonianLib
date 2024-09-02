#pragma once
#include "SuperResolution.hpp"

namespace DragonianLib
{
    namespace LibSuperResolution
    {
        class RealESRGan : public SuperResolution
        {
        public:

            RealESRGan(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
            ~RealESRGan() override;

            DragonianLib::Image& Infer(DragonianLib::Image& _Image, int64_t _BatchSize) const override;
        private:
            void Destory();
            RealESRGan(const RealESRGan&) = delete;
            RealESRGan(RealESRGan&&) = delete;
            RealESRGan& operator=(const RealESRGan&) = delete;
            RealESRGan& operator=(RealESRGan&&) = delete;

            Ort::Session* model = nullptr;
            Ort::Session* model_alpha = nullptr;
            long s_width = 64;
            long s_height = 64;
        };
    }
}