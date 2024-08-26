#pragma once
#include "EnvManager.hpp"
#include "Image-Video/ImgVideo.hpp"
namespace DragonianLib
{
    namespace LibSuperResolution
    {
        using ProgressCallback = std::function<void(size_t, size_t)>;

        struct Hparams
        {
            std::wstring RGBModel, AlphaModel;
            long InputWidth = 64;
            long InputHeight = 64;
            long Scale = 2;
        };

        class SuperResolution
        {
        public:
            SuperResolution(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider, ProgressCallback _Callback);
            virtual ~SuperResolution() = default;
            virtual DragonianLib::Image& Infer(DragonianLib::Image& _Image, int64_t _BatchSize) const;
        protected:
            DragonianLib::DragonianLibOrtEnv Env_;
            ProgressCallback Callback_;
            std::vector<Ort::AllocatedStringPtr> Names;
            char* inputNames = nullptr;
            char* outputNames = nullptr;
        private:
            SuperResolution(const SuperResolution&) = delete;
            SuperResolution(SuperResolution&&) = delete;
            SuperResolution& operator=(const SuperResolution&) = delete;
            SuperResolution& operator=(SuperResolution&&) = delete;
        };
    }
}