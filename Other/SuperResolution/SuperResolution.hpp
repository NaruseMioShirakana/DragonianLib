#pragma once
#include "EnvManager.hpp"
#include "Image-Video/ImgVideo.hpp"
namespace libsr
{
    using ProgressCallback = std::function<void(size_t, size_t)>;

    class RealESRGan
    {
    public:
        struct Hparams
        {
            std::wstring rgb, alpha;
            long s_width = 64;
            long s_height = 64;
        };

        RealESRGan(const Hparams& _Config, const ProgressCallback& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
    	~RealESRGan();

        DragonianLib::ImageSlicer& Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const;
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
        ProgressCallback _callback;
        DragonianLib::DragonianLibOrtEnv Env_;

        std::vector<const char*> inputNames = { "src" };
        std::vector<const char*> outputNames = { "img" };
    };
}
