#pragma once
#include "EnvManager.hpp"
#include "Image-Video/ImgVideo.hpp"
namespace libsr
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
        virtual DragonianLib::ImageSlicer& Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const;
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

    class RealESRGan : public SuperResolution
    {
    public:

        RealESRGan(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
    	~RealESRGan() override;

        DragonianLib::ImageSlicer& Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const override;
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

    class MoeSR : public SuperResolution
    {
    public:

        MoeSR(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
        ~MoeSR() override;

        DragonianLib::ImageSlicer& Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const override;
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
