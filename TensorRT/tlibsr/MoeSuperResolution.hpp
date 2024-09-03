#pragma once
#include "../TRTBase.hpp"
#include "Image-Video/ImgVideo.hpp"


namespace tlibsr
{
    using namespace TensorRTLib;

    class MoeSR
    {
    public:
        MoeSR(const std::wstring& RGBModel, long Scale, const TrtConfig& TrtSettings, ProgressCallback _Callback);
        ~MoeSR();

        DragonianLib::Image& Infer(DragonianLib::Image& _Image, const InferenceDeviceBuffer& _Buffer, int64_t _BatchSize) const;
    private:
        MoeSR(const MoeSR&) = delete;
        MoeSR(MoeSR&&) = delete;
        MoeSR& operator=(const MoeSR&) = delete;
        MoeSR& operator=(MoeSR&&) = delete;

        ProgressCallback Callback_;
        std::unique_ptr<TrtModel> Model = nullptr;
        long ScaleFactor = 2;
        DragonianLibSTL::Vector<DynaShapeSlice> DynaSetting{
	        {
	        	"DynaArg0",
	            nvinfer1::Dims4(1, 3, 64, 64),
	            nvinfer1::Dims4(1, 3, 64, 64),
	            nvinfer1::Dims4(1, 3, 64, 64)
	        }
        };
    };
}