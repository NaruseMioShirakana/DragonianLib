#include <iostream>

#ifndef DRAGONIANLIB_USE_SHARED_LIBS

#include "Libraries/Image-Video/ImgVideo.hpp"
#include "OnnxLibrary/SuperResolution/ImplSuperResolution.hpp"

#else

#endif


int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	using namespace DragonianLib;
	auto Image = ImageVideo::LoadAndSplitImageNorm(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestImages/Theresa.png)",
		256,
		256,
		248,
		248
	);

	std::get<0>(Image) + std::get<0>(Image);

	OnnxRuntime::SuperResolution::HyperParameters Parameters;
	Parameters.RGBModel = __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/x2_universal-fix1.onnx)";
	Parameters.Callback = DefaultProgressCallback(std::cout);

	auto Env = OnnxRuntime::CreateOnnxRuntimeEnvironment({
			Device::CUDA,
		});
	OnnxRuntime::SuperResolution::SuperResolutionBCRGBHW Model(
		Env,
		Parameters
	);

	Image = Model.Infer(Image, 1);

	ImageVideo::SaveBitmap(
		ImageVideo::CombineImage(Image, 248ll * 2, 248ll * 2),
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestImages/Theresa-SR.png)"
	);
#else

#endif
}