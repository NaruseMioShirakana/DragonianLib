#include "TensorRT/SuperResolution/MoeSuperResolution.hpp"
#include <iostream>

void ShowProgressBar(size_t progress, size_t total) {
	int barWidth = 70;
	float progressRatio = static_cast<float>(progress) / float(total);
	int pos = static_cast<int>(float(barWidth) * progressRatio);

	std::cout << "\r";
	std::cout.flush();
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
}

size_t TotalStep = 0;
void ProgressCb(size_t a, size_t)
{
	ShowProgressBar(a, TotalStep);
}

void ProgressCbS(size_t a, size_t b)
{
	ShowProgressBar(a, b);
}

template <typename _Function>
void WithTimer(_Function&& _Func)
{
	auto start = std::chrono::high_resolution_clock::now();
	_Func();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

int main()
{
	using namespace DragonianLib::TensorRTLib::SuperResolution;
	MoeSR Model{
		LR"(D:\VSGIT\hat_lite_seed0721.onnx)",
		4,
		DragonianLib::TensorRTLib::TrtConfig{
			LR"(D:\VSGIT\hat_lite_seed0721.trt)",
			DragonianLib::TensorRTLib::TrtConfig::DynaShapeSettings{
					{
						"DynaArg0",
					   nvinfer1::Dims4(1, 3, 64, 64),
					   nvinfer1::Dims4(1, 3, 128, 128),
					   nvinfer1::Dims4(1, 3, 192, 192)
					}
			},
			0,
			true,
			false,
			true,
			false,
			nvinfer1::ILogger::Severity::kWARNING,
			4
		},
		ProgressCbS
	};

	DragonianLib::ImageVideo::GdiInit();

	DragonianLib::ImageVideo::Image Image(
		LR"(D:\VSGIT\CG000000.BMP)",
		192,
		192,
		16,
		0.f,
		false/*,
		LR"(D:\VSGIT\CG000002-DEB.png)"*/
	);
	/*Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TN.png)", 1, 100))
		std::cout << "1-Complete!\n";
	Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TNN.png)", 1, 100))
		std::cout << "2-Complete!\n";*/

	WithTimer([&]() {
		Model.Infer(Image, 1);
		});

	if (Image.MergeWrite(LR"(D:\VSGIT\CG000000114514.BMP)", 4, 100))
		std::cout << "Complete!\n";

	DragonianLib::ImageVideo::GdiClose();
}