#include <iostream>

#include "Libraries/Stft/Stft.hpp"
#include "Libraries/AvCodec/SignalProcess.h"
#include "OnnxLibrary/UnitsEncoder/Register.hpp"

#ifndef DRAGONIANLIB_USE_SHARED_LIBS
#include "Libraries/AvCodec/AvCodec.h"
#include "NCNNLibrary/SingingVoiceConversion/Model/Vits-Svc.hpp"
#include "NCNNLibrary/UnitsEncoder/Register.hpp"
#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS

	auto AudioStream = DragonianLib::AvCodec::OpenInputStream(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal.wav)"
	);
	auto AudioOutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal-SVC-NCNN.wav)"
	);
	
	//vec-768-layer-12-f16
	const auto UnitsEncoder = DragonianLib::NCNN::UnitsEncoder::New(
		L"ContentVec-768-l12",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/vec)",
		{},
		16000,
		768
	);

	auto [Audio, SamplingRate] = AudioStream.DecodeAudio(2);
	auto Input =
		DragonianLib::Signal::ResampleKernel(DragonianLib::FunctionTransform::KaiserWindow<float>(32))(
			Audio[{"882000:1323000", "0"}].Transpose(-1, -2),
			SamplingRate,
			16000
			);
	Input.Evaluate();
	AudioOutStream.EncodeAll(
		Input.GetCRng(), 16000
	);
	
	auto F0 = DragonianLib::F0Extractor::New(L"Dio", nullptr)(
		Input, DragonianLib::F0Extractor::F0ExtractorParams()
		);

	auto Units = UnitsEncoder->Forward(Input.UnSqueeze(0));

#ifdef DRAGONIANLIB_USE_ORT_CHECK
	std::cout << (DragonianLib::OnnxRuntime::UnitsEncoder::New(
		L"HubertBase",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/vec-768-layer-12-f16.onnx)",
		DragonianLib::OnnxRuntime::CreateOnnxRuntimeEnvironment({})
	)(Input.UnSqueeze(0)) - Units).Abs().ReduceMax(-1).Evaluate();
#endif
	
#endif
}