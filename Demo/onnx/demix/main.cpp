#include <iostream>

#ifndef DRAGONIANLIB_USE_SHARED_LIBS

#include "Libraries/AvCodec/AvCodec.h"

#include "OnnxLibrary/Demix/Demix.hpp"
#include "OnnxLibrary/Demix/CascadedNet.hpp"

[[maybe_unused]] static void TestStft()
{
	using namespace DragonianLib;

	FunctionTransform::StftKernel Stft(
		2048, 512, 2048
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism.wav)"
	);
	auto [SrcAudio, SamplingRate] = AudioInStream.DecodeAudio(
		2, true
	);

	auto Spec = Stft.Execute(SrcAudio.UnSqueeze(0));

	auto Signal = Stft.Inverse(Spec).Evaluate();
	//auto Signal = Functional::MinMaxNormalize(Stft.Inverse(Spec), -1).Evaluate();

	std::cout << "IStft-Signal Min:   " << Signal.Abs().ReduceMin(-1).Evaluate() << "\n";
	std::cout << "IStft-Signal Max:   " << Signal.ReduceMax(-1).Evaluate() << "\n";
	std::cout << "Source Min:         " << SrcAudio.Abs().ReduceMin(-1).Evaluate() << "\n";
	std::cout << "Source Max:         " << SrcAudio.ReduceMax(-1).Evaluate() << "\n";
	auto Diff = Signal[{None, None, { 1024 + 256, 1024 + 256 + SrcAudio.Size(-1) }}] - SrcAudio;
	std::cout << "Epsilon Stft-IStft: " << Diff.Abs().ReduceMax(-1).Evaluate() << "\n";
	AvCodec::EncodeAudio(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-ISTFT.wav)",
		Signal.GetCRng(),
		SamplingRate,
		2,
		true
	);
}

#else

#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	TestStft();

	using namespace DragonianLib;
	const auto Env = OnnxRuntime::CreateEnvironment({ Device::CUDA, });
	Env->EnableMemPattern(true);
	OnnxRuntime::AudioDemix::CascadedNet Net(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/HP5_only_main_vocal.onnx)",
		Env,
		OnnxRuntime::AudioDemix::CascadedNet::GetPreDefinedHParams(L"4band_v2")
	);
	auto AudioInStream = AvCodec::OpenInputStream(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism.wav)"
	);
	auto [AudioData, SamplingRate] = AudioInStream.DecodeAudio(
		2, true
	);

	auto Results = Net.Forward(
		AudioData.UnSqueeze(0),
		{ SamplingRate, 85, 0.5f, 512, DefaultProgressCallback(std::cout) }
	);

	Results[0] = Functional::MinMaxNormalize(Results[0], -1, .8f, -.8f).Evaluate();
	Results[1] = Functional::MinMaxNormalize(Results[1], -1, .8f, -.8f).Evaluate();
	Results[0] = Results[0] - Results[0].Mean<true>(-1);
	Results[1] = Results[1] - Results[1].Mean<true>(-1);

	std::cout<<Results[0].ReduceMax(-1).Evaluate();
	auto OutputStream = AvCodec::OpenOutputStream(
		44100, __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Vocal.wav)"
	);
	OutputStream.EncodeAll(Results[0].GetCRng(), 44100, 2, true);
	OutputStream = AvCodec::OpenOutputStream(
		44100, __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Echoism-Instrument.wav)"
	);
	OutputStream.EncodeAll(Results[1].GetCRng(), 44100, 2, true);
#else

#endif
}