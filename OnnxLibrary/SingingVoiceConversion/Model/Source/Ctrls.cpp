#include "../Ctrls.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

Unit2Ctrl::Unit2Ctrl(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : SingingVoiceConversionModule(Params),
_MyBase(_Environment, Params.ModelPaths.at(L"Model"), _Logger)
{

};

SliceDatas Unit2Ctrl::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);



	return MyData;
}



_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End