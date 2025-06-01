#include "NCNNLibrary/SingingVoiceConversion/Model/Vits-Svc.hpp"
#include "NCNNLibrary/NCNNBase/Source/NCNNImpl.hpp"

_D_Dragonian_Lib_NCNN_Singing_Voice_Conversion_Header

VitsSvc::VitsSvc(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : SingingVoiceConversionModule(Params),
NCNNModel(Params.ModelPaths.at(L"Model"), Options, _AddCache, _Logger)
{

}

SliceDatas& VitsSvc::PreprocessNoise(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Float32 Scale,
	Int64 Seed
) const
{
	if (MyData.Noise && !MyData.Noise->Null())
	{
		const auto [BatchNoise, ChannelNoise, NoiseDims, NumFramesNoise] = MyData.Noise->Shape().RawArray();
		if (BatchSize != BatchNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise batch mismatch, expected: " + std::to_string(BatchSize) + ", got: " + std::to_string(BatchNoise));
		if (Channels != ChannelNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise channels mismatch, expected: " + std::to_string(Channels) + ", got: " + std::to_string(ChannelNoise));
		if (NoiseDims != _MyNoiseDims)
			_D_Dragonian_Lib_Throw_Exception("Invalid noise dims, expected: " + std::to_string(_MyNoiseDims) + ", got: " + std::to_string(NoiseDims));
		if (NumFramesNoise != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Noise = MyData.Noise->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			););
	}
	else
	{
		m_Logger->LogInfo(L"Noise not found, generating noise with param");
		SetRandomSeed(Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Noise = (Functional::Randn(
				IDim(BatchSize, Channels, _MyNoiseDims, TargetNumFrames)
			) * Scale);
		);
	}
	return MyData;
}

SliceDatas& VitsSvc::PreprocessStftNoise(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Float32 Scale
) const
{
	_D_Dragonian_Lib_Rethrow_Block(
		MyData.StftNoise = Functional::ConstantOf(
			IDim(BatchSize, Channels, _MyWindowSize, TargetNumFrames),
			Scale
		);
		);
	return MyData;
}

SoftVitsSvcV2::SoftVitsSvcV2(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(Params, Options, _AddCache, _Logger)
{

}

SliceDatas SoftVitsSvcV2::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessF0Embed(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, m_Logger);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> SoftVitsSvcV2::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto Length = InputDatas.UnitsLength;
	auto F0Embed = InputDatas.F0Embed;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (!Length || Length->Null())
		_D_Dragonian_Lib_Throw_Exception("Units length could not be null");
	if (!F0Embed || F0Embed->Null())
		_D_Dragonian_Lib_Throw_Exception("F0 embed could not be null");
	if (HasSpeakerMixLayer()) {
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker mix could not be null");
	}
	else if (HasSpeakerEmbedding()) {
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker id could not be null");
	}

	auto Extractor = m_NCNNNet->create_extractor();
	
	_D_Dragonian_Lib_Rethrow_Block(
		Unit = ExtractorInput(0, Extractor, Unit);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Length = ExtractorInput(1, Extractor, *Length);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0Embed = ExtractorInput(2, Extractor, *F0Embed);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerMix = ExtractorInput(3, Extractor, *SpeakerMix);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerId = ExtractorInput(3, Extractor, *SpeakerId);
		);

	Tensor<Float32, 4, Device::CPU> OutputAudio;
	_D_Dragonian_Lib_Rethrow_Block(
		OutputAudio = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(L"SoftVitsSvcV2 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	return OutputAudio;
}

SoftVitsSvcV3::SoftVitsSvcV3(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : SoftVitsSvcV2(Params, Options, _AddCache, _Logger)
{

}

SliceDatas SoftVitsSvcV3::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, m_Logger);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> SoftVitsSvcV3::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto Length = InputDatas.UnitsLength;
	auto F0 = InputDatas.F0;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (!Length || Length->Null())
		_D_Dragonian_Lib_Throw_Exception("Units length could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (HasSpeakerMixLayer()) {
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker mix could not be null");
	}
	else if (HasSpeakerEmbedding()) {
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker id could not be null");
	}

	auto Extractor = m_NCNNNet->create_extractor();

	_D_Dragonian_Lib_Rethrow_Block(
		Unit = ExtractorInput(0, Extractor, Unit);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Length = ExtractorInput(1, Extractor, *Length);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0 = ExtractorInput(2, Extractor, F0);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerMix = ExtractorInput(3, Extractor, *SpeakerMix);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerId = ExtractorInput(3, Extractor, *SpeakerId);
		);

	Tensor<Float32, 4, Device::CPU> OutputAudio;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputAudio = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(L"SoftVitsSvcV3 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	return OutputAudio;
}

SoftVitsSvcV4Beta::SoftVitsSvcV4Beta(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(Params, Options, _AddCache, _Logger)
{

}

SliceDatas SoftVitsSvcV4Beta::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, 0, m_Logger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, m_Logger);
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessStftNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.StftNoiseScale);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> SoftVitsSvcV4Beta::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto F0 = InputDatas.F0;
	auto Mel2Units = InputDatas.Mel2Units;
	auto StftNoise = InputDatas.StftNoise;
	auto Noise = InputDatas.Noise;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;
	auto Volume = InputDatas.Volume;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (!Mel2Units || Mel2Units->Null())
		_D_Dragonian_Lib_Throw_Exception("Mel2Units could not be null");
	if (!StftNoise || StftNoise->Null())
		_D_Dragonian_Lib_Throw_Exception("StftNoise could not be null");
	if (!Noise || Noise->Null())
		_D_Dragonian_Lib_Throw_Exception("Noise could not be null");
	if (HasSpeakerMixLayer()) {
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker mix could not be null");
	}
	else if (HasSpeakerEmbedding()) {
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker id could not be null");
	}
	if (HasVolumeEmbedding() && (!Volume || Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	auto Extractor = m_NCNNNet->create_extractor();

	_D_Dragonian_Lib_Rethrow_Block(
		Unit = ExtractorInput(0, Extractor, Unit);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0 = ExtractorInput(1, Extractor, F0);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Mel2Units = ExtractorInput(2, Extractor, *Mel2Units);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		StftNoise = ExtractorInput(3, Extractor, *StftNoise);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Noise = ExtractorInput(4, Extractor, *Noise);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerMix = ExtractorInput(5, Extractor, *SpeakerMix);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerId = ExtractorInput(5, Extractor, *SpeakerId);
		);

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			Volume = ExtractorInput(VolumeIndex, Extractor, *Volume);
		);

	Tensor<Float32, 4, Device::CPU> OutputAudio;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputAudio = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(L"SoftVitsSvcV4Beta Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	return OutputAudio;
}

SoftVitsSvcV4::SoftVitsSvcV4(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(Params, Options, _AddCache, _Logger)
{

}

SliceDatas SoftVitsSvcV4::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, 0, m_Logger);
	PreprocessUnVoice(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, m_Logger);
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> SoftVitsSvcV4::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto F0 = InputDatas.F0;
	auto Mel2Units = InputDatas.Mel2Units;
	auto UnVoice = InputDatas.UnVoice;
	auto Noise = InputDatas.Noise;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;
	auto Volume = InputDatas.Volume;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (!Mel2Units || Mel2Units->Null())
		_D_Dragonian_Lib_Throw_Exception("Mel2Units could not be null");
	if (!UnVoice || UnVoice->Null())
		_D_Dragonian_Lib_Throw_Exception("UnVoice could not be null");
	if (!Noise || Noise->Null())
		_D_Dragonian_Lib_Throw_Exception("Noise could not be null");
	if (HasSpeakerMixLayer()) {
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker mix could not be null");
	}
	else if (HasSpeakerEmbedding()) {
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker id could not be null");
	}
	if (HasVolumeEmbedding() && (!Volume || Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	auto Extractor = m_NCNNNet->create_extractor();

	_D_Dragonian_Lib_Rethrow_Block(
		Unit = ExtractorInput(0, Extractor, Unit);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0 = ExtractorInput(1, Extractor, F0);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Mel2Units = ExtractorInput(2, Extractor, *Mel2Units);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		UnVoice = ExtractorInput(3, Extractor, *UnVoice);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Noise = ExtractorInput(4, Extractor, *Noise);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerMix = ExtractorInput(5, Extractor, *SpeakerMix);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerId = ExtractorInput(5, Extractor, *SpeakerId);
		);

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			Volume = ExtractorInput(VolumeIndex, Extractor, *Volume);
		);

	Tensor<Float32, 4, Device::CPU> OutputAudio;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputAudio = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(L"SoftVitsSvcV4 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	return OutputAudio;
}

RetrievalBasedVitsSvc::RetrievalBasedVitsSvc(
	const HParams& Params,
	const NCNNOptions& Options,
	bool _AddCache,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(Params, Options, _AddCache, _Logger)
{

}

SliceDatas RetrievalBasedVitsSvc::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, m_Logger);
	PreprocessF0Embed(MyData, BatchSize, Channels, TargetNumFrames, 0.f, m_Logger);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, m_Logger);
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, m_Logger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> RetrievalBasedVitsSvc::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto Length = InputDatas.UnitsLength;
	auto F0Embed = InputDatas.F0Embed;
	auto F0 = InputDatas.F0;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;
	auto Noise = InputDatas.Noise;
	auto Volume = InputDatas.Volume;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (!Length || Length->Null())
		_D_Dragonian_Lib_Throw_Exception("Units length could not be null");
	if (!F0Embed || F0Embed->Null())
		_D_Dragonian_Lib_Throw_Exception("F0 embedding could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (HasSpeakerMixLayer()) {
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker mix could not be null");
	}
	else if (HasSpeakerEmbedding()) {
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("Speaker id could not be null");
	}
	if (!Noise || Noise->Null())
		_D_Dragonian_Lib_Throw_Exception("Noise could not be null");
	if (HasVolumeEmbedding() && (!Volume || Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	auto Extractor = m_NCNNNet->create_extractor();

	_D_Dragonian_Lib_Rethrow_Block(
		Unit = ExtractorInput(0, Extractor, Unit);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		Length = ExtractorInput(1, Extractor, *Length);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0Embed = ExtractorInput(2, Extractor, *F0Embed);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		F0 = ExtractorInput(3, Extractor, F0);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerMix = ExtractorInput(4, Extractor, *SpeakerMix);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			SpeakerId = ExtractorInput(4, Extractor, *SpeakerId);
		);

	const auto NoiseIdx = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 5 : 4;

	_D_Dragonian_Lib_Rethrow_Block(
		Noise = ExtractorInput(NoiseIdx, Extractor, *Noise);
	);

	const auto VolumeIndex = NoiseIdx + 1;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			Volume = ExtractorInput(VolumeIndex, Extractor, *Volume);
		);

	Tensor<Float32, 4, Device::CPU> OutputAudio;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputAudio = (ExtractorOutput<Float32, 4>)(0, Extractor);
	);

#ifdef _DEBUG
	m_Logger->LogInfo(L"RetrievalBasedVitsSvc Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	return OutputAudio;
}

_D_Dragonian_Lib_NCNN_Singing_Voice_Conversion_End