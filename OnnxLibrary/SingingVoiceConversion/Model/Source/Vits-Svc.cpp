#include "../Vits-Svc.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

VitsSvc::VitsSvc(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : SingingVoiceConversionModule(Params),
_MyBase(_Environment, Params.ModelPaths.at(L"Model"), _Logger)
{
	const auto OutputAxis = _MyOutputDims[0].Size();
	if (OutputAxis > 4 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis");
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
			).Evaluate(););
	}
	else
	{
		LogInfo(L"Noise not found, generating noise with param");
		SetRandomSeed(Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Noise = (Functional::Randn(
				IDim(BatchSize, Channels, _MyNoiseDims, TargetNumFrames)
			) * Scale).Evaluate();
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
		).Evaluate();
		);
	return MyData;
}

SoftVitsSvcV2::SoftVitsSvcV2(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(_Environment, Params, _Logger)
{
	if (_MyInputCount != 4 && _MyInputCount != 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 3-4, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));

	const auto& HubertShape = _MyInputDims[0];
	const auto& LengthShape = _MyInputDims[1];
	const auto& F0Shape = _MyInputDims[2];

	const auto HubertAxis = HubertShape.Size();
	const auto LengthAxis = LengthShape.Size();
	const auto F0Axis = F0Shape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (LengthAxis < 1 || LengthAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid length axis, expected: 1-3, got: " + std::to_string(LengthAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));

	if (_MyInputCount == 3 && HasSpeakerEmbedding())
		_D_Dragonian_Lib_Throw_Exception("Invalid hparams, speaker embedding is enabled, but speaker embedding layer is not found");
	if (_MyInputCount == 4 && !HasSpeakerEmbedding())
		_D_Dragonian_Lib_Throw_Exception("Invalid hparams, speaker embedding is disabled, but speaker embedding layer is found");

	if (_MyInputCount == 4)
	{
		const auto& SpeakerIdShape = _MyInputDims[3];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}

	auto& OutputShape = _MyOutputDims[0];
	const auto OutputAxis = OutputShape.Size();

	if (OutputAxis > 3 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis, expected: 1-3, got: " + std::to_string(OutputAxis));

	bool Found = false;
	for (auto& UnitDim : HubertShape)
		if (UnitDim == _MyUnitsDim)
		{
			Found = true;
			break;
		}

	if (!Found && HubertShape.Back() != -1)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert dims, expected: " + std::to_string(_MyUnitsDim) + " got: " + std::to_string(HubertShape.Back()));
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

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessF0Embed(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());

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

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Length,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"Length" },
				"Length",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*F0Embed,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0Embed",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[3],
					_MyInputDims[3],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[3],
					_MyInputDims[3],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(
		AudioTuple = RunModel(InputTensors);
	);

	auto& Audio = AudioTuple[0];
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	LogInfo(L"SoftVitsSvcV2 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

SoftVitsSvcV3::SoftVitsSvcV3(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : SoftVitsSvcV2(_Environment, Params, _Logger)
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

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());

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

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Length,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"Length" },
				"Length",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				F0,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[3],
					_MyInputDims[3],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[3],
					_MyInputDims[3],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(
		AudioTuple = RunModel(InputTensors);
	);

	auto& Audio = AudioTuple[0];
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	LogInfo(L"SoftVitsSvcV3 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

SoftVitsSvcV4Beta::SoftVitsSvcV4Beta(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(_Environment, Params, _Logger)
{
	if (Params.ExtendedParameters.contains(L"NoiseDims"))
		_MyNoiseDims = _wcstoi64(Params.ExtendedParameters.at(L"NoiseDims").c_str(), nullptr, 10);
	else
		LogInfo(L"NoiseDims not found, using default value: 192");

	if (Params.ExtendedParameters.contains(L"WindowSize"))
		_MyWindowSize = _wcstoi64(Params.ExtendedParameters.at(L"WindowSize").c_str(), nullptr, 10);
	else
		LogInfo(L"WindowSize not found, using default value: 2048");

	if (_MyInputCount < 5 || _MyInputCount > 7)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 5-7, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));

	auto& OutputShape = _MyOutputDims[0];
	const auto OutputAxis = OutputShape.Size();

	if (OutputAxis > 3 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis, expected: 1-3, got: " + std::to_string(OutputAxis));

	auto& HubertShape = _MyInputDims[0];
	auto& F0Shape = _MyInputDims[1];
	auto& Mel2UnitsShape = _MyInputDims[2];
	auto& StftNoise = _MyInputDims[3];
	auto& NoiseShape = _MyInputDims[4];

	const auto HubertAxis = HubertShape.Size();
	const auto F0Axis = F0Shape.Size();
	const auto Mel2UnitsAxis = Mel2UnitsShape.Size();
	const auto StftNoiseAxis = StftNoise.Size();
	const auto NoiseAxis = NoiseShape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (Mel2UnitsAxis < 1 || Mel2UnitsAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel2units axis, expected: 1-3, got: " + std::to_string(Mel2UnitsAxis));
	if (StftNoiseAxis < 2 || StftNoiseAxis > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid unvoice axis, expected: 1-3, got: " + std::to_string(StftNoiseAxis));
	if (NoiseAxis < 2 || NoiseAxis > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid noise axis, expected: 2-4, got: " + std::to_string(NoiseAxis));

	if (HasSpeakerMixLayer())
	{
		auto SpeakerMixShape = _MyInputDims[5];
		const auto SpeakerMixAxis = SpeakerMixShape.Size();
		if (SpeakerMixAxis < 2 || SpeakerMixAxis > 4)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix axis, expected: 2-4, got: " + std::to_string(SpeakerMixAxis));
		bool FoundSpeakerCount = false;
		for (auto& SpeakerMixDim : SpeakerMixShape)
			if (SpeakerMixDim == _MySpeakerCount)
			{
				FoundSpeakerCount = true;
				break;
			}
		if (!FoundSpeakerCount)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix dims, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(SpeakerMixShape.Back()));
	}
	else if (HasSpeakerEmbedding())
	{
		const auto& SpeakerIdShape = _MyInputDims[5];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		const auto VolumeAxis = VolumeShape.Size();
		if (VolumeAxis < 1 || VolumeAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid volume axis, expected: 1-3, got: " + std::to_string(VolumeAxis));
	}

	bool Found = false;
	for (auto& UnitDim : HubertShape)
		if (UnitDim == _MyUnitsDim)
		{
			Found = true;
			break;
		}

	if (!Found && HubertShape.Back() != -1)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert dims, expected: " + std::to_string(_MyUnitsDim) + " got: " + std::to_string(HubertShape.Back()));
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

	PreprocessUnits(MyData, BatchSize, Channels, 0, GetLoggerPtr());
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessStftNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.StftNoiseScale);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());

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

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				F0,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Mel2Units,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"Mel2Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*StftNoise,
				_MyInputTypes[3],
				_MyInputDims[3],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"StftNoise",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Noise,
				_MyInputTypes[4],
				_MyInputDims[4],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"NoiseDims" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[5],
					_MyInputDims[5],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[5],
					_MyInputDims[5],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*Volume,
					_MyInputTypes[VolumeIndex],
					_MyInputDims[VolumeIndex],
					{ L"Batch/Channel", L"Channel/Batch", L"Volume" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(
		AudioTuple = RunModel(InputTensors);
	);

	auto& Audio = AudioTuple[0];
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	LogInfo(L"SoftVitsSvcV4Beta Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

SoftVitsSvcV4::SoftVitsSvcV4(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(_Environment, Params, _Logger)
{
	if (Params.ExtendedParameters.contains(L"NoiseDims"))
		_MyNoiseDims = _wcstoi64(Params.ExtendedParameters.at(L"NoiseDims").c_str(), nullptr, 10);
	else
		LogInfo(L"NoiseDims not found, using default value: 192");

	if (_MyInputCount < 5 || _MyInputCount > 7)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 5-7, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));
	
	auto& OutputShape = _MyOutputDims[0];
	const auto OutputAxis = OutputShape.Size();

	if (OutputAxis > 3 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis, expected: 1-3, got: " + std::to_string(OutputAxis));

	auto& HubertShape = _MyInputDims[0];
	auto& F0Shape = _MyInputDims[1];
	auto& Mel2UnitsShape = _MyInputDims[2];
	auto& UnVoiceShape = _MyInputDims[3];
	auto& NoiseShape = _MyInputDims[4];

	const auto HubertAxis = HubertShape.Size();
	const auto F0Axis = F0Shape.Size();
	const auto Mel2UnitsAxis = Mel2UnitsShape.Size();
	const auto UnVoiceAxis = UnVoiceShape.Size();
	const auto NoiseAxis = NoiseShape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (Mel2UnitsAxis < 1 || Mel2UnitsAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel2units axis, expected: 1-3, got: " + std::to_string(Mel2UnitsAxis));
	if (UnVoiceAxis < 1 || UnVoiceAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid unvoice axis, expected: 1-3, got: " + std::to_string(UnVoiceAxis));
	if (NoiseAxis < 2 || NoiseAxis > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid noise axis, expected: 2-4, got: " + std::to_string(NoiseAxis));

	if (HasSpeakerMixLayer())
	{
		auto SpeakerMixShape = _MyInputDims[5];
		const auto SpeakerMixAxis = SpeakerMixShape.Size();
		if (SpeakerMixAxis < 2 || SpeakerMixAxis > 4)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix axis, expected: 2-4, got: " + std::to_string(SpeakerMixAxis));
		bool FoundSpeakerCount = false;
		for (auto& SpeakerMixDim : SpeakerMixShape)
			if (SpeakerMixDim == _MySpeakerCount)
			{
				FoundSpeakerCount = true;
				break;
			}
		if (!FoundSpeakerCount)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix dims, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(SpeakerMixShape.Back()));
	}
	else if (HasSpeakerEmbedding())
	{
		const auto& SpeakerIdShape = _MyInputDims[5];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		const auto VolumeAxis = VolumeShape.Size();
		if (VolumeAxis < 1 || VolumeAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid volume axis, expected: 1-3, got: " + std::to_string(VolumeAxis));
	}

	bool Found = false;
	for (auto& UnitDim : HubertShape)
		if (UnitDim == _MyUnitsDim)
		{
			Found = true;
			break;
		}

	if (!Found && HubertShape.Back() != -1)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert dims, expected: " + std::to_string(_MyUnitsDim) + " got: " + std::to_string(HubertShape.Back()));
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

	PreprocessUnits(MyData, BatchSize, Channels, 0, GetLoggerPtr());
	PreprocessUnVoice(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());

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

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				F0,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Mel2Units,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"Mel2Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*UnVoice,
				_MyInputTypes[3],
				_MyInputDims[3],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"UnVoice",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Noise,
				_MyInputTypes[4],
				_MyInputDims[4],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"NoiseDims" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[5],
					_MyInputDims[5],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[5],
					_MyInputDims[5],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	const auto VolumeIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 6 : 5;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*Volume,
					_MyInputTypes[VolumeIndex],
					_MyInputDims[VolumeIndex],
					{ L"Batch/Channel", L"Channel/Batch", L"Volume" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(
		AudioTuple = RunModel(InputTensors);
	);

	auto& Audio = AudioTuple[0];
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	LogInfo(L"SoftVitsSvcV4 Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

RetrievalBasedVitsSvc::RetrievalBasedVitsSvc(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(_Environment, Params, _Logger)
{
	if (Params.ExtendedParameters.contains(L"NoiseDims"))
		_MyNoiseDims = _wcstoi64(Params.ExtendedParameters.at(L"NoiseDims").c_str(), nullptr, 10);
	else
		LogInfo(L"NoiseDims not found, using default value: 192");

	if (_MyInputCount < 5 || _MyInputCount > 7)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 5-7, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));

	auto& OutputShape = _MyOutputDims[0];
	const auto OutputAxis = OutputShape.Size();

	if (OutputAxis > 3 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis, expected: 1-3, got: " + std::to_string(OutputAxis));

	const auto& HubertShape = _MyInputDims[0];
	const auto& LengthShape = _MyInputDims[1];
	const auto& F0EmbedShape = _MyInputDims[2];
	const auto& F0Shape = _MyInputDims[3];

	const auto HubertAxis = HubertShape.Size();
	const auto LengthAxis = LengthShape.Size();
	const auto F0EmbedAxis = F0EmbedShape.Size();
	const auto F0Axis = F0Shape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (LengthAxis < 1 || LengthAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid length axis, expected: 1-3, got: " + std::to_string(LengthAxis));
	if (F0EmbedAxis < 1 || F0EmbedAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));

	if (HasSpeakerMixLayer())
	{
		auto SpeakerMixShape = _MyInputDims[4];
		const auto SpeakerMixAxis = SpeakerMixShape.Size();
		if (SpeakerMixAxis < 2 || SpeakerMixAxis > 4)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix axis, expected: 2-4, got: " + std::to_string(SpeakerMixAxis));
		bool FoundSpeakerCount = false;
		for (auto& SpeakerMixDim : SpeakerMixShape)
			if (SpeakerMixDim == _MySpeakerCount)
			{
				FoundSpeakerCount = true;
				break;
			}
		if (!FoundSpeakerCount)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix dims, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(SpeakerMixShape.Back()));
	}
	else if (HasSpeakerEmbedding())
	{
		const auto& SpeakerIdShape = _MyInputDims[4];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}

	const auto NoiseIdx = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 5 : 4;
	const auto& NoiseShape = _MyInputDims[NoiseIdx];
	const auto NoiseAxis = NoiseShape.Size();

	if (NoiseAxis < 2 || NoiseAxis > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid noise axis, expected: 2-4, got: " + std::to_string(NoiseAxis));

	const auto VolumeIndex = NoiseIdx + 1;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		const auto VolumeAxis = VolumeShape.Size();
		if (VolumeAxis < 1 || VolumeAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid volume axis, expected: 1-3, got: " + std::to_string(VolumeAxis));
	}

	bool Found = false;
	for (auto& UnitDim : HubertShape)
		if (UnitDim == _MyUnitsDim)
		{
			Found = true;
			break;
		}

	if (!Found && HubertShape.Back() != -1)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert dims, expected: " + std::to_string(_MyUnitsDim) + " got: " + std::to_string(HubertShape.Back()));
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

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessF0Embed(MyData, BatchSize, Channels, TargetNumFrames, 0.f, GetLoggerPtr());
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());

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

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Length,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"Length" },
				"Length",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*F0Embed,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0Embed",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				F0,
				_MyInputTypes[3],
				_MyInputDims[3],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[4],
					_MyInputDims[4],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[4],
					_MyInputDims[4],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	const auto NoiseIdx = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 5 : 4;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Noise,
				_MyInputTypes[NoiseIdx],
				_MyInputDims[NoiseIdx],
				{ L"Batch/Channel", L"Channel/Batch", L"NoiseDims", L"AudioFrames" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	const auto VolumeIndex = NoiseIdx + 1;

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*Volume,
					_MyInputTypes[VolumeIndex],
					_MyInputDims[VolumeIndex],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(
		AudioTuple = RunModel(InputTensors);
	);

	auto& Audio = AudioTuple[0];
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	LogInfo(L"RetrievalBasedVitsSvc Inference finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End