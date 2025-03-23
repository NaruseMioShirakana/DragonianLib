#include "../Vits-Svc.hpp"

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

Tensor<Float32, 4, Device::CPU> VitsSvc::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	if (static_cast<Int64>(InputDatas.OrtValues.size()) != _MyInputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: " + std::to_string(_MyInputCount) + ", got: " + std::to_string(InputDatas.OrtValues.size()));

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(AudioTuple = _MyModel->Run(
		*_MyRunOptions,
		_MyInputNames.Data(),
		InputDatas.OrtValues.data(),
		_MyInputCount,
		_MyOutputNames.Data(),
		_MyOutputCount
	););

	auto Audio = std::move(AudioTuple[0]);
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	//const auto OutputElementCount = Audio.GetTensorTypeAndShapeInfo().GetElementCount();
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
	LogInfo(L"VitsSvc::Forward finished, time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - StartTime).count()) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
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

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	if (MyData.F0.Null() && (!MyData.F0Embed || MyData.F0Embed->Empty()))
		_D_Dragonian_Lib_Throw_Exception("F0 and F0Embed could not be null at the same time");

	if (MyData.SourceSampleCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample count, expected: > 0, got: " + std::to_string(MyData.SourceSampleCount));

	if (MyData.SourceSampleRate <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample rate, expected: > 0, got: " + std::to_string(MyData.SourceSampleRate));

	const auto InputAudioDuration = static_cast<Float32>(MyData.SourceSampleCount) /
		static_cast<Float32>(MyData.SourceSampleRate);
	const auto TargetNumFramesPerSecond = static_cast<Float32>(_MyOutputSamplingRate) /
		static_cast<Float32>(_MyHopSize);
	const auto TargetNumFrames = static_cast<Int64>(std::ceil(InputAudioDuration * TargetNumFramesPerSecond));

	if (TargetNumFrames != MyData.Units.Shape(-2))
		_D_Dragonian_Lib_Rethrow_Block(MyData.Units = MyData.Units.Interpolate<Operators::InterpolateMode::Nearest>(
			IDim(-2),
			{ IDim(TargetNumFrames) }
		).Evaluate(););

	const auto [Batch, Channel, NumFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	if (MyData.UnitsLength.has_value() && !MyData.UnitsLength->Null())
	{
		const auto [BatchLength, ChannelLength, Length] = MyData.UnitsLength->Shape().RawArray();
		if (BatchLength != Batch)
			_D_Dragonian_Lib_Throw_Exception("Units and units length batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchLength));
		if (ChannelLength != Channel)
			_D_Dragonian_Lib_Throw_Exception("Units and units length channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelLength));
		if (Length != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid units length length");
	}
	else
	{
		LogInfo(L"Units length not found, generating units length with units");
		using UnitsSizeType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.UnitsLength = UnitsSizeType::ConstantOf({ Batch, Channel, 1 }, NumFrames).Evaluate(););
	}

	if (HasSpeakerEmbedding() && MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
	{
		const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
		if (Batch != BatchSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
		if (Channel != ChannelSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
		if (LengthSpeakerId != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
	}
	else
	{
		LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
		using SpkType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
	}

	if (MyData.F0Embed.has_value() && !MyData.F0Embed->Null())
	{
		const auto [BatchF0Embed, ChannelF0Embed, NumFramesF0Embed] = MyData.F0Embed->Shape().RawArray();
		if (Batch != BatchF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0Embed));
		if (Channel != ChannelF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0Embed));
		if (NumFrames != NumFramesF0Embed)
			_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = MyData.F0Embed->Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"F0 embedding not found, generating f0 embedding with f0 tensor");
		if (MyData.F0.Null())
			_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		if (NumFramesF0 != NumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.F0 = MyData.F0.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););

		(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();

		_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = GetF0Embed(MyData.F0, static_cast<Float32>(_MyF0Bin), _MyF0MelMax, _MyF0MelMin););
	}

	auto& HubertShape = _MyInputDims[0];
	auto& LengthShape = _MyInputDims[1];
	auto& F0Shape = _MyInputDims[2];

	MyData.Clear();

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Units,
				_MyInputTypes[0],
				HubertShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.UnitsLength.value(),
				_MyInputTypes[1],
				LengthShape,
				{ L"Batch/Channel", L"Batch/Channel", L"Length" },
				"UnitsLength",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0Embed.value(),
				_MyInputTypes[2],
				F0Shape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0Embed",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerEmbedding())
	{
		auto& SpeakerIdShape = _MyInputDims[3];
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MyData.SpeakerId.value(),
					_MyInputTypes[3],
					SpeakerIdShape,
					{ L"Batch/Channel", L"Batch/Channel", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);
	}

	return MyData;
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

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	if (MyData.F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

	if (MyData.SourceSampleCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample count, expected: > 0, got: " + std::to_string(MyData.SourceSampleCount));

	if (MyData.SourceSampleRate <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample rate, expected: > 0, got: " + std::to_string(MyData.SourceSampleRate));

	const auto InputAudioDuration = static_cast<Float32>(MyData.SourceSampleCount) /
		static_cast<Float32>(MyData.SourceSampleRate);
	const auto TargetNumFramesPerSecond = static_cast<Float32>(_MyOutputSamplingRate) /
		static_cast<Float32>(_MyHopSize);
	const auto TargetNumFrames = static_cast<Int64>(std::ceil(InputAudioDuration * TargetNumFramesPerSecond));

	if (TargetNumFrames != MyData.Units.Shape(-2))
		_D_Dragonian_Lib_Rethrow_Block(MyData.Units = MyData.Units.Interpolate<Operators::InterpolateMode::Nearest>(
			IDim(-2),
			{ IDim(TargetNumFrames) }
		).Evaluate(););

	const auto [Batch, Channel, NumFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	if (MyData.UnitsLength.has_value() && !MyData.UnitsLength->Null())
	{
		const auto [BatchLength, ChannelLength, Length] = MyData.UnitsLength->Shape().RawArray();
		if (BatchLength != Batch)
			_D_Dragonian_Lib_Throw_Exception("Units and units length batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchLength));
		if (ChannelLength != Channel)
			_D_Dragonian_Lib_Throw_Exception("Units and units length channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelLength));
		if (Length != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid units length length");
	}
	else
	{
		LogInfo(L"Units length not found, generating units length with units");
		using UnitsSizeType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.UnitsLength = UnitsSizeType::ConstantOf({ Batch, Channel, 1 }, NumFrames).Evaluate(););
	}

	if (HasSpeakerEmbedding() && MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
	{
		const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
		if (Batch != BatchSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
		if (Channel != ChannelSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
		if (LengthSpeakerId != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
	}
	else
	{
		LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
		using SpkType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
	}

	const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
	if (Batch != BatchF0)
		_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
	if (Channel != ChannelF0)
		_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

	if (NumFramesF0 != NumFrames)
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.F0 = InterpolateUnVoicedF0(
				MyData.F0
			).Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate();
		);

	(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();

	auto& HubertShape = _MyInputDims[0];
	auto& LengthShape = _MyInputDims[1];
	auto& F0Shape = _MyInputDims[2];

	MyData.Clear();

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Units,
				_MyInputTypes[0],
				HubertShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.UnitsLength.value(),
				_MyInputTypes[1],
				LengthShape,
				{ L"Batch/Channel", L"Batch/Channel", L"Length" },
				"UnitsLength",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0,
				_MyInputTypes[2],
				F0Shape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerEmbedding())
	{
		auto& SpeakerIdShape = _MyInputDims[3];
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MyData.SpeakerId.value(),
					_MyInputTypes[3],
					SpeakerIdShape,
					{ L"Batch/Channel", L"Batch/Channel", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);
	}

	return MyData;
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

	if (Params.ExtendedParameters.contains(L"NoiseDims"))
		_MyNoiseDims = _wcstoi64(Params.ExtendedParameters.at(L"NoiseDims").c_str(), nullptr, 10);
	else
		LogInfo(L"NoiseDims not found, using default value: 192");

	if (_MyInputCount < 5 || _MyInputCount > 7)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 4-6, got: " + std::to_string(_MyInputCount));
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

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	if (MyData.F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

	if (HasVolumeEmbedding() && (!MyData.Volume || MyData.Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	if (MyData.SourceSampleCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample count, expected: > 0, got: " + std::to_string(MyData.SourceSampleCount));

	if (MyData.SourceSampleRate <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample rate, expected: > 0, got: " + std::to_string(MyData.SourceSampleRate));

	const auto InputAudioDuration = static_cast<Float32>(MyData.SourceSampleCount) /
		static_cast<Float32>(MyData.SourceSampleRate);
	const auto TargetNumFramesPerSecond = static_cast<Float32>(_MyOutputSamplingRate) /
		static_cast<Float32>(_MyHopSize);
	const auto TargetNumFrames = static_cast<Int64>(std::ceil(InputAudioDuration * TargetNumFramesPerSecond)) + 1;

	const auto [Batch, Channel, UnitFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	{
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		if (NumFramesF0 != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.F0 = InterpolateUnVoicedF0(
					MyData.F0
				).Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				).Evaluate();
			);

		(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();
	}

	if (MyData.Mel2Units && !MyData.Mel2Units->Null())
	{
		const auto [BatchMel2Units, ChannelMel2Units, NumFramesMel2Units] = MyData.Mel2Units->Shape().RawArray();
		if (Batch != BatchMel2Units)
			_D_Dragonian_Lib_Throw_Exception("Units and mel2units batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchMel2Units));
		if (Channel != ChannelMel2Units)
			_D_Dragonian_Lib_Throw_Exception("Units and mel2units channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelMel2Units));
		if (NumFramesMel2Units != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Mel2Units = MyData.Mel2Units->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"Mel2Units not found, generating mel2units with units");
		auto MyMel2Unit = Functional::Linspace(0.f, static_cast<Float32>(UnitFrames), TargetNumFrames).Cast<Int64>().Evaluate();
		if (MyMel2Unit[-1] > UnitFrames - 1)
			MyMel2Unit[-1] = UnitFrames - 1;
		const auto BatchChannels = Batch * Channel;
		if (BatchChannels > 1)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.UnSqueeze(0).Repeat({ BatchChannels }).View(Batch, Channel, TargetNumFrames).Evaluate();
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.View(Batch, Channel, TargetNumFrames).Evaluate();
			);

	}

	if (MyData.Noise && !MyData.Noise->Null())
	{
		const auto [BatchNoise, ChannelNoise, NoiseDims, NumFramesNoise] = MyData.Noise->Shape().RawArray();
		if (Batch != BatchNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchNoise));
		if (Channel != ChannelNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelNoise));
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
		SetRandomSeed(Params.Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Noise = Functional::Randn(IDim(Batch, Channel, _MyNoiseDims, TargetNumFrames)).Evaluate();
		);
	}

	if (abs(Params.NoiseScale - 1.f) > 1e-4f)
		(*MyData.Noise *= Params.NoiseScale).Evaluate();

	if (HasSpeakerMixLayer())
	{
		if (MyData.Speaker.has_value() && !MyData.Speaker->Null())
		{
			const auto [BatchSpeaker, ChannelSpeaker, NumFramesSpeaker, NumSpeakers] = MyData.Speaker->Shape().RawArray();
			if (Batch != BatchSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeaker));
			if (Channel != ChannelSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeaker));
			if (NumSpeakers != _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker count, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(NumSpeakers));
			if (NumFramesSpeaker != TargetNumFrames)
				_D_Dragonian_Lib_Rethrow_Block(MyData.Speaker = MyData.Speaker->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-2),
					{ IDim(TargetNumFrames) }
				).Evaluate(););
		}
		else
		{
			LogInfo(L"Speaker not found, generating speaker with mel2units");
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Speaker = Functional::Zeros(IDim(Batch, Channel, TargetNumFrames, _MySpeakerCount)).Evaluate();
			);
			(MyData.Speaker.value()[{":", ":", ":", std::to_string(Params.SpeakerId)}] = 1.f).Evaluate();
		}
	}
	else if (HasSpeakerEmbedding())
	{
		if (MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
		{
			const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
			if (Batch != BatchSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
			if (Channel != ChannelSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
			if (LengthSpeakerId != 1)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
		}
		else
		{
			LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
			using SpkType = Tensor<Int64, 3, Device::CPU>;
			_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
		}
	}

	if (HasVolumeEmbedding())
	{
		const auto [BatchVolume, ChannelVolume, NumFramesVolume] = MyData.Volume->Shape().RawArray();
		if (Batch != BatchVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchVolume));
		if (Channel != ChannelVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelVolume));
		if (NumFramesVolume != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Volume = MyData.Volume->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}

	auto& HubertShape = _MyInputDims[0];
	auto& F0Shape = _MyInputDims[1];
	auto& Mel2UnitsShape = _MyInputDims[2];
	auto& StftNoiseShape = _MyInputDims[3];
	auto& NoiseShape = _MyInputDims[4];

	MyData.Clear();

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Units,
				_MyInputTypes[0],
				HubertShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0,
				_MyInputTypes[1],
				F0Shape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Mel2Units.value(),
				_MyInputTypes[2],
				Mel2UnitsShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"Mel2Units",
				GetLoggerPtr()
			)
		);
	);


	{
		Tensor<Float32, 4, Device::CPU> STFTNoise;
		_D_Dragonian_Lib_Rethrow_Block(
			STFTNoise = Functional::ConstantOf(IDim(Batch, Channel, _MyWindowSize, TargetNumFrames), Params.StftNoiseScale).Evaluate();
		);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					STFTNoise,
					_MyInputTypes[3],
					StftNoiseShape,
					{ L"Batch/Channel", L"Batch/Channel", L"WindowSize", L"AudioFrames" },
					"STFTNoise",
					GetLoggerPtr()
				)
			);
		);
	}

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Noise.value(),
				_MyInputTypes[4],
				NoiseShape,
				{ L"Batch/Channel", L"Batch/Channel", L"NoiseDims", L"AudioFrames" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerEmbedding() || HasSpeakerMixLayer())
	{
		auto& SpeakerShape = _MyInputDims[5];
		if (HasSpeakerMixLayer())
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.Speaker.value(),
						_MyInputTypes[5],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"SpeakerCount" },
						"Speaker",
						GetLoggerPtr()
					)
				);
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.SpeakerId.value(),
						_MyInputTypes[5],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"SpeakerId" },
						"SpeakerId",
						GetLoggerPtr()
					)
				);
			);
	}

	const auto VolumeIndex = (HasSpeakerEmbedding() || HasSpeakerMixLayer()) ? 6 : 5;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MyData.Volume.value(),
					_MyInputTypes[VolumeIndex],
					VolumeShape,
					{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);
	}

	return MyData;
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
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 4-6, got: " + std::to_string(_MyInputCount));
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

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	if (MyData.F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

	if (HasVolumeEmbedding() && (!MyData.Volume || MyData.Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	if (MyData.SourceSampleCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample count, expected: > 0, got: " + std::to_string(MyData.SourceSampleCount));

	if (MyData.SourceSampleRate <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample rate, expected: > 0, got: " + std::to_string(MyData.SourceSampleRate));

	const auto InputAudioDuration = static_cast<Float32>(MyData.SourceSampleCount) /
		static_cast<Float32>(MyData.SourceSampleRate);
	const auto TargetNumFramesPerSecond = static_cast<Float32>(_MyOutputSamplingRate) /
		static_cast<Float32>(_MyHopSize);
	const auto TargetNumFrames = static_cast<Int64>(std::ceil(InputAudioDuration * TargetNumFramesPerSecond)) + 1;

	const auto [Batch, Channel, UnitFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	if (MyData.Mel2Units && !MyData.Mel2Units->Null())
	{
		const auto [BatchMel2Units, ChannelMel2Units, NumFramesMel2Units] = MyData.Mel2Units->Shape().RawArray();
		if (Batch != BatchMel2Units)
			_D_Dragonian_Lib_Throw_Exception("Units and mel2units batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchMel2Units));
		if (Channel != ChannelMel2Units)
			_D_Dragonian_Lib_Throw_Exception("Units and mel2units channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelMel2Units));
		if (NumFramesMel2Units != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Mel2Units = MyData.Mel2Units->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"Mel2Units not found, generating mel2units with units");
		auto MyMel2Unit = Functional::Linspace(0.f, static_cast<Float32>(UnitFrames), TargetNumFrames).Cast<Int64>().Evaluate();
		if (MyMel2Unit[-1] > UnitFrames - 1)
			MyMel2Unit[-1] = UnitFrames - 1;
		const auto BatchChannels = Batch * Channel;
		if (BatchChannels > 1)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.UnSqueeze(0).Repeat({ BatchChannels } ).View(Batch, Channel, TargetNumFrames).Evaluate();
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.View(Batch, Channel, TargetNumFrames).Evaluate();
			);
		
	}

	if (MyData.UnVoice && !MyData.UnVoice->Null())
	{
		const auto [BatchUnVoice, ChannelUnVoice, NumFramesUnVoice] = MyData.UnVoice->Shape().RawArray();
		if (Batch != BatchUnVoice)
			_D_Dragonian_Lib_Throw_Exception("Units and unvoice batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchUnVoice));
		if (Channel != ChannelUnVoice)
			_D_Dragonian_Lib_Throw_Exception("Units and unvoice channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelUnVoice));
		if (NumFramesUnVoice != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.UnVoice = MyData.UnVoice->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"UnVoice not found, generating unvoice with f0");
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.UnVoice = (MyData.F0 > 1e-4f).Cast<Float32>();
		);
		if (MyData.UnVoice->Shape(-1) != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.UnVoice = MyData.UnVoice->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}

	{
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		if (NumFramesF0 != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.F0 = InterpolateUnVoicedF0(
					MyData.F0
				).Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				).Evaluate();
			);

		(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();
	}

	if (MyData.Noise && !MyData.Noise->Null())
	{
		const auto [BatchNoise, ChannelNoise, NoiseDims, NumFramesNoise] = MyData.Noise->Shape().RawArray();
		if (Batch != BatchNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchNoise));
		if (Channel != ChannelNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelNoise));
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
		SetRandomSeed(Params.Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Noise = Functional::Randn(IDim(Batch, Channel, _MyNoiseDims, TargetNumFrames)).Evaluate();
		);
	}

	if (abs(Params.NoiseScale - 1.f) > 1e-4f)
		(*MyData.Noise *= Params.NoiseScale).Evaluate();

	if (HasSpeakerMixLayer())
	{
		if (MyData.Speaker.has_value() && !MyData.Speaker->Null())
		{
			const auto [BatchSpeaker, ChannelSpeaker, NumFramesSpeaker, NumSpeakers] = MyData.Speaker->Shape().RawArray();
			if (Batch != BatchSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeaker));
			if (Channel != ChannelSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeaker));
			if (NumSpeakers != _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker count, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(NumSpeakers));
			if (NumFramesSpeaker != TargetNumFrames)
				_D_Dragonian_Lib_Rethrow_Block(MyData.Speaker = MyData.Speaker->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-2),
					{ IDim(TargetNumFrames) }
				).Evaluate(););
		}
		else
		{
			LogInfo(L"Speaker not found, generating speaker with mel2units");
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Speaker = Functional::Zeros(IDim(Batch, Channel, TargetNumFrames, _MySpeakerCount)).Evaluate();
			);
			(MyData.Speaker.value()[{":", ":", ":", std::to_string(Params.SpeakerId)}] = 1.f).Evaluate();
		}
	}
	else if (HasSpeakerEmbedding())
	{
		if (MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
		{
			const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
			if (Batch != BatchSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
			if (Channel != ChannelSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
			if (LengthSpeakerId != 1)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
		}
		else
		{
			LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
			using SpkType = Tensor<Int64, 3, Device::CPU>;
			_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
		}
	}

	if (HasVolumeEmbedding())
	{
		const auto [BatchVolume, ChannelVolume, NumFramesVolume] = MyData.Volume->Shape().RawArray();
		if (Batch != BatchVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchVolume));
		if (Channel != ChannelVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelVolume));
		if (NumFramesVolume != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Volume = MyData.Volume->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}

	auto& HubertShape = _MyInputDims[0];
	auto& F0Shape = _MyInputDims[1];
	auto& Mel2UnitsShape = _MyInputDims[2];
	auto& UnVoiceShape = _MyInputDims[3];
	auto& NoiseShape = _MyInputDims[4];

	MyData.Clear();

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Units,
				_MyInputTypes[0],
				HubertShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0,
				_MyInputTypes[1],
				F0Shape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Mel2Units.value(),
				_MyInputTypes[2],
				Mel2UnitsShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"Mel2Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.UnVoice.value(),
				_MyInputTypes[3],
				UnVoiceShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"UnVoice",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Noise.value(),
				_MyInputTypes[4],
				NoiseShape,
				{ L"Batch/Channel", L"Batch/Channel", L"NoiseDims", L"AudioFrames" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerEmbedding() || HasSpeakerMixLayer())
	{
		auto& SpeakerShape = _MyInputDims[5];
		if (HasSpeakerMixLayer())
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.Speaker.value(),
						_MyInputTypes[5],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"SpeakerCount" },
						"Speaker",
						GetLoggerPtr()
					)
				);
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.SpeakerId.value(),
						_MyInputTypes[5],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"SpeakerId" },
						"SpeakerId",
						GetLoggerPtr()
					)
				);
			);
	}

	const auto VolumeIndex = (HasSpeakerEmbedding() || HasSpeakerMixLayer()) ? 6 : 5;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MyData.Volume.value(),
					_MyInputTypes[VolumeIndex],
					VolumeShape,
					{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);
	}

	return MyData;
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
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 4-6, got: " + std::to_string(_MyInputCount));
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

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	if (MyData.F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

	if (HasVolumeEmbedding() && (!MyData.Volume || MyData.Volume->Null()))
		_D_Dragonian_Lib_Throw_Exception("Volume could not be null");

	if (MyData.SourceSampleCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample count, expected: > 0, got: " + std::to_string(MyData.SourceSampleCount));

	if (MyData.SourceSampleRate <= 0)
		_D_Dragonian_Lib_Throw_Exception("Invalid source sample rate, expected: > 0, got: " + std::to_string(MyData.SourceSampleRate));

	const auto InputAudioDuration = static_cast<Float32>(MyData.SourceSampleCount) /
		static_cast<Float32>(MyData.SourceSampleRate);
	const auto TargetNumFramesPerSecond = static_cast<Float32>(_MyOutputSamplingRate) /
		static_cast<Float32>(_MyHopSize);
	const auto TargetNumFrames = static_cast<Int64>(std::ceil(InputAudioDuration * TargetNumFramesPerSecond));

	if (TargetNumFrames != MyData.Units.Shape(-2))
		_D_Dragonian_Lib_Rethrow_Block(MyData.Units = MyData.Units.Interpolate<Operators::InterpolateMode::Nearest>(
			IDim(-2),
			{ IDim(TargetNumFrames) }
		).Evaluate(););

	const auto [Batch, Channel, NumFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	if (MyData.UnitsLength.has_value() && !MyData.UnitsLength->Null())
	{
		const auto [BatchLength, ChannelLength, Length] = MyData.UnitsLength->Shape().RawArray();
		if (BatchLength != Batch)
			_D_Dragonian_Lib_Throw_Exception("Units and units length batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchLength));
		if (ChannelLength != Channel)
			_D_Dragonian_Lib_Throw_Exception("Units and units length channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelLength));
		if (Length != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid units length length");
	}
	else
	{
		LogInfo(L"Units length not found, generating units length with units");
		using UnitsSizeType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.UnitsLength = UnitsSizeType::ConstantOf({ Batch, Channel, 1 }, NumFrames).Evaluate(););
	}

	{
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		if (NumFramesF0 != NumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.F0 = InterpolateUnVoicedF0(
					MyData.F0
				).Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					{ IDim(NumFrames) }
				).Evaluate();
			);

		(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();
	}

	if (MyData.F0Embed.has_value() && !MyData.F0Embed->Null())
	{
		const auto [BatchF0Embed, ChannelF0Embed, NumFramesF0Embed] = MyData.F0Embed->Shape().RawArray();
		if (Batch != BatchF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0Embed));
		if (Channel != ChannelF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0Embed));
		if (NumFrames != NumFramesF0Embed)
			_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = MyData.F0Embed->Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"F0 embedding not found, generating f0 embedding with f0 tensor");
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = GetF0Embed(MyData.F0, static_cast<Float32>(_MyF0Bin), _MyF0MelMax, _MyF0MelMin););
	}

	if (HasSpeakerMixLayer())
	{
		if (MyData.Speaker.has_value() && !MyData.Speaker->Null())
		{
			const auto [BatchSpeaker, ChannelSpeaker, NumFramesSpeaker, NumSpeakers] = MyData.Speaker->Shape().RawArray();
			if (Batch != BatchSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeaker));
			if (Channel != ChannelSpeaker)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeaker));
			if (NumSpeakers != _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker count, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(NumSpeakers));
			if (NumFramesSpeaker != TargetNumFrames)
				_D_Dragonian_Lib_Rethrow_Block(MyData.Speaker = MyData.Speaker->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-2),
					{ IDim(TargetNumFrames) }
				).Evaluate(););
		}
		else
		{
			LogInfo(L"Speaker not found, generating speaker with mel2units");
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Speaker = Functional::Zeros(IDim(Batch, Channel, TargetNumFrames, _MySpeakerCount)).Evaluate();
			);
			(MyData.Speaker.value()[{":", ":", ":", std::to_string(Params.SpeakerId)}] = 1.f).Evaluate();
		}
	}
	else if (HasSpeakerEmbedding())
	{
		if (MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
		{
			const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
			if (Batch != BatchSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
			if (Channel != ChannelSpeakerId)
				_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
			if (LengthSpeakerId != 1)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
		}
		else
		{
			LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
			using SpkType = Tensor<Int64, 3, Device::CPU>;
			_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
		}
	}

	if (MyData.Noise && !MyData.Noise->Null())
	{
		const auto [BatchNoise, ChannelNoise, NoiseDims, NumFramesNoise] = MyData.Noise->Shape().RawArray();
		if (Batch != BatchNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchNoise));
		if (Channel != ChannelNoise)
			_D_Dragonian_Lib_Throw_Exception("Units and noise channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelNoise));
		if (NoiseDims != _MyNoiseDims)
			_D_Dragonian_Lib_Throw_Exception("Invalid noise dims, expected: " + std::to_string(_MyNoiseDims) + ", got: " + std::to_string(NoiseDims));
		if (NumFramesNoise != NumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Noise = MyData.Noise->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"Noise not found, generating noise with param");
		SetRandomSeed(Params.Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Noise = Functional::Randn(IDim(Batch, Channel, _MyNoiseDims, NumFrames)).Evaluate();
		);
	}

	if (abs(Params.NoiseScale - 1.f) > 1e-4f)
		(*MyData.Noise *= Params.NoiseScale).Evaluate();

	if (HasVolumeEmbedding())
	{
		const auto [BatchVolume, ChannelVolume, NumFramesVolume] = MyData.Volume->Shape().RawArray();
		if (Batch != BatchVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchVolume));
		if (Channel != ChannelVolume)
			_D_Dragonian_Lib_Throw_Exception("Units and volume channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelVolume));
		if (NumFramesVolume != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.Volume = MyData.Volume->Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			).Evaluate(););
	}

	auto& HubertShape = _MyInputDims[0];
	auto& LengthShape = _MyInputDims[1];
	auto& F0EmbedShape = _MyInputDims[2];
	auto& F0Shape = _MyInputDims[3];

	MyData.Clear();

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Units,
				_MyInputTypes[0],
				HubertShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.UnitsLength.value(),
				_MyInputTypes[1],
				LengthShape,
				{ L"Batch/Channel", L"Batch/Channel", L"Length" },
				"UnitsLength",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0Embed.value(),
				_MyInputTypes[2],
				F0EmbedShape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0Embed",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.F0,
				_MyInputTypes[3],
				F0Shape,
				{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerEmbedding() || HasSpeakerMixLayer())
	{
		auto& SpeakerShape = _MyInputDims[4];
		if (HasSpeakerMixLayer())
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.Speaker.value(),
						_MyInputTypes[4],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames", L"SpeakerCount" },
						"Speaker",
						GetLoggerPtr()
					)
				);
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						MyData.SpeakerId.value(),
						_MyInputTypes[4],
						SpeakerShape,
						{ L"Batch/Channel", L"Batch/Channel", L"SpeakerId" },
						"SpeakerId",
						GetLoggerPtr()
					)
				);
			);
	}

	const auto NoiseIndex = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 5 : 4;

	_D_Dragonian_Lib_Rethrow_Block(
		MyData.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MyData.Noise.value(),
				_MyInputTypes[NoiseIndex],
				_MyInputDims[NoiseIndex],
				{ L"Batch/Channel", L"Batch/Channel", L"NoiseDims", L"AudioFrames" },
				"Noise",
				GetLoggerPtr()
			)
		);
	);

	const auto VolumeIndex = NoiseIndex + 1;

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[VolumeIndex];
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MyData.Volume.value(),
					_MyInputTypes[VolumeIndex],
					VolumeShape,
					{ L"Batch/Channel", L"Batch/Channel", L"AudioFrames" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);
	}

	return MyData;
}


_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End