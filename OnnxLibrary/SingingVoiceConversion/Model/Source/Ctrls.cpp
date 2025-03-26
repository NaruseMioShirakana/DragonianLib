#include "../Ctrls.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

Unit2Ctrl::Unit2Ctrl(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : SingingVoiceConversionModule(Params),
_MyBase(_Environment, Params.ModelPaths.at(L"Ctrl"), _Logger), _MyMelBins(Params.MelBins)
{
	if (Params.MelBins < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel bins, expected: > 0, got: " + std::to_string(Params.MelBins));

	if (_MyInputCount < 3 || _MyInputCount > 6)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 3-6, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount < 1 || _MyOutputCount > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1-4, got: " + std::to_string(_MyOutputCount));

	auto& HubertShape = _MyInputDims[0];
	auto& Mel2UnitsShape = _MyInputDims[1];
	auto& F0Shape = _MyInputDims[2];

	const auto HubertAxis = HubertShape.Size();
	const auto Mel2UnitsAxis = Mel2UnitsShape.Size();
	const auto F0Axis = F0Shape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (Mel2UnitsAxis < 1 || Mel2UnitsAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel2units axis, expected: 1-3, got: " + std::to_string(Mel2UnitsAxis));

	if (HasVolumeEmbedding())
	{
		auto& VolumeShape = _MyInputDims[3];
		const auto VolumeAxis = VolumeShape.Size();
		if (VolumeAxis < 1 || VolumeAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid volume axis, expected: 1-3, got: " + std::to_string(VolumeAxis));
	}

	const auto SpeakerAxis = HasVolumeEmbedding() ? 4 : 3;

	if (HasSpeakerMixLayer())
	{
		auto SpeakerMixShape = _MyInputDims[SpeakerAxis];
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
		const auto& SpeakerIdShape = _MyInputDims[SpeakerAxis];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}
}

Unit2Ctrl::Unit2Ctrl(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger,
	bool
) : SingingVoiceConversionModule(Params),
_MyBase(_Environment, Params.ModelPaths.at(L"Ctrl"), _Logger), _MyMelBins(Params.MelBins)
{

}

std::optional<Tensor<Float32, 4, Device::CPU>>& Unit2Ctrl::PreprocessSpec(
	std::optional<Tensor<Float32, 4, Device::CPU>>& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Int64 Seed,
	Float32 Scale,
	const DLogger& Logger
) const
{
	if (!MyData || MyData->Null())
	{
		Logger->LogInfo(L"Generating random Noise");
		SetRandomSeed(Seed);
		_D_Dragonian_Lib_Rethrow_Block(
			MyData = Functional::Randn<Float32>(
				IDim(BatchSize, Channels, _MyMelBins, TargetNumFrames),
				0.f, 1.f
			);
		);
	}
	else
	{
		const auto [B, C, M, F] = MyData->Shape().RawArray();

		if (B != BatchSize)
			_D_Dragonian_Lib_Throw_Exception("Invalid batch size, expected: " + std::to_string(BatchSize) + ", got: " + std::to_string(B));

		if (C != Channels)
			_D_Dragonian_Lib_Throw_Exception("Invalid channels, expected: " + std::to_string(Channels) + ", got: " + std::to_string(C));

		if (F != TargetNumFrames)
			_D_Dragonian_Lib_Throw_Exception("Invalid frames, expected: " + std::to_string(TargetNumFrames) + ", got: " + std::to_string(F));

		if (M != _MyMelBins)
			_D_Dragonian_Lib_Throw_Exception("Invalid mel bins, expected: " + std::to_string(_MyMelBins) + ", got: " + std::to_string(M));
	}
	MyData->Evaluate();
	if (abs(Scale - 1.f) > 1e-4)
		*MyData *= Scale;
	MyData->Evaluate();

	return MyData;
}

SliceDatas Unit2Ctrl::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate, 1);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, 0, GetLoggerPtr());
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, GetLoggerPtr());
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, GetLoggerPtr());
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, GetLoggerPtr());

	return MyData;
}

OrtTuple Unit2Ctrl::Extract(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto Mel2Units = InputDatas.Mel2Units;
	auto F0 = InputDatas.F0;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;
	auto Volume = InputDatas.Volume;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (!Mel2Units || Mel2Units->Null())
		_D_Dragonian_Lib_Throw_Exception("Mel2Units could not be null");
	if (HasSpeakerMixLayer())
	{
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("SpeakerMix could not be null");
	}
	else if (HasSpeakerEmbedding())
	{
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("SpeakerId could not be null");
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
				*Mel2Units,
				_MyInputTypes[1],
				_MyInputDims[1],
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
				F0,
				_MyInputTypes[2],
				_MyInputDims[2],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (HasVolumeEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*Volume,
					_MyInputTypes[3],
					_MyInputDims[3],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
					"Volume",
					GetLoggerPtr()
				)
			);
		);

	const auto SpeakerAxis = HasVolumeEmbedding() ? 4 : 3;

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[SpeakerAxis],
					_MyInputDims[SpeakerAxis],
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
					_MyInputTypes[SpeakerAxis],
					_MyInputDims[SpeakerAxis],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	if (SpeakerAxis + 2 == _MyInputCount)
	{
		auto SourceNoise = (Tensor<Float32, 3, Device::CPU>::Randn(
			IDim(F0.Shape(0), F0.Shape(1), F0.Shape(2) * _MyHopSize),
			0.f, 1.f
		) * Params.StftNoiseScale).Evaluate();

		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					SourceNoise,
					_MyInputTypes[SpeakerAxis + 1],
					_MyInputDims[SpeakerAxis + 1],
					{ L"Batch/Channel", L"Channel/Batch", L"NoiseDims" },
					"Noise",
					GetLoggerPtr()
				)
			);
		);
	}

	OrtTuple Tuple;

	_D_Dragonian_Lib_Rethrow_Block(
		Tuple = RunModel(InputTensors);
	);

#ifdef _DEBUG
	const auto EndTime = std::chrono::high_resolution_clock::now();
	const auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();
	GetLoggerPtr()->LogInfo(L"Unit2Ctrl finished, time: " + std::to_wstring(Duration) + L"ms");
#endif

	return Tuple;
}

Naive::Naive(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : Unit2Ctrl(_Environment, Params, _Logger)
{
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));
}

Tensor<Float32, 4, Device::CPU> Naive::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
	OrtTuple MelTuple;

	_D_Dragonian_Lib_Rethrow_Block(MelTuple = Extract(Params, InputDatas););

	auto& Mel = MelTuple[0];
	auto OutputShape = Mel.GetTensorTypeAndShapeInfo().GetShape();
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
	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Mel), OutputDims););
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End