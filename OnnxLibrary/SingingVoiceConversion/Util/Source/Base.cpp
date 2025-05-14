#include "OnnxLibrary/SingingVoiceConversion/Util/Base.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

SingingVoiceConversionModule::SingingVoiceConversionModule(
	const HParams& Params
) :
	_MyOutputSamplingRate(Params.OutputSamplingRate), _MyUnitsDim(Params.UnitsDim), _MyHopSize(Params.HopSize),
	_MySpeakerCount(Params.SpeakerCount), _MySpecMax(Params.SpecMax), _MySpecMin(Params.SpecMin),
	_MyF0Bin(Params.F0Bin), _MyF0Max(Params.F0Max), _MyF0Min(Params.F0Min),
	_MyF0MelMax(1127.f * log(1.f + Params.F0Max / 700.f)), _MyF0MelMin(1127.f * log(1.f + Params.F0Min / 700.f)),
	_HasVolumeEmbedding(Params.HasVolumeEmbedding), _HasSpeakerEmbedding(Params.HasSpeakerEmbedding),
	_HasSpeakerMixLayer(Params.HasSpeakerMixLayer), _MyProgressCallback(Params.ProgressCallback)
{
	if (_MyOutputSamplingRate <= 4410)
		_D_Dragonian_Lib_Throw_Exception("Output sampling rate must be greater than 4410.");
	if (_MyUnitsDim <= 64)
		_D_Dragonian_Lib_Throw_Exception("Units dimension must be greater than 64.");
	if (_MyHopSize <= 64)
		_D_Dragonian_Lib_Throw_Exception("Hop size must be greater than 64.");
	if (_MySpeakerCount <= 0)
		_D_Dragonian_Lib_Throw_Exception("Speaker count must be greater than 0.");
	if (_MySpecMax <= _MySpecMin)
		_D_Dragonian_Lib_Throw_Exception("Spec max must be greater than spec min.");
	if (_MyF0Bin <= 0)
		_D_Dragonian_Lib_Throw_Exception("F0 bin must be greater than 0.");
	if (_MyF0Max <= _MyF0Min)
		_D_Dragonian_Lib_Throw_Exception("F0 max must be greater than f0 min.");
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::Inference(
	const Parameters& Params,
	const Tensor<Float32, 3, Device::CPU>& Audio,
	SizeType SourceSampleRate,
	const FeatureExtractor& UnitsEncoder,
	const PitchExtractor& F0Extractor,
	const PitchParameters& F0Params,
	std::optional<std::reference_wrapper<const Cluster>> UnitsCluster,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> AudioMask,
	SliceDatas* OutPointers
) const
{
	if (Audio.Null())
		_D_Dragonian_Lib_Throw_Exception("Audio could not be null");
	if (SourceSampleRate <= 4410)
		_D_Dragonian_Lib_Throw_Exception("Source sample rate must be greater than 4410.");
	if (UnitsEncoder->GetUnitsDims() != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitsEncoder->GetUnitsDims()));

	const auto [BatchSize, Channels, Length] = Audio.Shape().RawArray();
	if (Length < (SourceSampleRate >> 3))
		_D_Dragonian_Lib_Throw_Exception("Audio length too short. (min 125ms)");
	if (Length > SourceSampleRate * 30)
		_D_Dragonian_Lib_Throw_Exception("Audio length too long. (max 30s)");

	SliceDatas InferenceDatas;

	InferenceDatas.SourceSampleRate = SourceSampleRate;
	InferenceDatas.SourceSampleCount = Length;

	_D_Dragonian_Lib_Rethrow_Block(
		InferenceDatas.Units = (*UnitsEncoder)(Audio, SourceSampleRate, AudioMask);
	);

	if (UnitsCluster)
	{
		const auto AudioFrames = InferenceDatas.Units.Size(2);
		const auto ClusterRate = Operators::BinaryOperators::Clamp(Params.ClusterRate, 0.f, 1.f);
		if (ClusterRate > 1e-4)
		{
			auto ClusUnits = (UnitsCluster.value().get())->Search(
				InferenceDatas.Units.View(-1, _MyUnitsDim),
				static_cast<Long>(Params.SpeakerId)
			).View(BatchSize, Channels, AudioFrames, _MyUnitsDim);

			if (ClusterRate > 0.9999f)
				InferenceDatas.Units = std::move(ClusUnits);
			else
				InferenceDatas.Units = InferenceDatas.Units * (1 - ClusterRate) + ClusUnits * ClusterRate;
		}
	}

	_D_Dragonian_Lib_Rethrow_Block(
		InferenceDatas.F0 = (*F0Extractor)(Audio.View(-1, Length), F0Params).View(BatchSize, Channels, -1);
	);

	if (_HasVolumeEmbedding)
		_D_Dragonian_Lib_Rethrow_Block(
			InferenceDatas.Volume = ExtractVolume(
				Audio,
				SourceSampleRate * _MyHopSize / _MyOutputSamplingRate,
				SourceSampleRate >> 4
			);
		);

	InferenceDatas.GTAudio = Audio.View();
	InferenceDatas.GTSampleRate = SourceSampleRate;

	if (OutPointers)
	{
		*OutPointers = VPreprocess(Params, std::move(InferenceDatas));
		_D_Dragonian_Lib_Rethrow_Block(return Forward(Params, *OutPointers););
	}
	_D_Dragonian_Lib_Rethrow_Block(return Forward(Params, VPreprocess(Params, std::move(InferenceDatas))););
}

SliceDatas SingingVoiceConversionModule::Preprocess(
	const Parameters& Params,
	const SliceDatas& InferenceDatas
) const
{
	SliceDatas Ret;
	Ret.SourceSampleRate = InferenceDatas.SourceSampleRate;
	Ret.SourceSampleCount = InferenceDatas.SourceSampleCount;

	if (!InferenceDatas.Units.Null())
		Ret.Units = InferenceDatas.Units.Clone();

	if (!InferenceDatas.F0.Null())
		Ret.F0 = InferenceDatas.F0.Clone();

	if (InferenceDatas.Volume && !InferenceDatas.Volume->Null())
		Ret.Volume = InferenceDatas.Volume->Clone();

	if (InferenceDatas.UnVoice && !InferenceDatas.UnVoice->Null())
		Ret.UnVoice = InferenceDatas.UnVoice->Clone();

	if (InferenceDatas.F0Embed && !InferenceDatas.F0Embed->Null())
		Ret.F0Embed = InferenceDatas.F0Embed->Clone();

	if (InferenceDatas.UnitsLength && !InferenceDatas.UnitsLength->Null())
		Ret.UnitsLength = InferenceDatas.UnitsLength->Clone();

	if (InferenceDatas.SpeakerId && !InferenceDatas.SpeakerId->Null())
		Ret.SpeakerId = InferenceDatas.SpeakerId->Clone();

	if (InferenceDatas.Speaker && !InferenceDatas.Speaker->Null())
		Ret.Speaker = InferenceDatas.Speaker->Clone();

	if (InferenceDatas.Noise && !InferenceDatas.Noise->Null())
		Ret.Noise = InferenceDatas.Noise->Clone();

	if (InferenceDatas.Mel2Units && !InferenceDatas.Mel2Units->Null())
		Ret.Mel2Units = InferenceDatas.Mel2Units->Clone();

	if (InferenceDatas.GTSpec && !InferenceDatas.GTSpec->Null())
		Ret.GTSpec = InferenceDatas.GTSpec->Clone();

	if (InferenceDatas.GTAudio && !InferenceDatas.GTAudio->Null())
		Ret.GTAudio = InferenceDatas.GTAudio->Clone();

	Ret.GTSampleRate = InferenceDatas.GTSampleRate;

	return VPreprocess(Params, std::move(Ret));
}

Int64 SingingVoiceConversionModule::CalculateFrameCount(
	Int64 InputSampleCount,
	Int64 InputSamplingRate,
	Int64 Offset
) const noexcept
{
	const auto Num = InputSampleCount * _MyOutputSamplingRate;
	const auto Den = InputSamplingRate * _MyHopSize;
	if (Den == 0)
		return 0;
	return std::max((Num / Den) + Offset, 0ll);
}

SliceDatas& SingingVoiceConversionModule::Preprocess_(
	const Parameters& Params,
	SliceDatas& MyData
) const
{
	MyData = VPreprocess(Params, std::move(MyData));
	return MyData;
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::NormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec
) const
{
	return NormSpec(Spec, _MySpecMax, _MySpecMin);
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::DenormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec
) const
{
	return DenormSpec(Spec, _MySpecMax, _MySpecMin);
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::NormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec,
	float SpecMax,
	float SpecMin
)
{
	return (Spec - SpecMin) / (SpecMax - SpecMin) * 2 - 1;
}

Tensor<Float32, 4, Device::CPU> SingingVoiceConversionModule::DenormSpec(
	const Tensor<Float32, 4, Device::CPU>& Spec,
	float SpecMax,
	float SpecMin
)
{
	return (Spec + 1) / 2 * (SpecMax - SpecMin) + SpecMin;
}

Tensor<Float32, 3, Device::CPU> SingingVoiceConversionModule::ExtractVolume(
	const Tensor<Float32, 3, Device::CPU>& Audio,
	Int64 HopSize, Int64 WindowSize
)
{
	const auto [BatchSize, Channels, Length] = Audio.Shape().RawArray();

	const auto NumFrames = Length / HopSize + 1;
	const auto HalfWindowSize = WindowSize / 2;
	auto NewAudio = Audio.Padding(
		{ None,None, {HalfWindowSize, HalfWindowSize} },
		PaddingType::Reflect
	).Evaluate();
	auto NewAudio2 = NewAudio.Pow(2).Evaluate();
	const auto NewLength = NewAudio.Shape(2);

	auto VolumeShape = Dimensions<3>{ BatchSize, Channels, NumFrames };
	auto Volume = Tensor<Float32, 3, Device::CPU>::New(VolumeShape);

	for (SizeType i = 0; i < BatchSize; ++i)
	{
		for (SizeType j = 0; j < Channels; ++j)
		{
			auto AudioData = NewAudio.Data() + i * Channels * NewLength + j * NewLength;
			auto Audio2Data = NewAudio2.Data() + i * Channels * NewLength + j * NewLength;
			auto VolumeData = Volume.Data() + i * Channels * NumFrames + j * NumFrames;
			Volume.AppendTask(
				[AudioData, Audio2Data, VolumeData, HopSize, WindowSize, NumFrames]
				(std::shared_ptr<void>, std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
				{
					for (SizeType k = 0; k < NumFrames; ++k)
					{
						auto Mean = DragonianLibSTL::Average(
							AudioData + k * HopSize,
							AudioData + k * HopSize + WindowSize
						);
						auto MeanSquare = DragonianLibSTL::Average(
							Audio2Data + k * HopSize,
							Audio2Data + k * HopSize + WindowSize
						);
						VolumeData[k] = sqrt(Operators::BinaryOperators::ClampMin(float(MeanSquare - Mean * Mean), 0.f));
					}
				},
				NewAudio.Buffer(),
				NewAudio2.Buffer()
			);
		}
	}
	return Volume;
}

Tensor<Int64, 3, Device::CPU> SingingVoiceConversionModule::GetF0Embed(
	const Tensor<Float32, 3, Device::CPU>& F0,
	Float32 F0Bin, Float32 F0MelMax, Float32 F0MelMin
)
{
	auto F0Cont = F0.Continuous().Evaluate();
	const auto [BatchSize, Channels, Length] = F0Cont.Shape().RawArray();
	auto F0Embed = Tensor<Int64, 3, Device::CPU>::New({ BatchSize, Channels, Length });
	for (SizeType i = 0; i < BatchSize; ++i)
	{
		for (SizeType j = 0; j < Channels; ++j)
		{
			const auto F0Data = F0Cont.Data() + i * Channels * Length + j * Length;
			auto F0EmbedData = F0Embed.Data() + i * Channels * Length + j * Length;
			F0Embed.AppendTask(
				[F0Data, F0EmbedData, Length, F0Bin, F0MelMax, F0MelMin]
				(std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
				{
					for (SizeType k = 0; k < Length; ++k)
					{
						auto F0Mel = 1127.f * log(1.f + F0Data[k] / 700.f);
						if (F0Mel > 0.f)
							F0Mel = (F0Mel - F0MelMin) * (F0Bin - 2.f) / (F0MelMax - F0MelMin) + 1.f;
						F0Mel = std::max(F0Mel, 1.f);
						F0Mel = std::min(F0Mel, F0Bin - 1.f);
						F0EmbedData[k] = static_cast<Int64>(F0Mel);
					}
				},
				F0Cont.Buffer()
			);
		}
	}
	return F0Embed;
}

Tensor<Float32, 3, Device::CPU> SingingVoiceConversionModule::InterpolateUnVoicedF0(
	const Tensor<Float32, 3, Device::CPU>& F0
)
{
	auto F0Cont = F0.Continuous().Evaluate();
	const auto [BatchSize, Channels, Length] = F0Cont.Shape().RawArray();
	auto InterpolatedF0 = Tensor<Float32, 3, Device::CPU>::New({ BatchSize, Channels, Length });
	for (SizeType i = 0; i < BatchSize; ++i)
	{
		for (SizeType j = 0; j < Channels; ++j)
		{
			const auto F0Data = F0Cont.Data() + i * Channels * Length + j * Length;
			auto InterpolatedF0Data = InterpolatedF0.Data() + i * Channels * Length + j * Length;
			InterpolatedF0.AppendTask(
				[InterpolatedF0Data, F0Data, Length]
				(std::shared_ptr<void>) // NOLINT(performance-unnecessary-value-param)
				{
					constexpr Float32 epsilon = std::numeric_limits<Float32>::epsilon();

					Int64 firstNonZeroIndex = 0;
					while (firstNonZeroIndex < Length && F0Data[firstNonZeroIndex] < epsilon)
						++firstNonZeroIndex;

					if (firstNonZeroIndex == Length)
						return;
					for (Int64 i = 0; i < firstNonZeroIndex; ++i)
						InterpolatedF0Data[i] = F0Data[firstNonZeroIndex];

					Int64 start = firstNonZeroIndex;
					while (start < Length) {
						while (start < Length && F0Data[start] > epsilon)
						{
							InterpolatedF0Data[start] = F0Data[start];
							++start;
						}

						if (start == Length)
							break;
						--start;

						Int64 end = start + 1;
						while (end < Length && F0Data[end] < epsilon)
							++end;

						if (end < Length) 
						{
							float startValue = F0Data[start];
							float endValue = F0Data[end];
							Int64 gap = end - start;
							for (Int64 i = 1; i < gap; ++i)
								InterpolatedF0Data[start + i] = startValue + (endValue - startValue) * (float(i) / float(gap));
							start = end;
						}
						else 
						{
							for (Int64 i = start + 1; i < Length; ++i)
								InterpolatedF0Data[i] = F0Data[start];
							break;
						}
					}
				},
				F0Cont.Buffer()
			);
		}
	}
	return InterpolatedF0;
}

void SingingVoiceConversionModule::CheckParams(
	const SliceDatas& MyData
)
{
	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (MyData.SourceSampleCount <= 4410)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid source sample count, expected: > 4410, got: " +
			std::to_string(MyData.SourceSampleCount)
		);
	if (MyData.SourceSampleRate <= 4410)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid source sample rate, expected: > 4410, got: " +
			std::to_string(MyData.SourceSampleRate)
		);
}

SliceDatas& SingingVoiceConversionModule::PreprocessUnits(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	const DLogger& /*Logger*/
) const
{
	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception(
			"Units could not be null"
		);
	const auto [Batch, Channel, NumFrames, UnitDims] =
		MyData.Units.Shape().RawArray();
	if (Batch != BatchSize)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid batch size, expected: " +
			std::to_string(BatchSize) +
			", got: " +
			std::to_string(Batch)
		);
	if (Channel != Channels)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid channels, expected: " + 
			std::to_string(Channels) + 
			", got: " +
			std::to_string(Channel)
		);
	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception(
			"Invalid units dims, expected: " + 
			std::to_string(_MyUnitsDim) + 
			", got: " +
			std::to_string(UnitDims)
		);

	if (TargetNumFrames == 0)
		return MyData;

	if (NumFrames != TargetNumFrames)
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Units = MyData.Units.Interpolate<Operators::InterpolateMode::Nearest>(
				IDim(-2),
				{ IDim(TargetNumFrames) }
			);
		);

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessUnitsLength(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	const DLogger& Logger
)
{
	if (MyData.UnitsLength && !MyData.UnitsLength->Null())
	{
		const auto [BatchLength, ChannelLength, Length] =
			MyData.UnitsLength->Shape().RawArray();
		if (BatchLength != BatchSize)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and units length batch mismatch, expected: " +
				std::to_string(BatchSize) +
				", got: " +
				std::to_string(BatchLength)
			);
		if (ChannelLength != Channels)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and units length channels mismatch, expected: " +
				std::to_string(Channels) +
				", got: " +
				std::to_string(ChannelLength)
			);
		if (Length != 1)
			_D_Dragonian_Lib_Throw_Exception(
				"Invalid units length length"
			);
	}
	else
	{
		if (Logger)
			Logger->LogInfo(L"Units length not found, generating units length with units");

		using UnitsSizeType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.UnitsLength = UnitsSizeType::ConstantOf(
				{ BatchSize, Channels, 1 },
				TargetNumFrames
			);
		);
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessMel2Units(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	const DLogger& Logger
)
{
	if (MyData.Mel2Units && !MyData.Mel2Units->Null())
	{
		const auto [BatchMel2Units, ChannelMel2Units, NumFramesMel2Units] = 
			MyData.Mel2Units->Shape().RawArray();
		if (BatchSize != BatchMel2Units)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and mel2units batch mismatch, expected: " +
				std::to_string(BatchSize) +
				", got: " +
				std::to_string(BatchMel2Units)
			);
		if (Channels != ChannelMel2Units)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and mel2units channels mismatch, expected: " +
				std::to_string(Channels) +
				", got: " +
				std::to_string(ChannelMel2Units)
			);
		if (NumFramesMel2Units != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyData.Mel2Units->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}
	else
	{
		if (Logger)
			Logger->LogInfo(L"Mel2Units not found, generating mel2units with units");

		const auto UnitFrames = MyData.Units.Shape(-2);
		auto MyMel2Unit = Functional::Linspace(
			0.f, static_cast<Float32>(UnitFrames), TargetNumFrames, true
		).Cast<Int64>().ClampMax(UnitFrames - 1);
		const auto BatchChannels = BatchSize * Channels;
		if (BatchChannels > 1)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.UnSqueeze(0).Repeat(
					{ BatchChannels }
				).View(BatchSize, Channels, TargetNumFrames);
			);
		else
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Mel2Units = MyMel2Unit.View(
					BatchSize, Channels, TargetNumFrames
				);
			);
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessUnVoice(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	const DLogger& Logger
)
{
	if (MyData.UnVoice && !MyData.UnVoice->Null())
	{
		const auto [BatchUnVoice, ChannelUnVoice, NumFramesUnVoice] =
			MyData.UnVoice->Shape().RawArray();
		if (BatchSize != BatchUnVoice)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and unvoice batch mismatch, expected: " +
				std::to_string(BatchSize) +
				", got: " +
				std::to_string(BatchUnVoice)
			);
		if (Channels != ChannelUnVoice)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and unvoice channels mismatch, expected: " +
				std::to_string(Channels) +
				", got: " +
				std::to_string(ChannelUnVoice)
			);
		if (NumFramesUnVoice != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.UnVoice = MyData.UnVoice->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}
	else
	{
		if (Logger)
			Logger->LogInfo(L"UnVoice not found, generating unvoice with f0");
		if (MyData.F0.Null())
			_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

		_D_Dragonian_Lib_Rethrow_Block(
			MyData.UnVoice = (MyData.F0 > 1e-4f).Cast<Float32>();
		);
		if (MyData.UnVoice->Shape(-1) != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.UnVoice = MyData.UnVoice->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessF0(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Float32 F0Offset,
	bool InterpolateUnVoiced,
	Parameters::F0PreprocessMethod F0Method,
	void* UserParameters,
	const DLogger& /*Logger*/
)
{
	if (MyData.F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

	const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
	if (BatchSize != BatchF0)
		_D_Dragonian_Lib_Throw_Exception(
			"Units and f0 batch mismatch, expected: " +
			std::to_string(BatchSize) +
			", got: " +
			std::to_string(BatchF0)
		);
	if (Channels != ChannelF0)
		_D_Dragonian_Lib_Throw_Exception(
			"Units and f0 channels mismatch, expected: " +
			std::to_string(Channels) +
			", got: " +
			std::to_string(ChannelF0)
		);

	if (InterpolateUnVoiced)
		MyData.F0 = InterpolateUnVoicedF0(
			MyData.F0
		);

	if (NumFramesF0 != TargetNumFrames)
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.F0 = MyData.F0.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(TargetNumFrames) }
			);
		);

	if (F0Method || abs(F0Offset) > 1e-4f)
		MyData.SourceF0 = MyData.F0.Clone();

	if (abs(F0Offset) > 1e-4f)
		_D_Dragonian_Lib_Rethrow_Block((MyData.F0 *= std::pow(2.f, F0Offset / 12.f)););

	if (F0Method)
		_D_Dragonian_Lib_Rethrow_Block((MyData.F0 = F0Method(MyData.F0, UserParameters)););
	
	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessVolume(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	const DLogger& Logger
) const
{
	if (!_HasVolumeEmbedding)
		return MyData;
	if (MyData.Volume && !MyData.Volume->Null())
	{
		const auto [BatchVolume, ChannelVolume, NumFramesVolume] = MyData.Volume->Shape().RawArray();
		if (BatchSize != BatchVolume)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and volume batch mismatch, expected: " +
				std::to_string(BatchSize) +
				", got: " +
				std::to_string(BatchVolume)
			);
		if (Channels != ChannelVolume)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and volume channels mismatch, expected: " +
				std::to_string(Channels) +
				", got: " +
				std::to_string(ChannelVolume)
			);
		if (NumFramesVolume != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Volume = MyData.Volume->Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}
	else
	{
		if (Logger)
			Logger->LogInfo(L"Volume not found, generating volume with gt-audio");
		if (!MyData.GTAudio || MyData.GTAudio->Null())
			_D_Dragonian_Lib_Throw_Exception("GTAudio could not be null");
		if (MyData.GTSampleRate <= 4410)
			_D_Dragonian_Lib_Throw_Exception("GTAudio sample rate must be greater than 4410.");

		_D_Dragonian_Lib_Rethrow_Block(
			MyData.Volume = ExtractVolume(
				*MyData.GTAudio,
				MyData.GTSampleRate * _MyHopSize / _MyOutputSamplingRate,
				MyData.GTSampleRate >> 4
			);
		);
		if (MyData.Volume->Shape(-1) != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Volume = MyData.Volume->Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessF0Embed(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Float32 F0Offset,
	const DLogger& Logger
) const
{
	if (MyData.F0Embed && !MyData.F0Embed->Null())
	{
		const auto [BatchF0Embed, ChannelF0Embed, NumFramesF0Embed] = MyData.F0Embed->Shape().RawArray();
		if (BatchSize != BatchF0Embed)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and f0embed batch mismatch, expected: " +
				std::to_string(BatchSize) +
				", got: " +
				std::to_string(BatchF0Embed)
			);
		if (Channels != ChannelF0Embed)
			_D_Dragonian_Lib_Throw_Exception(
				"Units and f0embed channels mismatch, expected: " +
				std::to_string(Channels) +
				", got: " +
				std::to_string(ChannelF0Embed)
			);
		if (NumFramesF0Embed != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.F0Embed = MyData.F0Embed->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}
	else
	{
		if (Logger)
			Logger->LogInfo(L"F0Embed not found, generating f0embed with f0");
		if (MyData.F0.Null())
			_D_Dragonian_Lib_Throw_Exception("F0 could not be null");

		if (abs(F0Offset) > 1e-4f)
			_D_Dragonian_Lib_Rethrow_Block((MyData.F0 *= std::pow(2.f, F0Offset / 12.f)););
		_D_Dragonian_Lib_Rethrow_Block(
			MyData.F0Embed = GetF0Embed(
				MyData.F0,
				Float32(_MyF0Bin),
				_MyF0MelMax,
				_MyF0MelMin
			);
		);
		if (MyData.F0Embed->Shape(-1) != TargetNumFrames)
			_D_Dragonian_Lib_Rethrow_Block(
				MyData.F0Embed = MyData.F0Embed->Interpolate<Operators::InterpolateMode::Nearest>(
					IDim(-1),
					{ IDim(TargetNumFrames) }
				);
			);
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessSpeakerMix(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 TargetNumFrames,
	Int64 SpeakerId,
	const DLogger& Logger
) const
{
	if (HasSpeakerMixLayer())
	{
		if (MyData.Speaker.has_value() && !MyData.Speaker->Null())
		{
			const auto [BatchSpeaker, ChannelSpeaker, NumFrames, NumSpeakers] =
				MyData.Speaker->Shape().RawArray();
			if (BatchSize != BatchSpeaker)
				_D_Dragonian_Lib_Throw_Exception(
					"Units and speaker batch mismatch, expected: " + 
					std::to_string(BatchSize) + 
					", got: " + 
					std::to_string(BatchSpeaker)
				);
			if (Channels != ChannelSpeaker)
				_D_Dragonian_Lib_Throw_Exception(
					"Units and speaker channels mismatch, expected: " +
					std::to_string(Channels) +
					", got: " +
					std::to_string(ChannelSpeaker)
				);
			if (NumSpeakers != _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Invalid speaker count, expected: " +
					std::to_string(_MySpeakerCount) +
					", got: " +
					std::to_string(NumSpeakers)
				);
			if (NumFrames != TargetNumFrames)
				_D_Dragonian_Lib_Rethrow_Block(
					MyData.Speaker = MyData.Speaker->Interpolate<Operators::InterpolateMode::Nearest>(
						IDim(-2),
						{ IDim(TargetNumFrames) }
					);
				);
		}
		else
		{
			if (Logger)
				Logger->LogInfo(L"Speaker not found, generating speaker with param.speaker_id");
			if (SpeakerId < 0 || SpeakerId >= _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Invalid speaker id, expected: 0 ~ " +
					std::to_string(_MySpeakerCount - 1) +
					", got: " +
					std::to_string(SpeakerId)
				);

			_D_Dragonian_Lib_Rethrow_Block(
				MyData.Speaker = Functional::Zeros(
					IDim(BatchSize, Channels, TargetNumFrames, _MySpeakerCount)
				);
			);
			(MyData.Speaker.value()[{":", ":", ":", Range::Idx(SpeakerId)}]= 1.f);
		}
	}

	return MyData;
}

SliceDatas& SingingVoiceConversionModule::PreprocessSpeakerId(
	SliceDatas& MyData,
	Int64 BatchSize,
	Int64 Channels,
	Int64 /*TargetNumFrames*/,
	Int64 SpeakerId,
	const DLogger& Logger
) const
{
	if (HasSpeakerEmbedding() && !HasSpeakerMixLayer())
	{
		if (MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
		{
			const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] =
				MyData.SpeakerId->Shape().RawArray();
			if (BatchSize != BatchSpeakerId)
				_D_Dragonian_Lib_Throw_Exception(
					"Units and speaker id batch mismatch, expected: " +
					std::to_string(BatchSize) +
					", got: " +
					std::to_string(BatchSpeakerId)
				);
			if (Channels != ChannelSpeakerId)
				_D_Dragonian_Lib_Throw_Exception(
					"Units and speaker id channels mismatch, expected: " +
					std::to_string(Channels) +
					", got: " +
					std::to_string(ChannelSpeakerId)
				);
			if (LengthSpeakerId != 1)
				_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
		}
		else
		{
			if (Logger)
				Logger->LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
			if (SpeakerId < 0 || SpeakerId >= _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Invalid speaker id, expected: 0 ~ " +
					std::to_string(_MySpeakerCount - 1) +
					", got: " +
					std::to_string(SpeakerId)
				);

			using SpkType = Tensor<Int64, 3, Device::CPU>;
			_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf(
				{ BatchSize, Channels, 1 },
				SpeakerId
			););
		}
	}

	return MyData;
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End