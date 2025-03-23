#include "../Base.hpp"

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
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> AudioMask
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

	Audio.Evaluate();

	InferenceDatas.SourceSampleRate = SourceSampleRate;
	InferenceDatas.SourceSampleCount = Length;

	_D_Dragonian_Lib_Rethrow_Block(
		InferenceDatas.Units = (*UnitsEncoder)(Audio, SourceSampleRate, AudioMask).Evaluate();
	);

	if (UnitsCluster)
	{
		const auto AudioFrames = InferenceDatas.Units.Size(2);
		const auto ClusterRate = Operators::BinaryOperators::Clamp(Params.ClusterRate, 0.f, 1.f);
		if (ClusterRate > 1e-4)
		{
			auto ClusUnits = (UnitsCluster.value().get())->Search(InferenceDatas.Units.View(-1, _MyUnitsDim), static_cast<Long>(Params.SpeakerId)).View(BatchSize, Channels, AudioFrames, _MyUnitsDim).Evaluate();
			if (ClusterRate > 0.9999f)
				InferenceDatas.Units = std::move(ClusUnits);
			else
				InferenceDatas.Units = InferenceDatas.Units * (1 - ClusterRate) + ClusUnits * ClusterRate;
			InferenceDatas.Units.Evaluate();
		}
	}

	_D_Dragonian_Lib_Rethrow_Block(
		InferenceDatas.F0 = (*F0Extractor)(Audio.View(-1, Length), F0Params).View(BatchSize, Channels, -1).Evaluate();
	);

	if (_HasVolumeEmbedding)
		_D_Dragonian_Lib_Rethrow_Block(
			InferenceDatas.Volume = ExtractVolume(
				Audio,
				SourceSampleRate * _MyHopSize / _MyOutputSamplingRate,
				SourceSampleRate >> 4
			).Evaluate();
		);

	InferenceDatas.GTAudio = Audio.View();
	InferenceDatas.GTSampleRate = SourceSampleRate;

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
	auto NewAudio2 = NewAudio.Pow(2);
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
				}
			);
		}
	}
	return std::move(Volume.Evaluate());
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
				{
					for (SizeType k = 0; k < Length; ++k)
					{
						auto F0Mel = 1127.f * log(1.f + F0Data[k] / 700.f);
						if (F0Mel > 0.f)
							F0Mel = (F0Mel - F0MelMin) * (F0Bin - 2.f) / (F0MelMax - F0MelMin) + 1.f;
						if (F0Mel < 1.f)
							F0Mel = 1.f;
						if (F0Mel > F0Bin - 1.f)
							F0Mel = F0Bin - 1.f;
						F0EmbedData[k] = static_cast<Int64>(F0Mel);
					}
				}
			);
		}
	}
	return std::move(F0Embed.Evaluate());
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
				}
			);
		}
	}
	return std::move(InterpolatedF0.Evaluate());
}


_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End