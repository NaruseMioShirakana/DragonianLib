/**
 * @file Base.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Base class of SingingVoiceConversion
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Created <
 *	> 2025/3/22 NaruseMioShirakana Add SingingVoiceConversionModule <
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Util/Util.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

constexpr Int64 DefaultNoiseDim = 192;
constexpr Int64 DefaultWindowSize = 2048;

/**
 * @class SingingVoiceConversionModule
 * @brief Base class of SingingVoiceConversion, provides common functions and parameters, such as inference, preprocess, etc.
 *
 * Classes that inherit from this class must implement the following functions:
 * - Forward (virtual): Forward inference
 * - VPreprocess (virtual): Preprocess input datas for inference
 *
 * Comments:
 * - Extended parameters ["Key", ... ] means that you could add more parameters to HParams::ExtendedParameters with {"Key", "Value"}  
 * - Model path ["Key", ... ] means that you must add more model paths to HParams::ModelPaths with {"Key", "Value"}
 * - AUTOGEN means that if the parameter is not set, the model will automatically generate this parameter
 * - OPTIONAL means that the parameter is optional if your model does not have this layer
 * - REQUIRED means that the parameter is always required
 */
class SingingVoiceConversionModule
{
public:

	SingingVoiceConversionModule() = delete;
	SingingVoiceConversionModule(const HParams& Params);
	virtual ~SingingVoiceConversionModule() = default;

	SingingVoiceConversionModule(const SingingVoiceConversionModule&) = default;
	SingingVoiceConversionModule(SingingVoiceConversionModule&&) noexcept = default;
	SingingVoiceConversionModule& operator=(const SingingVoiceConversionModule&) = default;
	SingingVoiceConversionModule& operator=(SingingVoiceConversionModule&&) noexcept = default;

	/**
	 * @brief Inference
	 * @param Params Inference parameters, see SingingVoiceConversion::Parameters
	 * @param Audio Input audio, shape must be {BatchSize, Channels, SampleCount}
	 * @param SourceSampleRate Source sample rate
	 * @param UnitsEncoder Units encoder
	 * @param F0Extractor F0 extractor
	 * @param F0Params F0 parameters
	 * @param UnitsCluster Units cluster
	 * @param AudioMask Audio mask
	 * @param[Output] OutPointers If it is not null, the function will override the data in the OutPointers with preprocessed datas, It is useful when you need to get the preprocessed datas
	 * @return Tensor<Float32, 4, Device::CPU> Inference result, may be the mel spectrogram or the audio, if output is audio, shape must be {1, BatchSize, Channels, SampleCount}, if output is mel spectrogram, shape must be {BatchSize, Channels, MelBins, AudioFrames}
	 */
	Tensor<Float32, 4, Device::CPU> Inference(
		const Parameters& Params,
		const Tensor<Float32, 3, Device::CPU>& Audio,
		SizeType SourceSampleRate,
		const FeatureExtractor& UnitsEncoder,
		const PitchExtractor& F0Extractor,
		const PitchParameters& F0Params,
		std::optional<std::reference_wrapper<const Cluster>> UnitsCluster = std::nullopt,
		std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> AudioMask = std::nullopt,
		SliceDatas* OutPointers = nullptr
	) const;

	/**
	 * @brief Check arguments and preprocess input datas for inference, will copy the input datas and return the new datas
	 * @param Params Inference parameters, see SingingVoiceConversion::Parameters
	 * @param InferenceDatas Input datas, see SingingVoiceConversion::SliceDatas
	 * @return SliceDatas Preprocessed datas
	 */
	SliceDatas Preprocess(
		const Parameters& Params,
		const SliceDatas& InferenceDatas
	) const;

	/**
	 * @brief Forward inference
	 * @param Params Parameters, see SingingVoiceConversion::Parameters
	 * @param InputDatas Input datas, see SingingVoiceConversion::SliceDatas
	 * @return Tensor<Float32, 4, Device::CPU>
	 */
	virtual Tensor<Float32, 4, Device::CPU> Forward(
		const Parameters& Params,
		const SliceDatas& InputDatas
	) const = 0;

	/**
	 * @brief Check arguments and preprocess input datas for inference, this function will modify the input variables, input datas will be moved in this function
	 * @param Params Inference parameters, see SingingVoiceConversion::Parameters
	 * @param InputDatas Input datas, see SingingVoiceConversion::SliceDatas
	 * @return Preprocessed datas (moved and modified from input datas)
	 */
	virtual SliceDatas VPreprocess(
		const Parameters& Params,
		SliceDatas&& InputDatas
	) const = 0;

	/**
	 * @brief Calculate frame count with input sample count and sampling rate， The following equation is calculated: FrameCount = (InputSampleCount * OutputSamplingRate) / (InputSamplingRate * HopSize) + Offset
	 * @param InputSampleCount Input sample count
	 * @param InputSamplingRate Input sampling rate
	 * @param Offset Offset
	 * @return Int64 Frame count
	 */
	Int64 CalculateFrameCount(
		Int64 InputSampleCount,
		Int64 InputSamplingRate,
		Int64 Offset = 0
	) const noexcept;

	/**
	 * @brief Preprocess input datas for inference(override input datas)
	 * @param Params Inference parameters
	 * @param MyData Input datas
	 * @return Preprocessed datas(reference to input datas)
	 */
	SliceDatas& Preprocess_(
		const Parameters& Params,
		SliceDatas& MyData
	) const;

protected:
	Int64 _MyOutputSamplingRate = 32000;
	Int64 _MyUnitsDim = 256;
	Int64 _MyHopSize = 512;
	Int64 _MySpeakerCount = 1;
	Float32 _MySpecMax = 2.f;
	Float32 _MySpecMin = -12.f;
	Int64 _MyF0Bin = 256;
	Float32 _MyF0Max = 1100.0;
	Float32 _MyF0Min = 50.0;
	Float32 _MyF0MelMax = 1127.f * log(1.f + _MyF0Max / 700.f);
	Float32 _MyF0MelMin = 1127.f * log(1.f + _MyF0Min / 700.f);
	bool _HasVolumeEmbedding = false;
	bool _HasSpeakerEmbedding = false;
	bool _HasSpeakerMixLayer = false;

public:
	static Tensor<Float32, 4, Device::CPU> NormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec,
		float SpecMax,
		float SpecMin
	);

	static Tensor<Float32, 4, Device::CPU> DenormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec,
		float SpecMax,
		float SpecMin
	);

	Tensor<Float32, 4, Device::CPU> NormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec
	) const;

	Tensor<Float32, 4, Device::CPU> DenormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec
	) const;

	static Tensor<Float32, 3, Device::CPU> ExtractVolume(
		const Tensor<Float32, 3, Device::CPU>& Audio,
		Int64 HopSize, Int64 WindowSize
	);

	static Tensor<Int64, 3, Device::CPU> GetF0Embed(
		const Tensor<Float32, 3, Device::CPU>& F0,
		Float32 F0Bin, Float32 F0MelMax, Float32 F0MelMin
	);

	static Tensor<Float32, 3, Device::CPU> InterpolateUnVoicedF0(
		const Tensor<Float32, 3, Device::CPU>& F0
	);

protected:
	std::optional<ProgressCallback> _MyProgressCallback;

public:
	Int64 GetSamplingRate() const noexcept
	{
		return _MyOutputSamplingRate;
	}

	Int64 GetUnitsDim() const noexcept
	{
		return _MyUnitsDim;
	}

	Int64 GetHopSize() const noexcept
	{
		return _MyHopSize;
	}

	Int64 GetSpeakerCount() const noexcept
	{
		return _MySpeakerCount;
	}

	Float32 GetSpecMax() const noexcept
	{
		return _MySpecMax;
	}

	Float32 GetSpecMin() const noexcept
	{
		return _MySpecMin;
	}

	Int64 GetF0Bin() const noexcept
	{
		return _MyF0Bin;
	}

	Float32 GetF0Max() const noexcept
	{
		return _MyF0Max;
	}

	Float32 GetF0Min() const noexcept
	{
		return _MyF0Min;
	}

	Float32 GetF0MelMax() const noexcept
	{
		return _MyF0MelMax;
	}

	Float32 GetF0MelMin() const noexcept
	{
		return _MyF0MelMin;
	}

	bool HasVolumeEmbedding() const noexcept
	{
		return _HasVolumeEmbedding;
	}

	bool HasSpeakerEmbedding() const noexcept
	{
		return _HasSpeakerEmbedding;
	}

	bool HasSpeakerMixLayer() const noexcept
	{
		return _HasSpeakerMixLayer;
	}

protected:
	static void CheckParams(
		const SliceDatas& MyData
	);

public:
	SliceDatas& PreprocessUnits(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		const DLogger& Logger = nullptr
	) const;
	static SliceDatas& PreprocessUnitsLength(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		const DLogger& Logger = nullptr
	);
	static SliceDatas& PreprocessMel2Units(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		const DLogger& Logger = nullptr
	);
	static SliceDatas& PreprocessUnVoice(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		const DLogger& Logger = nullptr
	);
	static SliceDatas& PreprocessF0(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Float32 F0Offset,
		bool InterpolateUnVoiced,
		Parameters::F0PreprocessMethod F0Method,
		void* UserParameters,
		const DLogger& Logger = nullptr
	);
	SliceDatas& PreprocessVolume(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		const DLogger& Logger = nullptr
	) const;
	SliceDatas& PreprocessF0Embed(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Float32 F0Offset,
		const DLogger& Logger = nullptr
	) const;
	SliceDatas& PreprocessSpeakerMix(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Int64 SpeakerId,
		const DLogger& Logger = nullptr
	) const;
	SliceDatas& PreprocessSpeakerId(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Int64 SpeakerId,
		const DLogger& Logger = nullptr
	) const;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End