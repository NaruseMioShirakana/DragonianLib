﻿/**
 * FileName: SvcBase.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "TensorRT/TensorRTBase/TRTBase.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include "Libraries/Cluster/ClusterManager.hpp"

#define _D_Dragonian_Lib_TRT_Svc_Space_Header _D_Dragonian_TensorRT_Lib_Space_Header namespace SingingVoiceConversion {
#define _D_Dragonian_Lib_TRT_Svc_Space_End } _D_Dragonian_TensorRT_Lib_Space_End

_D_Dragonian_Lib_TRT_Svc_Space_Header

static inline std::vector<DynaShapeSlice> HubertDynaSetting{
	{"source", nvinfer1::Dims3(1, 1, 3200)	, nvinfer1::Dims3(1, 1, 32000), nvinfer1::Dims3(1, 1, 320000)}
};

using PitchParameters = F0Extractor::Parameters;
using PitchExtractor = F0Extractor::F0Extractor;
using Cluster = Cluster::Cluster;

/**
 * @struct DiffusionParameters
 * @brief Inference parameters for diffusion
 */
struct DiffusionParameters
{
	/**
	 * @brief Stride of the diffusion loop, every sample operation will skip (stride) steps, the diffusion step is ((end - begin) / stride), stride must be greater than (0) and less than (step)
	 */
	Int64 Stride = 1;

	/**
	 * @brief Begining of the diffusion loop, sample operation will start from (begin) step, begin must be greater equal than (0) and less equal than (end)
	 */
	Int64 Begin = 0;

	/**
	 * @brief End of the diffusion loop, sample operation will end at (end) step, end must be greater equal than (begin) and less equal than (max step)
	 */
	Int64 End = 1;

	/**
	 * @brief Sampler of the diffusion, it is the sampling method of the diffusion loop, the sampler must be one of the following:
	 *  - "Pndm"			(default)
	 *	- "DDim"			(implemented)
	 *  - "Eular"			(not implemented)
	 *	- "RK4"				(not implemented)
	 *	- "DPM-Solver"		(not implemented)
	 *	- "DPM-Solver++"	(not implemented)
	 */
	std::wstring Sampler = L"Pndm";

	/**
	 * @brief Mel factor, multiplied to the mel spectrogram, this argument is only used if the output audio has incorrect samples, this means that the mel spectrogram has incorrect unit, the mel factor is used to correct the mel spectrogram
	 */
	Float32 MelFactor = 1.f;

	/**
	 * @brief User parameters, this argument is used to pass user parameters to the diffusion model, the user parameters must be a pointer to the user parameters struct
	 */
	void* UserParameters = nullptr;
};

/**
 * @struct ReflowParameters
 * @brief Inference parameters for reflow
 */
struct ReflowParameters
{
	/**
	 * @brief Stride of the reflow loop, every sample operation will skip (stride) steps, the reflow step is ((end - begin) / stride), stride must be greater than (0) and less than (step)
	 */
	Float32 Stride = 0.2f;

	/**
	 * @brief Begining of the reflow loop, sample operation will start from (begin) step, begin must be greater than (0) and less equal than (end)
	 */
	Float32 Begin = 0.f;

	/**
	 * @brief End of the reflow loop, sample operation will end at (end) step, end must be greater equal than (begin) and less equal than (max step)
	 */
	Float32 End = 1.f;

	/**
	 * @brief Scale of the reflow, it is the scale of the reflow loop, the scale must be greater than (0)
	 */
	Float32 Scale = 1000.f;

	/**
	 * @brief Sampler of the reflow, it is the sampling method of the reflow loop, the sampler must be one of the following:
	 *  - "Eular"			(default)
	 *	- "RK4"				(implemented)
	 *	- "PECECE"			(implemented)
	 *	- "Heun"			(implemented)
	 */
	std::wstring Sampler = L"Eular";

	/**
	 * @brief Mel factor, multiplied to the mel spectrogram, this argument is only used if the output audio has incorrect samples, this means that the mel spectrogram has incorrect unit, the mel factor is used to correct the mel spectrogram
	 */
	Float32 MelFactor = 1.f;

	/**
	 * @brief User parameters, this argument is used to pass user parameters to the reflow model, the user parameters must be a pointer to the user parameters struct
	 */
	void* UserParameters = nullptr;
};

/**
 * @namespace PreDefinedF0PreprocessMethod
 * @brief Pre-defined f0 preprocess methods
 */
namespace PreDefinedF0PreprocessMethod
{
	/**
	 * @brief Log2 preprocess method
	 * @param F0 Input f0 tensor
	 * @param UserParameters Null
	 * @return Preprocessed f0 tensor
	 */
	Tensor<Float32, 3, Device::CPU> Log2(
		const Tensor<Float32, 3, Device::CPU>& F0,
		void* UserParameters
	);
}

/**
 * @struct Parameters
 * @brief Inference parameters for all models
 */
struct Parameters
{
	using F0PreprocessMethod = Tensor<Float32, 3, Device::CPU>(*)(
		const Tensor<Float32, 3, Device::CPU>& F0,
		void* UserParameters
		);

	/**
	 * @brief Noise scale factor, multiplied to the noise, this argument may be harmful to the output audio or helpful to the output audio. noise scale has no range, but it is recommended to be in the range of (0, 1)
	 */
	Float32 NoiseScale = 0.3f;

	/**
	 * @brief Speaker id, it is the index of the speaker embedding layer, if the model has speaker mixing layer, this argument is used to modify the speaker mixing tensor, if the model has no speaker mixing layer and has speaker embedding layer, this argument is used to select the speaker embedding feature, speaker id must be greater than (0) and less than (speaker count)
	 */
	Int64 SpeakerId = 0;

	/**
	 * @brief Pitch offset, f0 will be multiplied by (2 ^ (offset / 12)), the pitch offset has no range, but it is recommended to be in the range of midi pitch (-128, 128)
	 */
	Float32 PitchOffset = 0.f;

	/**
	 * @brief Random seed, this argument is used to generate random numbers, has unknown effect on the output audio, it depends on your luck
	 */
	Int64 Seed = 52468;

	/**
	 * @brief Cluster rate, it is the rate of the cluster, the cluster rate must be greater than (0) and less than (1)
	 */
	Float32 ClusterRate = 0.5f;

	/**
	 * @brief Whether the f0 has unvoice, if this flag is (true), F0 could have zero values, otherwise, zero values in F0 will be interpolated, this value MUST be set by user
	 */
	bool F0HasUnVoice = false;

	/**
	 * @brief Diffusion parameters
	 */
	DiffusionParameters Diffusion;

	/**
	 * @brief Reflow parameters
	 */
	ReflowParameters Reflow;

	/**
	 * @brief STFT noise scale for SoVitsSvc4.0-Beta, in general, this argument is not used
	 */
	Float32 StftNoiseScale = 0.8f;

	/**
	 * @brief F0 preprocess method, this argument is used to preprocess the f0 tensor, if the model not input the source f0(HZ), this argument must be set, otherwise, this argument must be (nullptr), here are some pre-defined f0 preprocess methods, see namespace PreDefinedF0PreprocessMethod
	 */
	F0PreprocessMethod F0Preprocess = nullptr;

	/**
	 * @brief User parameters of f0 preprocess method, this argument is used to pass user parameters to the model, the user parameters must be a pointer to the user parameters struct, if the model not input the source f0(HZ) and the f0 preprocess method has user parameters, this argument must be set, otherwise, this argument must be (nullptr)
	 */
	void* UserParameters = nullptr;
};

/**
 * @struct HParams
 * @brief Hyperparameters
 */
struct HParams
{
	/**
	 * @brief Model paths, key value pairs of model type and model path.
	 *
	 * if the model is a diffusion model, the model type must be following:
	 * - "Ctrl": if your diffusion model has only one onnx model, this is the only path you need to provide
	 * - "Ctrl": the encoder layer of the diffusion model.
	 * - "Denoiser": the denoiser layer of the diffusion model.
	 * - "NoisePredictor": the noise predictor layer of the diffusion model.
	 * - "AlphaCumprod": the alpha cumprod layer of the diffusion model. [optional]
	 *
	 * if the model is a vits based model, the model type must be following:
	 * - "Model": the model path of the vits based model.
	 *
	 * if the model is a reflow model, the model type must be following:
	 * - "Ctrl": the encoder layer of the reflow model.
	 * - "Velocity": the velocity layer of the reflow model.
	 *
	 * if the model is a ddsp model, the model type must be following:
	 * - "Ctrl": the source model of the ddsp model.
	 * - "Velocity": the velocity model of the reflow model.
	 */
	std::unordered_map<std::wstring, std::wstring> ModelPaths;

	TrtConfig TrtSettings;

	std::shared_ptr<TrtModel> HubertModel = nullptr;

	/**
	 * @brief Sampling rate of the output audio, it is the sampling rate of the model, not means the output audio will be resampled to this sampling rate, the output audio will be generated at this sampling rate
	 */
	Int64 OutputSamplingRate = 32000;

	/**
	 * @brief Units dimension, it is the dimension of the units, the units dimension must be greater than (0)
	 */
	Int64 UnitsDim = 256;

	/**
	 * @brief Hop size, it is the hop size of the model, the hop size must be greater than (0)
	 */
	Int64 HopSize = 512;

	/**
	 * @brief Speaker count, it is the count of the speaker, the speaker count must be greater than (0)
	 */
	Int64 SpeakerCount = 1;

	/**
	 * @brief Has volume embedding, it is the flag of the volume embedding layer, if the model has volume embedding layer, this flag must be (true), otherwise, this flag must be (false)
	 */
	bool HasVolumeEmbedding = false;

	/**
	 * @brief Has speaker embedding, it is the flag of the speaker embedding layer, if the model has speaker embedding layer, this flag must be (true), otherwise, this flag must be (false)
	 */
	bool HasSpeakerEmbedding = false;

	/**
	 * @brief Has speaker mix layer, it is the flag of the speaker mix layer, if the model has speaker mix layer, this flag must be (true), otherwise, this flag must be (false)
	 */
	bool HasSpeakerMixLayer = false;

	/**
	 * @brief Spec max, it is the maximum value of the spectrogram, the spec max must be greater than (spec min)
	 */
	float SpecMax = 2.f;

	/**
	 * @brief Spec min, it is the minimum value of the spectrogram, the spec min must be less than (spec max)
	 */
	float SpecMin = -12.f;

	/**
	 * @brief F0 bin, it is the bin count of the f0, the f0 bin must be greater than (0)
	 */
	Int64 F0Bin = 256;

	/**
	 * @brief F0 max, it is the maximum value of the f0, the f0 max must be greater than (f0 min)
	 */
	Float32 F0Max = 1100.0;

	/**
	 * @brief F0 min, it is the minimum value of the f0, the f0 min must be less than (f0 max)
	 */
	Float32 F0Min = 50.0;

	/**
	 * @brief Mel bins, it is the bin count of the mel spectrogram, the mel bins must be greater than (0)
	 */
	Int64 MelBins = 128;

	/**
	 * @brief Progress callback, it is the callback function for progress updates, if you need to get the progress of the inference, you must set this callback function, arguments of the callback function are (arg1, arg2), if arg1 is true, arg2 is the total steps of the inference, if arg1 is false, arg2 is the current step of the inference
	 */
	std::optional<ProgressCallback> ProgressCallback = std::nullopt;

	/**
	 * @brief Extented parameters, key value pairs of extended parameters, for example, the max step of the diffusion model("MaxStep": "1000")
	 */
	std::unordered_map<std::wstring, std::wstring> ExtendedParameters;
};

struct SliceDatas
{
	/**
	 * @brief Source sample rate, it is the sample rate of the source audio, the source sample rate must be greater than (0), this value MUST be set by user
	 */
	Int64 SourceSampleRate = 0;

	/**
	 * @brief Source sample count, it is the sample count of the source audio, the source audio sample count must be greater than (0), this value MUST be set by user
	 */
	Int64 SourceSampleCount = 0;

	/**
	 * @brief Units, it is the units tensor, the units tensor must be a 4D tensor with the shape of [batch size, channels, audio frames, units dims], this tensor MUST be set by user
	 */
	Tensor<Float32, 4, Device::CPU> Units;

	/**
	 * @brief F0, it is the f0 tensor, the f0 tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], this tensor MUST be set by user
	 */
	Tensor<Float32, 3, Device::CPU> F0;

	/**
	 * @brief Volume, it is the volume tensor, the volume tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], this tensor could be automatically generated if the model has volume embedding layer
	 */
	std::optional<Tensor<Float32, 3, Device::CPU>> Volume = std::nullopt;

	/**
	 * @brief UnVoice, it is the unvoice tensor, the unvoice tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], if the model has unvoice layer, this tensor MUST be set by user
	 */
	std::optional<Tensor<Float32, 3, Device::CPU>> UnVoice = std::nullopt;

	/**
	 * @brief F0Embed, it is the f0 embedding tensor, the f0 embedding tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], this tensor will be automatically generated if the model has f0 embedding layer and the f0 tensor is set by user
	 */
	std::optional<Tensor<Int64, 3, Device::CPU>> F0Embed = std::nullopt;

	/**
	 * @brief Units length, it is the units length tensor, the units length tensor must be a 3D tensor with the shape of [batch size, channels, 1], if it is not set, the units length will be set to (audio frames)
	 */
	std::optional<Tensor<Int64, 3, Device::CPU>> UnitsLength = std::nullopt;

	/**
	 * @brief Speaker id, it is the speaker id tensor, the speaker id tensor must be a 3D tensor with the shape of [batch size, channels, 1], if it is not set, the speaker id will be set to (param.speaker_id)
	 */
	std::optional<Tensor<Int64, 3, Device::CPU>> SpeakerId = std::nullopt;

	/**
	 * @brief Speaker, it is the speaker tensor, the speaker tensor must be a 4D tensor with the shape of [batch size, channels, audio frames, speaker count], if the model has speaker mix layer, the speaker tensor could be set by user or automatically generated with zeros and (speaker[:, :, :, param.speaker_id]) will be set to (1)
	 */
	std::optional<Tensor<Float32, 4, Device::CPU>> Speaker = std::nullopt;

	/**
	 * @brief Noise, it is the noise tensor, the noise tensor must be a 4D tensor with the shape of [batch size, channels, noise dims, audio frames], if it is not set, the noise will be automatically generated
	 */
	std::optional<Tensor<Float32, 4, Device::CPU>> Noise = std::nullopt;

	/**
	 * @brief Mel2Units, it is the mel to units tensor, the mel to units tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], if it is not set, the mel to units will be automatically generated, it is used to align the units tensor and the mel spectrogram (and f0)
	 */
	std::optional<Tensor<Int64, 3, Device::CPU>> Mel2Units = std::nullopt;

	/**
	 * @brief Mel, it is the mel tensor, the mel tensor must be a 4D tensor with the shape of [batch size, channels, mel bins, audio frame], if you need shallow diffusion inference, this tensor must be set by user.
	 */
	std::optional<Tensor<Float32, 4, Device::CPU>> GTSpec = std::nullopt;

	/**
	 * @brief GT audio, it is the ground truth audio tensor, the ground truth audio tensor must be a 3D tensor with the shape of [batch size, channels, audio frames], this tensor is automatically generated.
	 */
	std::optional<Tensor<Float32, 3, Device::CPU>> GTAudio = std::nullopt;

	/**
	 * @brief GT sample rate, it is the ground truth sample rate, the ground truth sample rate must be greater than (0).
	 */
	Int64 GTSampleRate = 0;

	/**
	 * @brief Stft noise, it is the stft noise tensor, the stft noise tensor must be a 4D tensor with the shape of [batch size, channels, stft dims, audio frames], it could not be set by user, it is used to generate noise for SoVitsSvc4.0-Beta
	 */
	std::optional<Tensor<Float32, 4, Device::CPU>> StftNoise = std::nullopt;

	/**
	 * @brief Source f0, it is the source f0 tensor, if the preprocessing function modifies the f0 tensor, this tensor will be set to the original f0 tensor, it could not be set by user, it is used to return the original f0 tensor for vocoder or user.
	 */
	std::optional<Tensor<Float32, 3, Device::CPU>> SourceF0 = std::nullopt;
};

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

	Tensor<Float32, 4, Device::CPU> ExtractUnits(
		const Tensor<Float32, 3, Device::CPU>& _PCMData,
		Int64 _SamplingRate = 16000
	) const;

	Tensor<Float32, 4, Device::CPU> Inference(
		const Parameters& Params,
		const Tensor<Float32, 3, Device::CPU>& Audio,
		SizeType SourceSampleRate,
		const PitchExtractor& F0Extractor,
		const PitchParameters& F0Params,
		std::optional<std::reference_wrapper<const Cluster>> UnitsCluster = std::nullopt,
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

	virtual void EmptyCache() = 0;

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
	std::shared_ptr<TrtModel> HubertModel;
	mutable std::unordered_map<Int64, std::shared_ptr<InferenceSession>> HubertSession;

	std::optional<ProgressCallback> _MyProgressCallback;
	DLogger _MyLogger = GetDefaultLogger();

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
	static SliceDatas& PreprocessNoise(
		SliceDatas& MyData,
		Int64 BatchSize,
		Int64 Channels,
		Int64 TargetNumFrames,
		Float32 Scale,
		Int64 Seed,
		const DLogger& Logger = nullptr
	);
};

_D_Dragonian_Lib_TRT_Svc_Space_End
