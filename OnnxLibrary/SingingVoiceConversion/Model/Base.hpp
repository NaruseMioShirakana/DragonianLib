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
 */

#pragma once
#include "OnnxLibrary/SingingVoiceConversion/Util/Util.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

class SingingVoiceConversionModule
{
public:
	/**
	 * @struct DiffusionParameters
	 * @brief Parameters for diffusion
	 */
	struct DiffusionParameters
    {
		size_t Stride = 1; ///< Stride of the diffusion
		size_t Step = 1; ///< Step of the diffusion
		std::wstring Sampler = L"Pndm"; ///< Sampler of the diffusion
		float MelFactor = 1.f; ///< Mel factor, multiplied to the mel spectrogram
    };

	/**
	 * @struct ReflowParameters
	 * @brief Parameters for reflow
	 */
	struct ReflowParameters
	{
		size_t Stride = 1; ///< Stride of the reflow
		size_t Step = 1; ///< Step of the reflow
		float Begin = 0.f; ///< Begin of the reflow
		float End = 1.f; ///< End of the reflow
		std::wstring Sampler = L"Eular"; ///< Sampler of the reflow
		float MelFactor = 1.f; ///< Mel factor, multiplied to the mel spectrogram
	};

	/**
	 * @struct Parameters
	 * @brief Parameters for inference
	 */
    struct Parameters
    {
		float NoiseScale = 0.3f; ///< Noise scale, multiplied to the noise
		int64_t SpeakerId = 0; ///< Speaker ID
		float ToneOffset = 0.f; ///< Tone offset for F0
		int64_t Seed = 52468; ///< Random seed
		
		DiffusionParameters Diffusion; ///< Diffusion parameters
		ReflowParameters Reflow; ///< Reflow parameters
		float StftNoiseScale = 0.8f; ///< STFT noise scale for SoVitsSvc4.0-Beta
    };

	struct HParams
	{
		std::unordered_map<std::wstring, std::wstring> ModelPaths; ///< Model paths
		Int64 OutputSamplingRate = 32000;
		Int64 UnitsDim = 256;
		Int64 HopSize = 512;
		Int64 SpeakerCount = 1;
		bool _HasVolumeEmbedding = false;
		bool _HasSpeakerEmbedding = false;
		bool _HasSpeakerMixLayer = false;
		float SpecMax = 2.f;
		float SpecMin = -12.f;
	};

	SingingVoiceConversionModule() = default;
	SingingVoiceConversionModule(const SingingVoiceConversionModule&) = default;
	SingingVoiceConversionModule(SingingVoiceConversionModule&&) noexcept = default;
	SingingVoiceConversionModule& operator=(const SingingVoiceConversionModule&) = default;
	SingingVoiceConversionModule& operator=(SingingVoiceConversionModule&&) noexcept = default;
	virtual ~SingingVoiceConversionModule() = default;

	Tensor<Float32, 4, Device::CPU> Inference(
		const Parameters& Params,
		const Tensor<Float32, 3, Device::CPU>& Audio,
		SizeType SamplingRate,
		const F0Extractor::F0Extractor& F0Extractor,
		const F0Extractor::Parameters& F0Params
	) const;

	virtual Tensor<Float32, 4, Device::CPU> Inference(
		const Parameters& Params,
		const Tensor<Float32, 4, Device::CPU>& Units,
		const Tensor<Float32, 3, Device::CPU>& F0,
		SizeType SourceSampleCount = 0,
		std::optional<Tensor<Float32, 3, Device::CPU>> Volume = std::nullopt,
		std::optional<Tensor<Float32, 4, Device::CPU>> Speaker = std::nullopt,
		std::optional<Tensor<Float32, 4, Device::CPU>> GTSpec = std::nullopt
	);

protected:
	bool _HasVolumeEmbedding = false;
	bool _HasSpeakerEmbedding = false;
	bool _HasSpeakerMixLayer = false;
	Int64 _MyOutputSamplingRate = 32000;
	Int64 _MyUnitsDim = 256;
	Int64 _MyHopSize = 512;
	Int64 _MySpeakerCount = 1;
	float _MySpecMax = 2.f;
	float _MySpecMin = -12.f;

public:
	static Tensor<Float32, 4, Device::CPU> NormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec,
		float SpecMax = 2.f,
		float SpecMin = -12.f
	);

	static Tensor<Float32, 4, Device::CPU> DenormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec,
		float SpecMax = 2.f,
		float SpecMin = -12.f
	);

	Tensor<Float32, 4, Device::CPU> NormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec
	) const;

	Tensor<Float32, 4, Device::CPU> DenormSpec(
		const Tensor<Float32, 4, Device::CPU>& Spec
	) const;
};

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End