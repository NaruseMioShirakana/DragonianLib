﻿#include "VitsSvc.hpp"
#include <random>
#include <regex>
#include "Util/Logger.h"

namespace tlibsvc {

	VitsSvc::~VitsSvc() { DragonianLibLogMessage(L"[Info] unloading VitsSvc Models"); }

	VitsSvc::VitsSvc(
		const VitsSvcConfig& _Hps,
		const ProgressCallback& _ProgressCallback
	)
	{
		MySamplingRate = std::max(_Hps.SamplingRate, 2000l);
		HopSize = std::max(_Hps.HopSize, 1);
		HiddenUnitKDims = std::max(_Hps.HiddenUnitKDims, 1ll);
		SpeakerCount = std::max(_Hps.SpeakerCount, 1ll);
		EnableVolume = _Hps.EnableVolume;
		EnableCharaMix = _Hps.EnableCharaMix;
		VitsSvcVersion = _Hps.TensorExtractor;

		ProgressFn = _ProgressCallback;

		if (!_Hps.Cluster.Type.empty())
		{
			ClusterCenterSize = _Hps.Cluster.ClusterCenterSize;
			try
			{
				Cluster = DragonianLib::GetCluster(_Hps.Cluster.Type, _Hps.Cluster.Path, HiddenUnitKDims, ClusterCenterSize);
				EnableCluster = true;
			}
			catch (std::exception& e)
			{
				DragonianLibErrorMessage(e.what());
				EnableCluster = false;
			}
		}

		try
		{
			DragonianLibLogMessage(L"[Info] loading VitsSvcModel Models");
			if (_Hps.HubertModel)
				HubertModel = _Hps.HubertModel;
			else
				HubertModel = std::make_shared<TrtModel>(_Hps.HubertPath, _Hps.TrtSettings.CacheFile, _Hps.TrtSettings.DLACore, _Hps.TrtSettings.Fallback, _Hps.TrtSettings.EnableFp16, _Hps.TrtSettings.EnableBf16, _Hps.TrtSettings.EnableInt8, _Hps.TrtSettings.VerboseLevel);
			VitsSvcModel = std::make_unique<TrtModel>(_Hps.ModelPath, _Hps.TrtSettings.CacheFile, _Hps.TrtSettings.DLACore, _Hps.TrtSettings.Fallback, _Hps.TrtSettings.EnableFp16, _Hps.TrtSettings.EnableBf16, _Hps.TrtSettings.EnableInt8, _Hps.TrtSettings.VerboseLevel);
			DragonianLibLogMessage(L"[Info] VitsSvcModel Models loaded");
		}
		catch (std::exception& _exception)
		{
			DragonianLibThrow(_exception.what());
		}
	}

	TensorXData VitsSvc::SoVits4Preprocess(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		const InferenceParams& Params,
		int64_t AudioSize
	) const
	{
		TensorXData SvcTensors;
		std::mt19937 gen(int(Params.Seed));
		std::normal_distribution<float> normal(0, 1);
		const auto HubertSize = HiddenUnit.Size();
		const auto HubertLen = int64_t(HubertSize) / int64_t(HiddenUnitKDims);
		auto FrameShape = nvinfer1::Dims2{ 1, int64_t(AudioSize * MySamplingRate / Params.SrcSamplingRate / HopSize) };
		auto HiddenUnitShape = nvinfer1::Dims3{ 1, HubertLen, int64_t(HiddenUnitKDims) };
		auto SpkShape = nvinfer1::Dims2{ FrameShape.d[1], int64_t(SpeakerCount) };
		auto NoiseShape = nvinfer1::Dims3{ 1, 192, FrameShape.d[1] };
		const auto NoiseSize = NoiseShape.d[1] * NoiseShape.d[2] * NoiseShape.d[0];

		SvcTensors.Data.HiddenUnit = HiddenUnit;
		SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(FrameShape.d[1])));
		for (auto& it : SvcTensors.Data.F0)
			it *= (float)pow(2.0, static_cast<double>(Params.Keys) / 12.0);
		SvcTensors.Data.Alignment = GetAligments(FrameShape.d[1], HubertLen);
		SvcTensors.Data.UnVoice = GetUV(F0);
		SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
		for (auto& it : SvcTensors.Data.Noise)
			it = normal(gen) * Params.NoiseScale;
		SvcTensors.Data.Speaker[0] = Params.SpeakerId;

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.HiddenUnit.Data(),
			HiddenUnitShape,
			"c",
			SvcTensors.Data.HiddenUnit.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.F0.Data(),
			FrameShape,
			"f0",
			SvcTensors.Data.F0.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.Alignment.Data(),
			FrameShape,
			"mel2ph",
			SvcTensors.Data.Alignment.Size() * sizeof(int64_t),
			nvinfer1::DataType::kINT64
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.UnVoice.Data(),
			FrameShape,
			"uv",
			SvcTensors.Data.UnVoice.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.Noise.Data(),
			NoiseShape,
			"noise",
			SvcTensors.Data.Noise.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		if (EnableCharaMix)
		{
			SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, FrameShape.d[1], Params.SpeakerId, SpeakerCount);

			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.SpkMap.Data(),
				SpkShape,
				"sid",
				SvcTensors.Data.SpkMap.Size() * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);
		}
		else
		{
			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.Speaker,
				TensorData::OneShape,
				"sid",
				sizeof(int64_t),
				nvinfer1::DataType::kINT64
			);
		}

		if (EnableVolume)
		{
			SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(FrameShape.d[1]));

			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.Volume.Data(),
				FrameShape,
				"vol",
				SvcTensors.Data.Volume.Size() * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);
		}

		return SvcTensors;
	}

	TensorXData VitsSvc::RVCTensorPreprocess(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		const InferenceParams& Params,
		int64_t AudioSize
	) const
	{
		TensorXData SvcTensors;
		std::mt19937 gen(int(Params.Seed));
		std::normal_distribution<float> normal(0, 1);
		auto HubertSize = HiddenUnit.Size();
		const auto HubertLen = int64_t(HubertSize) / int64_t(HiddenUnitKDims);
		auto FrameShape = nvinfer1::Dims2{ 1, int64_t(AudioSize * MySamplingRate / Params.SrcSamplingRate / HopSize) };
		auto HiddenUnitShape = nvinfer1::Dims3{ 1, HubertLen, int64_t(HiddenUnitKDims) };
		constexpr int64_t upSample = 2;
		const auto srcHubertSize = HiddenUnitShape.d[1];
		HiddenUnitShape.d[1] *= upSample;
		HubertSize *= upSample;
		FrameShape.d[1] = HiddenUnitShape.d[1];
		auto SpkShape = nvinfer1::Dims2{ FrameShape.d[1], int64_t(SpeakerCount) };
		auto NoiseShape = nvinfer1::Dims3{ 1, 192, FrameShape.d[1] };
		const auto NoiseSize = NoiseShape.d[1] * NoiseShape.d[2] * NoiseShape.d[0];

		SvcTensors.Data.HiddenUnit.Reserve(HubertSize);
		for (int64_t itS = 0; itS < srcHubertSize; ++itS)
			for (int64_t itSS = 0; itSS < upSample; ++itSS)
				SvcTensors.Data.HiddenUnit.Insert(SvcTensors.Data.HiddenUnit.end(), HiddenUnit.begin() + itS * (int64_t)HiddenUnitKDims, HiddenUnit.begin() + (itS + 1) * (int64_t)HiddenUnitKDims);

		SvcTensors.Data.Length[0] = HiddenUnitShape.d[1];
		SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(HiddenUnitShape.d[1])));
		for (auto& it : SvcTensors.Data.F0)
			it *= (float)pow(2.0, static_cast<double>(Params.Keys) / 12.0);
		SvcTensors.Data.NSFF0 = GetNSFF0(SvcTensors.Data.F0);
		SvcTensors.Data.Alignment = GetAligments(FrameShape.d[1], HubertLen);
		SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
		for (auto& it : SvcTensors.Data.Noise)
			it = normal(gen) * Params.NoiseScale;
		SvcTensors.Data.Speaker[0] = Params.SpeakerId;

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.HiddenUnit.Data(),
			HiddenUnitShape,
			"phone",
			SvcTensors.Data.HiddenUnit.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.Length,
			TensorData::OneShape,
			"phone_lengths",
			sizeof(int64_t),
			nvinfer1::DataType::kINT64
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.NSFF0.Data(),
			FrameShape,
			"pitch",
			SvcTensors.Data.NSFF0.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.F0.Data(),
			FrameShape,
			"pitchf",
			SvcTensors.Data.F0.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		if (EnableCharaMix)
		{
			SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, FrameShape.d[1], Params.SpeakerId, SpeakerCount);

			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.SpkMap.Data(),
				SpkShape,
				"ds",
				SvcTensors.Data.SpkMap.Size() * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);
		}
		else
		{
			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.Speaker,
				TensorData::OneShape,
				"ds",
				sizeof(int64_t),
				nvinfer1::DataType::kINT64
			);
		}

		SvcTensors.Tensors.EmplaceBack(
			SvcTensors.Data.Noise.Data(),
			NoiseShape,
			"rnd",
			SvcTensors.Data.Noise.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		if (EnableVolume)
		{
			SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(FrameShape.d[1]));

			SvcTensors.Tensors.EmplaceBack(
				SvcTensors.Data.Volume.Data(),
				FrameShape,
				"vol",
				SvcTensors.Data.Volume.Size() * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);
		}

		return SvcTensors;
	}

	TensorXData VitsSvc::Preprocess(
		const DragonianLibSTL::Vector<float>& HiddenUnit,
		const DragonianLibSTL::Vector<float>& F0,
		const DragonianLibSTL::Vector<float>& Volume,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
		const InferenceParams& Params,
		int64_t AudioSize
	) const
	{
		if (VitsSvcVersion == L"RVC")
			return RVCTensorPreprocess(
				HiddenUnit,
				F0,
				Volume,
				SpkMap,
				Params,
				AudioSize
			);
		if (VitsSvcVersion == L"SoVits4.0")
			return SoVits4Preprocess(
				HiddenUnit,
				F0,
				Volume,
				SpkMap,
				Params,
				AudioSize
			);
		DragonianLibNotImplementedError;
	}

	DragonianLibSTL::Vector<int16_t> VitsSvc::SliceInference(
		const SingleSlice& _Slice,
		const InferenceParams& _Params,
		ioBuffer_t& _IOBuffer
	) const
	{
		if (_Slice.IsNotMute)
		{
			DragonianLibSTL::Vector<float> _16KAudio;

			_16KAudio = InterpResample(_Slice.Audio, (int)(_Params.SrcSamplingRate), 16000, 32768.0f);
			const auto src_audio_length = _16KAudio.Size();
			bool NeedPadding = false;
#ifdef LIBSVC_CUDA_ONLY_PADDING
			if (_cur_execution_provider == ExecutionProviders::CUDA)
#endif
			{
				NeedPadding = _16KAudio.Size() % DRAGONIANLIB_PADDING_COUNT;
				const size_t WavPaddedSize = _16KAudio.Size() / DRAGONIANLIB_PADDING_COUNT + 1;
				if (NeedPadding)
					_16KAudio.Resize(WavPaddedSize * DRAGONIANLIB_PADDING_COUNT, 0.f);
			}

			DragonianLibSTL::Vector<TrtTensor> HubertInputTensors, HubertOutPuts;

			HubertInputTensors.EmplaceBack(
				_16KAudio.Data(),
				nvinfer1::Dims3(1, 1, (int64_t)_16KAudio.Size()),
				"source",
				_16KAudio.Size() * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);

			try {
				HubertOutPuts = HubertModel->Infer(
					HubertInputTensors,
					_IOBuffer[0].Reload(*HubertModel),
					{ "embed" }
				);
			}
			catch (std::exception& e)
			{
				DragonianLibThrow((std::string("Locate: Hubert\n") + e.what()));
			}

			HubertOutPuts[0].DeviceData2Host();
			const auto HubertSize = HubertOutPuts[0].GetElementCount();
			const auto HubertOutPutData = (float*)HubertOutPuts[0].Data;
			auto HubertOutPutShape = HubertOutPuts[0].Shape;
			if (HubertOutPutShape.d[2] != HiddenUnitKDims)
				DragonianLibThrow("HiddenUnitKDims UnMatch");

			DragonianLibSTL::Vector SrcHiddenUnits(HubertOutPutData, HubertOutPutData + HubertSize);

			int64_t SpeakerIdx = _Params.SpeakerId;
			if (SpeakerIdx >= SpeakerCount)
				SpeakerIdx = SpeakerCount;
			if (SpeakerIdx < 0)
				SpeakerIdx = 0;

			const auto max_cluster_size = int64_t((size_t)HubertOutPutShape.d[1] * src_audio_length / _16KAudio.Size());
			if (EnableCluster && _Params.ClusterRate > 0.001f)
			{
				const auto pts = Cluster->Search(SrcHiddenUnits.Data(), long(SpeakerIdx), max_cluster_size);
				for (int64_t indexs = 0; indexs < max_cluster_size * HiddenUnitKDims; ++indexs)
					SrcHiddenUnits[indexs] = SrcHiddenUnits[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
			}

			TensorXData InputTensors;

			if (NeedPadding)
			{
				DragonianLibSTL::Vector<float> F0Padded, VolumePadded;
				DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> SpeakerPadded;

				F0Padded = _Slice.F0;
				VolumePadded = _Slice.Volume;
				SpeakerPadded = _Slice.Speaker;
				const auto ScaleSamplingConut = _Params.SrcSamplingRate * DRAGONIANLIB_PADDING_COUNT / 16000;
				const auto SrcAudioLength = _Slice.Audio.Size();
				const size_t WavPaddedSize = (SrcAudioLength / ScaleSamplingConut + 1) * ScaleSamplingConut;
				const size_t AudioPadSize = WavPaddedSize - SrcAudioLength;
				const size_t PaddedF0Size = F0Padded.Size() + (F0Padded.Size() * AudioPadSize / SrcAudioLength);

				if (!F0Padded.Empty()) F0Padded.Resize(PaddedF0Size, 0.f);
				if (!VolumePadded.Empty()) VolumePadded.Resize(PaddedF0Size, 0.f);
				for (auto iSpeaker : SpeakerPadded)
				{
					if (!iSpeaker.Empty())
						iSpeaker.Resize(PaddedF0Size, 0.f);
				}
				InputTensors = Preprocess(SrcHiddenUnits, F0Padded, VolumePadded, SpeakerPadded, _Params, int64_t(WavPaddedSize));
			}
			else
				InputTensors = Preprocess(SrcHiddenUnits, _Slice.F0, _Slice.Volume, _Slice.Speaker, _Params, int64_t(_Slice.OrgLen));

			DragonianLibSTL::Vector<TrtTensor> finaOut;
			try
			{
				finaOut = VitsSvcModel->Infer(
					InputTensors.Tensors,
					_IOBuffer[1].Reload(*VitsSvcModel),
					{ "audio" }
				);
			}
			catch (std::exception& e)
			{
				DragonianLibThrow((std::string("Locate: VitsSvc\n") + e.what()));
			}

			auto VitsOutputAudioSize = finaOut[0].GetElementCount();
			DragonianLibSTL::Vector<int16_t> VitsOutput(VitsOutputAudioSize);
			{
				finaOut[0].DeviceData2Host();
				auto VitsOutputAudioData = (float*)finaOut[0].Data;
				auto OutputAudioData = VitsOutput.Data();
				const auto OutputAudioEnd = OutputAudioData + VitsOutput.Size();
				while (OutputAudioData != OutputAudioEnd)
					*(OutputAudioData++) = (int16_t)(*(VitsOutputAudioData++) * 32760.f);
			}

			const auto dstWavLen = (_Slice.OrgLen * int64_t(MySamplingRate)) / (int)(_Params.SrcSamplingRate);
			VitsOutput.Resize(dstWavLen, 0);
			return VitsOutput;
		}
		//Mute clips
		const auto len = size_t(_Slice.OrgLen * int64_t(MySamplingRate) / (int)(_Params.SrcSamplingRate));
		return { len, 0i16, GetMemoryProvider(DragonianLib::Device::CPU) };
	}
}