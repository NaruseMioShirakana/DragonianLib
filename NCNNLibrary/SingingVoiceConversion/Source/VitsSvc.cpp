#include "../VitsSvc.hpp"
#include <random>
#include <regex>
#include "Libraries/Util/Logger.h"
#include "TensorRT/SingingVoiceConversion/SvcBase.hpp"
#include "../../NCNNBase/NCNNBase.h"

_D_Dragonian_Lib_NCNN_Svc_Space_Header
//"c", "f0", "mel2ph", "uv", "noise", "sid", "vol" "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd", "vol"

void ModelDeleter(void* _Model)
{
	auto* Model = static_cast<NCNNModel*>(_Model);
	delete Model;
}

VitsSvc::~VitsSvc() { LogMessage(L"[Info] unloading VitsSvc Models"); }

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
			Cluster = DragonianLib::Cluster::GetCluster(_Hps.Cluster.Type, _Hps.Cluster.Path, HiddenUnitKDims, ClusterCenterSize);
			EnableCluster = true;
		}
		catch (std::exception& e)
		{
			LogError(UTF8ToWideString(e.what()));
			EnableCluster = false;
		}
	}

	try
	{
		LogMessage(L"[Info] loading VitsSvcModel Models");
		if (_Hps.HubertModel)
			HubertModel = _Hps.HubertModel;
		else
			HubertModel = std::shared_ptr<void>(
				new NCNNModel(_Hps.HubertPath, _Hps.DeviceId, _Hps.ThreadCount, _Hps.UseVulkan, _Hps.Precision, true),
				ModelDeleter
			);

		VitsSvcModel = std::shared_ptr<void>(
			new NCNNModel(_Hps.HubertPath, _Hps.DeviceId, _Hps.ThreadCount, _Hps.UseVulkan, _Hps.Precision, false),
			ModelDeleter
		);
		LogMessage(L"[Info] VitsSvcModel Models loaded");
	}
	catch (std::exception& _exception)
	{
		_D_Dragonian_Lib_Throw_Exception(_exception.what());
	}
}

TensorXData VitsSvc::SoVits4Preprocess(
	const DragonianLibSTL::Vector<float>& HiddenUnit,
	const DragonianLibSTL::Vector<float>& F0,
	const DragonianLibSTL::Vector<float>& Volume,
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
	const InferenceParams& Params,
	int32_t SourceSamplingRate,
	int64_t AudioSize
) const
{
	TensorXData SvcTensors;
	std::mt19937 gen(int(Params.Seed));
	std::normal_distribution<float> normal(0, 1);
	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int(HubertSize) / int(HiddenUnitKDims);

	int FrameShape = int(AudioSize * MySamplingRate / 16000 / HopSize);
	int HiddenUnitShape[] = { HubertLen, int(HiddenUnitKDims) };
	int SpkShape[] = { FrameShape, int(SpeakerCount) };
	int NoiseShape[] = { 192, FrameShape };
	const auto NoiseSize = NoiseShape[1] * NoiseShape[0];

	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(FrameShape)));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(Params.Keys) / 12.0);
	SvcTensors.Data.UnVoice = GetUV(F0);
	SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
	for (auto& it : SvcTensors.Data.Noise)
		it = normal(gen) * Params.NoiseScale;
	SvcTensors.Data.Speaker[0] = static_cast<int>(Params.SpeakerId);

	SvcTensors.Tensors.emplace_back(
		SvcTensors.Data.HiddenUnit.Data(),
		{ HiddenUnitShape[0], HiddenUnitShape[1], 0, 0 },
		2,
		SvcTensors.Data.HiddenUnit.Size() * sizeof(float)
	);

	SvcTensors.Tensors.emplace_back(
		SvcTensors.Data.F0.Data(),
		{ FrameShape, 0, 0, 0 },
		1,
		SvcTensors.Data.F0.Size() * sizeof(float)
	);

	SvcTensors.Tensors.emplace_back(
		SvcTensors.Data.UnVoice.Data(),
		{ FrameShape, 0, 0, 0 },
		1,
		SvcTensors.Data.UnVoice.Size() * sizeof(float)
	);

	SvcTensors.Tensors.emplace_back(
		SvcTensors.Data.Noise.Data(),
		{ NoiseShape[0], NoiseShape[1], 0, 0 },
		2,
		SvcTensors.Data.Noise.Size() * sizeof(float)
	);

	if (EnableCharaMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, FrameShape, Params.SpeakerId, SpeakerCount);
		SvcTensors.Tensors.emplace_back(
			SvcTensors.Data.SpkMap.Data(),
			{ SpkShape[0], SpkShape[1], 0, 0 },
			2,
			SvcTensors.Data.SpkMap.Size() * sizeof(float)
		);
	}
	else
	{
		SvcTensors.Tensors.emplace_back(
			SvcTensors.Data.Speaker,
			{ 1, 0, 0, 0 },
			1,
			sizeof(int)
		);
	}

	if (EnableVolume)
	{
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(FrameShape));
		SvcTensors.Tensors.emplace_back(
			SvcTensors.Data.Volume.Data(),
			{ FrameShape, 0, 0, 0 },
			1,
			SvcTensors.Data.Volume.Size() * sizeof(float)
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
	int32_t SourceSamplingRate,
	int64_t
) const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

TensorXData VitsSvc::Preprocess(
	const DragonianLibSTL::Vector<float>& HiddenUnit,
	const DragonianLibSTL::Vector<float>& F0,
	const DragonianLibSTL::Vector<float>& Volume,
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
	const InferenceParams& Params,
	int32_t SourceSamplingRate,
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
			SourceSamplingRate,
			AudioSize
		);
	if (VitsSvcVersion == L"SoVits4.0" || VitsSvcVersion == L"SoVits")
		return SoVits4Preprocess(
			HiddenUnit,
			F0,
			Volume,
			SpkMap,
			Params,
			SourceSamplingRate,
			AudioSize
		);
	_D_Dragonian_Lib_Not_Implemented_Error;
}

DragonianLibSTL::Vector<float> VitsSvc::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params
)
{
	if (_Slice.IsNotMute)
	{
		auto& _MyVitsSvcSession = *static_cast<NCNNModel*>(VitsSvcModel.get());
		auto& _MyHubertSession = *static_cast<NCNNModel*>(HubertModel.get());

		DragonianLibSTL::Vector<float> _16KAudio;
		_16KAudio = InterpResample<float>(_Slice.Audio, _Slice.SamplingRate, 16000);
		const auto _16KAudioSourceSize = _16KAudio.Size();
		const auto _16KAudioPaddedCount = _16KAudioSourceSize % DRAGONIANLIB_PADDING_COUNT ?
			(_16KAudioSourceSize / DRAGONIANLIB_PADDING_COUNT + 1) * DRAGONIANLIB_PADDING_COUNT :
			_16KAudioSourceSize;
		if (_16KAudioSourceSize != _16KAudioPaddedCount)
			_16KAudio.Resize(_16KAudioPaddedCount, 0.f);

		std::vector<Tensor> HubertInputTensors, HubertOutputTensors;

		HubertInputTensors.emplace_back(
			_16KAudio.Data(),
			{ static_cast<int>(_16KAudio.Size()), 1, 1, 1 },
			3,
			_16KAudio.Size() * sizeof(float)
		);

		try 
		{
			HubertOutputTensors = _MyHubertSession.Run(HubertInputTensors);
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Hubert\n") + e.what()));
		}

		auto& HubertOutputInfos = HubertOutputTensors[0];
		const auto HubertSize = HubertOutputInfos.BufferSize / sizeof(float);
		auto& HubertOutPutShape = HubertOutputInfos.Shape;
		if (HubertOutPutShape[1] != HiddenUnitKDims)
			_D_Dragonian_Lib_Throw_Exception("HiddenUnitKDims UnMatch");

		auto SrcHiddenUnits = static_cast<float*>(HubertOutputInfos.Buffer);

		int64_t SpeakerIdx = _Params.SpeakerId;
		if (SpeakerIdx >= SpeakerCount)
			SpeakerIdx = SpeakerCount;
		if (SpeakerIdx < 0)
			SpeakerIdx = 0;

		if (EnableCluster && _Params.ClusterRate > 0.001f)
		{
			const auto pts = Cluster->Search(
				SrcHiddenUnits, long(SpeakerIdx), HubertOutPutShape[1]
			);
			for (int64_t indexs = 0; indexs < HubertOutPutShape[1] * HiddenUnitKDims; ++indexs)
				SrcHiddenUnits[indexs] = SrcHiddenUnits[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
		}

		TensorXData InputTensors;
		if (_16KAudioSourceSize != _16KAudioPaddedCount)
		{
			const auto _MyF0Size = _Slice.F0.Size();
			const auto _PaddedF0Size = _MyF0Size * _16KAudioPaddedCount / _16KAudioSourceSize;
			auto _PaddedF0 = _Slice.F0;
			_PaddedF0.Resize(_PaddedF0Size, 0.f);
			auto _PaddedVolume = _Slice.Volume;
			_PaddedVolume.Resize(_PaddedF0Size, 0.f);
			DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _PaddedSpkMap;
			if (EnableCharaMix)
			{
				_PaddedSpkMap = _Slice.Speaker;
				for (auto& it : _PaddedSpkMap)
					it.Resize(_PaddedF0Size, 0.f);
			}
			InputTensors = Preprocess(
				SrcHiddenUnits, _PaddedF0, _PaddedVolume, _PaddedSpkMap, _Params, _Slice.SamplingRate, int64_t(_16KAudioPaddedCount)
			);
		}
		else
			InputTensors = Preprocess(
				SrcHiddenUnits, _Slice.F0, _Slice.Volume, _Slice.Speaker, _Params, _Slice.SamplingRate, int64_t(_16KAudioSourceSize)
			);

		try
		{
			if (!_MyVitsSvcSession->IsReady(InputTensors.Tensors))
				*_MyVitsSvcSession = VitsSvcModel->Construct(
					InputTensors.Tensors,
					{ "audio" }
				);
			for (size_t i = 0; i < InputTensors.InputData.size(); ++i)
				_MyVitsSvcSession->HostMemoryToDevice(i, InputTensors.InputData[i], InputTensors.Tensors[i].GetSize());
			_MyVitsSvcSession->Run();
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: VitsSvc\n") + e.what()));
		}

		auto VitsOutputAudioSize = _MyVitsSvcSession->GetOutputInfos()[0].GetElementCount();
		DragonianLibSTL::Vector<float> VitsOutputAudio(VitsOutputAudioSize);
		_MyVitsSvcSession->DeviceMemoryToHost(0, VitsOutputAudio.Data(), VitsOutputAudioSize * sizeof(float));

		const auto dstWavLen = (_Slice.OrgLen * int64_t(MySamplingRate)) / (int)(_Slice.SamplingRate);
		VitsOutputAudio.Resize(dstWavLen, 0.f);
		return VitsOutputAudio;
	}
	//Mute clips
	const auto len = size_t(_Slice.OrgLen * int64_t(MySamplingRate) / (int)(_Slice.SamplingRate));
	return { len, 0.f, GetMemoryProvider(DragonianLib::Device::CPU) };
}

_D_Dragonian_Lib_NCNN_Svc_Space_End