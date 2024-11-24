#include "../VitsSvc.hpp"
#include <random>
#include <regex>
#include "Libraries/Util/Logger.h"

_D_Dragonian_Lib_TRT_Svc_Space_Header

//"c", "f0", "mel2ph", "uv", "noise", "sid", "vol" "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd", "vol"

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
			HubertModel = std::make_shared<TrtModel>(
				_Hps.HubertPath,
				_Hps.TrtSettings.CacheFile.at(_Hps.HubertPath),
				_Hps.TrtSettings.DynaSetting,
				_Hps.TrtSettings.DLACore,
				_Hps.TrtSettings.Fallback,
				_Hps.TrtSettings.EnableFp16,
				_Hps.TrtSettings.EnableBf16,
				_Hps.TrtSettings.EnableInt8,
				_Hps.TrtSettings.VerboseLevel,
				_Hps.TrtSettings.OptimizationLevel
			);

		VitsSvcModel = std::make_unique<TrtModel>(
			_Hps.ModelPath,
			_Hps.TrtSettings.CacheFile.at(_Hps.ModelPath),
			_Hps.TrtSettings.DynaSetting,
			_Hps.TrtSettings.DLACore,
			_Hps.TrtSettings.Fallback,
			_Hps.TrtSettings.EnableFp16,
			_Hps.TrtSettings.EnableBf16,
			_Hps.TrtSettings.EnableInt8,
			_Hps.TrtSettings.VerboseLevel,
			_Hps.TrtSettings.OptimizationLevel
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
	const auto HubertLen = int64_t(HubertSize) / int64_t(HiddenUnitKDims);
	auto FrameShape = nvinfer1::Dims2{ 1, int64_t(AudioSize * MySamplingRate / SourceSamplingRate / HopSize) };
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

	SvcTensors.InputData.emplace_back(SvcTensors.Data.HiddenUnit.Data());
	SvcTensors.Tensors.emplace_back(
		HiddenUnitShape,
		"c",
		SvcTensors.Data.HiddenUnit.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.F0.Data());
	SvcTensors.Tensors.emplace_back(
		FrameShape,
		"f0",
		SvcTensors.Data.F0.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.Alignment.Data());
	SvcTensors.Tensors.emplace_back(
		FrameShape,
		"mel2ph",
		SvcTensors.Data.Alignment.Size() * sizeof(int64_t),
		nvinfer1::DataType::kINT64
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.UnVoice.Data());
	SvcTensors.Tensors.emplace_back(
		FrameShape,
		"uv",
		SvcTensors.Data.UnVoice.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.Noise.Data());
	SvcTensors.Tensors.emplace_back(
		NoiseShape,
		"noise",
		SvcTensors.Data.Noise.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	if (EnableCharaMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, FrameShape.d[1], Params.SpeakerId, SpeakerCount);

		SvcTensors.InputData.emplace_back(SvcTensors.Data.SpkMap.Data());
		SvcTensors.Tensors.emplace_back(
			SpkShape,
			"sid",
			SvcTensors.Data.SpkMap.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);
	}
	else
	{
		SvcTensors.InputData.emplace_back(SvcTensors.Data.Speaker);
		SvcTensors.Tensors.emplace_back(
			TensorData::OneShape,
			"sid",
			sizeof(int64_t),
			nvinfer1::DataType::kINT64
		);
	}

	if (EnableVolume)
	{
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(FrameShape.d[1]));

		SvcTensors.InputData.emplace_back(SvcTensors.Data.Volume.Data());
		SvcTensors.Tensors.emplace_back(
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
	int32_t SourceSamplingRate,
	int64_t
) const
{
	TensorXData SvcTensors;
	std::mt19937 gen(int(Params.Seed));
	std::normal_distribution<float> normal(0, 1);
	auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(HiddenUnitKDims);
	auto FrameShape = nvinfer1::Dims2{ 1, 0 };
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

	SvcTensors.InputData.emplace_back(SvcTensors.Data.HiddenUnit.Data());
	SvcTensors.Tensors.emplace_back(
		HiddenUnitShape,
		"phone",
		SvcTensors.Data.HiddenUnit.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.Length);
	SvcTensors.Tensors.emplace_back(
		TensorData::OneShape,
		"phone_lengths",
		sizeof(int64_t),
		nvinfer1::DataType::kINT64
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.NSFF0.Data());
	SvcTensors.Tensors.emplace_back(
		FrameShape,
		"pitch",
		SvcTensors.Data.NSFF0.Size() * sizeof(int64_t),
		nvinfer1::DataType::kINT64
	);

	SvcTensors.InputData.emplace_back(SvcTensors.Data.F0.Data());
	SvcTensors.Tensors.emplace_back(
		FrameShape,
		"pitchf",
		SvcTensors.Data.F0.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	if (EnableCharaMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, FrameShape.d[1], Params.SpeakerId, SpeakerCount);

		SvcTensors.InputData.emplace_back(SvcTensors.Data.SpkMap.Data());
		SvcTensors.Tensors.emplace_back(
			SpkShape,
			"ds",
			SvcTensors.Data.SpkMap.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);
	}
	else
	{
		SvcTensors.InputData.emplace_back(SvcTensors.Data.Speaker);
		SvcTensors.Tensors.emplace_back(
			TensorData::OneShape,
			"ds",
			sizeof(int64_t),
			nvinfer1::DataType::kINT64
		);
	}

	SvcTensors.InputData.emplace_back(SvcTensors.Data.Noise.Data());
	SvcTensors.Tensors.emplace_back(
		NoiseShape,
		"rnd",
		SvcTensors.Data.Noise.Size() * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	if (EnableVolume)
	{
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(FrameShape.d[1]));

		SvcTensors.InputData.emplace_back(SvcTensors.Data.Volume.Data());
		SvcTensors.Tensors.emplace_back(
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
	if (VitsSvcVersion == L"SoVits4.0")
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
		std::shared_ptr<InferenceSession> _MyVitsSvcSession = nullptr;
		std::shared_ptr<InferenceSession> _MyHubertSession = nullptr;

		DragonianLibSTL::Vector<float> _16KAudio;
		_16KAudio = InterpResample<float>(_Slice.Audio, _Slice.SamplingRate, 16000);
		const auto _16KAudioSourceSize = _16KAudio.Size();
		const auto _16KAudioPaddedCount = _16KAudioSourceSize % DRAGONIANLIB_PADDING_COUNT ?
			(_16KAudioSourceSize / DRAGONIANLIB_PADDING_COUNT + 1) * DRAGONIANLIB_PADDING_COUNT :
			_16KAudioSourceSize;
		if (_16KAudioSourceSize != _16KAudioPaddedCount)
			_16KAudio.Resize(_16KAudioPaddedCount, 0.f);

		// Dynamic Shape
		{
			auto Iter = VitsSvcSession.find(_16KAudioPaddedCount);
			if (Iter != VitsSvcSession.end())
				_MyVitsSvcSession = Iter->second;
			else
			{
				_MyVitsSvcSession = std::make_shared<InferenceSession>();
				VitsSvcSession[_16KAudioPaddedCount] = _MyVitsSvcSession;
			}
			Iter = HubertSession.find(_16KAudioPaddedCount);
			if (Iter != HubertSession.end())
				_MyHubertSession = Iter->second;
			else
			{
				_MyHubertSession = std::make_shared<InferenceSession>();
				HubertSession[_16KAudioPaddedCount] = _MyHubertSession;
			}
		}

		std::vector<ITensorInfo> HubertInputTensors;

		HubertInputTensors.emplace_back(
			nvinfer1::Dims3(1, 1, (int64_t)_16KAudio.Size()),
			"source",
			_16KAudio.Size() * sizeof(float),
			nvinfer1::DataType::kFLOAT
		);

		try {
			if (!_MyHubertSession->IsReady(HubertInputTensors))
				*_MyHubertSession = HubertModel->Construct(
					HubertInputTensors,
					{ "embed" }
				);
			_MyHubertSession->HostMemoryToDevice(0, _16KAudio.Data(), HubertInputTensors[0].GetSize());
			_MyHubertSession->Run();
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Hubert\n") + e.what()));
		}

		auto& HubertOutputInfos = _MyHubertSession->GetOutputInfos();
		const auto HubertSize = HubertOutputInfos[0].GetElementCount();
		auto& HubertOutPutShape = HubertOutputInfos[0].GetShape();
		if (HubertOutPutShape.d[2] != HiddenUnitKDims)
			_D_Dragonian_Lib_Throw_Exception("HiddenUnitKDims UnMatch");

		DragonianLibSTL::Vector<float> SrcHiddenUnits(HubertSize);
		_MyHubertSession->DeviceMemoryToHost(0, SrcHiddenUnits.Data(), HubertSize * sizeof(float));

		int64_t SpeakerIdx = _Params.SpeakerId;
		if (SpeakerIdx >= SpeakerCount)
			SpeakerIdx = SpeakerCount;
		if (SpeakerIdx < 0)
			SpeakerIdx = 0;

		if (EnableCluster && _Params.ClusterRate > 0.001f)
		{
			const auto pts = Cluster->Search(
				SrcHiddenUnits.Data(), long(SpeakerIdx), HubertOutPutShape.d[1]
			);
			for (int64_t indexs = 0; indexs < HubertOutPutShape.d[1] * HiddenUnitKDims; ++indexs)
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

void VitsSvc::EmptyCache()
{
	VitsSvcSession.clear();
	HubertSession.clear();
}

_D_Dragonian_Lib_TRT_Svc_Space_End