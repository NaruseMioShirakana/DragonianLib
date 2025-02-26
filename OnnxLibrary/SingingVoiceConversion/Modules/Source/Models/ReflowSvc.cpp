﻿#include "../../header/Models/ReflowSvc.hpp"
#include "../../header/InferTools/Sampler/SamplerManager.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"

#include <random>
#include <regex>

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

ReflowSvc::~ReflowSvc()
{
	LogInfo(L"Unloading ReflowSvc Models");
}

ReflowSvc::ReflowSvc(
	const Hparams& _Hps,
	const ProgressCallback& _ProgressCallback,
	ExecutionProviders ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : SingingVoiceConversion(_Hps.HubertPath, ExecutionProvider_, DeviceID_, ThreadCount_)
{
	ModelSamplingRate = std::max(_Hps.SamplingRate, 2000l);
	MelBins = std::max(_Hps.MelBins, 1ll);
	HopSize = std::max(_Hps.HopSize, 1);
	HiddenUnitKDims = std::max(_Hps.HiddenUnitKDims, 1ll);
	SpeakerCount = std::max(_Hps.SpeakerCount, 1ll);
	EnableVolume = _Hps.EnableVolume;
	EnableCharaMix = _Hps.EnableCharaMix;
	ReflowSvcVersion = _Hps.TensorExtractor;
	SpecMax = _Hps.SpecMax;
	SpecMin = _Hps.SpecMin;
	MaxStep = std::max(_Hps.MaxStep, 1ll);
	Scale = _Hps.Scale;
	VaeMode = _Hps.VaeMode;
	F0Min = _Hps.F0Min;
	F0Max = _Hps.F0Max;

	ProgressCallbackFunction = _ProgressCallback;

	if (!_Hps.Cluster.Type.empty())
	{
		ClusterCenterSize = _Hps.Cluster.ClusterCenterSize;
		try
		{
			Cluster = Cluster::GetCluster(_Hps.Cluster.Type, _Hps.Cluster.Path, HiddenUnitKDims, ClusterCenterSize);
			EnableCluster = true;
		}
		catch (std::exception& e)
		{
			LogWarn(UTF8ToWideString(e.what()));
			EnableCluster = false;
		}
	}

	//LoadModels
	try
	{
		LogInfo(L"Loading ReflowSvc Models");

		PreEncoder = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.Encoder.c_str(), *SessionOptions);
		VelocityFunction = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.VelocityFn.c_str(), *SessionOptions);
		PostDecoder = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.After.c_str(), *SessionOptions);

		LogInfo(L"ReflowSvc Models loaded");
	}
	catch (Ort::Exception& _exception)
	{
		_D_Dragonian_Lib_Throw_Exception(_exception.what());
	}

	TensorExtractor::LibSvcTensorExtractor::Others _others_param;
	_others_param.Memory = *MemoryInfo;

	try
	{
		Preprocessor = GetTensorExtractor(ReflowSvcVersion, 48000, ModelSamplingRate, HopSize, EnableCharaMix, EnableVolume, HiddenUnitKDims, SpeakerCount, _others_param);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

ReflowSvc::ReflowSvc(
	const Hparams& _Hps,
	const ProgressCallback& _ProgressCallback,
	const std::shared_ptr<DragonianLibOrtEnv>& Env_
) : SingingVoiceConversion(_Hps.HubertPath, Env_)
{
	ModelSamplingRate = std::max(_Hps.SamplingRate, 2000l);
	MelBins = std::max(_Hps.MelBins, 1ll);
	HopSize = std::max(_Hps.HopSize, 1);
	HiddenUnitKDims = std::max(_Hps.HiddenUnitKDims, 1ll);
	SpeakerCount = std::max(_Hps.SpeakerCount, 1ll);
	EnableVolume = _Hps.EnableVolume;
	EnableCharaMix = _Hps.EnableCharaMix;
	ReflowSvcVersion = _Hps.TensorExtractor;
	SpecMax = _Hps.SpecMax;
	SpecMin = _Hps.SpecMin;
	MaxStep = std::max(_Hps.MaxStep, 1ll);
	Scale = _Hps.Scale;
	VaeMode = _Hps.VaeMode;
	F0Min = _Hps.F0Min;
	F0Max = _Hps.F0Max;

	ProgressCallbackFunction = _ProgressCallback;

	if (!_Hps.Cluster.Type.empty())
	{
		ClusterCenterSize = _Hps.Cluster.ClusterCenterSize;
		try
		{
			Cluster = Cluster::GetCluster(_Hps.Cluster.Type, _Hps.Cluster.Path, HiddenUnitKDims, ClusterCenterSize);
			EnableCluster = true;
		}
		catch (std::exception& e)
		{
			LogWarn(UTF8ToWideString(e.what()));
			EnableCluster = false;
		}
	}

	//LoadModels
	try
	{
		LogInfo(L"Loading ReflowSvc Models");

		PreEncoder = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.Encoder.c_str(), *SessionOptions);
		VelocityFunction = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.VelocityFn.c_str(), *SessionOptions);
		PostDecoder = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.ReflowSvc.After.c_str(), *SessionOptions);

		LogInfo(L"ReflowSvc Models loaded");
	}
	catch (Ort::Exception& _exception)
	{
		_D_Dragonian_Lib_Throw_Exception(_exception.what());
	}

	TensorExtractor::LibSvcTensorExtractor::Others _others_param;
	_others_param.Memory = *MemoryInfo;

	try
	{
		Preprocessor = GetTensorExtractor(ReflowSvcVersion, 48000, ModelSamplingRate, HopSize, EnableCharaMix, EnableVolume, HiddenUnitKDims, SpeakerCount, _others_param);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

DragonianLibSTL::Vector<float> ReflowSvc::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params,
	size_t& _Process
) const
{
	if (!_Params.VocoderModel)
		_D_Dragonian_Lib_Throw_Exception("Missing Vocoder Model!");
	std::mt19937 gen(int(_Params.Seed));
	std::normal_distribution<float> normal(0, 1);
	auto step = (int64_t)_Params.Step;
	if (step > MaxStep) step = MaxStep;
	const auto SingleStepSkip = step;
	if (_Slice.IsNotMute)
	{
		auto RawWav = InterpResample<float>(_Slice.Audio, (int)(_Slice.SamplingRate), 16000);
		bool NeedPadding = false;
#ifdef LIBSVC_CUDA_ONLY_PADDING
		if (ModelExecutionProvider == ExecutionProviders::CUDA)
#endif
		{
			NeedPadding = RawWav.Size() % DRAGONIANLIB_PADDING_COUNT;
			const size_t WavPaddedSize = RawWav.Size() / DRAGONIANLIB_PADDING_COUNT + 1;
			if (NeedPadding)
				RawWav.Resize(WavPaddedSize * DRAGONIANLIB_PADDING_COUNT, 0.f);
		}

		const int64_t HubertInputShape[3] = { 1i64,1i64,(int64_t)RawWav.Size() };
		OrtTensors HubertInputTensors, HubertOutPuts;
		HubertInputTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, RawWav.Data(), RawWav.Size(), HubertInputShape, 3));
		try {
			HubertOutPuts = HubertModel->Run(Ort::RunOptions{ nullptr },
				hubertInput.data(),
				HubertInputTensors.data(),
				HubertInputTensors.size(),
				hubertOutput.data(),
				hubertOutput.size());
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: hubert\n") + e.what()));
		}
		const auto HubertSize = HubertOutPuts[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto HubertOutPutData = HubertOutPuts[0].GetTensorMutableData<float>();
		auto HubertOutPutShape = HubertOutPuts[0].GetTensorTypeAndShapeInfo().GetShape();
		HubertInputTensors.clear();
		if (HubertOutPutShape[2] != HiddenUnitKDims)
			_D_Dragonian_Lib_Throw_Exception("HiddenUnitKDims UnMatch");

		DragonianLibSTL::Vector srcHiddenUnits(HubertOutPutData, HubertOutPutData + HubertSize);
		int64_t SpeakerIdx = _Params.SpeakerId;
		if (SpeakerIdx >= SpeakerCount)
			SpeakerIdx = SpeakerCount;
		if (SpeakerIdx < 0)
			SpeakerIdx = 0;

		if (EnableCluster && _Params.ClusterRate > 0.001f)
		{
			const auto pts = Cluster->Search(
				srcHiddenUnits.Data(), long(SpeakerIdx), HubertOutPutShape[1]
			);
			for (int64_t indexs = 0; indexs < HubertOutPutShape[1] * HiddenUnitKDims; ++indexs)
				srcHiddenUnits[indexs] = srcHiddenUnits[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
		}
		OrtTensors finaOut;
		OrtTensors ReflowOut;

		TensorExtractor::LibSvcTensorExtractor::InferParams _Inference_Params;
		_Inference_Params.AudioSize = _Slice.Audio.Size();
		_Inference_Params.Chara = SpeakerIdx;
		_Inference_Params.NoiseScale = _Params.NoiseScale;
		_Inference_Params.DDSPNoiseScale = _Params.DDSPNoiseScale;
		_Inference_Params.Seed = int(_Params.Seed);
		_Inference_Params.upKeys = _Params.Keys;
		_Inference_Params.SrcSamplingRate = _Slice.SamplingRate;

		TensorExtractor::LibSvcTensorExtractor::Inputs InputTensors;

		if (NeedPadding)
		{
			auto CUDAF0 = _Slice.F0;
			auto CUDAVolume = _Slice.Volume;
			auto CUDASpeaker = _Slice.Speaker;
			const auto ScaleSamplingConut = _Slice.SamplingRate * DRAGONIANLIB_PADDING_COUNT / 16000;
			const auto SrcAudioLength = _Slice.Audio.Size();
			const size_t WavPaddedSize = (SrcAudioLength / ScaleSamplingConut + 1) * ScaleSamplingConut;
			const size_t AudioPadSize = WavPaddedSize - SrcAudioLength;
			const size_t PaddedF0Size = CUDAF0.Size() + (CUDAF0.Size() * AudioPadSize / SrcAudioLength);

			if (!CUDAF0.Empty()) CUDAF0.Resize(PaddedF0Size, 0.f);
			if (!CUDAVolume.Empty()) CUDAVolume.Resize(PaddedF0Size, 0.f);
			for (auto iSpeaker : CUDASpeaker)
			{
				if (!iSpeaker.Empty())
					iSpeaker.Resize(PaddedF0Size, 0.f);
			}
			_Inference_Params.AudioSize = WavPaddedSize;
			InputTensors = Preprocessor->Extract(srcHiddenUnits, CUDAF0, CUDAVolume, CUDASpeaker, _Inference_Params);
		}
		else
			InputTensors = Preprocessor->Extract(srcHiddenUnits, _Slice.F0, _Slice.Volume, _Slice.Speaker, _Inference_Params);
				
		OrtTensors EncoderOut;
		try {
			EncoderOut = PreEncoder->Run(Ort::RunOptions{ nullptr },
				InputTensors.InputNames,
				InputTensors.Tensor.data(),
				std::min(InputTensors.Tensor.size(), PreEncoder->GetInputCount()),
				OutputNamesEncoder.data(),
				PreEncoder->GetOutputCount());
		}
		catch (Ort::Exception& e1)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: encoder\n") + e1.what()));
		}

		OrtTensors SamplerInTensors;
		DragonianLibSTL::Vector<float> initial_noise(MelBins * InputTensors.Data.FrameShape[1], 0.0);
		long long noise_shape[4] = { 1,1,MelBins,InputTensors.Data.FrameShape[1] };
		const auto x_size = EncoderOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
		float t_end = std::max(std::min(1.f, _Params.TEnd), 0.002f);
		float t_start = std::max(std::min(_Params.TBegin, t_end), 0.f);
		if (x_size != 1)
		{
			auto x_it = EncoderOut[0].GetTensorMutableData<float>();
			auto x_end = EncoderOut[0].GetTensorMutableData<float>() + x_size;
			while (x_it != x_end) { (*(x_it++) *= t_start) += ((t_end - t_start) * normal(gen) * _Params.NoiseScale); }
			SamplerInTensors.emplace_back(std::move(EncoderOut[0]));
		}
		else
		{
			for (auto& it : initial_noise)
				it = normal(gen) * _Params.NoiseScale;
			SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, initial_noise.Data(), initial_noise.Size(), noise_shape, 4));
		}
		int64_t Time[] = { int64_t(t_start * Scale) };
		int64_t OneShape[] = { 1 };
		SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, Time, 1, OneShape, 1));
		SamplerInTensors.emplace_back(std::move(EncoderOut[1]));
		const auto dt = (t_end - t_start) / float(step);

		OrtTensors PredOut;

		if (abs(t_end - t_start) > std::numeric_limits<float>::epsilon())
			PredOut = Sampler::GetReflowSampler(_Params.ReflowSampler, VelocityFunction.get(), MelBins, ProgressCallbackFunction, MemoryInfo)->Sample(SamplerInTensors, step, dt, Scale, _Process);
		else
		{
			PredOut.emplace_back(std::move(SamplerInTensors[0]));
			ProgressCallbackFunction(_Process += SingleStepSkip, 1);
		}

		try
		{
			ReflowOut = PostDecoder->Run(Ort::RunOptions{ nullptr },
				afterInput.data(),
				PredOut.data(),
				afterInput.size(),
				afterOutput.data(),
				afterOutput.size());
		}
		catch (Ort::Exception& e1)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: pred\n") + e1.what()));
		}

		ReflowOut.emplace_back(std::move(EncoderOut[2]));

		try
		{
			finaOut = _Params.VocoderModel->Run(Ort::RunOptions{ nullptr },
				nsfInput.data(),
				ReflowOut.data(),
				_Params.VocoderModel->GetInputCount(),
				nsfOutput.data(),
				nsfOutput.size());
		}
		catch (Ort::Exception& e3)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Nsf\n") + e3.what()));
		}

		auto DiffOutputAudioSize = finaOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
		auto DiffOutputAudioData = finaOut[0].GetTensorData<float>();
		DragonianLibSTL::Vector DiffPCMOutput(DiffOutputAudioData, DiffOutputAudioData + DiffOutputAudioSize);

		const auto dstWavLen = (_Slice.OrgLen * int64_t(ModelSamplingRate)) / (int)(_Slice.SamplingRate);
		DiffPCMOutput.Resize(dstWavLen, 0.f);
		return DiffPCMOutput;
	}
	ProgressCallbackFunction(_Process += SingleStepSkip, 1);
	const auto len = size_t(_Slice.OrgLen * int64_t(ModelSamplingRate) / (int)(_Slice.SamplingRate));
	return { len, 0.f, TemplateLibrary::CPUAllocator() };
}

DragonianLibSTL::Vector<float> ReflowSvc::InferPCMData(
	const DragonianLibSTL::Vector<float>& _PCMData,
	long _SrcSamplingRate,
	const InferenceParams& _Params
) const
{
	if (!_Params.VocoderModel)
		_D_Dragonian_Lib_Throw_Exception("Missing Vocoder Model!");
	auto step = (int64_t)_Params.Step;
	if (step > MaxStep) step = MaxStep;
	auto hubertin = InterpResample<float>(_PCMData, _SrcSamplingRate, 16000);
	int64_t SpeakerIdx = _Params.SpeakerId;
	if (SpeakerIdx >= SpeakerCount)
		SpeakerIdx = SpeakerCount;
	if (SpeakerIdx < 0)
		SpeakerIdx = 0;
	std::mt19937 gen(int(_Params.Seed));
	std::normal_distribution<float> normal(0, 1);

	const int64_t inputShape[3] = { 1i64,1i64,(int64_t)hubertin.Size() };
	OrtTensors inputTensorshu;
	inputTensorshu.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, hubertin.Data(), hubertin.Size(), inputShape, 3));
	OrtTensors hubertOut;

	const auto RealSteps = step;
	ProgressCallbackFunction(0, RealSteps);

	try {
		hubertOut = HubertModel->Run(Ort::RunOptions{ nullptr },
			hubertInput.data(),
			inputTensorshu.data(),
			inputTensorshu.size(),
			hubertOutput.data(),
			hubertOutput.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: hubert\n") + e.what()));
	}
	const auto HubertSize = hubertOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto HubertOutPutData = hubertOut[0].GetTensorMutableData<float>();
	const auto HubertOutPutShape = hubertOut[0].GetTensorTypeAndShapeInfo().GetShape();
	inputTensorshu.clear();
	if (HubertOutPutShape[2] != HiddenUnitKDims)
		_D_Dragonian_Lib_Throw_Exception("HiddenUnitKDims UnMatch");

	DragonianLibSTL::Vector HiddenUnits(HubertOutPutData, HubertOutPutData + HubertSize);

	if (EnableCluster && _Params.ClusterRate > 0.001f)
	{
		const auto clus_size = HubertOutPutShape[1];
		const auto pts = Cluster->Search(HiddenUnits.Data(), long(SpeakerIdx), clus_size);
		for (size_t indexs = 0; indexs < HiddenUnits.Size(); ++indexs)
			HiddenUnits[indexs] = HiddenUnits[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
	}

	const auto HubertLen = int64_t(HubertSize) / HiddenUnitKDims;
	const int64_t F0Shape[] = { 1, int64_t(_PCMData.Size() / HopSize) };
	const int64_t HiddenUnitShape[] = { 1, HubertLen, HiddenUnitKDims };
	constexpr int64_t CharaEmbShape[] = { 1 };
	const int64_t CharaMixShape[] = { F0Shape[1], SpeakerCount };

	const auto F0Extractor = F0Extractor::GetF0Extractor(
		_Params.F0Method,
		_Params.F0ExtractorUserParameter
	);

	auto F0Data = F0Extractor->ExtractF0(
		_PCMData,
		{
			_SrcSamplingRate,
			HopSize * _SrcSamplingRate / ModelSamplingRate,
			_Params.F0Bins,
			_Params.F0Max,
			_Params.F0Min,
			_Params.F0ExtractorUserParameter
		}
	);
	for (auto& ifo : F0Data)
		ifo *= (float)pow(2.0, static_cast<double>(_Params.Keys) / 12.0);
	F0Data = Preprocessor->GetInterpedF0(InterpFunc(F0Data, long(F0Data.Size()), long(F0Shape[1])));
	DragonianLibSTL::Vector<int64_t> Alignment = Preprocessor->GetAligments(F0Shape[1], HubertLen);
	int64_t CharaEmb[] = { SpeakerIdx };

	OrtTensors EncoderTensors;

	EncoderTensors.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo,
		HiddenUnits.Data(),
		HubertSize,
		HiddenUnitShape,
		3
	));

	EncoderTensors.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo,
		Alignment.Data(),
		F0Shape[1],
		F0Shape,
		2
	));

	EncoderTensors.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo,
		F0Data.Data(),
		F0Shape[1],
		F0Shape,
		2
	));

	std::vector<const char*> InputNamesEncoder;
	DragonianLibSTL::Vector<float> Volume, SpkMap;

	if (EnableVolume)
	{
		InputNamesEncoder = { "hubert", "mel2ph", "f0", "volume", "spk_mix" };
		Volume = ExtractVolume(_PCMData, HopSize);
		if (abs(int64_t(Volume.Size()) - int64_t(F0Data.Size())) > 3)
			Volume = InterpFunc(ExtractVolume(_PCMData, HopSize), long(Volume.Size()), long(F0Shape[1]));
		else
			Volume.Resize(F0Data.Size(), 0.f);
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			Volume.Data(),
			F0Shape[1],
			F0Shape,
			2
		));
	}
	else
		InputNamesEncoder = { "hubert", "mel2ph", "f0", "spk_mix" };

	if (EnableCharaMix)
	{
		SpkMap = Preprocessor->GetCurrectSpkMixData(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>(), F0Shape[1], SpeakerIdx);
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			SpkMap.Data(),
			SpkMap.Size(),
			CharaMixShape,
			2
		));
	}
	else
	{
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			CharaEmb,
			1,
			CharaEmbShape,
			1
		));
	}

	OrtTensors EncoderOut;
	try {
		EncoderOut = PreEncoder->Run(Ort::RunOptions{ nullptr },
			InputNamesEncoder.data(),
			EncoderTensors.data(),
			std::min(EncoderTensors.size(), PreEncoder->GetInputCount()),
			OutputNamesEncoder.data(),
			PreEncoder->GetOutputCount());
	}
	catch (Ort::Exception& e1)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: encoder\n") + e1.what()));
	}

	size_t _Process = 0;
	OrtTensors SamplerInTensors;
	DragonianLibSTL::Vector<float> initial_noise(MelBins * F0Shape[1], 0.0);
	long long noise_shape[4] = { 1,1,MelBins,F0Shape[1] };
	const auto x_size = EncoderOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
	float t_end = std::max(std::min(1.f, _Params.TEnd), 0.002f);
	float t_start = std::max(std::min(_Params.TBegin, t_end - 0.001f), 0.f);
	if (x_size != 1)
	{
		auto x_it = EncoderOut[0].GetTensorMutableData<float>();
		auto x_end = EncoderOut[0].GetTensorMutableData<float>() + x_size;
		while (x_it != x_end) { (*(x_it++) *= t_start) += ((t_end - t_start) * normal(gen) * _Params.NoiseScale); }
		SamplerInTensors.emplace_back(std::move(EncoderOut[0]));
	}
	else
	{
		for (auto& it : initial_noise)
			it = normal(gen) * _Params.NoiseScale;
		SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, initial_noise.Data(), initial_noise.Size(), noise_shape, 4));
	}
	float Time[] = { t_start };
	int64_t OneShape[] = { 1 };
	SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, Time, 1, OneShape, 1));
	SamplerInTensors.emplace_back(std::move(EncoderOut[1]));
	const auto dt = (t_end - t_start) / float(step);

	OrtTensors PredOut;

	if (abs(t_end - t_start) > std::numeric_limits<float>::epsilon())
		PredOut = Sampler::GetReflowSampler(_Params.ReflowSampler, VelocityFunction.get(), MelBins, ProgressCallbackFunction, MemoryInfo)->Sample(SamplerInTensors, step, dt, Scale, _Process);
	else
	{
		PredOut.emplace_back(std::move(SamplerInTensors[0]));
		ProgressCallbackFunction(_Process += RealSteps, 1);
	}

	OrtTensors ReflowOut, finaOut;

	try
	{
		ReflowOut = PostDecoder->Run(Ort::RunOptions{ nullptr },
			afterInput.data(),
			PredOut.data(),
			afterInput.size(),
			afterOutput.data(),
			afterOutput.size());
	}
	catch (Ort::Exception& e1)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: pred\n") + e1.what()));
	}

	ReflowOut.emplace_back(std::move(EncoderOut[2]));
	try
	{
		finaOut = _Params.VocoderModel->Run(Ort::RunOptions{ nullptr },
			nsfInput.data(),
			ReflowOut.data(),
			_Params.VocoderModel->GetInputCount(),
			nsfOutput.data(),
			nsfOutput.size());
	}
	catch (Ort::Exception& e3)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Nsf\n") + e3.what()));
	}

	auto DiffOutputAudioSize = finaOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
	auto DiffOutputAudioData = finaOut[0].GetTensorData<float>();
	DragonianLibSTL::Vector DiffPCMOutput(DiffOutputAudioData, DiffOutputAudioData + DiffOutputAudioSize);

	const auto dstWavLen = (_PCMData.Size() * int64_t(ModelSamplingRate)) / _SrcSamplingRate;
	DiffPCMOutput.Resize(dstWavLen, 0.f);

	UNUSED(Volume.Size());
	UNUSED(SpkMap.Size());
	return DiffPCMOutput;
}

void ReflowSvc::NormMel(DragonianLibSTL::Vector<float>& MelSpec) const
{
	for (auto& it : MelSpec)
		it = (it - SpecMin) / (SpecMax - SpecMin) * 2 - 1;
}

DragonianLibSTL::Vector<float> ReflowSvc::ShallowDiffusionInference(
	DragonianLibSTL::Vector<float>& _16KAudioHubert,
	const InferenceParams& _Params,
	std::pair<DragonianLibSTL::Vector<float>, int64_t>& _Mel,
	const DragonianLibSTL::Vector<float>& _SrcF0,
	const DragonianLibSTL::Vector<float>& _SrcVolume,
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
	size_t& Process,
	int64_t SrcSize
) const
{
	if (!_Params.VocoderModel)
		_D_Dragonian_Lib_Throw_Exception("Missing Vocoder Model!");
	std::mt19937 gen(int(_Params.Seed));
	std::normal_distribution<float> normal(0, 1);
	auto step = (int64_t)_Params.Step;
	if (step > MaxStep) step = MaxStep;
	std::vector<const char*> InputNamesEncoder;
	const auto _Mel_Size = _Mel.second;

	OrtTensors HubertInputTensors, HubertOutputTensors;
	const int64_t HubertInputShape[3] = { 1i64,1i64,(int64_t)_16KAudioHubert.Size() };
	HubertInputTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, _16KAudioHubert.Data(), _16KAudioHubert.Size(), HubertInputShape, 3));
	try {
		HubertOutputTensors = HubertModel->Run(Ort::RunOptions{ nullptr },
			hubertInput.data(),
			HubertInputTensors.data(),
			HubertInputTensors.size(),
			hubertOutput.data(),
			hubertOutput.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: hubert\n") + e.what()));
	}

	int64_t SpeakerIdx = _Params.SpeakerId;
	if (SpeakerIdx >= SpeakerCount)
		SpeakerIdx = SpeakerCount;
	if (SpeakerIdx < 0)
		SpeakerIdx = 0;

	const auto HubertLength = HubertOutputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];
	const int64_t FrameShape[] = { 1, _Mel_Size };
	const int64_t CharaMixShape[] = { _Mel_Size, SpeakerCount };
	constexpr int64_t OneShape[] = { 1 };
	int64_t CharaEmb[] = { SpeakerIdx };

	auto Alignment = Preprocessor->GetAligments(_Mel_Size, HubertLength);
	Alignment.Resize(FrameShape[1]);
	auto F0Data = InterpFunc(_SrcF0, long(_SrcF0.Size()), long(FrameShape[1]));
	DragonianLibSTL::Vector<float> Volume, SpkMap;

	OrtTensors EncoderTensors;
	EncoderTensors.emplace_back(std::move(HubertOutputTensors[0]));
	EncoderTensors.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo,
		Alignment.Data(),
		FrameShape[1],
		FrameShape,
		2
	));

	EncoderTensors.emplace_back(Ort::Value::CreateTensor(
		*MemoryInfo,
		F0Data.Data(),
		FrameShape[1],
		FrameShape,
		2
	));

	if (EnableVolume)
	{
		InputNamesEncoder = { "hubert", "mel2ph", "f0", "volume", "spk_mix" };
		Volume = InterpFunc(_SrcVolume, long(_SrcVolume.Size()), long(FrameShape[1]));
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			Volume.Data(),
			FrameShape[1],
			FrameShape,
			2
		));
	}
	else
		InputNamesEncoder = { "hubert", "mel2ph", "f0", "spk_mix" };

	if (EnableCharaMix)
	{
		SpkMap = Preprocessor->GetCurrectSpkMixData(_SrcSpeakerMap, FrameShape[1], CharaEmb[0]);
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			SpkMap.Data(),
			SpkMap.Size(),
			CharaMixShape,
			2
		));
	}
	else
	{
		EncoderTensors.emplace_back(Ort::Value::CreateTensor(
			*MemoryInfo,
			CharaEmb,
			1,
			OneShape,
			1
		));
	}

	OrtTensors EncoderOut;
	try {
		EncoderOut = PreEncoder->Run(Ort::RunOptions{ nullptr },
			InputNamesEncoder.data(),
			EncoderTensors.data(),
			std::min(EncoderTensors.size(), PreEncoder->GetInputCount()),
			OutputNamesEncoder.data(),
			PreEncoder->GetOutputCount());
	}
	catch (Ort::Exception& e1)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: encoder\n") + e1.what()));
	}

	NormMel(_Mel.first);
	long long noise_shape[4] = { 1,1,MelBins,_Mel_Size };
	OrtTensors SamplerInTensors;
	float t_end = std::max(std::min(1.f, _Params.TEnd), 0.002f);
	float t_start = std::max(std::min(_Params.TBegin, t_end - 0.001f), 0.f);
	SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, _Mel.first.Data(), _Mel.first.Size(), noise_shape, 4));
	auto x_it = SamplerInTensors[0].GetTensorMutableData<float>();
	auto x_end = SamplerInTensors[0].GetTensorMutableData<float>() + _Mel.first.Size();
	while (x_it != x_end) { (*(x_it++) *= t_start) += ((t_end - t_start) * normal(gen) * _Params.NoiseScale); }
	float Time[] = { t_start };
	SamplerInTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, Time, 1, OneShape, 1));
	SamplerInTensors.emplace_back(std::move(EncoderOut[1]));
	const auto dt = (t_end - t_start) / float(step);

	OrtTensors PredOut;

	if (abs(t_end - t_start) > std::numeric_limits<float>::epsilon())
		PredOut = Sampler::GetReflowSampler(_Params.ReflowSampler, VelocityFunction.get(), MelBins, ProgressCallbackFunction, MemoryInfo)->Sample(SamplerInTensors, step, dt, Scale, Process);
	else
		PredOut.emplace_back(std::move(SamplerInTensors[0]));

	OrtTensors ReflowOut, finaOut;
	try
	{
		ReflowOut = PostDecoder->Run(Ort::RunOptions{ nullptr },
			afterInput.data(),
			PredOut.data(),
			afterInput.size(),
			afterOutput.data(),
			afterOutput.size());
	}
	catch (Ort::Exception& e1)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: pred\n") + e1.what()));
	}

	ReflowOut.emplace_back(std::move(EncoderTensors[2]));
	try
	{
		finaOut = _Params.VocoderModel->Run(Ort::RunOptions{ nullptr },
			nsfInput.data(),
			ReflowOut.data(),
			_Params.VocoderModel->GetInputCount(),
			nsfOutput.data(),
			nsfOutput.size());
	}
	catch (Ort::Exception& e3)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: Nsf\n") + e3.what()));
	}

	auto DiffOutputAudioSize = finaOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
	auto DiffOutputAudioData = finaOut[0].GetTensorData<float>();
	DragonianLibSTL::Vector DiffPCMOutput(DiffOutputAudioData, DiffOutputAudioData + DiffOutputAudioSize);

	UNUSED(Volume.Size());
	UNUSED(SpkMap.Size());
	return DiffPCMOutput;
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
