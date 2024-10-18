#include "../../header/Models/VitsSvc.hpp"
#include "../../header/Modules.hpp"
#include "F0Extractor/F0ExtractorManager.hpp"
#include "Util/Logger.h"
#include "Base.h"
#include <random>
#include <regex>

#include "Util/StringPreprocess.h"

LibSvcHeader
	VitsSvc::~VitsSvc()
{
	LogInfo(L"Unloading VitsSvc Models");
}

VitsSvc::VitsSvc(
	const Hparams& _Hps,
	const ProgressCallback& _ProgressCallback,
	ExecutionProviders ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) :
	SingingVoiceConversion(_Hps.HubertPath, ExecutionProvider_, DeviceID_, ThreadCount_)
{

	ModelSamplingRate = std::max(_Hps.SamplingRate, 2000l);
	HopSize = std::max(_Hps.HopSize, 1);
	HiddenUnitKDims = std::max(_Hps.HiddenUnitKDims, 1ll);
	SpeakerCount = std::max(_Hps.SpeakerCount, 1ll);
	EnableVolume = _Hps.EnableVolume;
	EnableCharaMix = _Hps.EnableCharaMix;
	VitsSvcVersion = _Hps.TensorExtractor;

	ProgressCallbackFunction = _ProgressCallback;

	if (!_Hps.Cluster.Type.empty())
	{
		ClusterCenterSize = _Hps.Cluster.ClusterCenterSize;
		try
		{
			Cluster = GetCluster(_Hps.Cluster.Type, _Hps.Cluster.Path, HiddenUnitKDims, ClusterCenterSize);
			EnableCluster = true;
		}
		catch (std::exception& e)
		{
			LogWarn(UTF8ToWideString(e.what()));
			EnableCluster = false;
		}
	}

	try
	{
		LogInfo(L"[Info] loading VitsSvcModel Models");
		VitsSvcModel = std::make_shared<Ort::Session>(*OnnxEnv, _Hps.VitsSvc.VitsSvc.c_str(), *SessionOptions);
		LogInfo(L"[Info] VitsSvcModel Models loaded");
	}
	catch (Ort::Exception& _exception)
	{
		DragonianLibThrow(_exception.what());
	}

	if (VitsSvcModel->GetInputCount() == 4 && VitsSvcVersion != L"SoVits3.0")
		VitsSvcVersion = L"SoVits2.0";

	LibSvcTensorExtractor::Others _others_param;
	_others_param.Memory = *MemoryInfo;
	try
	{
		Preprocessor = GetTensorExtractor(VitsSvcVersion, 48000, ModelSamplingRate, HopSize, EnableCharaMix, EnableVolume, HiddenUnitKDims, SpeakerCount, _others_param);
	}
	catch (std::exception& e)
	{
		DragonianLibThrow(e.what());
	}
}

DragonianLibSTL::Vector<float> VitsSvc::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params,
	size_t& _Process
) const
{
	if (_Slice.IsNotMute)
	{
		DragonianLibSTL::Vector<float> _16KAudio;

		_16KAudio = InterpResample<float>(_Slice.Audio, (int)_Slice.SamplingRate, 16000);
		const auto src_audio_length = _16KAudio.Size();
		bool NeedPadding = false;
#ifdef LIBSVC_CUDA_ONLY_PADDING
		if (ModelExecutionProvider == ExecutionProviders::CUDA)
#endif
		{
			NeedPadding = _16KAudio.Size() % DRAGONIANLIB_PADDING_COUNT;
			const size_t WavPaddedSize = _16KAudio.Size() / DRAGONIANLIB_PADDING_COUNT + 1;
			if (NeedPadding)
				_16KAudio.Resize(WavPaddedSize * DRAGONIANLIB_PADDING_COUNT, 0.f);
		}

		const int64_t HubertInputShape[3] = { 1i64,1i64,(int64_t)_16KAudio.Size() };
		OrtTensors HubertInputTensors, HubertOutPuts;
		HubertInputTensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, _16KAudio.Data(), _16KAudio.Size(), HubertInputShape, 3));
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
			DragonianLibThrow((std::string("Locate: hubert\n") + e.what()));
		}
		const auto HubertSize = HubertOutPuts[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto HubertOutPutData = HubertOutPuts[0].GetTensorMutableData<float>();
		auto HubertOutPutShape = HubertOutPuts[0].GetTensorTypeAndShapeInfo().GetShape();
		if (HubertOutPutShape[2] != HiddenUnitKDims)
			DragonianLibThrow("HiddenUnitKDims UnMatch");

		DragonianLibSTL::Vector SrcHiddenUnits(HubertOutPutData, HubertOutPutData + HubertSize);

		int64_t SpeakerIdx = _Params.SpeakerId;
		if (SpeakerIdx >= SpeakerCount)
			SpeakerIdx = SpeakerCount;
		if (SpeakerIdx < 0)
			SpeakerIdx = 0;

		const auto max_cluster_size = int64_t((size_t)HubertOutPutShape[1] * src_audio_length / _16KAudio.Size());
		if (EnableCluster && _Params.ClusterRate > 0.001f)
		{
			const auto pts = Cluster->Search(SrcHiddenUnits.Data(), long(SpeakerIdx), max_cluster_size);
			for (int64_t indexs = 0; indexs < max_cluster_size * HiddenUnitKDims; ++indexs)
				SrcHiddenUnits[indexs] = SrcHiddenUnits[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
		}

		LibSvcTensorExtractor::InferParams _Inference_Params;
		_Inference_Params.AudioSize = _Slice.Audio.Size();
		_Inference_Params.Chara = SpeakerIdx;
		_Inference_Params.NoiseScale = _Params.NoiseScale;
		_Inference_Params.DDSPNoiseScale = _Params.DDSPNoiseScale;
		_Inference_Params.Seed = int(_Params.Seed);
		_Inference_Params.upKeys = _Params.Keys;
		_Inference_Params.SrcSamplingRate = _Slice.SamplingRate;

		LibSvcTensorExtractor::Inputs InputTensors;

		if (NeedPadding)
		{
			auto PaddedF0 = _Slice.F0;
			auto PaddedVolume = _Slice.Volume;
			auto PaddedSpeaker = _Slice.Speaker;
			const auto ScaleSamplingConut = _Slice.SamplingRate * DRAGONIANLIB_PADDING_COUNT / 16000;
			const auto SrcAudioLength = _Slice.Audio.Size();
			const size_t WavPaddedSize = (SrcAudioLength / ScaleSamplingConut + 1) * ScaleSamplingConut;
			const size_t AudioPadSize = WavPaddedSize - SrcAudioLength;
			const size_t PaddedF0Size = PaddedF0.Size() + (PaddedF0.Size() * AudioPadSize / SrcAudioLength);

			if (!PaddedF0.Empty()) PaddedF0.Resize(PaddedF0Size, 0.f);
			if (!PaddedVolume.Empty()) PaddedVolume.Resize(PaddedF0Size, 0.f);
			for (auto iSpeaker : PaddedSpeaker)
			{
				if (!iSpeaker.Empty())
					iSpeaker.Resize(PaddedF0Size, 0.f);
			}
			_Inference_Params.AudioSize = WavPaddedSize;
			InputTensors = Preprocessor->Extract(SrcHiddenUnits, PaddedF0, PaddedVolume, PaddedSpeaker, _Inference_Params);
			_Params.CacheData(std::move(_16KAudio), std::move(PaddedF0), std::move(PaddedVolume), std::move(PaddedSpeaker));
		}
		else
			InputTensors = Preprocessor->Extract(SrcHiddenUnits, _Slice.F0, _Slice.Volume, _Slice.Speaker, _Inference_Params);

		OrtTensors finaOut;
		try
		{
			finaOut = VitsSvcModel->Run(Ort::RunOptions{ nullptr },
				InputTensors.InputNames,
				InputTensors.Tensor.data(),
				InputTensors.Tensor.size(),
				soVitsOutput.data(),
				soVitsOutput.size());
		}
		catch (Ort::Exception& e)
		{
			DragonianLibThrow((std::string("Locate: VitsSvc\n") + e.what()));
		}

		auto VitsOutputAudioSize = finaOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
		auto VitsOutputAudioData = finaOut[0].GetTensorData<float>();
		DragonianLibSTL::Vector VitsOutput(VitsOutputAudioData, VitsOutputAudioData + VitsOutputAudioSize);

		/*if (shallow_diffusion && stft_operator && _Params.UseShallowDiffusion)
		{
			auto PCMAudioBegin = finaOut[0].GetTensorData<float>();
			auto PCMAudioEnd = PCMAudioBegin + finaOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
			auto MelSpec = MelExtractor(PCMAudioBegin, PCMAudioEnd);
			auto ShallowParam = _Params;
			ShallowParam.SrcSamplingRate = _samplingRate;
			auto ShallowDiffusionOutput = shallow_diffusion->ShallowDiffusionInference(
				RawWav,
				ShallowParam,
				std::move(MelSpec[0]),
				NeedPadding ? CUDAF0 : _Slice.F0,
				NeedPadding ? CUDAVolume : _Slice.Volume,
				NeedPadding ? CUDASpeaker : _Slice.Speaker
			);
			ShallowDiffusionOutput.resize(dstWavLen, 0);
			return ShallowDiffusionOutput;
		}*/

		const auto dstWavLen = (_Slice.OrgLen * int64_t(ModelSamplingRate)) / _Slice.SamplingRate;
		VitsOutput.Resize(dstWavLen, 0.f);
		return VitsOutput;
	}
	//Mute clips
	const auto len = size_t(_Slice.OrgLen * int64_t(ModelSamplingRate) / _Slice.SamplingRate);
	return { len, 0.f, GetMemoryProvider(DragonianLib::Device::CPU) };
}

DragonianLibSTL::Vector<float> VitsSvc::InferPCMData(
	const DragonianLibSTL::Vector<float>& _PCMData,
	long _SrcSamplingRate,
	const InferenceParams& _Params
) const
{
	auto hubertin = InterpResample<float>(_PCMData, _SrcSamplingRate, 16000);
	int64_t SpeakerIdx = _Params.SpeakerId;
	if (SpeakerIdx >= SpeakerCount)
		SpeakerIdx = SpeakerCount;
	if (SpeakerIdx < 0)
		SpeakerIdx = 0;

	std::mt19937 gen(int(_Params.Seed));
	std::normal_distribution<float> normal(0, 1);
	float noise_scale = _Params.NoiseScale;
	float ddsp_noise_scale = _Params.DDSPNoiseScale;

	const int64_t inputShape[3] = { 1i64,1i64,(int64_t)hubertin.Size() };
	OrtTensors inputTensorshu;
	inputTensorshu.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, hubertin.Data(), hubertin.Size(), inputShape, 3));
	OrtTensors hubertOut;

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
		DragonianLibThrow((std::string("Locate: hubert\n") + e.what()));
	}
	auto HubertSize = hubertOut[0].GetTensorTypeAndShapeInfo().GetElementCount();
	auto HubertOutPutData = hubertOut[0].GetTensorMutableData<float>();
	auto HubertOutPutShape = hubertOut[0].GetTensorTypeAndShapeInfo().GetShape();
	inputTensorshu.clear();
	if (HubertOutPutShape[2] != HiddenUnitKDims)
		DragonianLibThrow("HiddenUnitKDims UnMatch");

	DragonianLibSTL::Vector HiddenUnitsSrc(HubertOutPutData, HubertOutPutData + HubertSize);

	if (EnableCluster && _Params.ClusterRate > 0.001f)
	{
		const auto clus_size = HubertOutPutShape[1];
		const auto pts = Cluster->Search(HiddenUnitsSrc.Data(), long(SpeakerIdx), clus_size);
		for (size_t indexs = 0; indexs < HiddenUnitsSrc.Size(); ++indexs)
			HiddenUnitsSrc[indexs] = HiddenUnitsSrc[indexs] * (1.f - _Params.ClusterRate) + pts[indexs] * _Params.ClusterRate;
	}

	const auto HubertLen = int64_t(HubertSize) / HiddenUnitKDims;
	int64_t F0Shape[] = { 1, int64_t(_PCMData.Size() / HopSize) };
	int64_t HiddenUnitShape[] = { 1, HubertLen, HiddenUnitKDims };
	constexpr int64_t LengthShape[] = { 1 };
	int64_t CharaEmbShape[] = { 1 };
	int64_t CharaMixShape[] = { F0Shape[1], SpeakerCount };
	int64_t RandnShape[] = { 1, 192, F0Shape[1] };
	const int64_t IstftShape[] = { 1, 2048, F0Shape[1] };
	int64_t RandnCount = F0Shape[1] * 192;
	const int64_t IstftCount = F0Shape[1] * 2048;

	DragonianLibSTL::Vector<float> RandnInput, IstftInput, UV, InterpedF0;
	DragonianLibSTL::Vector<int64_t> alignment;
	int64_t XLength[1] = { HubertLen };
	DragonianLibSTL::Vector<int64_t> Nsff0;
	int64_t Chara[] = { SpeakerIdx };
	DragonianLibSTL::Vector<float> charaMix;

	const auto F0Extractor = DragonianLib::GetF0Extractor(_Params.F0Method, ModelSamplingRate, HopSize);

	auto srcF0Data = F0Extractor->ExtractF0(_PCMData, _PCMData.Size() / HopSize);
	for (auto& ifo : srcF0Data)
		ifo *= (float)pow(2.0, static_cast<double>(_Params.Keys) / 12.0);
	DragonianLibSTL::Vector<float> HiddenUnits;
	DragonianLibSTL::Vector<float> F0Data;

	OrtTensors _Tensors;
	std::vector<const char*> SoVitsInput = soVitsInput;

	//Compatible with all versions
	if (VitsSvcVersion == L"SoVits3.0")
	{
		int64_t upSample = ModelSamplingRate / 16000;
		HiddenUnits.Reserve(HubertSize * (upSample + 1));
		for (int64_t itS = 0; itS < HiddenUnitShape[1]; ++itS)
			for (int64_t itSS = 0; itSS < upSample; ++itSS)
				HiddenUnits.Insert(HiddenUnits.end(), HiddenUnitsSrc.begin() + itS * HiddenUnitKDims, HiddenUnitsSrc.begin() + (itS + 1) * HiddenUnitKDims);
		HiddenUnitShape[1] *= upSample;
		HubertSize *= upSample;
		F0Data = Preprocessor->GetInterpedF0(DragonianLibSTL::InterpFunc(srcF0Data, long(srcF0Data.Size()), long(HiddenUnitShape[1])));
		F0Shape[1] = HiddenUnitShape[1];
		XLength[0] = HiddenUnitShape[1];
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, HiddenUnits.Data(), HubertSize, HiddenUnitShape, 3));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, XLength, 1, LengthShape, 1));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, F0Data.Data(), F0Data.Size(), F0Shape, 2));
	}
	else if (VitsSvcVersion == L"SoVits2.0")
	{
		HiddenUnits = std::move(HiddenUnitsSrc);
		F0Shape[1] = HiddenUnitShape[1];
		F0Data = InterpFunc(srcF0Data, long(srcF0Data.Size()), long(HiddenUnitShape[1]));
		XLength[0] = HiddenUnitShape[1];
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, HiddenUnits.Data(), HubertSize, HiddenUnitShape, 3));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, XLength, 1, LengthShape, 1));
		Nsff0 = Preprocessor->GetNSFF0(F0Data);
		_Tensors.emplace_back(Ort::Value::CreateTensor<long long>(*MemoryInfo, Nsff0.Data(), Nsff0.Size(), F0Shape, 2));
	}
	else if (VitsSvcVersion == L"RVC")
	{
		constexpr int64_t upSample = 2;
		HiddenUnits.Reserve(HubertSize * (upSample + 1));
		for (int64_t itS = 0; itS < HiddenUnitShape[1]; ++itS)
			for (int64_t itSS = 0; itSS < upSample; ++itSS)
				HiddenUnits.Insert(HiddenUnits.end(), HiddenUnitsSrc.begin() + itS * HiddenUnitKDims, HiddenUnitsSrc.begin() + (itS + 1) * HiddenUnitKDims);
		HiddenUnitShape[1] *= upSample;
		HubertSize *= upSample;
		F0Data = DragonianLibSTL::InterpFunc(srcF0Data, long(srcF0Data.Size()), long(HiddenUnitShape[1]));
		F0Shape[1] = HiddenUnitShape[1];
		XLength[0] = HiddenUnitShape[1];
		RandnCount = 192 * F0Shape[1];
		RandnShape[2] = F0Shape[1];
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, HiddenUnits.Data(), HubertSize, HiddenUnitShape, 3));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, XLength, 1, LengthShape, 1));
		InterpedF0 = Preprocessor->GetInterpedF0(F0Data);
		Nsff0 = Preprocessor->GetNSFF0(InterpedF0);
		_Tensors.emplace_back(Ort::Value::CreateTensor<long long>(*MemoryInfo, Nsff0.Data(), Nsff0.Size(), F0Shape, 2));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, InterpedF0.Data(), InterpedF0.Size(), F0Shape, 2));
		SoVitsInput = RVCInput;
		RandnInput = DragonianLibSTL::Vector(RandnCount, 0.f);
		for (auto& it : RandnInput)
			it = normal(gen) * noise_scale;
	}
	else
	{
		HiddenUnits = std::move(HiddenUnitsSrc);
		F0Data = DragonianLibSTL::InterpFunc(srcF0Data, long(srcF0Data.Size()), long(F0Shape[1]));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, HiddenUnits.Data(), HubertSize, HiddenUnitShape, 3));
		InterpedF0 = Preprocessor->GetInterpedF0(F0Data);
		alignment = Preprocessor->GetAligments(F0Shape[1], HubertLen);
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, InterpedF0.Data(), InterpedF0.Size(), F0Shape, 2));
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, alignment.Data(), InterpedF0.Size(), F0Shape, 2));
		if (VitsSvcVersion != L"SoVits4.0-DDSP")
		{
			UV = Preprocessor->GetUV(F0Data);
			SoVitsInput = { "c", "f0", "mel2ph", "uv", "noise", "sid" };
			_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, UV.Data(), UV.Size(), F0Shape, 2));
		}
		else
		{
			SoVitsInput = { "c", "f0", "mel2ph", "t_window", "noise", "sid" };
			IstftInput = DragonianLibSTL::Vector(IstftCount, ddsp_noise_scale);
			_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, IstftInput.Data(), IstftInput.Size(), IstftShape, 3));
		}
		RandnInput = DragonianLibSTL::Vector(RandnCount, 0.f);
		for (auto& it : RandnInput)
			it = normal(gen) * noise_scale;
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, RandnInput.Data(), RandnCount, RandnShape, 3));
	}


	if (EnableCharaMix)
	{
		CharaMixShape[0] = F0Shape[1];
		DragonianLibSTL::Vector charaMap(SpeakerCount, 0.f);
		charaMap[SpeakerIdx] = 1.f;
		charaMix.Reserve((SpeakerCount + 1) * F0Shape[1]);
		for (int64_t index = 0; index < F0Shape[1]; ++index)
			charaMix.Insert(charaMix.end(), charaMap.begin(), charaMap.end());
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, charaMix.Data(), charaMix.Size(), CharaMixShape, 2));
	}
	else
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, Chara, 1, CharaEmbShape, 1));

	if (VitsSvcVersion == L"RVC")
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, RandnInput.Data(), RandnCount, RandnShape, 3));

	DragonianLibSTL::Vector<float> VolumeData;

	if (EnableVolume)
	{
		SoVitsInput.emplace_back("vol");
		VolumeData = ExtractVolume(_PCMData, HopSize);
		VolumeData.Resize(F0Shape[1], 0.f);
		_Tensors.emplace_back(Ort::Value::CreateTensor(*MemoryInfo, VolumeData.Data(), F0Shape[1], F0Shape, 2));
	}

	OrtTensors finaOut;

	finaOut = VitsSvcModel->Run(Ort::RunOptions{ nullptr },
		SoVitsInput.data(),
		_Tensors.data(),
		_Tensors.size(),
		soVitsOutput.data(),
		soVitsOutput.size());

	const auto OutAudioLength = finaOut[0].GetTensorTypeAndShapeInfo().GetShape()[2];
	auto OutAudioData = finaOut[0].GetTensorData<float>();
	DragonianLibSTL::Vector TempVecWav(OutAudioData, OutAudioData + OutAudioLength);

	const auto dstWavLen = (_PCMData.Size() * int64_t(ModelSamplingRate)) / _SrcSamplingRate;
	TempVecWav.Resize(dstWavLen, 0.f);

	UNUSED(IstftInput.Size());
	UNUSED(VolumeData.Size());
	UNUSED(UV.Size());
	UNUSED(alignment.Size());
	UNUSED(charaMix.Size());
	return TempVecWav;
}

LibSvcEnd