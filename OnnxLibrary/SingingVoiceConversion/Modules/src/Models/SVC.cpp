#include "../../header/Models/SVC.hpp"
#include "Base.h"
#include "F0Extractor/F0ExtractorManager.hpp"

LibSvcHeader

SingingVoiceConversion::SingingVoiceConversion(
	const std::wstring& HubertPath_,
	const ExecutionProviders& ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : LibSvcModule(ExecutionProvider_, DeviceID_, ThreadCount_)
{
	HubertModel = RefOrtCachedModel(HubertPath_, *OrtApiEnv);
}

SingingVoiceConversion::~SingingVoiceConversion() = default;

DragonianLibSTL::Vector<float> SingingVoiceConversion::InferPCMData(
	const DragonianLibSTL::Vector<float>& _PCMData,
	long _SrcSamplingRate,
	const InferenceParams& _Params
) const
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::ShallowDiffusionInference(
	DragonianLibSTL::Vector<float>& _16KAudioHubert,
	const InferenceParams& _Params,
	std::pair<DragonianLibSTL::Vector<float>,
	int64_t>& _Mel,
	const DragonianLibSTL::Vector<float>& _SrcF0,
	const DragonianLibSTL::Vector<float>& _SrcVolume,
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _SrcSpeakerMap,
	size_t& Process,
	int64_t SrcSize
) const
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params,
	size_t& _Process
) const
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::ExtractVolume(
	const DragonianLibSTL::Vector<float>& _Audio
) const
{
	DragonianLibSTL::Vector<float> Audio;
	Audio.Reserve(_Audio.Size() * 2);
	Audio.Insert(Audio.end(), HopSize, _Audio[0]);
	Audio.Insert(Audio.end(), _Audio.begin(), _Audio.end());
	Audio.Insert(Audio.end(), HopSize, _Audio[_Audio.Size() - 1]);
	const size_t n_frames = (_Audio.Size() / HopSize) + 1;
	DragonianLibSTL::Vector<float> volume(n_frames);
	for (auto& i : Audio)
		i = powf(i, 2);
	int64_t index = 0;
	for (auto& i : volume)
	{
		i = sqrt((float)DragonianLibSTL::Average(Audio.begin() + index * HopSize, Audio.begin() + (index + 1) * HopSize));
		++index;
	}
	return volume;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::ExtractVolume(
	const DragonianLibSTL::Vector<float>& _Audio,
	int _HopSize
)
{
	DragonianLibSTL::Vector<float> Audio;
	Audio.Reserve(_Audio.Size() * 2);
	Audio.Insert(Audio.end(), _HopSize, float(_Audio[0]));
	for (const auto i : _Audio)
		Audio.EmplaceBack((float)i);
	Audio.Insert(Audio.end(), _HopSize, float(_Audio[_Audio.Size() - 1]));
	const size_t n_frames = (_Audio.Size() / _HopSize) + 1;
	DragonianLibSTL::Vector<float> volume(n_frames);
	for (auto& i : Audio)
		i = powf(i, 2);
	int64_t index = 0;
	for (auto& i : volume)
	{
		i = sqrt((float)DragonianLibSTL::Average(Audio.begin() + index * _HopSize, Audio.begin() + (index + 1) * _HopSize));
		++index;
	}
	return volume;
}

SingleAudio SingingVoiceConversion::GetAudioSlice(
	const DragonianLibSTL::Vector<float>& _InputPCM,
	const DragonianLibSTL::Vector<size_t>& _SlicePos,
	const SlicerSettings& _SlicerConfig
)
{
	SingleAudio audio_slice;
	for (size_t i = 1; i < _SlicePos.Size(); i++)
	{
		SingleSlice _CurSlice;
		_CurSlice.SamplingRate = _SlicerConfig.SamplingRate;
		const bool is_not_mute = abs(DragonianLibSTL::Average((_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]))) > _SlicerConfig.Threshold;
		_CurSlice.IsNotMute = is_not_mute;
		_CurSlice.OrgLen = _SlicePos[i] - _SlicePos[i - 1];
		if (is_not_mute)
			_CurSlice.Audio = { (_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]) };
		else
			_CurSlice.Audio.Clear();
		audio_slice.Slices.EmplaceBack(std::move(_CurSlice));
	}
	return audio_slice;
}

void SingingVoiceConversion::PreProcessAudio(
	const SingleAudio& _Input,
	int _SamplingRate,
	int _HopSize,
	const std::wstring& _F0Method
)
{
	const auto F0Extractor = GetF0Extractor(_F0Method, _SamplingRate, _HopSize);
	const auto num_slice = _Input.Slices.Size();
	for (size_t i = 0; i < num_slice; ++i)
	{
		if (_Input.Slices[i].IsNotMute)
		{
			_Input.Slices[i].F0 = F0Extractor->ExtractF0(_Input.Slices[i].Audio, _Input.Slices[i].Audio.Size() / _HopSize);
			_Input.Slices[i].Volume = ExtractVolume(_Input.Slices[i].Audio, _HopSize);
		}
		else
		{
			_Input.Slices[i].F0.Clear();
			_Input.Slices[i].Volume.Clear();
		}
		_Input.Slices[i].Speaker.Clear();
	}
}

int SingingVoiceConversion::GetHopSize() const
{
	return HopSize;
}

int64_t SingingVoiceConversion::GetHiddenUnitKDims() const
{
	return HiddenUnitKDims;
}

int64_t SingingVoiceConversion::GetSpeakerCount() const
{
	return SpeakerCount;
}

bool SingingVoiceConversion::SpeakerMixEnabled() const
{
	return EnableCharaMix;
}

const std::wstring& SingingVoiceConversion::GetUnionSvcVer() const
{
	DragonianLibNotImplementedError;
}

int64_t SingingVoiceConversion::GetMaxStep() const
{
	DragonianLibNotImplementedError;
}

int64_t SingingVoiceConversion::GetMelBins() const
{
	DragonianLibNotImplementedError;
}

void SingingVoiceConversion::NormMel(
	DragonianLibSTL::Vector<float>& MelSpec
) const
{
	DragonianLibNotImplementedError;
}

std::unordered_map<std::wstring, std::shared_ptr<Ort::Session>> GlobalOrtModelCache;

std::shared_ptr<Ort::Session>& SingingVoiceConversion::RefOrtCachedModel(
	const std::wstring& Path_,
	const DragonianLibOrtEnv& Env_
)
{
	const auto ID = L"EP:" + std::to_wstring(Env_.GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_.GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_.GetCurThreadCount()) +
		L" PATH:" + Path_;
	auto Iter = GlobalOrtModelCache.find(ID);
	if (Iter != GlobalOrtModelCache.end())
		return Iter->second;
	return GlobalOrtModelCache[ID] = std::make_shared<Ort::Session>(*Env_.GetEnv(), Path_.c_str(), *Env_.GetSessionOptions());
}

void SingingVoiceConversion::UnRefOrtCachedModel(
	const std::wstring& Path_,
	const DragonianLibOrtEnv& Env_
)
{
	const auto ID = L"EP:" + std::to_wstring(Env_.GetCurProvider()) +
		L" DEVICE:" + std::to_wstring(Env_.GetCurDeviceID()) +
		L" THREAD:" + std::to_wstring(Env_.GetCurThreadCount()) +
		L" PATH:" + Path_;
	auto Iter = GlobalOrtModelCache.find(ID);
	if (Iter != GlobalOrtModelCache.end())
		GlobalOrtModelCache.erase(Iter);
}

void SingingVoiceConversion::ClearModelCache()
{
	GlobalOrtModelCache.clear();
}

DragonianLibSTL::Vector<float> VocoderInfer(
	DragonianLibSTL::Vector<float>& Mel,
	DragonianLibSTL::Vector<float>& F0,
	int64_t MelBins,
	int64_t MelSize,
	const Ort::MemoryInfo* Mem,
	const std::shared_ptr<Ort::Session>& _VocoderModel
)
{
	if (!_VocoderModel)
		DragonianLibThrow("Missing Vocoder Model!");

	const int64_t MelShape[] = { 1i64,MelBins,MelSize };
	const int64_t FrameShape[] = { 1,MelSize };
	OrtTensors Tensors;
	Tensors.emplace_back(Ort::Value::CreateTensor(
		*Mem,
		Mel.Data(),
		Mel.Size(),
		MelShape,
		3)
	);
	Tensors.emplace_back(Ort::Value::CreateTensor(
		*Mem,
		F0.Data(),
		FrameShape[1],
		FrameShape,
		2)
	);
	const DragonianLibSTL::Vector nsfInput = { "c", "f0" };
	const DragonianLibSTL::Vector nsfOutput = { "audio" };
	
	Tensors = _VocoderModel->Run(Ort::RunOptions{ nullptr },
		nsfInput.Data(),
		Tensors.data(),
		_VocoderModel->GetInputCount(),
		nsfOutput.Data(),
		nsfOutput.Size());
	const auto AudioSize = Tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2];
	const auto OutputData = Tensors[0].GetTensorData<float>();
	return { OutputData , OutputData + AudioSize };
}

LibSvcEnd