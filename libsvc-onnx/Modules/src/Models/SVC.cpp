#include "Models/SVC.hpp"
#include "Base.h"
#include "F0Extractor/F0ExtractorManager.hpp"

LibSvcHeader
	SingingVoiceConversion::SingingVoiceConversion(const ExecutionProviders& ExecutionProvider_, unsigned DeviceID_, unsigned ThreadCount_) : LibSvcModule(ExecutionProvider_, DeviceID_, ThreadCount_)
{

}

SingingVoiceConversion::~SingingVoiceConversion()
{
	delete hubert;
	hubert = nullptr;
}

DragonianLibSTL::Vector<int16_t> SingingVoiceConversion::InferPCMData(
	const DragonianLibSTL::Vector<int16_t>& _PCMData,
	long _SrcSamplingRate,
	const InferenceParams& _Params
) const
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<int16_t> SingingVoiceConversion::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params,
	size_t& _Process
) const
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::ExtractVolume(const DragonianLibSTL::Vector<double>& _Audio) const
{
	DragonianLibSTL::Vector<double> Audio;
	Audio.Reserve(_Audio.Size() * 2);
	Audio.Insert(Audio.end(), HopSize, _Audio[0]);
	Audio.Insert(Audio.end(), _Audio.begin(), _Audio.end());
	Audio.Insert(Audio.end(), HopSize, _Audio[_Audio.Size() - 1]);
	const size_t n_frames = (_Audio.Size() / HopSize) + 1;
	DragonianLibSTL::Vector<float> volume(n_frames);
	for (auto& i : Audio)
		i = pow(i, 2);
	int64_t index = 0;
	for (auto& i : volume)
	{
		i = sqrt((float)DragonianLibSTL::Average(Audio.begin() + index * HopSize, Audio.begin() + (index + 1) * HopSize));
		++index;
	}
	return volume;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::ExtractVolume(const DragonianLibSTL::Vector<int16_t>& _Audio, int _HopSize)
{
	DragonianLibSTL::Vector<double> Audio;
	Audio.Reserve(_Audio.Size() * 2);
	Audio.Insert(Audio.end(), _HopSize, double(_Audio[0]) / 32768.);
	for (const auto i : _Audio)
		Audio.EmplaceBack((double)i / 32768.);
	Audio.Insert(Audio.end(), _HopSize, double(_Audio[_Audio.Size() - 1]) / 32768.);
	const size_t n_frames = (_Audio.Size() / _HopSize) + 1;
	DragonianLibSTL::Vector<float> volume(n_frames);
	for (auto& i : Audio)
		i = pow(i, 2);
	int64_t index = 0;
	for (auto& i : volume)
	{
		i = sqrt((float)DragonianLibSTL::Average(Audio.begin() + index * _HopSize, Audio.begin() + (index + 1) * _HopSize));
		++index;
	}
	return volume;
}

SingleAudio SingingVoiceConversion::GetAudioSlice(const DragonianLibSTL::Vector<int16_t>& _InputPCM, const DragonianLibSTL::Vector<size_t>& _SlicePos, const SlicerSettings& _SlicerConfig)
{
	SingleAudio audio_slice;
	for (size_t i = 1; i < _SlicePos.Size(); i++)
	{
		SingleSlice _CurSlice;
		const bool is_not_mute = abs(DragonianLibSTL::Average((_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]))) > _SlicerConfig.Threshold;
		_CurSlice.IsNotMute = is_not_mute;
		_CurSlice.OrgLen = long(_SlicePos[i] - _SlicePos[i - 1]);
		if (is_not_mute)
			_CurSlice.Audio = { (_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]) };
		else
			_CurSlice.Audio.Clear();
		audio_slice.Slices.EmplaceBack(std::move(_CurSlice));
	}
	return audio_slice;
}

void SingingVoiceConversion::PreProcessAudio(SingleAudio& _Input, int _SamplingRate, int _HopSize, const std::wstring& _F0Method)
{
	const auto F0Extractor = DragonianLib::GetF0Extractor(_F0Method, _SamplingRate, _HopSize);
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

LibSvcEnd