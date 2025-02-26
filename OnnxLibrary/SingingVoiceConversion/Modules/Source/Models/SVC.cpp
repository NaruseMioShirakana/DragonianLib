#include "../../header/Models/SVC.hpp"
#include "Libraries/Base.h"
#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

SingingVoiceConversion::SingingVoiceConversion(
	const std::wstring& HubertPath_,
	const ExecutionProviders& ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : LibSvcModule(ExecutionProvider_, DeviceID_, ThreadCount_)
{
	HubertModel = RefOrtCachedModel(HubertPath_, *OrtApiEnv);
}

SingingVoiceConversion::SingingVoiceConversion(
	const std::wstring& HubertPath_,
	const std::shared_ptr<DragonianLibOrtEnv>& Env_
) : LibSvcModule(Env_)
{
	HubertModel = RefOrtCachedModel(HubertPath_, *Env_);
}

SingingVoiceConversion::~SingingVoiceConversion() = default;

DragonianLibSTL::Vector<float> SingingVoiceConversion::InferPCMData(
	const DragonianLibSTL::Vector<float>& _PCMData,
	long _SrcSamplingRate,
	const InferenceParams& _Params
) const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
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
	_D_Dragonian_Lib_Not_Implemented_Error;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::InferenceWithCrossFade(
	const DragonianLibSTL::ConstantRanges<float>& _Audio,
	long _SrcSamplingRate,
	const InferenceParams& _Params,
	const CrossFadeParams& _Crossfade,
	const F0Extractor::F0ExtractorParams& _F0Params,
	const std::wstring& _F0Method,
	const F0Extractor::NetF0ExtractorSetting& _F0ExtractorLoadParameter,
	double _DbThreshold
) const
{
	const auto SourceSamplingRate = _SrcSamplingRate;
	const auto CrossFadeSamples = static_cast<ptrdiff_t>(_Crossfade.CrossFadeTime * (float)SourceSamplingRate);
	const auto SliceSize = static_cast<size_t>(_Crossfade.WindowSize * (float)SourceSamplingRate) - CrossFadeSamples;
	if (static_cast<long long>(SliceSize) < CrossFadeSamples / 2)
		_D_Dragonian_Lib_Throw_Exception("Slice time is too low to inference!");

	TemplateLibrary::Vector<float> Audio{ _Audio.Begin(), _Audio.End() };
	Audio.Resize((Audio.Size() / SliceSize + 1) * SliceSize, 0.f);
	auto SlicePos = DragonianLibSTL::Arange(0ull, Audio.Size() + SliceSize, SliceSize);
	auto Slices = GetAudioSlice(Audio, SlicePos, SourceSamplingRate, _DbThreshold);

	for (size_t i = 0; i < Slices.Slices.Size() - 1; ++i)
	{
		Slices.Slices[i].OrgLen += static_cast<int32_t>(CrossFadeSamples);
		if (Slices.Slices[i].IsNotMute)
		{
			if (Slices.Slices[i + 1].IsNotMute)
				Slices.Slices[i].Audio.Insert(
					Slices.Slices[i].Audio.End(),
					Slices.Slices[i + 1].Audio.Begin(),
					Slices.Slices[i + 1].Audio.Begin() + CrossFadeSamples
				);
			else
				Slices.Slices[i].Audio.Resize(
					Slices.Slices[i].Audio.Size() + CrossFadeSamples,
					0.f
				);
		}
	}
	Slices.Slices.Back().OrgLen += static_cast<int32_t>(CrossFadeSamples);
	Slices.Slices.Back().Audio.Resize(
		Slices.Slices.Back().Audio.Size() + CrossFadeSamples,
		0.f
	);

	PreProcessAudio(
		Slices,
		_F0Params,
		_F0Method,
		_F0ExtractorLoadParameter
	);

	std::vector<float> FadeInWindow(CrossFadeSamples), FadeOutWindow(CrossFadeSamples);
	{
		double Current = 0.;
		double Frequency = 3.1415926535 / double(CrossFadeSamples);
		for (int64_t i = 0; i < CrossFadeSamples; ++i) {
			auto Reg = (float)(0.5 * (1. + sin(Frequency * Current + 3.1415926535 / 2.)));
			Reg *= Reg;
			FadeOutWindow[i] = Reg;
			FadeInWindow[i] = 1.f - Reg;
			Current += 1.;
		}
	}

	ProgressCallbackFunction(0, Slices.Slices.Size());
	size_t Proc = 0;

	DragonianLibSTL::Vector<float> TotalOutPutAudio;
	TotalOutPutAudio.Reserve(Slices.Slices.Size() * SliceSize + SliceSize);
	DragonianLibSTL::Vector<float> LastData;
	for (auto& Slice : Slices.Slices)
	{
		auto OutPutAudio = DragonianLibSTL::InterpResample<float>(SliceInference(Slice, _Params, Proc), static_cast<long>(GetSamplingRate()), static_cast<long>(SourceSamplingRate));
		if (!LastData.Empty())
			for (int64_t i = 0; i < CrossFadeSamples; ++i)
				OutPutAudio[i] = OutPutAudio[i] * FadeInWindow[i] + LastData[i + SliceSize] * FadeOutWindow[i];
		TotalOutPutAudio.Insert(
			TotalOutPutAudio.End(),
			OutPutAudio.Begin(),
			OutPutAudio.Begin() + static_cast<ptrdiff_t>(SliceSize)
		);
		LastData = std::move(OutPutAudio);
	}

	return TotalOutPutAudio;
}

DragonianLibSTL::Vector<float> SingingVoiceConversion::SliceInference(
	const SingleSlice& _Slice,
	const InferenceParams& _Params,
	size_t& _Process
) const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
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
		i = sqrt((float)DragonianLibSTL::Average(
			(Audio.begin() + index * HopSize).Get(),
			(Audio.begin() + (index + 1) * HopSize).Get()
		));
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
		i = sqrt((float)DragonianLibSTL::Average(
			(Audio.begin() + index * _HopSize).Get(),
			(Audio.begin() + (index + 1) * _HopSize).Get()
		));
		++index;
	}
	return volume;
}

SingleAudio SingingVoiceConversion::GetAudioSlice(
	const DragonianLibSTL::Vector<float>& _InputPCM,
	const DragonianLibSTL::Vector<size_t>& _SlicePos,
	long _SamplingRate,
	double _Threshold
)
{
	SingleAudio audio_slice;
	const auto _SliceHopSize = std::max(_SamplingRate / 20, 320l);
	for (size_t i = 1; i < _SlicePos.Size(); i++)
	{
		SingleSlice _CurSlice;
		_CurSlice.SamplingRate = _SamplingRate;
		_CurSlice.OrgLen = _SlicePos[i] - _SlicePos[i - 1];
		bool is_not_mute = false;
		if (_CurSlice.OrgLen > static_cast<size_t>(_SliceHopSize))
			for (size_t _SliceBegin = _SlicePos[i - 1]; _SliceBegin <= _SlicePos[i] - _SliceHopSize; _SliceBegin += _SliceHopSize)
			{
				auto _SliceBuffer = _InputPCM.Data() + _SliceBegin;
				const auto _SliceDb = AvCodec::CalculateDB(_SliceBuffer, _SliceBuffer + _SliceHopSize);
				if (_SliceDb > _Threshold)
				{
					is_not_mute = true;
					break;
				}
			}
		else
			is_not_mute = AvCodec::CalculateDB((_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i])) > _Threshold;
		_CurSlice.IsNotMute = is_not_mute;
		
		if (is_not_mute)
			_CurSlice.Audio = { (_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]) };
		else
			_CurSlice.Audio.Clear();
		audio_slice.Slices.EmplaceBack(std::move(_CurSlice));
	}
	return audio_slice;
}

SingleAudio SingingVoiceConversion::GetAudioSlice(
	const DragonianLibSTL::ConstantRanges<float>& _InputPCM,
	const DragonianLibSTL::ConstantRanges<size_t>& _SlicePos,
	long _SamplingRate,
	double _Threshold
)
{
	SingleAudio audio_slice;
	const auto _SliceHopSize = std::max(_SamplingRate / 20, 320l);
	for (size_t i = 1; i < _SlicePos.Size(); i++)
	{
		SingleSlice _CurSlice;
		_CurSlice.SamplingRate = _SamplingRate;
		_CurSlice.OrgLen = _SlicePos[i] - _SlicePos[i - 1];
		bool is_not_mute = false;
		if (_CurSlice.OrgLen > static_cast<size_t>(_SliceHopSize))
			for (size_t _SliceBegin = _SlicePos[i - 1]; _SliceBegin <= _SlicePos[i] - _SliceHopSize; _SliceBegin += _SliceHopSize)
			{
				auto _SliceBuffer = _InputPCM.Data() + _SliceBegin;
				const auto _SliceDb = AvCodec::CalculateDB(_SliceBuffer, _SliceBuffer + _SliceHopSize);
				if (_SliceDb > _Threshold)
				{
					is_not_mute = true;
					break;
				}
			}
		else
			is_not_mute = AvCodec::CalculateDB((_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i])) > _Threshold;
		_CurSlice.IsNotMute = is_not_mute;
		
		if (is_not_mute)
			_CurSlice.Audio = { (_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i]) };
		else
			_CurSlice.Audio.Clear();
		audio_slice.Slices.EmplaceBack(std::move(_CurSlice));
	}
	return audio_slice;
}

void SingingVoiceConversion::PreProcessAudio(
	SingleAudio& _Input,
	const F0Extractor::F0ExtractorParams& _Params,
	const std::wstring& _F0Method,
	const F0Extractor::NetF0ExtractorSetting& _F0ExtractorLoadParameter
)
{
	const auto F0Extractor = F0Extractor::GetF0Extractor(
		_F0Method,
		&_F0ExtractorLoadParameter
	);
	const auto num_slice = _Input.Slices.Size();
	for (size_t i = 0; i < num_slice; ++i)
	{
		if (_Input.Slices[i].IsNotMute)
		{
			_Input.Slices[i].F0 = F0Extractor->ExtractF0(_Input.Slices[i].Audio, _Params);
			_Input.Slices[i].Volume = ExtractVolume(_Input.Slices[i].Audio, _Params.HopSize);
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
	_D_Dragonian_Lib_Not_Implemented_Error;
}

int64_t SingingVoiceConversion::GetMaxStep() const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

int64_t SingingVoiceConversion::GetMelBins() const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

void SingingVoiceConversion::NormMel(
	DragonianLibSTL::Vector<float>& MelSpec
) const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
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
		_D_Dragonian_Lib_Throw_Exception("Missing Vocoder Model!");

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

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End
