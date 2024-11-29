#include "../SvcBase.hpp"
#include "Libraries/AvCodec/AvCodec.h"

_D_Dragonian_Lib_NCNN_Svc_Space_Header

constexpr float f0_max = 1100.0;
constexpr float f0_min = 50.0;
float f0_mel_min = 1127.f * log(1.f + f0_min / 700.f);
float f0_mel_max = 1127.f * log(1.f + f0_max / 700.f);

DragonianLibSTL::Vector<float> SvcBase::GetCurrectSpkMixData(
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
	size_t dst_len, int64_t curspk, int64_t _NSpeaker
)
{
	DragonianLibSTL::Vector<float> mixData;
	mixData.Reserve(_NSpeaker * dst_len);
	if (_input.Empty())
	{
		DragonianLibSTL::Vector<float> LenData(_NSpeaker, 0.0);
		LenData[curspk] = 1.0;
		for (size_t i = 0; i < dst_len; ++i)
			mixData.Insert(mixData.end(), LenData.begin(), LenData.end());
	}
	else
	{
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _spkMap;
		for (size_t i = 0; i < _input.Size() && i < size_t(_NSpeaker); ++i)
			_spkMap.EmplaceBack(InterpFunc(_input[i], long(_input[i].Size()), long(dst_len)));
		LinearCombination(_spkMap, curspk);
		const auto curnspk = _input.Size();
		if (curnspk < size_t(_NSpeaker))
		{
			DragonianLibSTL::Vector<float> LenData(_NSpeaker - curnspk, 0.0);
			for (size_t i = 0; i < dst_len; ++i)
			{
				for (size_t j = 0; j < curnspk; ++j)
					mixData.EmplaceBack(_spkMap[j][i]);
				mixData.Insert(mixData.end(), LenData.begin(), LenData.end());
			}
		}
		else
			for (size_t i = 0; i < dst_len; ++i)
				for (size_t j = 0; j < size_t(_NSpeaker); ++j)
					mixData.EmplaceBack(_spkMap[j][i]);
	}
	return mixData;
}

DragonianLibSTL::Vector<float> SvcBase::GetSpkMixData(
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input,
	size_t dst_len, size_t spk_count
)
{
	DragonianLibSTL::Vector<float> mixData;
	mixData.Reserve(spk_count * dst_len);
	if (_input.Empty())
	{
		DragonianLibSTL::Vector<float> LenData(spk_count, 0.0);
		LenData[0] = 1.0;
		for (size_t i = 0; i < dst_len; ++i)
			mixData.Insert(mixData.end(), LenData.begin(), LenData.end());
	}
	else
	{
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _spkMap;
		for (size_t i = 0; i < _input.Size() && i < spk_count; ++i)
			_spkMap.EmplaceBack(InterpFunc(_input[i], long(_input[i].Size()), long(dst_len)));
		LinearCombination(_spkMap, 0);
		const auto curnspk = _input.Size();
		if (curnspk < spk_count)
		{
			DragonianLibSTL::Vector<float> LenData(spk_count - curnspk, 0.0);
			for (size_t i = 0; i < dst_len; ++i)
			{
				for (size_t j = 0; j < curnspk; ++j)
					mixData.EmplaceBack(_spkMap[j][i]);
				mixData.Insert(mixData.end(), LenData.begin(), LenData.end());
			}
		}
		else
			for (size_t i = 0; i < dst_len; ++i)
				for (size_t j = 0; j < spk_count; ++j)
					mixData.EmplaceBack(_spkMap[j][i]);
	}
	return mixData;
}

DragonianLibSTL::Vector<int64_t> SvcBase::GetNSFF0(
	const DragonianLibSTL::Vector<float>& F0
)
{
	const auto f0Len = F0.Size();
	DragonianLibSTL::Vector<int64_t> NSFF0(f0Len);
	for (size_t i = 0; i < f0Len; ++i)
	{
		constexpr int f0_bin = 256;
		float f0_mel = 1127.f * log(1.f + F0[i] / 700.f);
		if (f0_mel > 0.f)
			f0_mel = (f0_mel - f0_mel_min) * (float(f0_bin) - 2.f) / (f0_mel_max - f0_mel_min) + 1.f;
		if (f0_mel < 1.f)
			f0_mel = 1.f;
		if (f0_mel > float(f0_bin) - 1.f)
			f0_mel = float(f0_bin) - 1.f;
		NSFF0[i] = (int64_t)round(f0_mel);
	}
	return NSFF0;
}

DragonianLibSTL::Vector<float> SvcBase::GetInterpedF0(
	const DragonianLibSTL::Vector<float>& F0
)
{
	const auto specLen = F0.Size();
	DragonianLibSTL::Vector<float> Of0(specLen, 0.0);

	float last_value = 0.0;
	for (size_t i = 0; i < specLen; ++i)
	{
		if (F0[i] <= 0.f)
		{
			size_t j = i + 1;
			for (; j < specLen; ++j)
			{
				if (F0[j] > 0.f)
					break;
			}
			if (j < specLen - 1)
			{
				if (last_value > 0.f)
				{
					const auto step = (F0[j] - F0[i - 1]) / float(j - i);
					for (size_t k = i; k < j; ++k)
						Of0[k] = float(F0[i - 1] + step * float(k - i + 1));
				}
				else
					for (size_t k = i; k < j; ++k)
						Of0[k] = float(F0[j]);
				i = j;
			}
			else
			{
				for (size_t k = i; k < specLen; ++k)
					Of0[k] = float(last_value);
				i = specLen;
			}
		}
		else
		{
			if (i == 0)
			{
				Of0[i] = float(F0[i]);
				continue;
			}
			Of0[i] = float(F0[i - 1]);
			last_value = F0[i];
		}
	}
	return Of0;
}

DragonianLibSTL::Vector<float> SvcBase::GetUV(
	const DragonianLibSTL::Vector<float>& F0
)
{
	const auto specLen = F0.Size();
	DragonianLibSTL::Vector<float> ruv(specLen, 1.0);
	for (size_t i = 0; i < specLen; ++i)
	{
		if (F0[i] < 0.001f)
			ruv[i] = 0.f;
	}
	return ruv;
}

DragonianLibSTL::Vector<int64_t> SvcBase::GetAligments(
	size_t specLen,
	size_t hubertLen
)
{
	DragonianLibSTL::Vector mel2ph(specLen + 1, 0ll);

	size_t startFrame = 0;
	const double ph_durs = static_cast<double>(specLen) / static_cast<double>(hubertLen);
	for (size_t iph = 0; iph < hubertLen; ++iph)
	{
		const auto endFrame = static_cast<size_t>(round(static_cast<double>(iph) * ph_durs + ph_durs));
		for (auto j = startFrame; j < endFrame + 1; ++j)
			mel2ph[j] = static_cast<long long>(iph) + 1;
		startFrame = endFrame + 1;
	}
	return mel2ph;
}

DragonianLibSTL::Vector<float> SvcBase::GetInterpedF0log(
	const DragonianLibSTL::Vector<float>& rF0,
	bool enable_log
)
{
	const auto specLen = rF0.Size();
	DragonianLibSTL::Vector<float> F0(specLen);
	DragonianLibSTL::Vector<float> Of0(specLen, 0.0);
	for (size_t i = 0; i < specLen; ++i)
	{
		if (enable_log)
			F0[i] = log2(rF0[i]);
		else
			F0[i] = rF0[i];
		if (isnan(F0[i]) || isinf(F0[i]))
			F0[i] = 0.f;
	}

	float last_value = 0.0;
	for (size_t i = 0; i < specLen; ++i)
	{
		if (F0[i] <= 0.f)
		{
			size_t j = i + 1;
			for (; j < specLen; ++j)
			{
				if (F0[j] > 0.f)
					break;
			}
			if (j < specLen - 1)
			{
				if (last_value > 0.f)
				{
					const auto step = (F0[j] - F0[i - 1]) / float(j - i);
					for (size_t k = i; k < j; ++k)
						Of0[k] = float(F0[i - 1] + step * float(k - i + 1));
				}
				else
					for (size_t k = i; k < j; ++k)
						Of0[k] = float(F0[j]);
				i = j;
			}
			else
			{
				for (size_t k = i; k < specLen; ++k)
					Of0[k] = float(last_value);
				i = specLen;
			}
		}
		else
		{
			Of0[i] = float(F0[i - 1]);
			last_value = F0[i];
		}
	}
	return Of0;
}

DragonianLibSTL::Vector<float> SvcBase::ExtractVolume(
	const DragonianLibSTL::Vector<float>& _Audio,
	int _HopSize
)
{
	DragonianLibSTL::Vector<double> Audio;
	Audio.Reserve(_Audio.Size() * 2);
	Audio.Insert(Audio.end(), _HopSize, double(_Audio[0]));
	for (const auto i : _Audio)
		Audio.EmplaceBack((double)i);
	Audio.Insert(Audio.end(), _HopSize, double(_Audio[_Audio.Size() - 1]));
	const size_t n_frames = (_Audio.Size() / _HopSize) + 1;
	DragonianLibSTL::Vector<float> volume(n_frames);
	for (auto& i : Audio)
		i = pow(i, 2);
	int64_t index = 0;
	for (auto& i : volume)
	{
		i = sqrt((float)DragonianLibSTL::Average(
			Audio.begin().Get() + index * _HopSize,
			Audio.begin().Get() + (index + 1) * _HopSize
		));
		++index;
	}
	return volume;
}

SingleAudio SvcBase::GetAudioSlice(
	const DragonianLibSTL::Vector<float>& _InputPCM,
	const DragonianLibSTL::Vector<size_t>& _SlicePos,
	double Threshold
)
{
	SingleAudio audio_slice;
	for (size_t i = 1; i < _SlicePos.Size(); i++)
	{
		constexpr auto _SliceHopSize = 1024;
		SingleSlice _CurSlice;
		bool is_not_mute = false;
		if (_CurSlice.OrgLen > _SliceHopSize)
			for (size_t _SliceBegin = _SlicePos[i - 1]; _SliceBegin <= _SlicePos[i] - _SliceHopSize; _SliceBegin += _SliceHopSize)
			{
				auto _SliceBuffer = _InputPCM.Data() + _SliceBegin;
				const auto _SliceDb = AvCodec::CalculateDB(_SliceBuffer, _SliceBuffer + _SliceHopSize);
				if (_SliceDb > Threshold)
				{
					is_not_mute = true;
					break;
				}
			}
		else
			is_not_mute = AvCodec::CalculateDB((_InputPCM.Data() + _SlicePos[i - 1]), (_InputPCM.Data() + _SlicePos[i])) > Threshold;
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

void SvcBase::PreProcessAudio(
	SingleAudio& _Input, int _SamplingRate, int _HopSize,
	const std::wstring& _F0Method, const void* UserParameter
)
{
	const auto F0Extractor = F0Extractor::GetF0Extractor(_F0Method, UserParameter);
	const auto num_slice = _Input.Slices.Size();
	for (size_t i = 0; i < num_slice; ++i)
	{
		_Input.Slices[i].SamplingRate = _SamplingRate;
		if (_Input.Slices[i].IsNotMute)
		{
			_Input.Slices[i].F0 = F0Extractor->ExtractF0(
				_Input.Slices[i].Audio,
				{
					_SamplingRate,
					_HopSize
				}
			);
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

DragonianLibSTL::Vector<float> SvcBase::InferenceAudio(
	const DragonianLibSTL::Vector<float>& _Audio,
	const InferenceParams& _Params,
	int64_t _SourceSamplingRate,
	size_t _SliceTime,
	bool _Refersh
)
{
	if (_Refersh)
		EmptyCache();

	const auto SourceSamplingRate = _SourceSamplingRate;
	const auto CrossFadeSamples = SourceSamplingRate / 10;
	const auto SliceSize = (SourceSamplingRate * _SliceTime) - CrossFadeSamples;
	if (static_cast<long long>(SliceSize) < CrossFadeSamples / 2)
		_D_Dragonian_Lib_Throw_Exception("Slice time is too low to inference!");

	auto Audio = _Audio;
	Audio.Resize((Audio.Size() / SliceSize + 1) * SliceSize, 0.f);
	auto SlicePos = DragonianLibSTL::Arange(0ull, Audio.Size() + SliceSize, SliceSize);
	auto Slices = GetAudioSlice(Audio, SlicePos, _Params.Threshold);

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
		static_cast<int>(SourceSamplingRate),
		_Params.MuteCheckHopSize,
		_Params.F0Method,
		_Params.UserParameters
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

	ProgressFn(0, Slices.Slices.Size());
	size_t Proc = 0;

	DragonianLibSTL::Vector<float> TotalOutPutAudio;
	TotalOutPutAudio.Reserve(Slices.Slices.Size() * SliceSize);
	DragonianLibSTL::Vector<float> LastData;
	for (auto& Slice : Slices.Slices)
	{
		auto OutPutAudio = DragonianLibSTL::InterpResample<float>(SliceInference(Slice, _Params), static_cast<long>(MySamplingRate), static_cast<long>(SourceSamplingRate));
		if (!LastData.Empty())
			for (int64_t i = 0; i < CrossFadeSamples; ++i)
				OutPutAudio[i] = OutPutAudio[i] * FadeInWindow[i] + LastData[i + SliceSize] * FadeOutWindow[i];
		TotalOutPutAudio.Insert(
			TotalOutPutAudio.End(),
			OutPutAudio.Begin(),
			OutPutAudio.Begin() + static_cast<ptrdiff_t>(SliceSize)
		);
		LastData = std::move(OutPutAudio);
		ProgressFn(++Proc, 0);
	}

	return TotalOutPutAudio;
}

_D_Dragonian_Lib_NCNN_Svc_Space_End