#include "../../../header/InferTools/TensorExtractor/BaseTensorExtractor.hpp"
#include "Base.h"

LibSvcHeader

LibSvcTensorExtractor::LibSvcTensorExtractor(uint64_t _srcsr, uint64_t _sr, uint64_t _hop, bool _smix, bool _volume, uint64_t _hidden_size, uint64_t _nspeaker, const Others& _other)
{
	//_SrcSamplingRate = _srcsr;
	_SamplingRate = _sr;
	_HopSize = _hop;
	_SpeakerMix = _smix;
	_Volume = _volume;
	_HiddenSize = _hidden_size;
	_NSpeaker = _nspeaker;
	f0_bin = _other.f0_bin;
	f0_max = _other.f0_max;
	f0_min = _other.f0_min;
	f0_mel_min = 1127.f * log(1.f + f0_min / 700.f);
	f0_mel_max = 1127.f * log(1.f + f0_max / 700.f);
	Memory = _other.Memory;
}

LibSvcTensorExtractor::Inputs LibSvcTensorExtractor::Extract(
	const DragonianLibSTL::Vector<float>& HiddenUnit,
	const DragonianLibSTL::Vector<float>& F0,
	const DragonianLibSTL::Vector<float>& Volume,
	const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap,
	Params params
)
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::GetCurrectSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, int64_t curspk) const
{
	DragonianLibSTL::Vector<float> mixData;
	mixData.Reserve(_NSpeaker * dst_len);
	if(_input.Empty())
	{
		DragonianLibSTL::Vector<float> LenData(_NSpeaker, 0.0);
		LenData[curspk] = 1.0;
		for (size_t i = 0; i < dst_len; ++i)
			mixData.Insert(mixData.end(), LenData.begin(), LenData.end());
	}
	else
	{
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _spkMap;
		for (size_t i = 0; i < _input.Size() && i < _NSpeaker; ++i)
			_spkMap.EmplaceBack(InterpFunc(_input[i], long(_input[i].Size()), long(dst_len)));
		LinearCombination(_spkMap, curspk);
		const auto curnspk = _input.Size();
		if (curnspk < _NSpeaker)
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
				for (size_t j = 0; j < _NSpeaker; ++j)
					mixData.EmplaceBack(_spkMap[j][i]);
	}
	return mixData;
}

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::GetSpkMixData(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _input, size_t dst_len, size_t spk_count)
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

DragonianLibSTL::Vector<int64_t> LibSvcTensorExtractor::GetNSFF0(const DragonianLibSTL::Vector<float>& F0) const
{
	const auto f0Len = F0.Size();
	DragonianLibSTL::Vector<int64_t> NSFF0(f0Len);
	for (size_t i = 0; i < f0Len; ++i)
	{
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

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::GetInterpedF0(const DragonianLibSTL::Vector<float>& F0)
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

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::InterpUVF0(const DragonianLibSTL::Vector<float>& F0, size_t PaddedIndex)
{
	if (PaddedIndex == size_t(-1))
		PaddedIndex = F0.Size();
	DragonianLibSTL::Vector<double> NUVF0;
	DragonianLibSTL::Vector<double> UVF0Indices, NUVF0Indices;
	UVF0Indices.Reserve(F0.Size());
	NUVF0.Reserve(F0.Size());
	NUVF0Indices.Reserve(F0.Size());
	if(F0[0] < 0.0001f)
	{
		NUVF0.EmplaceBack(0);
		NUVF0Indices.EmplaceBack(0);
	}
	for (size_t i = 1; i < PaddedIndex; ++i)
	{
		if (F0[i] < 0.0001f)
			UVF0Indices.EmplaceBack((double)i);
		else
		{
			NUVF0.EmplaceBack((double)F0[i]);
			NUVF0Indices.EmplaceBack((double)i);
		}
	}
	if (UVF0Indices.Empty() || NUVF0Indices.Empty())
		return F0;

	NUVF0Indices.EmplaceBack(F0.Size());
	NUVF0.EmplaceBack(0.);
	DragonianLibSTL::Vector<double> UVF0(F0.Size());
	DragonianLibSTL::Vector<float> Of0 = F0;
	interp1(NUVF0Indices.Data(), NUVF0.Data(), (int)NUVF0.Size(),
		UVF0Indices.Data(), (int)UVF0Indices.Size(), UVF0.Data());
	for (size_t i = 0; i < UVF0Indices.Size(); ++i)
		Of0[size_t(UVF0Indices[i])] = (float)UVF0[i];
	return Of0;
}

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::GetUV(const DragonianLibSTL::Vector<float>& F0)
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

DragonianLibSTL::Vector<int64_t> LibSvcTensorExtractor::GetAligments(size_t specLen, size_t hubertLen)
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

DragonianLibSTL::Vector<float> LibSvcTensorExtractor::GetInterpedF0log(const DragonianLibSTL::Vector<float>& rF0, bool enable_log)
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

LibSvcEnd