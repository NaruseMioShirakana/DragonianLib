#include "BaseF0Extractor.hpp"
#include "Base.h"

DragonianLib::BaseF0Extractor::BaseF0Extractor(int sampling_rate, int hop_size, int n_f0_bins, double max_f0, double min_f0) :
	fs(sampling_rate),
	hop(hop_size),
	f0_bin(n_f0_bins),
	f0_max(max_f0),
	f0_min(min_f0)
{
	f0_mel_min = (1127.0 * log(1.0 + f0_min / 700.0));
	f0_mel_max = (1127.0 * log(1.0 + f0_max / 700.0));
}

DragonianLibSTL::Vector<float> DragonianLib::BaseF0Extractor::ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength)
{
	DragonianLibNotImplementedError;
}

DragonianLibSTL::Vector<float> DragonianLib::BaseF0Extractor::ExtractF0(const DragonianLibSTL::Vector<float>& PCMData, size_t TargetLength)
{
	DragonianLibSTL::Vector<double> PCMVector(PCMData.Size());
	for (size_t i = 0; i < PCMData.Size(); ++i)
		PCMVector[i] = double(PCMData[i]);
	return ExtractF0(PCMVector, TargetLength);
}

DragonianLibSTL::Vector<float> DragonianLib::BaseF0Extractor::ExtractF0(const DragonianLibSTL::Vector<int16_t>& PCMData, size_t TargetLength)
{
	DragonianLibSTL::Vector<double> PCMVector(PCMData.Size());
	for (size_t i = 0; i < PCMData.Size(); ++i)
		PCMVector[i] = double(PCMData[i]);
	return ExtractF0(PCMVector, TargetLength);
}