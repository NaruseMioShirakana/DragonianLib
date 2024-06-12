﻿#include "F0Extractor/HarvestF0Extractor.hpp"
#include "matlabfunctions.h"
#include "harvest.h"
#include "stonemask.h"

DragonianLibF0ExtractorHeader
	HarvestF0Extractor::HarvestF0Extractor(int sampling_rate, int hop_size, int n_f0_bins, double max_f0, double min_f0):
	BaseF0Extractor(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0)
{
}

void HarvestF0Extractor::InterPf0(size_t TargetLength)
{
    const auto f0Len = refined_f0.Size();
    if (abs((int64_t)TargetLength - (int64_t)f0Len) < 3)
    {
        refined_f0.Resize(TargetLength, 0.0);
	    return;
    }
    for (size_t i = 0; i < f0Len; ++i) if (refined_f0[i] < 0.001) refined_f0[i] = NAN;

    auto xi = DragonianLibSTL::Arange(0., (double)f0Len * (double)TargetLength, (double)f0Len, (double)TargetLength);
    while (xi.Size() < TargetLength) xi.EmplaceBack(*(xi.End() - 1) + ((double)f0Len / (double)TargetLength));
    while (xi.Size() > TargetLength) xi.PopBack();

	auto x0 = DragonianLibSTL::Arange(0., (double)f0Len);
    while (x0.Size() < f0Len) x0.EmplaceBack(*(x0.End() - 1) + 1.);
    while (x0.Size() > f0Len) x0.PopBack();

    auto raw_f0 = DragonianLibSTL::Vector<double>(xi.Size());
    interp1(x0.Data(), refined_f0.Data(), static_cast<int>(x0.Size()), xi.Data(), (int)xi.Size(), raw_f0.Data());

    for (size_t i = 0; i < xi.Size(); i++) if (isnan(raw_f0[i])) raw_f0[i] = 0.0;
    refined_f0 = std::move(raw_f0);
}

DragonianLibSTL::Vector<float> HarvestF0Extractor::ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength)
{
    compute_f0(PCMData.Data(), PCMData.Size());
    InterPf0(TargetLength);
    DragonianLibSTL::Vector<float> f0(refined_f0.Size());
    for (size_t i = 0; i < refined_f0.Size(); ++i) f0[i] = (float)refined_f0[i];
    return f0;
}

void HarvestF0Extractor::compute_f0(const double* PCMData, size_t PCMLen)
{
    HarvestOption Doption;
    InitializeHarvestOption(&Doption);
    Doption.f0_ceil = f0_max;
    Doption.f0_floor = f0_min;
    Doption.frame_period = 1000.0 * hop / fs;

    const size_t f0Length = GetSamplesForHarvest(int(fs), (int)PCMLen, Doption.frame_period);
	auto temporal_positions = DragonianLibSTL::Vector<double>(f0Length);
	auto raw_f0 = DragonianLibSTL::Vector<double>(f0Length);
    refined_f0 = DragonianLibSTL::Vector<double>(f0Length);
    Harvest(PCMData, (int)PCMLen, int(fs), &Doption, temporal_positions.Data(), raw_f0.Data());
    StoneMask(PCMData, (int)PCMLen, int(fs), temporal_positions.Data(), raw_f0.Data(), (int)f0Length, refined_f0.Data());
}
DragonianLibF0ExtractorEnd