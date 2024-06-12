#include "Base.h"
#ifdef DRAGONIANLIB_ONNXRT_LIB
#include "EnvManager.hpp"
#include "F0Extractor/NetF0Predictors.hpp"
#include "matlabfunctions.h"
#include "F0Extractor/DioF0Extractor.hpp"

DragonianLibF0ExtractorHeader

Ort::Session* MelPEModel = nullptr;
DragonianLibOrtEnv* MelPEEnv = nullptr;

Ort::Session* RmvPEModel = nullptr;
DragonianLibOrtEnv* RmvPEEnv = nullptr;

inline double NetPredictorAverage(const double* begin, const double* end)
{
	const auto mp = double(end - begin);
	double sum = 0.;
	while (begin != end)
		sum += *(begin++);
	return sum / mp;
}

RMVPEF0Extractor::RMVPEF0Extractor(int sampling_rate, int hop_size, int n_f0_bins, double max_f0, double min_f0) :
	BaseF0Extractor(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0)
{
	
}

DragonianLibSTL::Vector<float> RMVPEF0Extractor::ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength)
{
	if (!RmvPEModel)
		return DioF0Extractor((int)fs, (int)hop, (int)f0_bin, f0_max, f0_min).ExtractF0(PCMData, TargetLength);

	const double step = double(fs) / 16000.;
	const auto window_len = (size_t)round(step);
	const auto half_window_len = window_len / 2;
	const auto f0_size = size_t((double)PCMData.Size() / step);
	const auto pcm_idx_size = PCMData.Size() - 2;
	DragonianLibSTL::Vector pcm(f0_size, 0.f);
	auto idx = double(half_window_len + 1);
	for (size_t i = 0; i < f0_size; ++i)
	{
		const auto index = size_t(round(idx));
		if(index + half_window_len > pcm_idx_size)
			break;
		if (half_window_len == 0)
			pcm[i] = (float)PCMData[index];
		else
			pcm[i] = (float)NetPredictorAverage(&PCMData[index - half_window_len], &PCMData[index + half_window_len]);
		idx += step;
	}
	std::vector<Ort::Value> Tensors;
	const int64_t pcm_shape[] = { 1, (int64_t)pcm.Size() };
	constexpr int64_t one_shape[] = { 1 };
	float threshold[] = { 0.03f };
	Tensors.emplace_back(Ort::Value::CreateTensor(*RmvPEEnv->GetMemoryInfo(), pcm.Data(), pcm.Size(), pcm_shape, 2));
	Tensors.emplace_back(Ort::Value::CreateTensor(*RmvPEEnv->GetMemoryInfo(), threshold, 1, one_shape, 1));

	const auto out = RmvPEModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorData<float>();
	refined_f0 = DragonianLibSTL::Vector<double>(osize);
	for (size_t i = 0; i < osize; ++i) refined_f0[i] = ((of0[i] > 0.001f) ? (double)out[0].GetTensorData<float>()[i] : NAN);
	InterPf0(TargetLength);
	DragonianLibSTL::Vector<float> finaF0(refined_f0.Size());
	for (size_t i = 0; i < refined_f0.Size(); ++i) finaF0[i] = isnan(refined_f0[i]) ? 0 : (float)refined_f0[i];
	return finaF0;
}

void RMVPEF0Extractor::InterPf0(size_t TargetLength)
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

MELPEF0Extractor::MELPEF0Extractor(int sampling_rate, int hop_size, int n_f0_bins, double max_f0, double min_f0) :
	BaseF0Extractor(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0)
{

}

void MELPEF0Extractor::InterPf0(size_t TargetLength)
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

DragonianLibSTL::Vector<float> MELPEF0Extractor::ExtractF0(const DragonianLibSTL::Vector<double>& PCMData, size_t TargetLength)
{
	if (!MelPEModel)
		return DioF0Extractor((int)fs, (int)hop, (int)f0_bin, f0_max, f0_min).ExtractF0(PCMData, TargetLength);

	const double step = double(fs) / 16000.;
	const auto window_len = (size_t)round(step);
	const auto half_window_len = window_len / 2;
	const auto f0_size = size_t((double)PCMData.Size() / step);
	const auto pcm_idx_size = PCMData.Size() - 2;
	DragonianLibSTL::Vector pcm(f0_size, 0.f);
	auto idx = double(half_window_len + 1);
	for (size_t i = 0; i < f0_size; ++i)
	{
		const auto index = size_t(round(idx));
		if (index + half_window_len > pcm_idx_size)
			break;
		if (half_window_len == 0)
			pcm[i] = (float)PCMData[index];
		else
			pcm[i] = (float)NetPredictorAverage(&PCMData[index - half_window_len], &PCMData[index + half_window_len]);
		idx += step;
	}
	std::vector<Ort::Value> Tensors;
	const int64_t pcm_shape[] = { 1, (int64_t)pcm.Size() };
	Tensors.emplace_back(Ort::Value::CreateTensor(*MelPEEnv->GetMemoryInfo(), pcm.Data(), pcm.Size(), pcm_shape, 2));

	const auto out = MelPEModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorData<float>();
	refined_f0 = DragonianLibSTL::Vector<double>(osize);
	for (size_t i = 0; i < osize; ++i) refined_f0[i] = ((of0[i] > 0.001f) ? (double)out[0].GetTensorData<float>()[i] : NAN);
	InterPf0(TargetLength);
	DragonianLibSTL::Vector<float> finaF0(refined_f0.Size());
	for (size_t i = 0; i < refined_f0.Size(); ++i) finaF0[i] = isnan(refined_f0[i]) ? 0 : (float)refined_f0[i];
	return finaF0;
}

void LoadNetPEModel(bool MelPE, const wchar_t* _ModelPath, unsigned ThreadCount, unsigned DeviceID, unsigned Provider)
{
	UnloadNetPEModel(MelPE);
	try
	{
		if (MelPE)
		{
			MelPEEnv = new DragonianLibOrtEnv(ThreadCount, DeviceID, Provider);
			MelPEModel = new Ort::Session(*MelPEEnv->GetEnv(), _ModelPath, *MelPEEnv->GetSessionOptions());
		}
		else
		{
			RmvPEEnv = new DragonianLibOrtEnv(ThreadCount, DeviceID, Provider);
			RmvPEModel = new Ort::Session(*RmvPEEnv->GetEnv(), _ModelPath, *RmvPEEnv->GetSessionOptions());
		}
	}
	catch (std::exception& e)
	{
		UnloadNetPEModel(MelPE);
		DragonianLibThrow(e.what());
	}
}

void UnloadNetPEModel(bool MelPE)
{
	if(MelPE)
	{
		delete MelPEModel;
		delete MelPEEnv;
		MelPEModel = nullptr;
		MelPEEnv = nullptr;
	}
	else
	{
		delete RmvPEModel;
		delete RmvPEEnv;
		RmvPEModel = nullptr;
		RmvPEEnv = nullptr;
	}
}

DragonianLibF0ExtractorEnd

#endif