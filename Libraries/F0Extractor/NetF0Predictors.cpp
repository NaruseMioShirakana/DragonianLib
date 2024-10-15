#ifdef DRAGONIANLIB_ONNXRT_LIB
#include "Base.h"
#include "EnvManager.hpp"
#include "F0Extractor/NetF0Predictors.hpp"
#include "F0Extractor/DioF0Extractor.hpp"
#include "Util/StringPreprocess.h"
#include "onnxruntime_cxx_api.h"

DragonianLibF0ExtractorHeader

std::shared_ptr<Ort::Session> MelPEModel = nullptr;
std::shared_ptr<Ort::Session> RmvPEModel = nullptr;
std::shared_ptr<DragonianLibOrtEnv> MelPEEnv = nullptr;
std::shared_ptr<DragonianLibOrtEnv> RmvPEEnv = nullptr;

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
	constexpr int64_t one_shape[] = { 1 };
	float threshold[] = { 0.03f };
	Tensors.emplace_back(Ort::Value::CreateTensor(*RmvPEEnv->GetMemoryInfo(), pcm.Data(), pcm.Size(), pcm_shape, 2));
	Tensors.emplace_back(Ort::Value::CreateTensor(*RmvPEEnv->GetMemoryInfo(), threshold, 1, one_shape, 1));

	auto out = RmvPEModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorMutableData<float>();

	for (size_t i = 0; i < osize; ++i) if (of0[i] < 0.001f) of0[i] = NAN;
	DragonianLibSTL::Vector<float> OutPut(TargetLength);
	DragonianLibSTL::Resample(of0, osize, OutPut.Data(), OutPut.Size());
	for (auto& f0 : OutPut) if (isnan(f0)) f0 = 0.f;
	return OutPut;
}

MELPEF0Extractor::MELPEF0Extractor(int sampling_rate, int hop_size, int n_f0_bins, double max_f0, double min_f0) :
	BaseF0Extractor(sampling_rate, hop_size, n_f0_bins, max_f0, min_f0)
{

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

	auto out = MelPEModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorMutableData<float>();

	for (size_t i = 0; i < osize; ++i) if (of0[i] < 0.001f) of0[i] = NAN;
	DragonianLibSTL::Vector<float> OutPut(TargetLength);
	DragonianLibSTL::Resample(of0, osize, OutPut.Data(), OutPut.Size());
	for (auto& f0 : OutPut) if (isnan(f0)) f0 = 0.f;
	return OutPut;
}

void LoadFCPEModel(const char* FCPEModelPath, const std::shared_ptr<DragonianLibOrtEnv>& Env)
{
	MelPEEnv = Env;
	MelPEModel = RefOrtCachedModel(UTF8ToWideString(FCPEModelPath), *MelPEEnv);
}

void LoadRMVPEModel(const char* RMVPEModelPath, const std::shared_ptr<DragonianLibOrtEnv>& Env)
{
	RmvPEEnv = Env;
	RmvPEModel = RefOrtCachedModel(UTF8ToWideString(RMVPEModelPath), *RmvPEEnv);
}

void UnloadFCPEModel()
{
	MelPEModel.reset();
	MelPEEnv.reset();
}

void UnloadRMVPEModel()
{
	RmvPEModel.reset();
	RmvPEEnv.reset();
}

DragonianLibF0ExtractorEnd

#endif
