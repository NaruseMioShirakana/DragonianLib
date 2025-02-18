#ifdef DRAGONIANLIB_ONNXRT_LIB
#include "Libraries/Base.h"
#include "../NetF0Predictors.hpp"
#include "../DioF0Extractor.hpp"
#include "Libraries/Util/StringPreprocess.h"
#include "onnxruntime_cxx_api.h"

_D_Dragonian_Lib_F0_Extractor_Header

DioF0Extractor _MyDioExtractor;

RMVPEF0Extractor::RMVPEF0Extractor(const std::wstring& _ModelPath, const std::shared_ptr<DragonianLibOrtEnv>& _OrtEnv) :
	BaseF0Extractor(), _MyModel(RefOrtCachedModel(_ModelPath, *_OrtEnv)), _MyOrtEnv(_OrtEnv) {}

Vector<float> RMVPEF0Extractor::ExtractF0(const Vector<double>& PCMData, const F0ExtractorParams& Params)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(PCMData, Params);

	return ExtractF0(
		InterpResample<float>(PCMData, Params.SamplingRate, 16000),
		{
			16000,
			Params.HopSize * 16000 / Params.SamplingRate,
			Params.F0Bins,
			Params.F0Max,
			Params.F0Min,
			Params.UserParameter
		}
	);
}

Vector<float> RMVPEF0Extractor::ExtractF0(const Vector<int16_t>& PCMData, const F0ExtractorParams& Params)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(InterpResample<double>(PCMData, Params.SamplingRate, 16000), Params);

	return ExtractF0(
		InterpResample<float>(PCMData, Params.SamplingRate, 16000),
		{
			16000,
			Params.HopSize * 16000 / Params.SamplingRate,
			Params.F0Bins,
			Params.F0Max,
			Params.F0Min,
			Params.UserParameter
		}
	);
}

Vector<float> RMVPEF0Extractor::ExtractF0(
	const Vector<float>& PCMData,
	const F0ExtractorParams& Params
)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(InterpResample<double>(PCMData, Params.SamplingRate, 16000), Params);

	if (Params.SamplingRate != 16000)
		return ExtractF0(
			InterpResample<float>(PCMData, Params.SamplingRate, 16000),
			{
				16000,
				Params.HopSize * 16000 / Params.SamplingRate,
				Params.F0Bins,
				Params.F0Max,
				Params.F0Min,
				Params.UserParameter
			}
		);

	if (Params.HopSize < 80)
		_D_Dragonian_Lib_Throw_Exception("HopSize Too Low!");

	std::vector<Ort::Value> Tensors;
	const int64_t pcm_shape[] = { 1, (int64_t)PCMData.Size() };
	constexpr int64_t one_shape[] = { 1 };
	float threshold[] = { 0.03f };
	Tensors.emplace_back(Ort::Value::CreateTensor(*_MyOrtEnv->GetMemoryInfo(), const_cast<float*>(PCMData.Data()), PCMData.Size(), pcm_shape, 2));
	Tensors.emplace_back(Ort::Value::CreateTensor(*_MyOrtEnv->GetMemoryInfo(), threshold, 1, one_shape, 1));

	auto out = _MyModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorMutableData<float>();
	auto TargetLength = (size_t)ceil(double(PCMData.Size()) / double(Params.HopSize));

	for (size_t i = 0; i < osize; ++i) if (of0[i] < 0.001f) of0[i] = NAN;
	Vector<float> OutPut(TargetLength);
	Resample(of0, osize, OutPut.Data(), OutPut.Size());
	for (auto& f0 : OutPut) if (isnan(f0)) f0 = 0.f;
	return OutPut;
}

MELPEF0Extractor::MELPEF0Extractor(const std::wstring& _ModelPath, const std::shared_ptr<DragonianLibOrtEnv>& _OrtEnv) :
	BaseF0Extractor(), _MyModel(RefOrtCachedModel(_ModelPath, *_OrtEnv)), _MyOrtEnv(_OrtEnv) {}

Vector<float> MELPEF0Extractor::ExtractF0(const Vector<double>& PCMData, const F0ExtractorParams& Params)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(PCMData, Params);

	return ExtractF0(
		InterpResample<float>(PCMData, Params.SamplingRate, 16000),
		{
			16000,
			Params.HopSize * 16000 / Params.SamplingRate,
			Params.F0Bins,
			Params.F0Max,
			Params.F0Min,
			Params.UserParameter
		}
	);
}

Vector<float> MELPEF0Extractor::ExtractF0(const Vector<int16_t>& PCMData, const F0ExtractorParams& Params)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(InterpResample<double>(PCMData, Params.SamplingRate, 16000), Params);

	return ExtractF0(
		InterpResample<float>(PCMData, Params.SamplingRate, 16000),
		{
			16000,
			Params.HopSize * 16000 / Params.SamplingRate,
			Params.F0Bins,
			Params.F0Max,
			Params.F0Min,
			Params.UserParameter
		}
	);
}

Vector<float> MELPEF0Extractor::ExtractF0(
	const Vector<float>& PCMData,
	const F0ExtractorParams& Params
)
{
	if (!_MyModel)
		return _MyDioExtractor.ExtractF0(InterpResample<double>(PCMData, Params.SamplingRate, 16000), Params);

	if (Params.SamplingRate != 16000)
		return ExtractF0(
			InterpResample<float>(PCMData, Params.SamplingRate, 16000),
			{
				16000,
				Params.HopSize * 16000 / Params.SamplingRate,
				Params.F0Bins,
				Params.F0Max,
				Params.F0Min,
				Params.UserParameter
			}
		);

	if (Params.HopSize < 80)
		_D_Dragonian_Lib_Throw_Exception("HopSize Too Low!");

	std::vector<Ort::Value> Tensors;
	const int64_t pcm_shape[] = { 1, (int64_t)PCMData.Size() };
	Tensors.emplace_back(Ort::Value::CreateTensor(*_MyOrtEnv->GetMemoryInfo(), const_cast<float*>(PCMData.Data()), PCMData.Size(), pcm_shape, 2));

	auto out = _MyModel->Run(Ort::RunOptions{ nullptr },
		InputNames.Data(),
		Tensors.data(),
		Tensors.size(),
		OutputNames.Data(),
		OutputNames.Size());

	const auto osize = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto of0 = out[0].GetTensorMutableData<float>();
	auto TargetLength = (size_t)ceil(double(PCMData.Size()) / double(Params.HopSize));

	for (size_t i = 0; i < osize; ++i) if (of0[i] < 0.001f) of0[i] = NAN;
	Vector<float> OutPut(TargetLength);
	Resample(of0, osize, OutPut.Data(), OutPut.Size());
	for (auto& f0 : OutPut) if (isnan(f0)) f0 = 0.f;
	return OutPut;
}

_D_Dragonian_Lib_F0_Extractor_End

#endif
