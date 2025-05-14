#include "OnnxLibrary/SingingVoiceConversion/Model/Reflow-Svc.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/Samplers.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header
	ReflowSvc::ReflowSvc(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : Unit2Ctrl(_Environment, Params, _Logger)
{
	if (!Params.ModelPaths.contains(L"Velocity"))
		_D_Dragonian_Lib_Throw_Exception("Velocity is required for ReflowSvc");
	_MyVelocity = RefOnnxRuntimeModel(Params.ModelPaths.at(L"Velocity"), GetDlEnvPtr());
}

Tensor<Float32, 4, Device::CPU> ReflowSvc::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const

{
	auto Tuple = Extract(Params, InputDatas);

#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto& Unit = InputDatas.Units;
	auto& F0 = InputDatas.F0;
	auto Mel = InputDatas.GTSpec;

	const bool OutputHasSpec = Tuple[0].GetTensorTypeAndShapeInfo().GetElementCount() != 1;
	Ort::Value Spec{ nullptr };

	if (OutputHasSpec && abs(Params.NoiseScale) < 1e-4f)
	{
		Spec = std::move(Tuple[0]);
		if ((Params.Reflow.End - Params.Reflow.Begin) / Params.Reflow.Stride <= 0.99f)
		{
			LogInfo(L"Reflow step is zero or negative, skip reflow");
			auto OutputShape = Spec.GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputAxis = OutputShape.size();
			Dimensions<4> OutputDims;
			if (OutputAxis == 1)
				OutputDims = { 1, 1, 1, OutputShape[0] };
			else if (OutputAxis == 2)
				OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
			else if (OutputAxis == 3)
				OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
			else if (OutputAxis == 4)
				OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };
			_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Spec), OutputDims););
		}
	}
	else
	{
		const auto BatchSize = Unit.Shape(0);
		const auto Channels = Unit.Shape(1);
		const auto TargetNumFrames = F0.Shape(2);
		PreprocessSpec(Mel, BatchSize, Channels, TargetNumFrames, Params.Seed, Params.NoiseScale, GetLoggerPtr());

		if (OutputHasSpec)
		{
			const auto Mel2Scale = Params.Reflow.Begin * Params.Reflow.Scale / 1000.f;

			auto Mel2 = Tensor<Float32, 4, Device::CPU>::FromBuffer(
				Mel->Shape(),
				Tuple[0].GetTensorMutableData<Float32>(),
				Tuple[0].GetTensorTypeAndShapeInfo().GetElementCount()
			);

			(*Mel += (Mel2 * Mel2Scale));
		}

		if ((Params.Reflow.End - Params.Reflow.Begin) / Params.Reflow.Stride <= 0.f)
		{
			LogInfo(L"Reflow step is zero or negative, skip reflow");
			return std::move(*Mel);
		}

		Mel = Mel->Contiguous().Evaluate();
		Spec = Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			Mel->Data(),
			Mel->ElementCount(),
			Mel->Shape().Data(),
			Mel->Rank()
		);
	}

	auto Condition = std::move(Tuple[1]);

	Ort::Value OMel{ nullptr };

	_D_Dragonian_Lib_Rethrow_Block(
		OMel = GetReflowSampler(Params.Reflow.Sampler)(
			std::move(Spec),
			std::move(Condition),
			Params.Reflow,
			*_MyRunOptions,
			*_MyMemoryInfo,
			_MyVelocity,
			_MyProgressCallback,
			GetLoggerPtr()
			);
	);

	auto OutputShape = OMel.GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;
	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

#ifdef _DEBUG
	const auto EndTime = std::chrono::high_resolution_clock::now();
	const auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();
	GetLoggerPtr()->LogInfo(L"Reflow finished, time: " + std::to_wstring(Duration) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(OMel), OutputDims););
}


_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End