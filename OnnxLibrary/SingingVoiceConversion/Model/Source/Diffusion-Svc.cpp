#include "OnnxLibrary/SingingVoiceConversion/Model/Diffusion-Svc.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"
#include "OnnxLibrary/SingingVoiceConversion/Model/Samplers.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

ProphesierDiffusion::ProphesierDiffusion(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : Unit2Ctrl(_Environment, Params, _Logger, false)
{
	if (Params.ModelPaths.contains(L"Denoiser"))
		_MyDenoiser = RefOnnxRuntimeModel(Params.ModelPaths.at(L"Denoiser"), _Environment);
	if (Params.ModelPaths.contains(L"NoisePredictor"))
		_MyNoisePredictor = RefOnnxRuntimeModel(Params.ModelPaths.at(L"NoisePredictor"), _Environment);
	if (Params.ModelPaths.contains(L"AlphaCumprod"))
		_MyAlphaCumprod = RefOnnxRuntimeModel(Params.ModelPaths.at(L"AlphaCumprod"), _Environment);

	if (_MyInputCount < 4 || _MyInputCount > 6)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 3-5, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount < 1 || _MyOutputCount > 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1-4, got: " + std::to_string(_MyOutputCount));

	auto& HubertShape = _MyInputDims[0];
	auto& Mel2UnitsShape = _MyInputDims[1];

	if (HasSpeakerMixLayer())
	{
		auto SpeakerMixShape = _MyInputDims[2];
		const auto SpeakerMixAxis = SpeakerMixShape.Size();
		if (SpeakerMixAxis < 2 || SpeakerMixAxis > 4)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix axis, expected: 2-4, got: " + std::to_string(SpeakerMixAxis));
		bool FoundSpeakerCount = false;
		for (auto& SpeakerMixDim : SpeakerMixShape)
			if (SpeakerMixDim == _MySpeakerCount)
			{
				FoundSpeakerCount = true;
				break;
			}
		if (!FoundSpeakerCount)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker mix dims, expected: " + std::to_string(_MySpeakerCount) + ", got: " + std::to_string(SpeakerMixShape.Back()));
	}
	else if (HasSpeakerEmbedding())
	{
		const auto& SpeakerIdShape = _MyInputDims[2];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));
	}

	const auto F0Index = (HasSpeakerMixLayer() || HasSpeakerEmbedding()) ? 3 : 2;
	auto& F0Shape = _MyInputDims[F0Index];

	const auto HubertAxis = HubertShape.Size();
	const auto Mel2UnitsAxis = Mel2UnitsShape.Size();
	const auto F0Axis = F0Shape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (Mel2UnitsAxis < 1 || Mel2UnitsAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel2units axis, expected: 1-3, got: " + std::to_string(Mel2UnitsAxis));

	if (_MyInputCount > 4)
	{
		auto& MelDims = _MyInputDims[_MyInputDims.Size() - 2];
		bool Found = false;
		for (auto& MelDim : MelDims)
			if (MelDim == _MyMelBins)
			{
				Found = true;
				break;
			}
		if (!Found)
			_D_Dragonian_Lib_Throw_Exception("Invalid mel dims, expected: " + std::to_string(_MyMelBins) + ", got: " + std::to_string(MelDims.Back()));
	}
}

Tensor<Float32, 4, Device::CPU> ProphesierDiffusion::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
#ifdef _DEBUG
	const auto StartTime = std::chrono::high_resolution_clock::now();
#endif

	auto Unit = InputDatas.Units;
	auto Mel2Units = InputDatas.Mel2Units;
	auto SpeakerId = InputDatas.SpeakerId;
	auto SpeakerMix = InputDatas.Speaker;
	auto F0 = InputDatas.F0;
	auto Mel = InputDatas.GTSpec;

	if (Unit.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");
	if (F0.Null())
		_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
	if (!Mel2Units || Mel2Units->Null())
		_D_Dragonian_Lib_Throw_Exception("Mel2Units could not be null");
	if (HasSpeakerMixLayer())
	{
		if (!SpeakerMix || SpeakerMix->Null())
			_D_Dragonian_Lib_Throw_Exception("SpeakerMix could not be null");
	}
	else if (HasSpeakerEmbedding())
	{
		if (!SpeakerId || SpeakerId->Null())
			_D_Dragonian_Lib_Throw_Exception("SpeakerId could not be null");
	}

	{
		const auto BatchSize = Unit.Shape(0);
		const auto Channels = Unit.Shape(1);
		const auto TargetNumFrames = F0.Shape(2);
		PreprocessSpec(Mel, BatchSize, Channels, TargetNumFrames, Params.Seed, Params.NoiseScale, GetLoggerPtr());
	}

	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				Unit,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"UnitsDims" },
				"Units",
				GetLoggerPtr()
			)
		);
	);

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				*Mel2Units,
				_MyInputTypes[1],
				_MyInputDims[1],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"Mel2Units",
				GetLoggerPtr()
			)
		);
	);

	if (HasSpeakerMixLayer())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerMix,
					_MyInputTypes[2],
					_MyInputDims[2],
					{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames", L"SpeakerCount" },
					"SpeakerMix",
					GetLoggerPtr()
				)
			);
		);
	else if (HasSpeakerEmbedding())
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*SpeakerId,
					_MyInputTypes[2],
					_MyInputDims[2],
					{ L"Batch/Channel", L"Channel/Batch", L"SpeakerId" },
					"SpeakerId",
					GetLoggerPtr()
				)
			);
		);

	const auto F0Axis = (HasSpeakerEmbedding() || HasSpeakerMixLayer()) ? 3 : 2;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				F0,
				_MyInputTypes[F0Axis],
				_MyInputDims[F0Axis],
				{ L"Batch/Channel", L"Channel/Batch", L"AudioFrames" },
				"F0",
				GetLoggerPtr()
			)
		);
	);

	if (_MyInputCount > 4)
	{
		const auto MelAxis = F0Axis + 1;

		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					*Mel,
					_MyInputTypes[MelAxis],
					_MyInputDims[MelAxis],
					{ L"Batch/Channel", L"Channel/Batch", L"MelBins", L"AudioFrames" },
					"Mel",
					GetLoggerPtr()
				)
			);
		);

		int64_t Stride[] = { Params.Diffusion.Stride };
		constexpr int64_t StrideShape[] = { 1 };

		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				Ort::Value::CreateTensor<long long>(
					*_MyMemoryInfo,
					Stride,
					1,
					StrideShape,
					1
				)
			);
		);
	}

	OrtTuple OutputTensors;

	_D_Dragonian_Lib_Rethrow_Block(OutputTensors = RunModel(InputTensors););

	if (_MyDenoiser && _MyInputCount <= 4)
	{
		if ((Params.Diffusion.End - Params.Diffusion.Begin) / Params.Diffusion.Stride <= 0)
		{
			LogInfo(L"Diffusion step is zero or negative, skip diffusion");
			return std::move(*Mel);
		}

		auto Spec = Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			Mel->Data(),
			Mel->ElementCount(),
			Mel->Shape().Data(),
			Mel->Rank()
		);

		auto Condition = std::move(OutputTensors[0]);

		_D_Dragonian_Lib_Rethrow_Block(
			OutputTensors[0] = GetDiffusionSampler(Params.Diffusion.Sampler)(
				std::move(Spec),
				std::move(Condition),
				Params.Diffusion,
				*_MyRunOptions,
				*_MyMemoryInfo,
				_MyDenoiser,
				_MyNoisePredictor,
				_MyAlphaCumprod,
				_MyProgressCallback,
				GetLoggerPtr()
				);
		);
	}

	auto& OMel = OutputTensors[0];
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
	GetLoggerPtr()->LogInfo(L"ProphesierDiffusion finished, time: " + std::to_wstring(Duration) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(OMel), OutputDims););
}

DiffusionSvc::DiffusionSvc(
	const OnnxRuntimeEnvironment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : Unit2Ctrl(_Environment, Params, _Logger)
{
	if (!Params.ModelPaths.contains(L"Denoiser"))
		_D_Dragonian_Lib_Throw_Exception("Denoiser is required for DiffusionSvc");
	_MyDenoiser = RefOnnxRuntimeModel(Params.ModelPaths.at(L"Denoiser"), GetDlEnvPtr());
	if (Params.ModelPaths.contains(L"AlphaCumprod"))
		_MyAlphaCumprod = RefOnnxRuntimeModel(Params.ModelPaths.at(L"AlphaCumprod"), GetDlEnvPtr());
	if (Params.ModelPaths.contains(L"NoisePredictor"))
		_MyNoisePredictor = RefOnnxRuntimeModel(Params.ModelPaths.at(L"NoisePredictor"), GetDlEnvPtr());
}

Tensor<Float32, 4, Device::CPU> DiffusionSvc::Forward(
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

	Ort::Value Spec{ nullptr };
	//const bool OutputHasSpec = (Tuple[0].GetTensorTypeAndShapeInfo().GetElementCount() != 1) && _MyOutputCount == 3;

	if ((_MyOutputCount == 3) && abs(Params.NoiseScale) < 1e-4f)
	{
		Spec = std::move(Tuple[2]);
		if ((Params.Diffusion.End - Params.Diffusion.Begin) / Params.Diffusion.Stride <= 0)
		{
			LogInfo(L"Diffusion step is zero or negative, skip diffusion");
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

		if (_MyOutputCount == 3)
		{
			const auto Mel2Scale = 1.f - Params.NoiseScale;
			if (Mel2Scale > 0.f)
			{
				auto Mel2 = Tensor<Float32, 4, Device::CPU>::FromBuffer(
					Mel->Shape(),
					Tuple[2].GetTensorMutableData<Float32>(),
					Tuple[2].GetTensorTypeAndShapeInfo().GetElementCount()
				);

				(*Mel += (Mel2 * Mel2Scale)).Evaluate();
			}
		}

		if ((Params.Diffusion.End - Params.Diffusion.Begin) / Params.Diffusion.Stride <= 0)
		{
			LogInfo(L"Diffusion step is zero or negative, skip diffusion");
			return std::move(*Mel);
		}

		Spec = Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			Mel->Data(),
			Mel->ElementCount(),
			Mel->Shape().Data(),
			Mel->Rank()
		);
	}

	auto Condition = std::move(Tuple[0]);

	Ort::Value OMel{ nullptr };

	_D_Dragonian_Lib_Rethrow_Block(
		OMel = GetDiffusionSampler(Params.Diffusion.Sampler)(
			std::move(Spec),
			std::move(Condition),
			Params.Diffusion,
			*_MyRunOptions,
			*_MyMemoryInfo,
			_MyDenoiser,
			_MyNoisePredictor,
			_MyAlphaCumprod,
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
	GetLoggerPtr()->LogInfo(L"Diffusion finished, time: " + std::to_wstring(Duration) + L"ms");
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(OMel), OutputDims););
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End