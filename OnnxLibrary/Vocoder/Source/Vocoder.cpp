#include "../Vocoder.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::Vocoder",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

VocoderBase::VocoderBase(
	const std::wstring& _Path,
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate,
	Int64 _MelBins,
	const std::shared_ptr<Logger>& _Logger
) : _MyBase(_Environment, _Path, _Logger), _MySamplingRate(_SamplingRate), _MyMelBins(_MelBins)
{
	const bool InvalidInputCount = _MyInputCount != 1 && _MyInputCount != 2;
	const bool InvalidOutputCount = _MyOutputCount != 1;
	if (InvalidInputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected 1 or 2, got " + std::to_string(_MyInputCount));
	if (InvalidOutputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected 1, got " + std::to_string(_MyOutputCount));
	const bool InvalidInputShape = _MyInputDims[0].Size() < 2 || _MyInputDims[0].Size() > 4 ||
		(_MyInputCount == 2 && (_MyInputDims[1].Size() > 3 || _MyInputDims[1].Empty()));
	const bool InvalidOutputShape = _MyOutputDims[0].Size() < 1 || _MyOutputDims[0].Size() > 3;
	if (InvalidInputShape)
		_D_Dragonian_Lib_Throw_Exception("Invalid input shape, expected 2 to 4 and 1 to 3, got " + std::to_string(_MyInputDims[0].Size()) + " and " + std::to_string(_MyInputDims[1].Size()));
	if (InvalidOutputShape)
		_D_Dragonian_Lib_Throw_Exception("Invalid output shape, expected 1 to 3, got " + std::to_string(_MyOutputDims[0].Size()));

	for (auto [Axis, Dim] : Enumrate(_MyInputDims[0]))
		if (Dim == _MyMelBins)
		{
			_MyBinAxis = Axis + 4 - static_cast<Int64>(_MyInputDims[0].Size());
			break;
		}

	if (_MyBinAxis == -1)
	{
		if (_MyInputDims[0].Back() == -1)
		{
			LogWarn(L"Could not found the mel bins axis in the input dims, use the last axis as the mel bins axis!");
			_MyInputDims[0].Back() = _MyMelBins;
			_MyBinAxis = 3;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid input dims, could not found the mel bins axis");
	}
	if (_MyBinAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid mel bins axis, expected 2 or greater, got " + std::to_string(_MyBinAxis));
}

Tensor<Float32, 3, Device::CPU> VocoderBase::Forward(
	const Tensor<Float32, 4, Device::CPU>& _Mel,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0
) const
{
	_D_Dragonian_Lib_Rethrow_Block(return Inference(_Mel, _F0););
}

Tensor<Float32, 3, Device::CPU> VocoderBase::Inference(
	const Tensor<Float32, 4, Device::CPU>& _Mel,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _F0
) const
{
#ifdef _DEBUG
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
#endif

	if (_MyInputCount == 2)
	{
		const auto FrameAxis = _MyBinAxis == 2 ? 3 : 2;
		if (!_F0.has_value() || _F0.value().get().Null())
			_D_Dragonian_Lib_Throw_Exception("F0 is required");

		if (_F0->get().Size(0) != _Mel.Size(0))
			_D_Dragonian_Lib_Throw_Exception(
				"Batch/Channel of F0 and Mel is mismatched, excepted f0: " +
				std::to_string(_Mel.Size(0)) +
				", got: " +
				std::to_string(_F0->get().Size(0))
			);
		if (_F0->get().Size(1) != _Mel.Size(1))
			_D_Dragonian_Lib_Throw_Exception(
				"Channel/Batch of F0 and Mel is mismatched, excepted f0: " +
				std::to_string(_Mel.Size(1)) +
				", got: " +
				std::to_string(_F0->get().Size(1))
			);
		if (_F0->get().Size(2) != _Mel.Size(FrameAxis))
			_D_Dragonian_Lib_Throw_Exception(
				"Frames of F0 and Mel is mismatched, excepted f0: " +
				std::to_string(_Mel.Size(FrameAxis)) +
				", got: " +
				std::to_string(_F0->get().Size(2))
			);
	}

	InputTensorsType Inputs;

	auto MelCont = _Mel.Continuous().Evaluate();
	_D_Dragonian_Lib_Rethrow_Block(
		Inputs.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				MelCont,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"MelBins", L"MelFrames" },
				"Mel",
				GetLoggerPtr()
			)
		);
	);

	if (_MyInputCount == 2)
	{
		auto F0Cont = _F0->get().Continuous().Evaluate();
		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					F0Cont,
					_MyInputTypes[1],
					_MyInputDims[1],
					{ L"Batch/Channel", L"Batch/Channel", L"MelFrames" },
					"F0",
					GetLoggerPtr()
				)
			);
		);
	}

	std::vector<Ort::Value> Outputs;

	_D_Dragonian_Lib_Rethrow_Block(Outputs = RunModel(Inputs););

	const auto& OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputRank = OutputShape.size();
	Dimensions<3> AudioShape;
	if (OutputRank == 3)
		AudioShape = { OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputRank == 2)
		AudioShape = { 1, OutputShape[0], OutputShape[1] };
	else if (OutputRank == 1)
		AudioShape = { 1, 1, OutputShape[0] };

#ifdef _DEBUG
	LogInfo(
		L"Vocoder Forward Inference With Mel Shape: [" +
		std::to_wstring(MelCont.Shape(0)) + L", " +
		std::to_wstring(MelCont.Shape(1)) + L", " +
		std::to_wstring(MelCont.Shape(2)) + L", " +
		std::to_wstring(MelCont.Shape(3)) + L"], Cost Time: " +
		std::to_wstring(
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - TimeBegin
			).count()
		) +
		L"ms"
	);
#endif

	_D_Dragonian_Lib_Rethrow_Block(
		return CreateTensorViewFromOrtValue<Float32>(
			std::move(Outputs[0]),
			AudioShape
		).Evaluate();
	);
}

_D_Dragonian_Lib_Onnx_Vocoder_End