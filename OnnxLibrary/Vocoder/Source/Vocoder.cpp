#include "../Vocoder.hpp"

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
	const bool InvalidInputShape = _MyInputDims[0].Size() < 2 || _MyInputDims[0].Size() > 4 || _MyInputDims[1].Size() < 1 || _MyInputDims[1].Size() > 3;
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
			LogWarn(L"Could not found the units axis in the output dims, use the last axis as the units axis!");
			_MyInputDims[0].Back() = _MyMelBins;
			_MyBinAxis = 3;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid output dims, could not found the units axis");
	}
	if (_MyBinAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid units axis, expected 2 or greater, got " + std::to_string(_MyBinAxis));

	if (_MyBinAxis != 2)
		LogWarn(L"Units axis is not the last axis, operations may be slow!");
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

	bool M2A = _MyBinAxis == 2;
	Int64 _MelShape[4] = { 1, 1, 1, 1 };
	Int64 _F0Shape[3] = { 1, 1, 1 };
	const auto MelInputAxisCount = static_cast<Int64>(_MyInputDims[0].Size());	// 4,  3,  2,  1
	const auto MIBI = MelInputAxisCount - 4;							// 0, -1, -2, -3
	const auto MICI = MelInputAxisCount - 3;							// 1,  0, -1, -2
	const auto MIMI = MelInputAxisCount - (M2A ? 2 : 1);				// 2,  1,  0, -1
	const auto MIFI = MelInputAxisCount - (M2A ? 1 : 2);				// 3,  2,  1,  0

	const auto MelCont = _Mel.Continuous().Evaluate();
	if (MIBI >= 0 && _MyInputDims[0][MIBI] != -1 && MelCont.Size(0) != _MyInputDims[0][MIBI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input batch size");
	if (MICI >= 0 && _MyInputDims[0][MICI] != -1 && MelCont.Size(1) != _MyInputDims[0][MICI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input channel count");
	if (MIMI >= 0 && _MyInputDims[0][MIMI] != -1 && MelCont.Size(2) != _MyInputDims[0][MIMI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input mel bins");
	if (MIFI >= 0 && _MyInputDims[0][MIFI] != -1 && MelCont.Size(3) != _MyInputDims[0][MIFI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input frames");

	std::optional<Tensor<Float32, 3, Device::CPU>> F0Cont = std::nullopt;
	if (_MyInputCount == 2)
	{
		if (!_F0.has_value())
			_D_Dragonian_Lib_Throw_Exception("F0 is required");

		const auto F0InputAxisCount = static_cast<Int64>(_MyInputDims[1].Size());
		const auto FIBI = F0InputAxisCount - 3;
		const auto FICI = F0InputAxisCount - 2;
		const auto FISI = F0InputAxisCount - 1;

		if (FIBI >= 0 && _MyInputDims[1][FIBI] != -1 && _F0->get().Size(0) != _MyInputDims[1][FIBI])
			_D_Dragonian_Lib_Throw_Exception("Invalid f0 batch size");
		if (FICI >= 0 && _MyInputDims[1][FICI] != -1 && _F0->get().Size(1) != _MyInputDims[1][FICI])
			_D_Dragonian_Lib_Throw_Exception("Invalid f0 channel count");
		if (FISI >= 0 && _MyInputDims[1][FISI] != -1 && _F0->get().Size(2) != _MyInputDims[1][FISI])
			_D_Dragonian_Lib_Throw_Exception("Invalid f0 frame count");
		_D_Dragonian_Lib_Rethrow_Block(F0Cont = _F0->get().Continuous().Evaluate(););
	}

	if (MelInputAxisCount == 4)
	{
		_MelShape[0] = MelCont.Size(0);
		_MelShape[1] = MelCont.Size(1);
		_MelShape[2] = MelCont.Size(2);
		_MelShape[3] = MelCont.Size(3);
	}
	else if (MelInputAxisCount == 3)
	{
		_MelShape[0] = MelCont.Size(1);
		_MelShape[1] = MelCont.Size(2);
		_MelShape[2] = MelCont.Size(3);
	}
	else if (MelInputAxisCount == 2)
	{
		_MelShape[0] = MelCont.Size(2);
		_MelShape[1] = MelCont.Size(3);
	}
	else if (MelInputAxisCount == 1)
	{
		_MelShape[0] = MelCont.Size(3);
	}

	std::vector<Ort::Value> Inputs;

	_D_Dragonian_Lib_Rethrow_Block(Inputs.emplace_back(
		Ort::Value::CreateTensor<float>(
			*_MyMemoryInfo,
			MelCont.Data(),
			MelCont.ElementCount(),
			_MelShape,
			MelInputAxisCount
		)
	););

	if (F0Cont.has_value())
	{
		const auto F0InputAxisCount = static_cast<Int64>(_MyInputDims[1].Size());
		if (F0InputAxisCount == 3)
		{
			_F0Shape[0] = F0Cont->Size(0);
			_F0Shape[1] = F0Cont->Size(1);
			_F0Shape[2] = F0Cont->Size(2);
		}
		else if (F0InputAxisCount == 2)
		{
			_F0Shape[0] = F0Cont->Size(1);
			_F0Shape[1] = F0Cont->Size(2);
		}
		else if (F0InputAxisCount == 1)
		{
			_F0Shape[0] = F0Cont->Size(2);
		}
		_D_Dragonian_Lib_Rethrow_Block(Inputs.emplace_back(
			Ort::Value::CreateTensor<float>(
				*_MyMemoryInfo,
				F0Cont->Data(),
				F0Cont->ElementCount(),
				_F0Shape,
				F0InputAxisCount
			)
		););
	}

	std::vector<Ort::Value> Outputs;

	_D_Dragonian_Lib_Rethrow_Block(Outputs = _MyModel->Run(
		*_MyRunOptions,
		_MyInputNames.Data(),
		Inputs.data(),
		_MyInputCount,
		_MyOutputNames.Data(),
		_MyOutputCount
	););

	const auto OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	Dimensions<3> AudioShape{ 1, 1, 1 };
	if (OutputShape.size() == 3)
	{
		AudioShape[0] = OutputShape[0];
		AudioShape[1] = OutputShape[1];
		AudioShape[2] = OutputShape[2];
	}
	else if (OutputShape.size() == 2)
	{
		AudioShape[1] = OutputShape[0];
		AudioShape[2] = OutputShape[1];
	}
	else if (OutputShape.size() == 1)
	{
		AudioShape[2] = OutputShape[0];
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Invalid output dims");

#ifdef _DEBUG
	LogInfo(
		L"Vocoder Forward Inference With Mel Shape: [" +
		std::to_wstring(_MelShape[0]) + L", " +
		std::to_wstring(_MelShape[1]) + L", " +
		std::to_wstring(_MelShape[2]) + L", " +
		std::to_wstring(_MelShape[3]) + L"], Cost Time: " +
		std::to_wstring(
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - TimeBegin
			).count()
		) +
		L"ms"
	);
#endif

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<float>(
		std::move(Outputs[0]),
		AudioShape
	).Evaluate(););
}


_D_Dragonian_Lib_Onnx_Vocoder_End