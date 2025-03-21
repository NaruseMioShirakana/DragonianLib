#include "../Hubert.hpp"

_D_Dragonian_Lib_Onnx_UnitsEncoder_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::UnitsEncoder",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

HubertBase::HubertBase(
	const std::wstring& _Path,
	const OnnxRuntimeEnviroment& _Enviroment,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	const std::shared_ptr<Logger>& _Logger
) : _MyBase(_Enviroment, _Path, _Logger), _MySamplingRate(_SamplingRate), _MyUnitsDims(_UnitsDims)
{
	const bool InvalidInputCount = _MyInputCount < 1 || _MyInputCount > 2;
	const bool InvalidOutputCount = _MyOutputCount != 1;
	if (InvalidInputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count");
	if (InvalidOutputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count");
	const bool InvalidInputDims = _MyInputDims[0].Size() > 3 || _MyInputDims[0].Empty() ||
		(_MyInputCount == 2 && (_MyInputDims[1].Size() > 2 || _MyInputDims[1].Empty()));
	const bool InvalidOutputDims = _MyOutputDims[0].Size() > 4 || _MyOutputDims[0].Size() < 2;
	if (InvalidInputDims)
		_D_Dragonian_Lib_Throw_Exception("Invalid input dims");
	if (InvalidOutputDims)
		_D_Dragonian_Lib_Throw_Exception("Invalid output dims");

	for (auto [Axis, Dim] : Enumrate(_MyOutputDims[0]))
		if (Dim == _MyUnitsDims)
		{
			_MyUnitsAxis = Axis + 4 - static_cast<Int64>(_MyOutputDims[0].Size());
			break;
		}

	if (_MyUnitsAxis == -1)
	{
		if (_MyOutputDims[0].Back() == -1)
		{
			LogWarn(L"Could not found the units axis in the output dims, use the last axis as the units axis!");
			_MyOutputDims[0].Back() = _MyUnitsDims;
			_MyUnitsAxis = 3;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Invalid output dims");
	}
	if (_MyUnitsAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid units axis");

	if (_MyUnitsAxis != 3)
		LogWarn(L"Units axis is not the last axis, operations may be slow!");
}

Tensor<Float32, 4, Device::CPU> HubertBase::InferenceModel(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
	const auto AudioInputAxisCount = static_cast<Int64>(_MyInputDims[0].Size());	// 3,  2,  1
	const auto AIBI = AudioInputAxisCount - 3;								// 0, -1, -2
	const auto AICI = AudioInputAxisCount - 2;								// 1,  0, -1
	const auto AISI = AudioInputAxisCount - 1;								// 2,  1,  0

	std::optional<Tensor<Float32, 3, Device::CPU>> Audio = std::nullopt;
	if (_SamplingRate != _MySamplingRate)
		_D_Dragonian_Lib_Rethrow_Block(
			Audio = _PCMData.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(2),
				IScale(double(_MySamplingRate) / double(_SamplingRate))
			).Evaluate();
		);
	else
		_D_Dragonian_Lib_Rethrow_Block(
			Audio = _PCMData.Continuous().Evaluate();
		);

	if (!Audio.has_value())
		_D_Dragonian_Lib_Throw_Exception("Invalid audio data");

	if (AIBI >= 0 && _MyInputDims[0][AIBI] != -1 && Audio->Size(0) != _MyInputDims[0][AIBI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input batch size");
	if (AICI >= 0 && _MyInputDims[0][AICI] != -1 && Audio->Size(1) != _MyInputDims[0][AICI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input channel count");
	if (AISI >= 0 && _MyInputDims[0][AISI] != -1 && Audio->Size(2) != _MyInputDims[0][AISI])
		_D_Dragonian_Lib_Throw_Exception("Invalid input sample count");

	Int64 MaskShape[3] = { 1, 1, 1 };
	Int64 AudioShape[3] = { 1, 1, 1 };

	std::optional<Tensor<Float32, 3, Device::CPU>> Mask = std::nullopt;
	if (_MyInputCount == 2)
	{
		if (!_Mask.has_value())
			_D_Dragonian_Lib_Throw_Exception("Mask is required");

		const auto MaskInputAxisCount = static_cast<Int64>(_MyInputDims[1].Size());
		const auto MIBI = MaskInputAxisCount - 3;
		const auto MICI = MaskInputAxisCount - 2;
		const auto MISI = MaskInputAxisCount - 1;
		if (MIBI >= 0 && _MyInputDims[1][MIBI] != -1 && _Mask->get().Size(0) != _MyInputDims[1][MIBI])
			_D_Dragonian_Lib_Throw_Exception("Invalid mask batch size");
		if (MICI >= 0 && _MyInputDims[1][MICI] != -1 && _Mask->get().Size(1) != _MyInputDims[1][MICI])
			_D_Dragonian_Lib_Throw_Exception("Invalid mask channel count");
		if (MISI >= 0 && _MyInputDims[1][MISI] != -1 && _Mask->get().Size(2) != _MyInputDims[1][MISI])
			_D_Dragonian_Lib_Throw_Exception("Invalid mask sample count");
		_D_Dragonian_Lib_Rethrow_Block(Mask = _Mask->get().BroadCast2AndCpy(Audio.value()).Evaluate(););
	}

	if (AudioInputAxisCount == 3)
	{
		AudioShape[0] = Audio->Size(0);
		AudioShape[1] = Audio->Size(1);
		AudioShape[2] = Audio->Size(2);
	}
	else if (AudioInputAxisCount == 2)
	{
		AudioShape[0] = Audio->Size(1);
		AudioShape[1] = Audio->Size(2);
	}
	else if (AudioInputAxisCount == 1)
	{
		AudioShape[0] = Audio->Size(2);
	}

	std::vector<Ort::Value> Inputs;

	_D_Dragonian_Lib_Rethrow_Block(Inputs.emplace_back(
		Ort::Value::CreateTensor<float>(
			*_MyMemoryInfo,
			Audio->Data(),
			Audio->ElementCount(),
			AudioShape,
			AudioInputAxisCount
		)
	););

	if (Mask.has_value())
	{
		const auto MaskInputAxisCount = static_cast<Int64>(_MyInputDims[1].Size());

		if (MaskInputAxisCount == 3)
		{
			MaskShape[0] = Mask->Size(0);
			MaskShape[1] = Mask->Size(1);
			MaskShape[2] = Mask->Size(2);
		}
		else if (MaskInputAxisCount == 2)
		{
			MaskShape[0] = Mask->Size(1);
			MaskShape[1] = Mask->Size(2);
		}
		else if (MaskInputAxisCount == 1)
		{
			MaskShape[0] = Mask->Size(2);
		}

		_D_Dragonian_Lib_Rethrow_Block(Inputs.emplace_back(
			Ort::Value::CreateTensor<float>(
				*_MyMemoryInfo,
				Mask->Data(),
				Mask->ElementCount(),
				MaskShape,
				MaskInputAxisCount
			)
		););
	}

	std::vector<Ort::Value> Outputs;

	_D_Dragonian_Lib_Rethrow_Block(Outputs = _MyModel->Run(
		Ort::RunOptions{ nullptr },
		_MyInputNames.Data(),
		Inputs.data(),
		_MyInputCount,
		_MyOutputNames.Data(),
		_MyOutputCount
	););

	const auto OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	Dimensions<4> UnitShape{ 1, 1, 1, 1 };
	if (OutputShape.size() == 4)
	{
		UnitShape[0] = OutputShape[0];
		UnitShape[1] = OutputShape[1];
		UnitShape[2] = OutputShape[2];
		UnitShape[3] = OutputShape[3];
	}
	else if (OutputShape.size() == 3)
	{
		UnitShape[1] = OutputShape[0];
		UnitShape[2] = OutputShape[1];
		UnitShape[3] = OutputShape[2];
	}
	else if (OutputShape.size() == 2)
	{
		UnitShape[2] = OutputShape[0];
		UnitShape[3] = OutputShape[1];
	}
	else
		_D_Dragonian_Lib_Throw_Exception("Invalid output dims");

	Tensor<Float32, 4, Device::CPU> UnitsOutput;

	_D_Dragonian_Lib_Rethrow_Block(UnitsOutput = CreateTensorViewFromOrtValue<float>(
		std::move(Outputs[0]),
		UnitShape
	););

	if (_MyUnitsAxis == 2)
		return UnitsOutput.Permute({ 0, 1, 3, 2 }).Evaluate();
	return std::move(UnitsOutput.Evaluate());
}

Tensor<Float32, 4, Device::CPU> HubertBase::Forward(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
	_D_Dragonian_Lib_Rethrow_Block(return InferenceModel(_PCMData, _SamplingRate, _Mask););
}

_D_Dragonian_Lib_Onnx_UnitsEncoder_End