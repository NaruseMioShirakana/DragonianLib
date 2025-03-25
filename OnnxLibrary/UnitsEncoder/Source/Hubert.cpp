#include "../Hubert.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

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
	const OnnxRuntimeEnvironment& _Environment,
	Int64 _SamplingRate,
	Int64 _UnitsDims,
	const std::shared_ptr<Logger>& _Logger
) : _MyBase(_Environment, _Path, _Logger), _MySamplingRate(_SamplingRate), _MyUnitsDims(_UnitsDims)
{
	const bool InvalidInputCount = _MyInputCount < 1 || _MyInputCount > 2;
	const bool InvalidOutputCount = _MyOutputCount != 1;
	if (InvalidInputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected 1 or 2, got " + std::to_string(_MyInputCount));
	if (InvalidOutputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected 1, got " + std::to_string(_MyOutputCount));
	const bool InvalidInputDims = _MyInputDims[0].Size() > 3 || _MyInputDims[0].Empty() ||
		(_MyInputCount == 2 && (_MyInputDims[1].Size() > 3 || _MyInputDims[1].Empty()));
	const bool InvalidOutputDims = _MyOutputDims[0].Size() > 4 || _MyOutputDims[0].Size() < 2;
	if (InvalidInputDims)
		_D_Dragonian_Lib_Throw_Exception("Invalid input dims, expected 1 to 3, got " + std::to_string(_MyInputDims[0].Size()) +
			(_MyInputCount == 2 ? ", " + std::to_string(_MyInputDims[1].Size()) : ""));
	if (InvalidOutputDims)
		_D_Dragonian_Lib_Throw_Exception("Invalid output dims, expected 2 to 4, got " + std::to_string(_MyOutputDims[0].Size()));

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
			_D_Dragonian_Lib_Throw_Exception("Invalid output dims, could not found the units axis");
	}
	if (_MyUnitsAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid units axis, expected 2 or greater, got " + std::to_string(_MyUnitsAxis));
}

Tensor<Float32, 4, Device::CPU> HubertBase::InferenceModel(
	const Tensor<Float32, 3, Device::CPU>& _PCMData,
	Int64 _SamplingRate,
	std::optional<std::reference_wrapper<const Tensor<Float32, 3, Device::CPU>>> _Mask
) const
{
#ifdef _DEBUG
	const auto TimeBegin = std::chrono::high_resolution_clock::now();
#endif

	if (_PCMData.Null())
		_D_Dragonian_Lib_Throw_Exception("PCMData is Null, please check the input tensor");

	if (_MyInputCount == 2)
	{
		const auto FrameAxis = _MyUnitsAxis == 2 ? 3 : 2;
		if (!_Mask.has_value() || _Mask.value().get().Null())
			_D_Dragonian_Lib_Throw_Exception("Mask is required");

		if (_Mask->get().Size(0) != _PCMData.Size(0))
			_D_Dragonian_Lib_Throw_Exception(
				"Batch/Channel of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(0)) +
				", got: " +
				std::to_string(_Mask->get().Size(0))
			);
		if (_Mask->get().Size(1) != _PCMData.Size(1))
			_D_Dragonian_Lib_Throw_Exception(
				"Channel/Batch of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(1)) +
				", got: " +
				std::to_string(_Mask->get().Size(1))
			);
		if (_Mask->get().Size(2) != _PCMData.Size(FrameAxis))
			_D_Dragonian_Lib_Throw_Exception(
				"Frames of Mask and PCMData is mismatched, excepted mask: " +
				std::to_string(_PCMData.Size(FrameAxis)) +
				", got: " +
				std::to_string(_Mask->get().Size(2))
			);
	}

	InputTensorsType Inputs;

	auto AudioCont = _PCMData.Continuous().Evaluate();
	if (_SamplingRate != _MySamplingRate)
		_D_Dragonian_Lib_Rethrow_Block(
			AudioCont = AudioCont.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(2),
				IScale(double(_MySamplingRate) / double(_SamplingRate))
			).Evaluate();
		);

	_D_Dragonian_Lib_Rethrow_Block(
		Inputs.Emplace(
			CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				AudioCont,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch/Channel", L"Channel/Batch", L"SampleCount" },
				"Audio",
				GetLoggerPtr()
			)
		);
	);

	if (_MyInputCount == 2)
	{
		auto MaskCont = _Mask->get().Continuous().Evaluate();
		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					MaskCont,
					_MyInputTypes[1],
					_MyInputDims[1],
					{ L"Batch/Channel", L"Batch/Channel", L"SampleCount" },
					"Mask",
					GetLoggerPtr()
				)
			);
		);
	}

	std::vector<Ort::Value> Outputs;

	_D_Dragonian_Lib_Rethrow_Block(Outputs = RunModel(Inputs); );

	const auto OutputShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const auto OutputRank = OutputShape.size();
	Dimensions<4> UnitShape;
	if (OutputRank == 4)
		UnitShape = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };
	else if (OutputRank == 3)
		UnitShape = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputRank == 2)
		UnitShape = { 1, 1, OutputShape[0], OutputShape[1] };
	else
		_D_Dragonian_Lib_Throw_Exception("Invalid output dims");

#ifdef _DEBUG
	LogInfo(
		L"Units Encoder Forward Inference With Audio Shape: [" +
		std::to_wstring(AudioCont.Shape(0)) + L", " +
		std::to_wstring(AudioCont.Shape(1)) + L", " +
		std::to_wstring(AudioCont.Shape(2)) + L"], Cost Time: " +
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
		UnitShape
	););
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