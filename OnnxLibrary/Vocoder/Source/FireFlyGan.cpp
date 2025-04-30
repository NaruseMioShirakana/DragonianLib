#include "OnnxLibrary/Vocoder/FireFlyGan.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Onnx_Vocoder_Header

namespace FireflyArchitecture
{

	Encoder::Encoder(
		const OnnxRuntimeEnvironment& _Environment,
		const std::wstring& _ModelPath,
		const std::shared_ptr<Logger>& _Logger,
		int64_t SamplingRate,
		int64_t NumCodebooks
	) : OnnxModelBase(_Environment, _ModelPath, _Logger), _MySampleingRate(SamplingRate), _MyNumCodebooks(NumCodebooks)
	{

	}

	Tensor<Int64, 4, Device::CPU> Encoder::Forward(
		const Tensor<Float32, 3, Device::CPU>& _Audio
	) const
	{
		if (_Audio.Null())
			_D_Dragonian_Lib_Throw_Exception("Input tensor is null.");
		InputTensorsType InputTensors;

		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					_Audio,
					_MyInputTypes[0],
					_MyInputDims[0],
					{ L"Batch/Channel", L"Channel/Batch", L"SampleCount" },
					"Audio",
					GetLoggerPtr()
				)
			);
		);

		OrtTuple OutputTensors;

		_D_Dragonian_Lib_Rethrow_Block(
			OutputTensors = RunModel(
				InputTensors
			);
		);

		Dimensions<4> Shape;
		auto OShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
		if (OShape.size() == 4)
			Shape = { OShape[0], OShape[1], OShape[2], OShape[3] };
		else if (OShape.size() == 3)
			Shape = { 1, OShape[0], OShape[1], OShape[2] };
		else if (OShape.size() == 2)
			Shape = { 1, 1, OShape[0], OShape[1] };
		else if (OShape.size() == 1)
			Shape = { 1, 1, 1, OShape[0] };
		else
			_D_Dragonian_Lib_Throw_Exception("Output shape is invalid.");

		_D_Dragonian_Lib_Rethrow_Block(
			return CreateTensorViewFromOrtValue<Int64>(
				std::move(OutputTensors[0]),
				Shape
			);
		);
	}

	Decoder::Decoder(
		const OnnxRuntimeEnvironment& _Environment,
		const std::wstring& _ModelPath,
		const std::shared_ptr<Logger>& _Logger,
		int64_t SamplingRate,
		int64_t NumCodebooks
	) : OnnxModelBase(_Environment, _ModelPath, _Logger), _MySampleingRate(SamplingRate), _MyNumCodebooks(NumCodebooks)
	{
		
	}

	Tensor<Float32, 3, Device::CPU> Decoder::Forward(
		const Tensor<Int64, 4, Device::CPU>& _Indices
	) const
	{
		if (_Indices.Null())
			_D_Dragonian_Lib_Throw_Exception("Input tensor is null.");
		InputTensorsType InputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			InputTensors.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					_Indices,
					_MyInputTypes[0],
					_MyInputDims[0],
					{ L"Batch/Channel", L"Channel/Batch", L"Codebook", L"SampleCount" },
					"Indices",
					GetLoggerPtr()
				)
			);
		);
		OrtTuple OutputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			OutputTensors = RunModel(
				InputTensors
			);
		);
		Dimensions<3> Shape;
		auto OShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
		if (OShape.size() == 3)
			Shape = { OShape[0], OShape[1], OShape[2] };
		else if (OShape.size() == 2)
			Shape = { 1, OShape[0], OShape[1] };
		else if (OShape.size() == 1)
			Shape = { 1, 1, OShape[0] };
		else
			_D_Dragonian_Lib_Throw_Exception("Output shape is invalid.");
		_D_Dragonian_Lib_Rethrow_Block(
			return CreateTensorViewFromOrtValue<Float32>(
				std::move(OutputTensors[0]),
				Shape
			);
		);
	}


}


_D_Dragonian_Lib_Onnx_Vocoder_End