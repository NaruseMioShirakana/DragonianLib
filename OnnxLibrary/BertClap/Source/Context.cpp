#include "OnnxLibrary/BertClap/Context.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Bert_Clap_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::ContextModel",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

ContextModel::ContextModel(
	const OnnxRuntimeEnvironment& _Environment,
	const std::wstring& _Path,
	const std::shared_ptr<Logger>& _Logger
) : OnnxModelBase(_Environment, _Path, _Logger)
{
	if (_MyInputCount < 1 || _MyInputCount > 3)
		_D_Dragonian_Lib_Throw_Exception("Input count must be 1~3!");
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Output count must be 1!");
}

Tensor<Float32, 3, Device::CPU> ContextModel::Forward(
	const Tensor<Int64, 2, Device::CPU>& TokenIds,
	std::optional<Tensor<Int64, 2, Device::CPU>> TokenTypeIds,
	std::optional<Tensor<Int64, 2, Device::CPU>> AttentionMask,
	std::optional<Tensor<Int64, 2, Device::CPU>> Aligment
) const
{
	if (TokenIds.Null())
		_D_Dragonian_Lib_Throw_Exception("TokenIds could not be null!");

	if (Aligment.has_value() && Aligment->HasValue() && Aligment->Size(0) != TokenIds.Size(0))
		_D_Dragonian_Lib_Throw_Exception("Batch size mis match!");

	if (_MyInputCount > 1)
	{
		if (!TokenTypeIds.has_value() || TokenTypeIds->Null())
			_D_Dragonian_Lib_Throw_Exception("TokenTypeIds could not be null!");

		if (TokenIds.Shape(0) != TokenTypeIds->Shape(0) ||
			TokenIds.Shape(1) != TokenTypeIds->Shape(1))
			_D_Dragonian_Lib_Throw_Exception("TokenIds and TokenTypeIds shape mismatch!");
	}
	
	if (_MyInputCount == 3)
	{
		if (!AttentionMask.has_value() || AttentionMask->Null())
			_D_Dragonian_Lib_Throw_Exception("AttentionMask could not be null!");
		if (AttentionMask.value().Shape(0) != TokenIds.Shape(0) ||
			AttentionMask.value().Shape(1) != TokenIds.Shape(1))
			_D_Dragonian_Lib_Throw_Exception("AttentionMask shape mismatch!");
	}

	InputTensorsType Inputs;

	_D_Dragonian_Lib_Rethrow_Block(
		Inputs.Emplace(
			CheckAndTryCreateValueFromTensor(
				*GetMemoryInfo(),
				TokenIds,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"BatchSize", L"TokenLength" },
				"TokenIds",
				GetLoggerPtr()
			)
		);
	);

	auto Axis = 1;

	if (_MyInputCount == 3)
	{
		++Axis;
		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					AttentionMask.value(),
					_MyInputTypes[1],
					_MyInputDims[1],
					{ L"BatchSize", L"TokenLength" },
					"AttentionMask",
					GetLoggerPtr()
				)
			);
		);
	}

	if (_MyInputCount > 1)
		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					*TokenTypeIds,
					_MyInputTypes[Axis],
					_MyInputDims[Axis],
					{ L"BatchSize", L"TokenLength" },
					"TokenTypeIds",
					GetLoggerPtr()
				)
			);
		);

	OrtTuple OutputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputTensors = RunModel(
			Inputs
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

	auto Raw = CreateTensorViewFromOrtValue<Float>(
		std::move(OutputTensors[0]),
		Shape
	);

	if (!(Aligment.has_value() && Aligment->HasValue()))
		return Raw;

	auto New = Functional::Empty(
		Dimensions{ Raw.Size(0), Aligment->Size(1), Raw.Size(2) }
	);

	auto AligCont = Aligment->Contiguous();
	auto AligData = AligCont.Data();
	auto NewData = New.Data();
	auto RawData = Raw.Data();

	auto [Batch, TokenSize, BertSize] = New.Size().RawArray();
	for (SizeType b = 0; b < Batch; ++b)
	{
		auto CurAligData = AligData + b * AligCont.Stride(0);
		auto CurNewData = NewData + b * New.Stride(0);
		auto CurRawData = RawData + b * Raw.Stride(0);
		New.AppendTask([TokenSize, BertSize, CurNewData, CurRawData, CurAligData]
		(std::shared_ptr<void>, std::shared_ptr<void>)  // NOLINT(performance-unnecessary-value-param)
			{
				for (SizeType t = 0; t < TokenSize; ++t)
				{
					memcpy(
						CurNewData + t * BertSize,
						CurRawData + CurAligData[t] * BertSize,
						BertSize * sizeof(float)
					);
				}
			},
			Raw.Buffer(),
			AligCont.Buffer()
		);
	}

	return New;
}

_D_Dragonian_Lib_Lib_Bert_Clap_End