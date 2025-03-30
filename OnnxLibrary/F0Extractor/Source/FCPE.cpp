#include "../FCPE.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_F0_Extractor_Header

FCPE::FCPE(
	const void* _ModelHParams
) : OnnxModelBase(*(const OnnxRuntime::OnnxRuntimeEnvironment*)((const PEModelHParams*)_ModelHParams)->Enviroment,
	((const PEModelHParams*)_ModelHParams)->ModelPath,
	*(const DragonianLib::DLogger*)((const PEModelHParams*)_ModelHParams)->Logger),
	_MySamplingRate(((const PEModelHParams*)_ModelHParams)->SamplingRate)
{

}

Tensor<Float32, 2, Device::CPU> FCPE::ExtractF0(
	const Tensor<Float32, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
) const
{
	if (PCMData.Null())
		_D_Dragonian_Lib_Throw_Exception("PCMData is Null!");
	if (Params.HopSize < 80)
		_D_Dragonian_Lib_Throw_Exception("HopSize Too Low!");

	auto AudioCont = PCMData.View();
	if (Params.SamplingRate != _MySamplingRate)
		AudioCont.Interpolate<Operators::InterpolateMode::Linear>(
			IDim(1),
			IScale(double(_MySamplingRate) / double(Params.SamplingRate))
		).Evaluate();

	OnnxRuntime::InputTensorsType Inputs;

	_D_Dragonian_Lib_Rethrow_Block(
		Inputs.Emplace(
			OnnxRuntime::CheckAndTryCreateValueFromTensor(
				*_MyMemoryInfo,
				AudioCont,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch", L"SampleCount" },
				"Audio",
				GetLoggerPtr()
			)
		);
	);

	constexpr Int64 TSShape = 1;
	Float Threshold = Params.Threshold;
	if (_MyInputCount == 2)
	{
		Inputs.Emplace(
			Ort::Value::CreateTensor(
				*_MyMemoryInfo,
				&Threshold,
				1,
				&TSShape,
				1
			)
		);
	}

	OnnxRuntime::OrtTuple Outputs;

	_D_Dragonian_Lib_Rethrow_Block(
		Outputs = RunModel(
			Inputs
		);
	);

	auto OutShape = Outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	Dimensions<2> OutDims;
	if (OutShape.size() == 2)
		OutDims = { OutShape[0], OutShape[1]};
	else if (OutShape.size() == 1)
		OutDims = { 1, OutShape[0] };
	else
		_D_Dragonian_Lib_Throw_Exception("Output Shape Error!");

	auto Output = OnnxRuntime::CreateTensorViewFromOrtValue<Float>(
		std::move(Outputs[0]),
		OutDims
	);

	auto InputCount = PCMData.Size(1);
	auto TargetLength = (Int64)ceil(double(InputCount) / double(Params.HopSize));
	if (Output.Size(1) != TargetLength)
		_D_Dragonian_Lib_Rethrow_Block(
			Output = Output.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(1),
				IScale(double(TargetLength) / double(Output.Size(1)))
			).Evaluate();
		);

	return Output;
}

Tensor<Float32, 2, Device::CPU> FCPE::ExtractF0(
	const Tensor<Float64, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
) const
{
	return ExtractF0(PCMData.Cast<Float32>(), Params);
}

Tensor<Float32, 2, Device::CPU> FCPE::ExtractF0(
	const Tensor<Int16, 2, Device::CPU>& PCMData,
	const F0ExtractorParams& Params
) const
{
	return ExtractF0(PCMData.Cast<Float32>(), Params);
}

_D_Dragonian_Lib_F0_Extractor_End