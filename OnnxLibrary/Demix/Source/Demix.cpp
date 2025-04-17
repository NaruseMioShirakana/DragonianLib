#include "../Demix.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Onnx_Demix_Header

Demix::Demix(
	const std::wstring& ModelPath,
	const OnnxRuntimeEnvironment& Environment,
	const HyperParameters& HParams,
	const DLogger& Logger
) : OnnxModelBase(Environment, ModelPath, Logger), _MySamplingRate(HParams.Demix.SamplingRate),
_MySubBandCount(HParams.Demix.SubBandCount), _MyStftBins(HParams.Demix.StftBins), _ComplexAsChannel(HParams.Demix.ComplexAsChannel),
_MyStftKernel(HParams.Demix.NumStft, HParams.Demix.HopSize, HParams.Demix.WindowSize, HParams.Demix.Center, HParams.Demix.Padding)
{

}

TemplateLibrary::Vector<SignalTensor> Demix::Forward(
	const SignalTensor& Signal,
	const Parameters& Params
) const
{
	//[BatchSize, ChannelCount, SampleCount] -> [BatchSize, ChannelCount, FrameCount, StftBins]

	auto SignalView = Signal.View();

	if (Params.SamplingRate != _MySamplingRate)
		SignalView = SignalView.Interpolate<Operators::InterpolateMode::Linear>(
			IDim(-1),
			IScale(double(_MySamplingRate) / double(Params.SamplingRate))
		);

	auto Spec = _MyStftKernel.Execute(
		SignalView
	);

	auto RealSpec = Spec.ViewAs<Float32>().View(
		Spec.Size(0),
		Spec.Size(1),
		Spec.Size(2),
		Spec.Size(3),
		2
	);
	
	auto Rest = Spec.Size(2);
	SizeType Offset = 0;
	const auto HopLength = Params.SegmentSize / 2;
	const auto Remainder = Rest % HopLength;
	const auto Padding = Remainder ? 2 * Params.SegmentSize - Remainder : Params.SegmentSize;
	Rest += Padding;
	RealSpec = RealSpec.Pad(
		PaddingCounts{
			NPAD,
			NPAD,
			PadCount{ 0, Padding }
		},
		PaddingType::Zero
	);

	std::vector<Tensor<Complex32, 5, Device::CPU>> Ret;
	if (Params.Progress)
		Params.Progress(true, Rest);

	while (Rest >= Params.SegmentSize)
	{
		auto Segment = RealSpec.Slice(
			{
				None, None,
				{ Offset, Offset + Params.SegmentSize },
				None, None
			}
		).Contiguous().Evaluate();
		_D_Dragonian_Lib_Rethrow_Block(
			Ret.emplace_back(
				ExecuteModel(
					Segment,
					Params
				)
			);
		);
		Rest -= HopLength;
		Offset += HopLength;
		if (Params.Progress)
			Params.Progress(false, Offset);
	}
	auto Tensor = Functional::ICat(
		Ret,
		-2
	);
	Tensor = Tensor.ReversedSlice({ None, { 0, Spec.Size(-2) } });

	const auto OSize = Tensor.Size(0);
	TemplateLibrary::Vector<SignalTensor> Result;
	for (SizeType i = 0; i < OSize; ++i)
		Result.EmplaceBack(_MyStftKernel.Inverse(Tensor[i]));
	return Result;
}

Tensor<Complex32, 5, Device::CPU> Demix::ExecuteModel(
	const Tensor<Float32, 5, Device::CPU>& RealSpec,
	const Parameters& Params
) const
{
	InputTensorsType InputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		InputTensors.Emplace(
			CheckAndTryCreateValueFromTensor(
				*GetMemoryInfo(),
				RealSpec,
				_MyInputTypes[0],
				_MyInputDims[0],
				{ L"Batch", L"Channel", L"Bins", L"Frames", L"Complex" },
				"Spec",
				GetLoggerPtr()
			)
		);
	);

	OrtTuple OutputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputTensors = RunModel(InputTensors);
	);

	const auto OShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
	Dimensions<6> Shape;
	Shape.Assign(OShape.data());

	_D_Dragonian_Lib_Rethrow_Block(
		return CreateTensorViewFromOrtValue<Float>(
			std::move(OutputTensors[0]),
			Shape
		).ViewAs<Complex32>().Squeeze(-1);
	);
}


_D_Dragonian_Lib_Onnx_Demix_End