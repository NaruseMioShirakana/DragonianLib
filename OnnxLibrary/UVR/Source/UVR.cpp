#include "../UVR.hpp"

#include <iostream>

#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Onnx_UVR_Header

DLogger& GetDefaultLogger() noexcept
{
	static DLogger _MyLogger = std::make_shared<Logger>(
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerId() + L"::UVR",
		_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger()->GetLoggerLevel(),
		nullptr
	);
	return _MyLogger;
}

CascadedNet::HParams CascadedNet::GetPreDefinedHParams(
	const std::wstring& Name
)
{
	if (Name == L"4band_v2")
	{
		return HParams{
			672,
			8,
			637,
			512,
			{
				{7350,80,640,0,85,-1,-1,25,53},
				{7350,80,320,4,87,25,12,31,62},
				{14700,160,512,17,216,48,24,139,210},
				{44100,480,960,78,383,130,86,-1,-1}
			},
			44100,
			668,
			672,
			128
		};
	}
	if (Name == L"4band_v3")
	{
		return HParams{
			672,
			8,
			530,
			512,
			{
				{7350,80,640,0,85,-1,-1,25,53},
				{7350,80,320,4,87,25,12,31,62},
				{14700,160,512,17,216,48,24,139,210},
				{44100,480,960,78,383,130,86,-1,-1}
			},
			44100,
			668,
			672,
			64
		};
	}
	_D_Dragonian_Lib_Throw_Exception("Unknown HParams.");
}

CascadedNet::CascadedNet(
	const std::wstring& ModelPath,
	const OnnxRuntimeEnvironment& Environment,
	HParams Setting,
	const DLogger& Logger
) : OnnxModelBase(Environment, ModelPath, Logger), _MySetting(std::move(Setting))
{
	_MyStftKernels.reserve(_MySetting.Bands.size());
	for (const auto& Band : _MySetting.Bands)
		_MyStftKernels.emplace_back(
			static_cast<Int>(Band.FFTSize),
			static_cast<Int>(Band.HopSize),
			static_cast<Int>(Band.FFTSize)
		);
	_MyPaddingLeft = _MySetting.Offset;
	_MyRoiSize = _MySetting.WindowSize - _MyPaddingLeft * 2;
	if (_MyRoiSize == 0)
		_MyRoiSize = _MySetting.WindowSize;
}

void HighPass(
	const Tensor<Complex32, 3, Device::CPU>& _Signal,
	Int64 BinStart,
	Int64 BinStop
)
{
	auto G = 1.f;
	for (auto B : TemplateLibrary::Ranges(BinStart, BinStop))
	{
		G -= 1.f / Float32(BinStart - BinStop);
		_Signal[{None, B}].Ignore() *= G;
	}
	_Signal[{None, { None, BinStop + 1 }}].Ignore() = 0.f;
}

void LowPass(
	const Tensor<Complex32, 3, Device::CPU>& _Signal,
	Int64 BinStart,
	Int64 BinStop
)
{
	auto G = 1.f;
	for (auto B : TemplateLibrary::Ranges(BinStart, BinStop))
	{
		G -= 1.f / Float32(BinStop - BinStart);
		_Signal[{None, B}].Ignore() *= G;
	}
	_Signal[{None, { BinStop, None }}].Ignore() = 0.f;
}

std::tuple<FltTensor, Cpx32Tensor, Cpx32Tensor, Cpx32Tensor, Int64, Int64, Float32, Int64> CascadedNet::Preprocess(
	const Tensor<Float32, 2, Device::CPU>& Signal,
	Int64 SamplingRate
) const
{
	const auto BandSize = static_cast<Int64>(_MySetting.Bands.size());
	const auto Channels = Signal.Shape()[0];
	auto SignalCont = Signal.Contiguous().Evaluate();

	std::vector<Tensor<Complex32, 3, Device::CPU>> StftResults;
	for (Int64 b = 0; b < BandSize; ++b)
	{
		const auto& Band = _MySetting.Bands[b];
		auto SignalView = SignalCont.Detach();
		if (Band.SamplingRate != SamplingRate)
			_D_Dragonian_Lib_Rethrow_Block(
				SignalView = SignalView.Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					IScale(double(Band.SamplingRate) / double(SamplingRate))
				);
			);
		StftResults.emplace_back(_MyStftKernels[b].Execute(SignalView.UnSqueeze(0)).Squeeze(0));
	}
	//[Channels, FrameCount, FFTSize, 2(Real, Imag)]
	Int64 FrameCount = INT64_MAX;
	for (const auto& StftResult : StftResults)
		if (StftResult.Shape(1) < FrameCount)
			FrameCount = StftResult.Shape()[1];

	//[Channels, FrameCount, Bins + 1, 2(Real, Imag)]
	auto ComplexSpec = Functional::Zeros<Complex32>(
		IDim(Channels, FrameCount, _MySetting.Bins + 1)
	).Evaluate();
	Int64 Offset = 0;
	const auto& BandConf = _MySetting.Bands;
	for (auto BandIdx : TemplateLibrary::Ranges(BandConf.size()))
	{
		auto H = BandConf[BandIdx].CropStop - BandConf[BandIdx].CropStart;
		ComplexSpec[{None, None, { Offset, Offset + H }}].Ignore().TensorAssign(
			StftResults[BandIdx][{None, {None, FrameCount}, { BandConf[BandIdx].CropStart, BandConf[BandIdx].CropStop }}].Ignore()
		);
		Offset += H;
	}
	ComplexSpec.Evaluate();
	if (_MySetting.PreFilterStart > 0)
	{
		if (BandConf.size() == 1)
		{
			Float32 G = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart, _MySetting.PreFilterStop))
			{
				G -= 1.f / Float32(_MySetting.PreFilterStop - _MySetting.PreFilterStart);
				ComplexSpec[{None, None, B }].Ignore() *= G;
			}
			ComplexSpec[{None, None, { _MySetting.PreFilterStop, None }}].Ignore() = 0.f;
		}
		else
		{
			Float32 GP = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart + 1, _MySetting.PreFilterStop))
			{
				GP = powf(10.f, -(Float32(B) - Float32(_MySetting.PreFilterStart)) * (3.5f - GP) / 20.f);
				ComplexSpec[{None, None, B }].Ignore() *= GP;
			}
		}
	}

	const auto PolarComplexSpec = ComplexSpec.ATan2().Evaluate();
	auto Magnitude = PolarComplexSpec.Real().Ignore().Clone().Evaluate();
	auto Phase = PolarComplexSpec.Imag().Ignore().Clone().Evaluate();
	auto Coef = Magnitude.View(-1).ReduceMax(0).Evaluate().Item();
	Magnitude /= Coef;

	const auto PaddingRight = _MyRoiSize - (FrameCount % _MyRoiSize) + _MyPaddingLeft;
	const auto NumWindow = (Int64)ceil(float(FrameCount) / float(_MyRoiSize));
	auto MagnitudePadded = Magnitude.Padding(
		{ None, Range{ _MyPaddingLeft, PaddingRight }, None },
		PaddingType::Zero
	);

	const auto InputHighEndH = (BandConf.back().FFTSize / 2 - BandConf.back().CropStop) + 
		(_MySetting.PreFilterStop - _MySetting.PreFilterStart);
	auto InputHighEnd = StftResults.back()[{
		None, None, { BandConf.back().FFTSize / 2 - InputHighEndH, BandConf.back().FFTSize / 2 }
	}];

	return {
		MagnitudePadded.Transpose().Contiguous(),
		(Phase.Transpose() * Complex32(0, 1)).Exp(),
		ComplexSpec.Transpose().Contiguous(),
		InputHighEnd.Transpose().Contiguous(),
		NumWindow, FrameCount, Coef, InputHighEndH
	};
}

FltTensor CascadedNet::Spec2Audio(
	const Tensor<Complex32, 3, Device::CPU>& Spec,
	const Tensor<Complex32, 3, Device::CPU>& InputHighEnd,
	Int64 InputHighEndH
) const
{
	auto Mirror = Spec[{
		None,
		{
			_MySetting.PreFilterStart - 10 - InputHighEnd.Size(1),
			_MySetting.PreFilterStart - 10
		}
	}].Reverse(1).Abs();
	Mirror *= (InputHighEnd.ATan2().Imag() * Complex32(0, 1)).Exp().Evaluate();
	auto Mask = InputHighEnd.Abs().Real() <= Mirror.Abs().Real();
	auto HighEnd = Mirror.Clone();
	HighEnd.MaskedFill(Mask, InputHighEnd).Evaluate();

	Int64 Offset = 0;
	const auto BandSize = static_cast<Int64>(_MySetting.Bands.size());
	Tensor<Float32, 3, Device::CPU> Audio;
	for (auto BandIdx : TemplateLibrary::Ranges(BandSize))
	{
		const auto& Band = _MySetting.Bands[BandIdx];
		const auto H = Band.CropStop - Band.CropStart;
		auto BSpec = Functional::Ones<Complex32>(
			IDim(2, Band.FFTSize / 2 + 1, Spec.Shape(2))
		).Evaluate();
		BSpec[{None, { Band.CropStart, Band.CropStop }}].Ignore().TensorAssign(
			Spec[{None, { Offset, Offset + H }}].Ignore()
		).Evaluate();
		Offset += H;

		if (BandIdx == BandSize - 1)
		{
			if (InputHighEndH)
			{
				const auto MaxBin = Band.FFTSize / 2;
				BSpec[{None, { MaxBin - InputHighEndH, MaxBin }}].Ignore().TensorAssign(
					HighEnd[{None, { None, InputHighEndH }}].Ignore()
				).Evaluate();
			}
			if (Band.HpfStart > 0)
				HighPass(BSpec, Band.HpfStart, Band.HpfStop - 1);
			if (BandSize == 1)
			{
				Audio = FunctionTransform::StftKernel::Inverse(
					BSpec.UnSqueeze(0).Transpose(),
					Band.HopSize
				);
			}
			else
			{
				auto Temp = FunctionTransform::StftKernel::Inverse(
					BSpec.UnSqueeze(0).Transpose(),
					Band.HopSize
				);
				Audio = Audio.Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					IDim(Temp.Size(-1))
				) + Temp;
			}
		}
		else
		{
			if (BandIdx == 0)
			{
				LowPass(BSpec, Band.LpfStart, Band.LpfStop);
				Audio = FunctionTransform::StftKernel::Inverse(
					BSpec.UnSqueeze(0).Transpose(),
					Band.HopSize
				);
				if (_MySetting.Bands[BandIdx + 1].SamplingRate != Band.SamplingRate)
					Audio = Audio.Interpolate<Operators::InterpolateMode::Linear>(
						IDim(-1),
						IScale(double(_MySetting.Bands[BandIdx + 1].SamplingRate) / double(Band.SamplingRate))
					);
			}
			else
			{
				HighPass(BSpec, Band.HpfStart, Band.HpfStop - 1);
				LowPass(BSpec, Band.LpfStart, Band.LpfStop);
				auto Temp = FunctionTransform::StftKernel::Inverse(
					BSpec.UnSqueeze(0).Transpose(),
					Band.HopSize
				);
				if (Audio.Size(-1) != Temp.Size(-1))
					Audio = Audio.Interpolate<Operators::InterpolateMode::Linear>(
						IDim(-1),
						IDim(Temp.Size(-1))
					) + Temp;
			}
		}
	}
	return Audio.Evaluate();
}

std::pair<FltTensor, FltTensor> CascadedNet::Forward(
	const Tensor<Float32, 2, Device::CPU>& Signal,
	Int64 SplitBin,
	Float32 Value,
	Int64 SamplingRate
) const
{
	auto [
		Mangitude, Phase,
			XSpecM, InputHighEnd,
			NumWindow, Frames, Coef, InputHighEndH
	] = Preprocess(Signal, SamplingRate);
	Mangitude.Evaluate();
	Phase.Evaluate();
	XSpecM.Evaluate();
	InputHighEnd.Evaluate();

	constexpr Int64 One = 1;
	const Int64 InputShape[4]{ 1, 2, _MySetting.Bins + 1, _MySetting.WindowSize };
	const auto OutShape = Dimensions{ 1, 2, _MySetting.Bins + 1, _MyRoiSize };
	Ort::Value Inputs[3]{
		Ort::Value{nullptr},
		Ort::Value::CreateTensor(*_MyMemoryInfo, &SplitBin, 1, &One, 1),
		Ort::Value::CreateTensor(*_MyMemoryInfo, &Value, 1, &One, 1)
	};
	std::vector<Tensor<Float32, 4, Device::CPU>> OutputTensors;
	for (Int64 i = 0; i < NumWindow; ++i)
	{
		const auto Start = i * _MyRoiSize;
		auto MagWin = Mangitude[{None, None, Range{ Start, Start + _MySetting.WindowSize }}].Contiguous().Evaluate();
		Inputs[0] = Ort::Value::CreateTensor(
			*_MyMemoryInfo, MagWin.Data(), MagWin.ElementCount(), InputShape, 4
		);
		OrtTuple Outputs;
		_D_Dragonian_Lib_Rethrow_Block(Outputs = RunModel(Inputs););
		_D_Dragonian_Lib_Rethrow_Block(
			OutputTensors.emplace_back(
				CreateTensorViewFromOrtValue<Float32>(std::move(Outputs[0]), OutShape)
			);
		);
	}
	auto Pred =
		(Functional::ICat(OutputTensors, -1).Evaluate().Squeeze(0) * Coef)[{None, None, { None, Frames }}];

	auto YSpec = (Pred * Phase).Evaluate();
	auto VSpec = (XSpecM - YSpec).Evaluate();

	return { Spec2Audio(VSpec, InputHighEnd, InputHighEndH), Spec2Audio(YSpec, InputHighEnd, InputHighEndH) };
}

_D_Dragonian_Lib_Onnx_UVR_End