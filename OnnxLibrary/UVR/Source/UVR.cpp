#include "../UVR.hpp"

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
}

Tensor<Float32, 3, Device::CPU> CascadedNet::Preprocess(
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
			FrameCount = StftResult.Shape()[2];

	//[Channels, FrameCount, Bins + 1, 2(Real, Imag)]
	auto Result = Functional::Empty<Complex32>(
		IDim(Channels, FrameCount, _MySetting.Bins + 1)
	);
	Int64 Offset = 0;
	const auto& BandConf = _MySetting.Bands;
	for (auto BandIdx : TemplateLibrary::Ranges(BandConf.size()))
	{
		auto H = BandConf[BandIdx].CropStop - BandConf[BandIdx].CropStart;
		Result[{None, None, { Offset, Offset + H }}].Ignore().TensorAssign(
			StftResults[BandIdx][{None, {None, FrameCount}, { BandConf[BandIdx].CropStart, BandConf[BandIdx].CropStop }}].Ignore()
		);
		Offset += H;
	}
	Result.Evaluate();
	if (_MySetting.PreFilterStart > 0)
	{
		if (BandConf.size() == 1)
		{
			Float32 G = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart, _MySetting.PreFilterStop))
			{
				G -= 1.f / Float32(_MySetting.PreFilterStop - _MySetting.PreFilterStart);
				Result[{None, None, { B, B + 1 }}].Ignore() *= G;
			}
			Result[{None, None, { _MySetting.PreFilterStop, None }}].Ignore() = 0.f;
		}
		else
		{
			Float32 GP = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart + 1, _MySetting.PreFilterStop))
			{
				GP = powf(10.f, -(Float32(B) - Float32(_MySetting.PreFilterStart)) * (3.5f - GP) / 20.f);
				Result[{None, None, { B, B + 1 }}].Ignore() *= GP;
			}
		}
	}

	Result = Result.ATan2().Evaluate();
	auto ResultReal = Result.ViewAs<Float32>().View(IDim(Channels, FrameCount, _MySetting.Bins + 1, 2));
	auto Magnitude = ResultReal[{None, None, None, "0"}].Squeeze(-1).Ignore().Contiguous();
	auto Phase = ResultReal[{None, None, None, "1"}].Squeeze(-1).Ignore().Contiguous();
	Float Coef = *Magnitude.View(-1).ReduceMax(0).Evaluate().Data();
	Magnitude /= Coef;

	const auto PaddingLeft = _MySetting.Offset;
	auto RoiSize = _MySetting.WindowSize - PaddingLeft * 2;
	if (RoiSize == 0)
		RoiSize = _MySetting.WindowSize;
	const auto PaddingRight = RoiSize - (FrameCount % RoiSize) + PaddingLeft;
	//const auto NumWindow = (Int64)ceil(float(FrameCount) / float(RoiSize));
	auto MagnitudePadded = Magnitude.Padding(
		IRanges(None, Range{ PaddingLeft, PaddingRight }, None),
		PaddingType::Zero
	);
	return MagnitudePadded.Transpose().Contiguous().View(IDim(-1, RoiSize, _MySetting.Bins + 1));
}

_D_Dragonian_Lib_Onnx_UVR_End