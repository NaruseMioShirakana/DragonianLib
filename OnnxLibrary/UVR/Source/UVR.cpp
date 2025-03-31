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
			{
				{7350,80,640,0,85,-1,-1,25,53},
				{7350,80,320,4,87,25,12,31,62},
				{14700,160,512,17,216,48,24,139,210},
				{44100,480,960,78,383,130,86,-1,-1}
			},
			44100,
			668,
			672
		};
	}
	if (Name == L"4band_v3")
	{
		return HParams{
			672,
			8,
			530,
			{
				{7350,80,640,0,85,-1,-1,25,53},
				{7350,80,320,4,87,25,12,31,62},
				{14700,160,512,17,216,48,24,139,210},
				{44100,480,960,78,383,130,86,-1,-1}
			},
			44100,
			668,
			672
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
	auto SignalCont = Signal.Contiguous().Evaluate();

	std::vector<Tensor<Complex32, 3, Device::CPU>> StftResults;
	for (Int64 b = 0; b < BandSize; ++b)
	{
		const auto& Band = _MySetting.Bands[b];
		auto SignalView = SignalCont.View();
		if (Band.SamplingRate != SamplingRate)
			_D_Dragonian_Lib_Rethrow_Block(
				SignalView = SignalView.Interpolate<Operators::InterpolateMode::Linear>(
					IDim(-1),
					IScale(double(Band.SamplingRate) / double(SamplingRate))
				);
			);
		StftResults.emplace_back(_MyStftKernels[b].Execute(SignalView.UnSqueeze(0)).Squeeze(0));
	}
	Int64 Length = INT64_MAX;
	for (const auto& StftResult : StftResults)
		if (StftResult.Shape()[2] < Length)
			Length = StftResult.Shape()[2];

	auto Result = Functional::Empty<Complex32>(
		IDim(2, Length, _MySetting.Bins + 1)
	);
	Int64 Offset = 0;
	const auto& BandConf = _MySetting.Bands;
	for (auto BandIdx : TemplateLibrary::Ranges(BandConf.size()))
	{
		auto H = BandConf[BandIdx].CropStop - BandConf[BandIdx].CropStart;
		Result[{None, { Offset, Offset + H }, None}].TensorAssign(
			StftResults[BandIdx][{None, { BandConf[BandIdx].CropStart, BandConf[BandIdx].CropStop }, None}]
		);
		Offset += H;
	}

	if (_MySetting.PreFilterStart > 0)
	{
		if (BandConf.size() == 1)
		{
			Float32 G = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart, _MySetting.PreFilterStop))
			{
				G -= 1.f / Float32(_MySetting.PreFilterStop - _MySetting.PreFilterStart);
				Result[{None, None, { B, B + 1 }}] *= G;
			}
			Result[{None, None, { _MySetting.PreFilterStop, None }}] = 0.f;
		}
		else
		{
			Float32 GP = 1.f;
			for (auto B : TemplateLibrary::Ranges(_MySetting.PreFilterStart + 1, _MySetting.PreFilterStop))
			{
				GP = powf(10.f, -(Float32(B) - Float32(_MySetting.PreFilterStart)) * (3.5f - GP) / 20.f);
				Result[{None, None, { B, B + 1 }}] *= GP;
			}
		}
	}

	Result = Result.ATan2();
	auto ResultReal = Result.ViewAs<Float32>().View(IDim(2, Length, _MySetting.Bins + 1, 2));
	auto Magnitude = ResultReal[{None, None, None, "0"}].Squeeze(-1).Contiguous();
	auto Phase = ResultReal[{None, None, None, "1"}].Squeeze(-1).Contiguous();


}

_D_Dragonian_Lib_Onnx_UVR_End