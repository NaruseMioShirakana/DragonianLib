#include "../MoeSuperResolution.hpp"
#include "Base.h"

_D_Dragonian_Lib_TRT_Sr_Space_Header

ImageVideo::Image& MoeSR::Infer(ImageVideo::Image& _Image, int64_t _BatchSize)
{
	UNUSED(_BatchSize);
	_BatchSize = 1;
	size_t progress = 0;

	auto InputWidth = _Image.GetWidth();
	auto InputHeight = _Image.GetHeight();
	const auto pixCount = InputWidth * InputHeight;
	const auto dataSize = pixCount * 3ull;
	const auto TotalScale = (size_t)uint32_t(ScaleFactor * ScaleFactor);

	_Image.Transpose();

	auto& imgRGB = _Image.data.rgb;
	auto& imgAlpha = _Image.data.alpha;
	DragonianLibSTL::Vector<float> OutRGB;
	OutRGB.Reserve(imgRGB.Size() * TotalScale);

	const size_t progressMax = imgAlpha.Size() / pixCount;
	Callback_(progress, progressMax);

	std::vector<ITensorInfo> InputTensorsInfo;
	InputTensorsInfo.emplace_back(
		nvinfer1::Dims4(_BatchSize, 3, InputHeight, InputWidth),
		Model->GetInputNames()[0],
		dataSize * _BatchSize * sizeof(float),
		nvinfer1::DataType::kFLOAT
	);

	try
	{
		if (!_MySession.IsReady(InputTensorsInfo))
			_MySession = Model->Construct(
				InputTensorsInfo,
				Model->GetOutputNames()
			);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}

	while (progress < progressMax)
	{
		if (progress + _BatchSize > progressMax)
			_BatchSize = int64_t(progressMax - progress);
		if (_BatchSize == 0)
			break;

		try
		{
			_MySession.HostMemoryToDevice(
				0, imgRGB.Data() + (dataSize * progress),
				dataSize * _BatchSize * sizeof(float)
			);
			_MySession.Run();
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}

		std::vector<float> OutData(TotalScale * dataSize);
		_MySession.DeviceMemoryToHost(0, OutData.data(), OutData.size() * sizeof(float));
		OutRGB.Insert(OutRGB.End(), OutData.data(), OutData.data() + OutData.size());
		progress += _BatchSize;
		Callback_(progress, progressMax);
	}

	_Image.data.rgb = std::move(OutRGB);
	imgAlpha.Clear();
	imgAlpha.Resize(imgRGB.Size() / 3, 1.f);
	_Image.Transpose(ScaleFactor);
	return _Image;
}

MoeSR::MoeSR(
	const std::wstring& RGBModel,
	long Scale,
	const TrtConfig& TrtSettings,
	ProgressCallback _Callback
)
{
	Callback_ = std::move(_Callback);
	ScaleFactor = Scale;
	try
	{
		Model = std::make_unique<TrtModel>(
			RGBModel,
			TrtSettings.CacheFile.at(RGBModel),
			TrtSettings.DynaSetting,
			TrtSettings.DLACore,
			TrtSettings.Fallback,
			TrtSettings.EnableFp16,
			TrtSettings.EnableBf16,
			TrtSettings.EnableInt8,
			TrtSettings.VerboseLevel,
			TrtSettings.OptimizationLevel
		);
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception(e.what());
	}
}

MoeSR::~MoeSR() = default;

_D_Dragonian_Lib_TRT_Sr_Space_End