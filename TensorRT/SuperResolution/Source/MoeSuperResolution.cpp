#include "TensorRT/SuperResolution/MoeSuperResolution.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

_D_Dragonian_Lib_TRT_Sr_Space_Header

std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> MoeSR::Infer(
	const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Bitmap
)
{
	//constexpr auto _BatchSize = 1ll;
	auto& [Bitmap, SourceHeight, SourceWidth] = _Bitmap;
	auto RGBChannel = Bitmap.ReversedSlice({ "0:3" }).Permute(0, 1, 4, 2, 3).Contiguous();
	auto AlphaChannel = Bitmap.ReversedSlice(
		{ "3" }
	).Ignore().Interpolate<Operators::InterpolateMode::Bilinear>(
		Dimensions{ 2ll, 3ll },
		IScale(double(_MyScaleH), double(_MyScaleW))
	);

	RGBChannel.Evaluate();
	const auto BatchCount = RGBChannel.Size(0) * RGBChannel.Size(1);
	const auto RGBData = RGBChannel.Data();
	const auto WindowHeight = RGBChannel.Size(3);
	const auto WindowWidth = RGBChannel.Size(4);
	const auto BatchStride = WindowHeight * WindowWidth * 3;
	auto Ret = ImageVideo::NormalizedImage5D::New({
			RGBChannel.Size(0),
			RGBChannel.Size(1),
			3,
			WindowHeight * _MyScaleH,
			WindowWidth * _MyScaleW,
		});
	auto RetData = Ret.Data();
	if (_MyCallback)
		_MyCallback(true, BatchCount);

	std::vector<ITensorInfo> InputTensorsInfo;
	InputTensorsInfo.emplace_back(
		nvinfer1::Dims4(1, 3, WindowHeight, WindowWidth),
		Model->GetInputNames()[0],
		BatchStride * sizeof(float),
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

	for (SizeType Batch = 0; Batch < BatchCount; ++Batch)
	{
		const auto CurData = RGBData + Batch * BatchStride;
		const auto InputSize = BatchStride;
		const auto OutputSize = InputSize * _MyScaleH * _MyScaleW;

		try
		{
			_MySession.HostMemoryToDevice(
				0, CurData,
				InputSize * sizeof(float)
			);
			_MySession.Run();
			_MySession.DeviceMemoryToHost(
				0, RetData,
				OutputSize * sizeof(float)
			);
		}
		catch (std::exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception(e.what());
		}

		RetData += OutputSize;
		if (_MyCallback)
			_MyCallback(false, Batch);
	}
	return std::make_tuple(
		Functional::Cat(
			Ret.Permute(0, 1, 3, 4, 2),
			AlphaChannel,
			-1
		).Evaluate(),
		SourceHeight * _MyScaleH,
		SourceWidth * _MyScaleW
	);
}

MoeSR::MoeSR(
	const std::wstring& RGBModel,
	Int64 Scale,
	const TrtConfig& TrtSettings,
	ProgressCallback _Callback
)
{
	_MyCallback = std::move(_Callback);
	_MyScaleH = _MyScaleW = Scale;
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