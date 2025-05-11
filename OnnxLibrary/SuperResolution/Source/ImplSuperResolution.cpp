#include "OnnxLibrary/SuperResolution/ImplSuperResolution.hpp"

_D_Dragonian_Lib_Lib_Super_Resolution_Header

SuperResolutionBase::SuperResolutionBase(
	const OnnxRuntimeEnvironment& _Environment,
	const HyperParameters& _Parameters,
	const DLogger& _Logger
) : SuperResolution(_Parameters), OnnxModelBase(_Environment, _Parameters.RGBModel, _Logger)
{

}

std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> SuperResolutionBCRGBHW::Infer(
	const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
	int64_t _BatchSize
) const
{
	_BatchSize = std::max(_BatchSize, 1ll);
	auto& [Bitmap, SourceHeight, SourceWidth] = _Image;
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

	for (SizeType Batch = 0; Batch < BatchCount; Batch += _BatchSize)
	{
		const auto CurBatchSize = std::min(_BatchSize, BatchCount - Batch);
		const auto CurData = RGBData + Batch * BatchStride;

		Int64 Shape[4] = { CurBatchSize, 3, WindowHeight, WindowWidth };
		const auto InputSize = CurBatchSize * BatchStride;
		OrtTuple Tensors, Outputs;
		Tensors.emplace_back(
			Ort::Value::CreateTensor(
				*GetMemoryInfo(),
				CurData,
				InputSize,
				Shape,
				4
			)
		);

		_D_Dragonian_Lib_Rethrow_Block(
			Outputs = RunModel(Tensors);
		);

		const auto OutputCount = Outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto OutputData = Outputs[0].GetTensorData<float>();
		if (std::cmp_not_equal(OutputCount, _MyScaleH * _MyScaleW * InputSize))
			_D_Dragonian_Lib_Throw_Exception("Scale mismatch");
		memcpy(RetData, OutputData, OutputCount * sizeof(float));
		RetData += OutputCount;
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

std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> SuperResolutionBHWCRGB::Infer(
	const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
	int64_t _BatchSize
) const
{
	_BatchSize = std::max(_BatchSize, 1ll);
	auto& [Bitmap, SourceHeight, SourceWidth] = _Image;
	auto RGBChannel = Bitmap.ReversedSlice({ "0:3" }).Contiguous();
	auto AlphaChannel = Bitmap.ReversedSlice(
		{ "3" }
	).Ignore().Interpolate<Operators::InterpolateMode::Bilinear>(
		Dimensions{ 2ll, 3ll },
		IScale(double(_MyScaleH), double(_MyScaleW))
	);

	RGBChannel.Evaluate();
	const auto BatchCount = RGBChannel.Size(0) * RGBChannel.Size(1);
	const auto RGBData = RGBChannel.Data();
	const auto WindowHeight = RGBChannel.Size(2);
	const auto WindowWidth = RGBChannel.Size(3);
	const auto BatchStride = WindowHeight * WindowWidth * 3;
	auto Ret = ImageVideo::NormalizedImage5D::New({
			RGBChannel.Size(0),
			RGBChannel.Size(1),
			WindowHeight * _MyScaleH,
			WindowWidth * _MyScaleW,
			3
		});
	auto RetData = Ret.Data();
	if (_MyCallback)
		_MyCallback(true, BatchCount);

	for (SizeType Batch = 0; Batch < BatchCount; Batch += _BatchSize)
	{
		const auto CurBatchSize = std::min(_BatchSize, BatchCount - Batch);
		const auto CurData = RGBData + Batch * BatchStride;

		Int64 Shape[4] = { CurBatchSize, WindowHeight, WindowWidth, 3 };
		const auto InputSize = CurBatchSize * BatchStride;
		OrtTuple Tensors, Outputs;
		Tensors.emplace_back(
			Ort::Value::CreateTensor(
				*GetMemoryInfo(),
				CurData,
				InputSize,
				Shape,
				4
			)
		);

		_D_Dragonian_Lib_Rethrow_Block(
			Outputs = RunModel(Tensors);
		);

		const auto OutputCount = Outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto OutputData = Outputs[0].GetTensorData<float>();
		if (std::cmp_not_equal(OutputCount, _MyScaleH * _MyScaleW * InputSize))
			_D_Dragonian_Lib_Throw_Exception("Scale mismatch");
		memcpy(RetData, OutputData, OutputCount * sizeof(float));
		RetData += OutputCount;
		if (_MyCallback)
			_MyCallback(false, Batch);
	}
	return std::make_tuple(
		Functional::Cat(
			Ret,
			AlphaChannel,
			-1
		).Evaluate(),
		SourceHeight * _MyScaleH,
		SourceWidth * _MyScaleW
	);
}

std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> SuperResolutionBCRGBAHW::Infer(
	const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
	int64_t _BatchSize
) const
{
	_BatchSize = std::max(_BatchSize, 1ll);
	auto& [Bitmap, SourceHeight, SourceWidth] = _Image;
	auto RGBChannel = Bitmap.Permute(0, 1, 4, 2, 3).Contiguous().Evaluate();
	const auto BatchCount = RGBChannel.Size(0) * RGBChannel.Size(1);
	const auto RGBData = RGBChannel.Data();
	const auto WindowHeight = RGBChannel.Size(3);
	const auto WindowWidth = RGBChannel.Size(4);
	const auto BatchStride = WindowHeight * WindowWidth * 4;
	auto Ret = ImageVideo::NormalizedImage5D::New({
			RGBChannel.Size(0),
			RGBChannel.Size(1),
			4,
			WindowHeight * _MyScaleH,
			WindowWidth * _MyScaleW,
		});
	auto RetData = Ret.Data();
	if (_MyCallback)
		_MyCallback(true, BatchCount);

	for (SizeType Batch = 0; Batch < BatchCount; Batch += _BatchSize)
	{
		const auto CurBatchSize = std::min(_BatchSize, BatchCount - Batch);
		const auto CurData = RGBData + Batch * BatchStride;

		Int64 Shape[4] = { CurBatchSize, 4, WindowHeight, WindowWidth };
		const auto InputSize = CurBatchSize * BatchStride;
		OrtTuple Tensors, Outputs;
		Tensors.emplace_back(
			Ort::Value::CreateTensor(
				*GetMemoryInfo(),
				CurData,
				InputSize,
				Shape,
				4
			)
		);

		_D_Dragonian_Lib_Rethrow_Block(
			Outputs = RunModel(Tensors);
		);

		const auto OutputCount = Outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto OutputData = Outputs[0].GetTensorData<float>();
		if (std::cmp_not_equal(OutputCount, _MyScaleH * _MyScaleW * InputSize))
			_D_Dragonian_Lib_Throw_Exception("Scale mismatch");
		memcpy(RetData, OutputData, OutputCount * sizeof(float));
		RetData += OutputCount;
		if (_MyCallback)
			_MyCallback(false, Batch);
	}
	return std::make_tuple(
		Ret.Permute(0, 1, 3, 4, 2).Evaluate(),
		SourceHeight * _MyScaleH,
		SourceWidth * _MyScaleW
	);
}

std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> SuperResolutionBHWCRGBA::Infer(
	const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
	int64_t _BatchSize
) const
{
	_BatchSize = std::max(_BatchSize, 1ll);
	auto& [Bitmap, SourceHeight, SourceWidth] = _Image;
	auto RGBChannel = Bitmap.Contiguous().Evaluate();
	const auto BatchCount = RGBChannel.Size(0) * RGBChannel.Size(1);
	const auto RGBData = RGBChannel.Data();
	const auto WindowHeight = RGBChannel.Size(2);
	const auto WindowWidth = RGBChannel.Size(3);
	const auto BatchStride = WindowHeight * WindowWidth * 4;
	auto Ret = ImageVideo::NormalizedImage5D::New({
			RGBChannel.Size(0),
			RGBChannel.Size(1),
			WindowHeight * _MyScaleH,
			WindowWidth * _MyScaleW,
			4
		});
	auto RetData = Ret.Data();
	if (_MyCallback)
		_MyCallback(true, BatchCount);

	for (SizeType Batch = 0; Batch < BatchCount; Batch += _BatchSize)
	{
		const auto CurBatchSize = std::min(_BatchSize, BatchCount - Batch);
		const auto CurData = RGBData + Batch * BatchStride;

		Int64 Shape[4] = { CurBatchSize, WindowHeight, WindowWidth, 4 };
		const auto InputSize = CurBatchSize * BatchStride;
		OrtTuple Tensors, Outputs;
		Tensors.emplace_back(
			Ort::Value::CreateTensor(
				*GetMemoryInfo(),
				CurData,
				InputSize,
				Shape,
				4
			)
		);

		_D_Dragonian_Lib_Rethrow_Block(
			Outputs = RunModel(Tensors);
		);

		const auto OutputCount = Outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
		const auto OutputData = Outputs[0].GetTensorData<float>();
		if (std::cmp_not_equal(OutputCount, _MyScaleH * _MyScaleW * InputSize))
			_D_Dragonian_Lib_Throw_Exception("Scale mismatch");
		memcpy(RetData, OutputData, OutputCount * sizeof(float));
		RetData += OutputCount;
		if (_MyCallback)
			_MyCallback(false, Batch);
	}
	return std::make_tuple(
		Ret.Evaluate(),
		SourceHeight * _MyScaleH,
		SourceWidth * _MyScaleW
	);
}

_D_Dragonian_Lib_Lib_Super_Resolution_End