#include "MoeSuperResolution.hpp"
#include "Base.h"

namespace tlibsr
{
	DragonianLib::Image& MoeSR::Infer(DragonianLib::Image& _Image, const InferenceDeviceBuffer& _Buffer, int64_t _BatchSize) const
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
		while (progress < progressMax)
		{
			if (progress + _BatchSize > progressMax)
				_BatchSize = int64_t(progressMax - progress);
			if (_BatchSize == 0)
				break;

			DragonianLibSTL::Vector<TrtTensor> Tensors, outTensors;
			Tensors.EmplaceBack(
				imgRGB.Data() + (dataSize * progress),
				nvinfer1::Dims4(_BatchSize, 3, InputHeight, InputWidth),
				Model->GetInputNames()[0],
				dataSize * _BatchSize * sizeof(float),
				nvinfer1::DataType::kFLOAT
			);

			try
			{
				outTensors = Model->Infer(
					Tensors,
					_Buffer,
					Model->GetOutputNames()
				);
			}
			catch (std::exception& e)
			{
				DragonianLibThrow(e.what());
			}

			outTensors[0].DeviceData2Host();
			const auto outData = (float*)outTensors[0].Data;
			OutRGB.Insert(OutRGB.End(), outData, outData + TotalScale * dataSize);

			progress += _BatchSize;
			Callback_(progress, progressMax);
		}

		_Image.data.rgb = std::move(OutRGB);
		imgAlpha.Clear();
		imgAlpha.Resize(imgRGB.Size() / 3, 1.f);
		_Image.Transpose(ScaleFactor);
		return _Image;
	}

	MoeSR::MoeSR(const std::wstring& RGBModel, long Scale, const TrtConfig& TrtSettings, ProgressCallback _Callback)
	{
		Callback_ = std::move(_Callback);
		ScaleFactor = Scale;
		try
		{
			Model = std::make_unique<TrtModel>(
				RGBModel,
				TrtSettings.CacheFile,
				DynaSetting,
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
			DragonianLibThrow(e.what());
		}
	}

	MoeSR::~MoeSR() = default;
}