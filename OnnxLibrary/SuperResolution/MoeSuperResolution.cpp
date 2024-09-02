#include "MoeSuperResolution.hpp"
#include "Base.h"

namespace DragonianLib
{
    namespace LibSuperResolution
    {
		void MoeSR::Destory()
		{
			delete Model;
			Model = nullptr;
		}

		DragonianLib::Image& MoeSR::Infer(DragonianLib::Image& _Image, int64_t _BatchSize) const
		{
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

				int64_t shape[4] = { _BatchSize, 3, InputHeight, InputWidth };

				std::vector<Ort::Value> Tensors, outTensors;
				Tensors.emplace_back(
					Ort::Value::CreateTensor(
						*Env_.GetMemoryInfo(),
						imgRGB.Data() + (dataSize * progress),
						dataSize * _BatchSize,
						shape,
						4
					)
				);

				try
				{
					outTensors = Model->Run(Ort::RunOptions{ nullptr },
						&inputNames,
						Tensors.data(),
						1,
						&outputNames,
						1
					);
				}
				catch (Ort::Exception& e)
				{
					DragonianLibThrow(e.what());
				}

				const auto outData = outTensors[0].GetTensorData<float>();
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

		MoeSR::MoeSR(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : SuperResolution(_ThreadCount, _DeviceID, _Provider, std::move(_Callback))
		{
			ScaleFactor = _Config.Scale;
			try
			{
				Model = new Ort::Session(*Env_.GetEnv(), _Config.RGBModel.c_str(), *Env_.GetSessionOptions());
			}
			catch (Ort::Exception& e)
			{
				Destory();
				DragonianLibThrow(e.what());
			}

			const auto allocator = Ort::AllocatorWithDefaultOptions();
			Names.emplace_back(Model->GetInputNameAllocated(0, allocator));
			Names.emplace_back(Model->GetOutputNameAllocated(0, allocator));
			inputNames = Names[0].get();
			outputNames = Names[1].get();
		}

		MoeSR::~MoeSR()
		{
			Destory();
		}
    }
}