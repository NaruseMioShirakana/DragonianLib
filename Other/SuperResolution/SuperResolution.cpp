#include "SuperResolution.hpp"
#include "Base.h"
#include "Util/StringPreprocess.h"

namespace libsr {

	const auto allocator = Ort::AllocatorWithDefaultOptions();

	DragonianLib::ImageSlicer& SuperResolution::Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const
	{
		DragonianLibNotImplementedError;
	}

	SuperResolution::SuperResolution(unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider, ProgressCallback _Callback)
		: Env_(_ThreadCount, _DeviceID, _Provider), Callback_(std::move(_Callback))
	{

	}

	RealESRGan::RealESRGan(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : SuperResolution(_ThreadCount, _DeviceID, _Provider, std::move(_Callback))
	{
		s_width = _Config.InputWidth;
		s_height = _Config.InputHeight;

		try
		{
			model = new Ort::Session(*Env_.GetEnv(), _Config.RGBModel.c_str(), *Env_.GetSessionOptions());
			model_alpha = new Ort::Session(*Env_.GetEnv(), _Config.AlphaModel.c_str(), *Env_.GetSessionOptions());
		}
		catch (Ort::Exception& e)
		{
			Destory();
			DragonianLibThrow(e.what());
		}
		
		Names.emplace_back(model->GetInputNameAllocated(0, allocator));
		Names.emplace_back(model->GetOutputNameAllocated(0, allocator));
		inputNames = Names[0].get();
		outputNames = Names[1].get();
	}

	void RealESRGan::Destory()
	{
		delete model;
		delete model_alpha;
		model_alpha = nullptr;
		model = nullptr;
	}

	RealESRGan::~RealESRGan()
	{
		Destory();
	}

	DragonianLib::ImageSlicer& RealESRGan::Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const
	{
		size_t progress = 0;
		//auto img = DragonianLib::ImageSlicer(_path, s_width, s_height, tile_pad, 0.f, false);

		const auto s_len = s_width * s_height;
		auto& imgRGB = _Image.data.rgb;
		auto& imgAlpha = _Image.data.alpha;
		const size_t progressMax = imgAlpha.Size() / s_len;

		Callback_(progress, progressMax);

		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _imgOutRGB, _imgOutAlpha;
		_imgOutRGB.Reserve(progressMax); _imgOutAlpha.Reserve(progressMax);

		while(progress < progressMax)
		{
			if (progress + _BatchSize > progressMax)
				_BatchSize = int64_t(progressMax - progress);
			if (_BatchSize == 0)
				break;

			int64_t shape[4] = { _BatchSize,s_height,s_width,3 };
			DragonianLibSTL::Vector ImageI(imgRGB.Data() + (s_len * 3ll * progress), imgRGB.Data() + (s_len * 3ll * (progress + _BatchSize)));
			std::vector<Ort::Value> Tensors, outTensors;
			Tensors.emplace_back(Ort::Value::CreateTensor(*Env_.GetMemoryInfo(), ImageI.Data(), ImageI.Size(), shape, 4));
			//auto BeginTime = clock();
			try
			{
				outTensors = model->Run(Ort::RunOptions{ nullptr },
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
			//printf("Cost Time(Sec): %lf\n", (double(clock()) - double(BeginTime)) / 1000.);
			const auto outShape = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
			const auto outData = outTensors[0].GetTensorData<float>();

			for (int64_t j = 0; j < _BatchSize; j++)
				_imgOutRGB.EmplaceBack(outData + j * outShape[1] * outShape[2] * 3, outData + (j + 1) * outShape[1] * outShape[2] * 3);

			shape[3] = 1;
			Tensors.clear();
			ImageI = DragonianLibSTL::Vector(imgAlpha.Data() + (s_len * progress), imgAlpha.Data() + (s_len * (progress + _BatchSize)));

			Tensors.emplace_back(Ort::Value::CreateTensor(*Env_.GetMemoryInfo(), ImageI.Data(), ImageI.Size(), shape, 4));

			try
			{
				outTensors = model_alpha->Run(Ort::RunOptions{ nullptr },
					& inputNames,
					Tensors.data(),
					1,
					& outputNames,
					1
				);
			}
			catch (Ort::Exception& e)
			{
				DragonianLibThrow(e.what());
			}
			const auto outShapeAlpha = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
			const auto outDataAlpha = outTensors[0].GetTensorData<float>();

			for (int64_t j = 0; j < _BatchSize; j++)
				_imgOutAlpha.EmplaceBack(outDataAlpha + j * outShapeAlpha[1] * outShapeAlpha[2], outDataAlpha + (j + 1) * outShapeAlpha[1] * outShapeAlpha[2]);

			progress += _BatchSize;
			Callback_(progress, progressMax);
		}
		imgRGB.Clear();
		imgAlpha.Clear();
		for (size_t i = 0; i < _imgOutAlpha.Size(); ++i)
		{
			imgRGB.Insert(imgRGB.end(), _imgOutRGB[i].begin(), _imgOutRGB[i].end());
			imgAlpha.Insert(imgAlpha.end(), _imgOutAlpha[i].begin(), _imgOutAlpha[i].end());
		}

		return _Image;
		/*
		for (int64_t i = 0; i < int64_t(s_width);)
		{
			if (progress + batch_size > progressMax)
				batch_size = int64_t(progressMax - progress);
			if (batch_size == 0)
				break;
			DragonianLibSTL::Vector<float> rgb, alpha;
			rgb.reserve(s_height * int64_t(s_width) * 3);
			alpha.reserve(s_height * int64_t(s_width));
			int64_t shape[4] = { batch_size,s_height,s_width,3 };
			DragonianLibSTL::Vector<float> ImageI;
			ImageI.reserve(int64_t(s_height) * s_width * 3ll * batch_size);

			for (int64_t j = 0; j < s_height; j++)
				ImageI.insert(ImageI.end(), imgRGB[j + i].begin(), imgRGB[j + i].end());

			DragonianLibSTL::Vector<Ort::Value> Tensors;


			Tensors.emplace_back(Ort::Value::CreateTensor(*memory_info, ImageI.data(), ImageI.size(), shape, 4));
			try
			{
				const auto outTensors = model->Run(Ort::RunOptions{ nullptr },
					inputNames.data(),
					Tensors.data(),
					Tensors.size(),
					outputNames.data(),
					1
				);
				const auto outShape = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
				const auto outData = outTensors[0].GetTensorData<float>();
				for (int64_t j = 0; j < batch_size; j++)
					imgRGB[j + i] = { outData + j * outShape[1] * outShape[2] * 3,outData + (j + 1) * outShape[1] * outShape[2] * 3 };

				shape[3] = 1;
				Tensors.clear();
				ImageI.clear();
				ImageI.reserve(64ll * 64ll * 1ll * batch_size);
				for (int64_t j = 0; j < batch_size; j++)
					ImageI.insert(ImageI.end(), imgAlpha[j + i].begin(), imgAlpha[j + i].end());
				Tensors.emplace_back(Ort::Value::CreateTensor(*memory_info, ImageI.data(), ImageI.size(), shape, 4));

				const auto outTensorsAlpha = model->Run(Ort::RunOptions{ nullptr },
					inputNames.data(),
					Tensors.data(),
					Tensors.size(),
					outputNames.data(),
					1
				);
				const auto outShapeAlpha = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
				const auto outDataAlpha = outTensors[0].GetTensorData<float>();
				for (int64_t j = 0; j < batch_size; j++)
					imgAlpha[j + i] = { outDataAlpha + j * outShapeAlpha[1] * outShapeAlpha[2] * 3,outDataAlpha + (j + 1) * outShapeAlpha[1] * outShapeAlpha[2] * 3 };

				progress += batch_size;
				i += batch_size;
				_callback(progress, progressMax);
			}catch (Ort::Exception& e)
			{
				DragonianLibThrow(e.what());
			}
		}
		img.write((_outputPath + _path.substr(_path.rfind(L'\\'), _path.rfind(L'.')) + std::to_wstring(unsigned long long(_path.data())) + L".mid").c_str(), scale);

		 */
	}

	void MoeSR::Destory()
	{
		delete Model;
		Model = nullptr;
	}

	DragonianLib::ImageSlicer& MoeSR::Infer(DragonianLib::ImageSlicer& _Image, int64_t _BatchSize) const
	{
		_BatchSize = 1;
		auto InputWidth = _Image.GetWidth();
		auto InputHeight = _Image.GetHeight();
		size_t progress = 0;
		const auto pixCount = InputWidth * InputHeight;
		auto& imgRGB = _Image.data.rgb;
		auto& imgAlpha = _Image.data.alpha;
		const size_t progressMax = imgAlpha.Size() / pixCount;
		Callback_(progress, progressMax);

		DragonianLibSTL::Vector<float> imageInRGB(imgRGB.Size()), imgInAlpha;
		imgInAlpha.Reserve(imgAlpha.Size() * 3);
		imgInAlpha.Insert(imgInAlpha.end(), imgAlpha.begin(), imgAlpha.end());
		imgInAlpha.Insert(imgInAlpha.end(), imgAlpha.begin(), imgAlpha.end());
		imgInAlpha.Insert(imgInAlpha.end(), imgAlpha.begin(), imgAlpha.end());
		const auto dataSize = pixCount * 3ull;

		for (size_t n = 0; n < imgRGB.Size(); n += dataSize)
		{
			auto imageInRGBPtr = imageInRGB.Data() + n;
			auto imageSrcPtr = imgRGB.Data() + n;
			for (size_t i = 0; i < 3; ++i)
				for (size_t j = i; j < dataSize; j += 3)
					*(imageInRGBPtr++) = imageSrcPtr[j];
		}

		const auto TotalScale = (size_t)uint32_t(ScaleFactor * ScaleFactor);
		imgRGB = { imgRGB.Size() * TotalScale, imgRGB.GetAllocator() };
		imgAlpha.Clear();
		imgAlpha.Reserve(imgAlpha.Size() * TotalScale);
		

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
					imageInRGB.Data() + (dataSize * progress),
					dataSize * _BatchSize,
					shape,
					4
				)
			);

			try
			{
				outTensors = Model->Run(Ort::RunOptions{ nullptr },
					& inputNames,
					Tensors.data(),
					1,
					& outputNames,
					1
				);
			}
			catch (Ort::Exception& e)
			{
				DragonianLibThrow(e.what());
			}

			for(size_t n = 0; n < _BatchSize * TotalScale * dataSize; n += dataSize * TotalScale)
			{
				auto imageRGBPtr = imgRGB.Data() + dataSize * progress + n;
				auto imageOutPtr = outTensors[0].GetTensorData<float>() + n;
				for (size_t i = 0; i < 3; ++i)
					for (size_t j = i; j < dataSize * TotalScale; j += 3)
						imageRGBPtr[j] = *(imageOutPtr++);
			}

			Tensors[0] = Ort::Value::CreateTensor(
				*Env_.GetMemoryInfo(),
				imgInAlpha.Data() + (dataSize * progress),
				dataSize * _BatchSize,
				shape,
				4
			);

			try
			{
				outTensors = Model->Run(Ort::RunOptions{ nullptr },
					& inputNames,
					Tensors.data(),
					1,
					& outputNames,
					1
				);
			}
			catch (Ort::Exception& e)
			{
				DragonianLibThrow(e.what());
			}

			for (size_t n = 0; n < _BatchSize * TotalScale * dataSize; n += dataSize * TotalScale)
				imgAlpha.Insert(imgAlpha.end(), outTensors[0].GetTensorData<float>() + n, outTensors[0].GetTensorData<float>() + n + dataSize * TotalScale);

			progress += _BatchSize;
			Callback_(progress, progressMax);
		}

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

		auto allocator = Ort::AllocatorWithDefaultOptions();
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