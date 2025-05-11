//#include "OnnxLibrary/SuperResolution/Real-ESRGan.hpp"
//
//namespace DragonianLib
//{
//	namespace LibSuperResolution
//	{
//		RealESRGan::RealESRGan(const Hparams& _Config, ProgressCallback _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : SuperResolution(_ThreadCount, _DeviceID, _Provider, std::move(_Callback))
//		{
//			s_width = _Config.InputWidth;
//			s_height = _Config.InputHeight;
//			const auto allocator = Ort::AllocatorWithDefaultOptions();
//			try
//			{
//				model = new Ort::Session(*Env_->GetEnv(), _Config.RGBModel.c_str(), *Env_->GetSessionOptions());
//				model_alpha = new Ort::Session(*Env_->GetEnv(), _Config.AlphaModel.c_str(), *Env_->GetSessionOptions());
//			}
//			catch (Ort::Exception& e)
//			{
//				Destory();
//				_D_Dragonian_Lib_Throw_Exception(e.what());
//			}
//
//			Names.emplace_back(model->GetInputNameAllocated(0, allocator));
//			Names.emplace_back(model->GetOutputNameAllocated(0, allocator));
//			inputNames = Names[0].get();
//			outputNames = Names[1].get();
//		}
//
//		void RealESRGan::Destory()
//		{
//			delete model;
//			delete model_alpha;
//			model_alpha = nullptr;
//			model = nullptr;
//		}
//
//		RealESRGan::~RealESRGan()
//		{
//			Destory();
//		}
//
//		ImageVideo::Image& RealESRGan::Infer(ImageVideo::Image& _Image, int64_t _BatchSize) const
//		{
//			size_t progress = 0;
//			//auto img = DragonianLib::ImageSlicer(_path, s_width, s_height, tile_pad, 0.f, false);
//
//			const auto s_len = s_width * s_height;
//			auto& imgRGB = _Image.data.rgb;
//			auto& imgAlpha = _Image.data.alpha;
//			const size_t progressMax = imgAlpha.Size() / s_len;
//
//			Callback_(progress, progressMax);
//
//			DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> _imgOutRGB, _imgOutAlpha;
//			_imgOutRGB.Reserve(progressMax); _imgOutAlpha.Reserve(progressMax);
//
//			while (progress < progressMax)
//			{
//				if (progress + _BatchSize > progressMax)
//					_BatchSize = int64_t(progressMax - progress);
//				if (_BatchSize == 0)
//					break;
//
//				int64_t shape[4] = { _BatchSize,s_height,s_width,3 };
//				DragonianLibSTL::Vector ImageI(imgRGB.Data() + (s_len * 3ll * progress), imgRGB.Data() + (s_len * 3ll * (progress + _BatchSize)));
//				std::vector<Ort::Value> Tensors, outTensors;
//				Tensors.emplace_back(Ort::Value::CreateTensor(*Env_->GetMemoryInfo(), ImageI.Data(), ImageI.Size(), shape, 4));
//				//auto BeginTime = clock();
//				try
//				{
//					outTensors = model->Run(Ort::RunOptions{ nullptr },
//						&inputNames,
//						Tensors.data(),
//						1,
//						&outputNames,
//						1
//					);
//				}
//				catch (Ort::Exception& e)
//				{
//					_D_Dragonian_Lib_Throw_Exception(e.what());
//				}
//				//printf("Cost Time(Sec): %lf\n", (double(clock()) - double(BeginTime)) / 1000.);
//				const auto outShape = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
//				const auto outData = outTensors[0].GetTensorData<float>();
//
//				for (int64_t j = 0; j < _BatchSize; j++)
//					_imgOutRGB.EmplaceBack(outData + j * outShape[1] * outShape[2] * 3, outData + (j + 1) * outShape[1] * outShape[2] * 3);
//
//				shape[3] = 1;
//				Tensors.clear();
//				ImageI = DragonianLibSTL::Vector(imgAlpha.Data() + (s_len * progress), imgAlpha.Data() + (s_len * (progress + _BatchSize)));
//
//				Tensors.emplace_back(Ort::Value::CreateTensor(*Env_->GetMemoryInfo(), ImageI.Data(), ImageI.Size(), shape, 4));
//
//				try
//				{
//					outTensors = model_alpha->Run(Ort::RunOptions{ nullptr },
//						&inputNames,
//						Tensors.data(),
//						1,
//						&outputNames,
//						1
//					);
//				}
//				catch (Ort::Exception& e)
//				{
//					_D_Dragonian_Lib_Throw_Exception(e.what());
//				}
//				const auto outShapeAlpha = outTensors[0].GetTensorTypeAndShapeInfo().GetShape();
//				const auto outDataAlpha = outTensors[0].GetTensorData<float>();
//
//				for (int64_t j = 0; j < _BatchSize; j++)
//					_imgOutAlpha.EmplaceBack(outDataAlpha + j * outShapeAlpha[1] * outShapeAlpha[2], outDataAlpha + (j + 1) * outShapeAlpha[1] * outShapeAlpha[2]);
//
//				progress += _BatchSize;
//				Callback_(progress, progressMax);
//			}
//			imgRGB.Reserve(_imgOutRGB.Size() * _imgOutRGB[0].Size());
//			imgRGB.Clear();
//			imgAlpha.Reserve(_imgOutAlpha.Size() * _imgOutAlpha[0].Size());
//			imgAlpha.Clear();
//			for (size_t i = 0; i < _imgOutAlpha.Size(); ++i)
//			{
//				imgRGB.Insert(imgRGB.end(), _imgOutRGB[i].begin(), _imgOutRGB[i].end());
//				imgAlpha.Insert(imgAlpha.end(), _imgOutAlpha[i].begin(), _imgOutAlpha[i].end());
//			}
//
//			return _Image;
//		}
//	}
//}