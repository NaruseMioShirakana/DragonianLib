#include <random>
#include <regex>
#include "Libraries/Util/Logger.h"
#include "TensorRT/SingingVoiceConversion/VitsSvc.hpp"

_D_Dragonian_Lib_TRT_Svc_Space_Header

//"c", "f0", "mel2ph", "uv", "noise", "sid", "vol" "phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd", "vol"

VitsSvc::VitsSvc(
	const HParams& _Hps
) : SingingVoiceConversionModule(_Hps)
{
	try
	{
		if (_Hps.HubertModel)
			HubertModel = _Hps.HubertModel;
		else
			HubertModel = std::make_shared<TrtModel>(
				_Hps.ModelPaths.at(L"UnitsEncoder"),
				_Hps.TrtSettings.CacheFile.at(_Hps.ModelPaths.at(L"UnitsEncoder")),
				_Hps.TrtSettings.DynaSetting,
				_Hps.TrtSettings.DLACore,
				_Hps.TrtSettings.Fallback,
				_Hps.TrtSettings.EnableFp16,
				_Hps.TrtSettings.EnableBf16,
				_Hps.TrtSettings.EnableInt8,
				_Hps.TrtSettings.VerboseLevel,
				_Hps.TrtSettings.OptimizationLevel
			);

		VitsSvcModel = std::make_unique<TrtModel>(
			_Hps.ModelPaths.at(L"Model"),
			_Hps.TrtSettings.CacheFile.at(_Hps.ModelPaths.at(L"Model")),
			_Hps.TrtSettings.DynaSetting,
			_Hps.TrtSettings.DLACore,
			_Hps.TrtSettings.Fallback,
			_Hps.TrtSettings.EnableFp16,
			_Hps.TrtSettings.EnableBf16,
			_Hps.TrtSettings.EnableInt8,
			_Hps.TrtSettings.VerboseLevel,
			_Hps.TrtSettings.OptimizationLevel
		);
	}
	catch (std::exception& _exception)  // NOLINT(bugprone-empty-catch)
	{
		_D_Dragonian_Lib_Throw_Exception(_exception.what());
	}
}

std::vector<ITensorInfo> Rvc::SvcPreprocess(
	const SliceDatas& MyData,
	const Parameters& Params
) const
{
	auto FrameShape = nvinfer1::Dims2{ 1, MyData.F0.Size(2) };
	auto HiddenUnitShape = nvinfer1::Dims3{ 1, MyData.Units.Size(2), MyData.Units.Size(3) };
	auto SpkShape = nvinfer1::Dims2{ FrameShape.d[1], int64_t(_MySpeakerCount) };
	auto NoiseShape = nvinfer1::Dims3{ 1, 192, FrameShape.d[1] };

	std::vector<ITensorInfo> Tensors;
	Tensors.emplace_back(
		HiddenUnitShape,
		"phone",
		MyData.Units.ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.Units.Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"phone_lengths",
		MyData.UnitsLength->ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.UnitsLength->Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"pitch",
		MyData.F0Embed->ElementCount() * sizeof(int64_t),
		nvinfer1::DataType::kINT64,
		MyData.F0Embed->Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"pitchf",
		MyData.F0.ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.F0.Data()
	);

	if (_HasSpeakerMixLayer)
	{
		Tensors.emplace_back(
			SpkShape,
			"ds",
			MyData.Speaker->ElementCount() * sizeof(float),
			nvinfer1::DataType::kFLOAT,
			MyData.Speaker->Data()
		);
	}
	else if (_HasSpeakerEmbedding)
	{
		Tensors.emplace_back(
			nvinfer1::Dims64(1, { 1 }),
			"ds",
			sizeof(int64_t),
			nvinfer1::DataType::kINT64,
			MyData.SpeakerId->Data()
		);
	}

	Tensors.emplace_back(
		NoiseShape,
		"rnd",
		MyData.Noise->ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.Noise->Data()
	);

	return Tensors;
}

std::vector<ITensorInfo> VitsSvc::SvcPreprocess(
	const SliceDatas& MyData,
	const Parameters& Params
) const
{
	auto FrameShape = nvinfer1::Dims2{ 1, MyData.F0.Size(2) };
	auto HiddenUnitShape = nvinfer1::Dims3{ 1, MyData.Units.Size(2), MyData.Units.Size(3) };
	auto SpkShape = nvinfer1::Dims2{ FrameShape.d[1], int64_t(_MySpeakerCount) };
	auto NoiseShape = nvinfer1::Dims3{ 1, 192, FrameShape.d[1] };

	std::vector<ITensorInfo> Tensors;
	Tensors.emplace_back(
		HiddenUnitShape,
		"c",
		MyData.Units.ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.Units.Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"f0",
		MyData.F0.ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.F0.Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"mel2ph",
		MyData.Mel2Units->ElementCount() * sizeof(int64_t),
		nvinfer1::DataType::kINT64,
		MyData.Mel2Units->Data()
	);

	Tensors.emplace_back(
		FrameShape,
		"uv",
		MyData.UnVoice->ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.UnVoice->Data()
	);

	Tensors.emplace_back(
		NoiseShape,
		"noise",
		MyData.Noise->ElementCount() * sizeof(float),
		nvinfer1::DataType::kFLOAT,
		MyData.Noise->Data()
	);

	if (_HasSpeakerMixLayer)
	{
		Tensors.emplace_back(
			SpkShape,
			"sid",
			MyData.Speaker->ElementCount() * sizeof(float),
			nvinfer1::DataType::kFLOAT,
			MyData.Speaker->Data()
		);
	}
	else if (_HasSpeakerEmbedding)
	{
		Tensors.emplace_back(
			nvinfer1::Dims64(1, { 1 }),
			"sid",
			sizeof(int64_t),
			nvinfer1::DataType::kINT64,
			MyData.SpeakerId->Data()
		);
	}

	if (_HasVolumeEmbedding)
	{
		Tensors.emplace_back(
			FrameShape,
			"vol",
			MyData.Volume->ElementCount() * sizeof(float),
			nvinfer1::DataType::kFLOAT,
			MyData.Volume->Data()
		);
	}

	return Tensors;
}

SliceDatas Rvc::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);
	PreprocessUnitsLength(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, _MyLogger);
	PreprocessF0Embed(MyData, BatchSize, Channels, TargetNumFrames, 0.f, _MyLogger);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed, _MyLogger);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, _MyLogger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, _MyLogger);
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);
	return MyData;
}


SliceDatas VitsSvc::VPreprocess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);
	CheckParams(MyData);
	const auto TargetNumFrames = CalculateFrameCount(MyData.SourceSampleCount, MyData.SourceSampleRate);
	const auto BatchSize = MyData.Units.Shape(0);
	const auto Channels = MyData.Units.Shape(1);

	PreprocessUnits(MyData, BatchSize, Channels, 0, _MyLogger);
	PreprocessUnVoice(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);
	PreprocessF0(MyData, BatchSize, Channels, TargetNumFrames, Params.PitchOffset, !Params.F0HasUnVoice, Params.F0Preprocess, Params.UserParameters, _MyLogger);
	PreprocessMel2Units(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);
	PreprocessNoise(MyData, BatchSize, Channels, TargetNumFrames, Params.NoiseScale, Params.Seed, _MyLogger);
	PreprocessSpeakerMix(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, _MyLogger);
	PreprocessSpeakerId(MyData, BatchSize, Channels, TargetNumFrames, Params.SpeakerId, _MyLogger);
	PreprocessVolume(MyData, BatchSize, Channels, TargetNumFrames, _MyLogger);

	return MyData;
}

Tensor<Float32, 4, Device::CPU> VitsSvc::Forward(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
	std::shared_ptr<InferenceSession> _MyVitsSvcSession = nullptr;
	{
		auto Iter = VitsSvcSession.find(InputDatas.Units.Size(2));
		if (Iter != VitsSvcSession.end())
			_MyVitsSvcSession = Iter->second;
		else
		{
			_MyVitsSvcSession = std::make_shared<InferenceSession>();
			VitsSvcSession[InputDatas.Units.Size(2)] = _MyVitsSvcSession;
		}
	}

	auto InputTensors = SvcPreprocess(
		InputDatas,
		Params
	);

	try
	{
		if (!_MyVitsSvcSession->IsReady(InputTensors))
			*_MyVitsSvcSession = VitsSvcModel->Construct(
				InputTensors,
				{ "audio" }
			);
		for (size_t i = 0; i < InputTensors.size(); ++i)
			_MyVitsSvcSession->HostMemoryToDevice(i, InputTensors[i].GetData(), InputTensors[i].GetSize());
		_MyVitsSvcSession->Run();
	}
	catch (std::exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: VitsSvc\n") + e.what()));
	}

	auto VitsOutputAudioSize = _MyVitsSvcSession->GetOutputInfos()[0].GetElementCount();
	auto Output = Tensor<Float32, 4, Device::CPU>::New(
		{1, 1, 1, VitsOutputAudioSize }
	);
	return Output;
}

void VitsSvc::EmptyCache()
{
	VitsSvcSession.clear();
	HubertSession.clear();
}

_D_Dragonian_Lib_TRT_Svc_Space_End