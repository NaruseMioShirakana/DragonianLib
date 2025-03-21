#include "../../../header/InferTools/TensorExtractor/TensorExtractor.hpp"
#include <random>

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_Header

LibSvcTensorExtractor::Inputs SoVits2TensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;

	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, HubertLen };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	// SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };

	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.Length[0] = HubertLen;
	SvcTensors.Data.F0 = InterpFunc(F0, (long)F0.Size(), (long)HubertLen);
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.NSFF0 = GetNSFF0(SvcTensors.Data.F0);
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Length,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.NSFF0.Data(),
		SvcTensors.Data.NSFF0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Speaker,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.InputNames = InputNames.data();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs SoVits3TensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;

	auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	// SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };
	const int64_t upSample = int64_t(_SamplingRate) / 16000;
	const auto srcHubertSize = SvcTensors.Data.HiddenUnitShape[1];
	SvcTensors.Data.HiddenUnitShape[1] *= upSample;
	HubertSize *= upSample;
	SvcTensors.Data.FrameShape[1] = SvcTensors.Data.HiddenUnitShape[1];
	SvcTensors.Data.HiddenUnit.Reserve(HubertSize * (upSample + 1));
	for (int64_t itS = 0; itS < srcHubertSize; ++itS)
		for (int64_t itSS = 0; itSS < upSample; ++itSS)
			SvcTensors.Data.HiddenUnit.Insert(SvcTensors.Data.HiddenUnit.end(), HiddenUnit.begin() + itS * (int64_t)_HiddenSize, HiddenUnit.begin() + (itS + 1) * (int64_t)_HiddenSize);
	SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.HiddenUnitShape[1])));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.Speaker[0] = params.Chara;
	SvcTensors.Data.Length[0] = SvcTensors.Data.HiddenUnitShape[1];

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Length,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Speaker,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.InputNames = InputNames.data();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs SoVits4TensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;
	std::mt19937 gen(int(params.Seed));
	std::normal_distribution<float> normal(0, 1);
	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };
	SvcTensors.Data.NoiseShape = { 1, 192, SvcTensors.Data.FrameShape[1] };
	const auto NoiseSize = SvcTensors.Data.NoiseShape[1] * SvcTensors.Data.NoiseShape[2] * SvcTensors.Data.NoiseShape[0];

	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.FrameShape[1])));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.Alignment = GetAligments(SvcTensors.Data.FrameShape[1], HubertLen);
	SvcTensors.Data.UnVoice = GetUV(F0);
	SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
	for (auto& it : SvcTensors.Data.Noise)
		it = normal(gen) * params.NoiseScale;
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Alignment.Data(),
		SvcTensors.Data.Alignment.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.UnVoice.Data(),
		SvcTensors.Data.UnVoice.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Noise.Data(),
		SvcTensors.Data.Noise.Size(),
		SvcTensors.Data.NoiseShape.Data(),
		3
	));

	if (_SpeakerMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, SvcTensors.Data.FrameShape[1], params.Chara);
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.SpkMap.Data(),
			SvcTensors.Data.SpkMap.Size(),
			SvcTensors.Data.SpkShape.Data(),
			2
		));
	}
	else
	{
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Speaker,
			1,
			SvcTensors.Data.OneShape,
			1
		));
	}

	if (_Volume)
	{
		SvcTensors.InputNames = InputNamesVol.data();
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(SvcTensors.Data.FrameShape[1]));
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Volume.Data(),
			SvcTensors.Data.FrameShape[1],
			SvcTensors.Data.FrameShape.Data(),
			2
		));
	}
	else
		SvcTensors.InputNames = InputNames.data();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs SoVits4DDSPTensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;
	std::mt19937 gen(int(params.Seed));
	std::normal_distribution<float> normal(0, 1);
	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };
	SvcTensors.Data.NoiseShape = { 1, 192, SvcTensors.Data.FrameShape[1] };
	const auto NoiseSize = SvcTensors.Data.NoiseShape[1] * SvcTensors.Data.NoiseShape[2] * SvcTensors.Data.NoiseShape[0];
	SvcTensors.Data.DDSPNoiseShape = { 1, 2048, SvcTensors.Data.FrameShape[1] };
	const int64_t IstftCount = SvcTensors.Data.FrameShape[1] * 2048;

	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.FrameShape[1])));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.Alignment = GetAligments(SvcTensors.Data.FrameShape[1], HubertLen);
	SvcTensors.Data.DDSPNoise = DragonianLibSTL::Vector(IstftCount, params.DDSPNoiseScale);
	SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
	for (auto& it : SvcTensors.Data.Noise)
		it = normal(gen) * params.NoiseScale;
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Alignment.Data(),
		SvcTensors.Data.Alignment.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.DDSPNoise.Data(),
		SvcTensors.Data.DDSPNoise.Size(),
		SvcTensors.Data.DDSPNoiseShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Noise.Data(),
		SvcTensors.Data.Noise.Size(),
		SvcTensors.Data.NoiseShape.Data(),
		3
	));

	if (_SpeakerMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, SvcTensors.Data.FrameShape[1], params.Chara);
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.SpkMap.Data(),
			SvcTensors.Data.SpkMap.Size(),
			SvcTensors.Data.SpkShape.Data(),
			2
		));
	}
	else
	{
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Speaker,
			1,
			SvcTensors.Data.OneShape,
			1
		));
	}

	if (_Volume)
	{
		SvcTensors.InputNames = InputNamesVol.data();
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(SvcTensors.Data.FrameShape[1]));
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Volume.Data(),
			SvcTensors.Data.FrameShape[1],
			SvcTensors.Data.FrameShape.Data(),
			2
		));
	}
	else
		SvcTensors.InputNames = InputNames.data();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs RVCTensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;
	std::mt19937 gen(int(params.Seed));
	std::normal_distribution<float> normal(0, 1);
	auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	constexpr int64_t upSample = 2;
	const auto srcHubertSize = SvcTensors.Data.HiddenUnitShape[1];
	SvcTensors.Data.HiddenUnitShape[1] *= upSample;
	HubertSize *= upSample;
	SvcTensors.Data.FrameShape[1] = SvcTensors.Data.HiddenUnitShape[1];
	SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };
	SvcTensors.Data.NoiseShape = { 1, 192, SvcTensors.Data.FrameShape[1] };
	const auto NoiseSize = SvcTensors.Data.NoiseShape[1] * SvcTensors.Data.NoiseShape[2] * SvcTensors.Data.NoiseShape[0];

	SvcTensors.Data.HiddenUnit.Reserve(HubertSize);
	for (int64_t itS = 0; itS < srcHubertSize; ++itS)
		for (int64_t itSS = 0; itSS < upSample; ++itSS)
			SvcTensors.Data.HiddenUnit.Insert(SvcTensors.Data.HiddenUnit.end(), HiddenUnit.begin() + itS * (int64_t)_HiddenSize, HiddenUnit.begin() + (itS + 1) * (int64_t)_HiddenSize);
	SvcTensors.Data.Length[0] = SvcTensors.Data.HiddenUnitShape[1];
	SvcTensors.Data.F0 = GetInterpedF0(InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.HiddenUnitShape[1])));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.NSFF0 = GetNSFF0(SvcTensors.Data.F0);
	SvcTensors.Data.Alignment = GetAligments(SvcTensors.Data.FrameShape[1], HubertLen);
	SvcTensors.Data.Noise = DragonianLibSTL::Vector(NoiseSize, 0.f);
	for (auto& it : SvcTensors.Data.Noise)
		it = normal(gen) * params.NoiseScale;
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Length,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.NSFF0.Data(),
		SvcTensors.Data.NSFF0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	if (_SpeakerMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, SvcTensors.Data.FrameShape[1], params.Chara);
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.SpkMap.Data(),
			SvcTensors.Data.SpkMap.Size(),
			SvcTensors.Data.SpkShape.Data(),
			2
		));
	}
	else
	{
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Speaker,
			1,
			SvcTensors.Data.OneShape,
			1
		));
	}

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Noise.Data(),
		SvcTensors.Data.Noise.Size(),
		SvcTensors.Data.NoiseShape.Data(),
		3
	));

	if (_Volume)
	{
		SvcTensors.InputNames = InputNamesVol.data();
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(SvcTensors.Data.FrameShape[1]));
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Volume.Data(),
			SvcTensors.Data.FrameShape[1],
			SvcTensors.Data.FrameShape.Data(),
			2
		));
	}
	else
		SvcTensors.InputNames = InputNames.data();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs DiffSvcTensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;
	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };

	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.F0 = InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.FrameShape[1]));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.F0 = GetInterpedF0log(SvcTensors.Data.F0, true);
	SvcTensors.Data.Alignment = GetAligments(SvcTensors.Data.FrameShape[1], HubertLen);
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Alignment.Data(),
		SvcTensors.Data.Alignment.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Speaker,
		1,
		SvcTensors.Data.OneShape,
		1
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.InputNames = InputNames.data();

	SvcTensors.OutputNames = OutputNames.data();

	SvcTensors.OutputCount = OutputNames.size();

	return SvcTensors;
}

LibSvcTensorExtractor::Inputs DiffusionSvcTensorExtractor::Extract(const DragonianLibSTL::Vector<float>& HiddenUnit, const DragonianLibSTL::Vector<float>& F0, const DragonianLibSTL::Vector<float>& Volume, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& SpkMap, Params params)
{
	Inputs SvcTensors;
	const auto HubertSize = HiddenUnit.Size();
	const auto HubertLen = int64_t(HubertSize) / int64_t(_HiddenSize);
	SvcTensors.Data.FrameShape = { 1, int64_t(params.AudioSize * _SamplingRate / params.SrcSamplingRate / _HopSize) };
	SvcTensors.Data.HiddenUnitShape = { 1, HubertLen, int64_t(_HiddenSize) };
	SvcTensors.Data.SpkShape = { SvcTensors.Data.FrameShape[1], int64_t(_NSpeaker) };
	//auto Padding = params.Padding * SvcTensors.Data.FrameShape[1] / F0.Size();
	//if (params.Padding == size_t(-1))
	//	Padding = size_t(-1);
	
	SvcTensors.Data.HiddenUnit = HiddenUnit;
	SvcTensors.Data.F0 = InterpFunc(F0, long(F0.Size()), long(SvcTensors.Data.FrameShape[1]));
	for (auto& it : SvcTensors.Data.F0)
		it *= (float)pow(2.0, static_cast<double>(params.upKeys) / 12.0);
	SvcTensors.Data.Alignment = GetAligments(SvcTensors.Data.FrameShape[1], HubertLen);
	SvcTensors.Data.Speaker[0] = params.Chara;

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.HiddenUnit.Data(),
		HubertSize,
		SvcTensors.Data.HiddenUnitShape.Data(),
		3
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.Alignment.Data(),
		SvcTensors.Data.Alignment.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
		Memory,
		SvcTensors.Data.F0.Data(),
		SvcTensors.Data.F0.Size(),
		SvcTensors.Data.FrameShape.Data(),
		2
	));

	if (_Volume)
	{
		SvcTensors.InputNames = InputNamesVol.data();
		SvcTensors.Data.Volume = InterpFunc(Volume, long(Volume.Size()), long(SvcTensors.Data.FrameShape[1]));
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Volume.Data(),
			SvcTensors.Data.FrameShape[1],
			SvcTensors.Data.FrameShape.Data(),
			2
		));
	}
	else
		SvcTensors.InputNames = InputNames.data();

	if (_SpeakerMix)
	{
		SvcTensors.Data.SpkMap = GetCurrectSpkMixData(SpkMap, SvcTensors.Data.FrameShape[1], params.Chara);
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.SpkMap.Data(),
			SvcTensors.Data.SpkMap.Size(),
			SvcTensors.Data.SpkShape.Data(),
			2
		));
	}
	else
	{
		SvcTensors.Tensor.emplace_back(Ort::Value::CreateTensor(
			Memory,
			SvcTensors.Data.Speaker,
			1,
			SvcTensors.Data.OneShape,
			1
		));
	}

	SvcTensors.OutputNames = OutputNames.data();

	SvcTensors.OutputCount = OutputNames.size();

	return SvcTensors;
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Tensor_Extrator_End
