#include "../Vits-Svc.hpp"

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_Header

VitsSvc::VitsSvc(
	const OnnxRuntimeEnviroment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : SingingVoiceConversionModule(Params),
_MyBase(_Environment, Params.ModelPaths.at(L"Model"), _Logger)
{
	if (Params.ExtendedParameters.contains(L"NoiseDims"))
		_MyNoiseDims = _wcstoi64(Params.ExtendedParameters.at(L"NoiseDims").c_str(), nullptr, 10);
	else
		LogInfo(L"NoiseDims not found, using default value: 192");

	const auto OutputAxis = _MyOutputDims[0].Size();
	if (OutputAxis > 4 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis");
}

Tensor<Float32, 4, Device::CPU> VitsSvc::Inference(
	const Parameters& Params,
	const SliceDatas& InputDatas
) const
{
	if (static_cast<Int64>(InputDatas.OrtValues.size()) != _MyInputCount)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: " + std::to_string(_MyInputCount) + ", got: " + std::to_string(InputDatas.OrtValues.size()));

	OrtTuple AudioTuple;

	_D_Dragonian_Lib_Rethrow_Block(AudioTuple = _MyModel->Run(
		*_MyRunOptions,
		_MyInputNames.Data(),
		InputDatas.OrtValues.data(),
		_MyInputCount,
		_MyOutputNames.Data(),
		_MyOutputCount
	););

	auto Audio = std::move(AudioTuple[0]);
	auto OutputShape = Audio.GetTensorTypeAndShapeInfo().GetShape();
	//const auto OutputElementCount = Audio.GetTensorTypeAndShapeInfo().GetElementCount();
	const auto OutputAxis = OutputShape.size();
	Dimensions<4> OutputDims;

	if (OutputAxis == 1)
		OutputDims = { 1, 1, 1, OutputShape[0] };
	else if (OutputAxis == 2)
		OutputDims = { 1, 1, OutputShape[0], OutputShape[1] };
	else if (OutputAxis == 3)
		OutputDims = { 1, OutputShape[0], OutputShape[1], OutputShape[2] };
	else if (OutputAxis == 4)
		OutputDims = { OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3] };

	_D_Dragonian_Lib_Rethrow_Block(return CreateTensorViewFromOrtValue<Float32>(std::move(Audio), OutputDims););
}

SoftVitsSvcV2::SoftVitsSvcV2(
	const OnnxRuntimeEnviroment& _Environment,
	const HParams& Params,
	const std::shared_ptr<Logger>& _Logger
) : VitsSvc(_Environment, Params, _Logger)
{
	if (_MyInputCount != 4)
		_D_Dragonian_Lib_Throw_Exception("Invalid input count, expected: 4, got: " + std::to_string(_MyInputCount));
	if (_MyOutputCount != 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output count, expected: 1, got: " + std::to_string(_MyOutputCount));

	auto& HubertShape = _MyInputDims[0];
	auto& LengthShape = _MyInputDims[1];
	auto& F0Shape = _MyInputDims[2];
	auto& SpeakerIdShape = _MyInputDims[3];

	const auto HubertAxis = HubertShape.Size();
	const auto LengthAxis = LengthShape.Size();
	const auto F0Axis = F0Shape.Size();
	const auto SpeakerIdAxis = SpeakerIdShape.Size();

	if (HubertAxis > 4 || HubertAxis < 2)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert axis, expected: 2-4, got: " + std::to_string(HubertAxis));
	if (LengthAxis < 1 || LengthAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid length axis, expected: 1-3, got: " + std::to_string(LengthAxis));
	if (F0Axis < 1 || F0Axis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid f0 axis, expected: 1-3, got: " + std::to_string(F0Axis));
	if (SpeakerIdAxis < 1 || SpeakerIdAxis > 3)
		_D_Dragonian_Lib_Throw_Exception("Invalid speaker id axis, expected: 1-3, got: " + std::to_string(SpeakerIdAxis));

	auto& OutputShape = _MyOutputDims[0];
	const auto OutputAxis = OutputShape.Size();

	if (OutputAxis > 3 || OutputAxis < 1)
		_D_Dragonian_Lib_Throw_Exception("Invalid output axis, expected: 1-3, got: " + std::to_string(OutputAxis));

	bool Found = false;
	for (auto& UnitDim : HubertShape)
		if (UnitDim == _MyUnitsDim)
		{
			Found = true;
			break;
		}

	if (!Found && HubertShape.Back() != -1)
		_D_Dragonian_Lib_Throw_Exception("Invalid hubert dims, expected: " + std::to_string(_MyUnitsDim), " got: " + std::to_string(HubertShape.Back()));
}

SliceDatas SoftVitsSvcV2::PreProcess(
	const Parameters& Params,
	SliceDatas&& InputDatas
) const
{
	auto MyData = std::move(InputDatas);

	if (MyData.Units.Null())
		_D_Dragonian_Lib_Throw_Exception("Units could not be null");

	const auto [Batch, Channel, NumFrames, UnitDims] = MyData.Units.Shape().RawArray();

	if (UnitDims != _MyUnitsDim)
		_D_Dragonian_Lib_Throw_Exception("Invalid units dims, expected: " + std::to_string(_MyUnitsDim) + ", got: " + std::to_string(UnitDims));

	if (MyData.UnitsLength.has_value() && !MyData.UnitsLength->Null())
	{
		const auto [BatchLength, ChannelLength, Length] = MyData.UnitsLength->Shape().RawArray();
		if (BatchLength != Batch)
			_D_Dragonian_Lib_Throw_Exception("Units and units length batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchLength));
		if (ChannelLength != Channel)
			_D_Dragonian_Lib_Throw_Exception("Units and units length channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelLength));
		if (Length != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid units length length");
	}
	else
	{
		LogInfo(L"Units length not found, generating units length with units");
		using UnitsSizeType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.UnitsLength = UnitsSizeType::ConstantOf({ Batch, Channel, 1 }, NumFrames).Evaluate(););
	}

	if (HasSpeakerEmbedding() && MyData.SpeakerId.has_value() && !MyData.SpeakerId->Null())
	{
		const auto [BatchSpeakerId, ChannelSpeakerId, LengthSpeakerId] = MyData.SpeakerId->Shape().RawArray();
		if (Batch != BatchSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchSpeakerId));
		if (Channel != ChannelSpeakerId)
			_D_Dragonian_Lib_Throw_Exception("Units and speaker id channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelSpeakerId));
		if (LengthSpeakerId != 1)
			_D_Dragonian_Lib_Throw_Exception("Invalid speaker id length");
	}
	else
	{
		LogInfo(L"Speaker id not found, generating speaker id with param.speaker_id");
		using SpkType = Tensor<Int64, 3, Device::CPU>;
		_D_Dragonian_Lib_Rethrow_Block(MyData.SpeakerId = SpkType::ConstantOf({ Batch, Channel, 1 }, Params.SpeakerId).Evaluate(););
	}

	if (MyData.F0Embed.has_value() && !MyData.F0Embed->Null())
	{
		const auto [BatchF0Embed, ChannelF0Embed, NumFramesF0Embed] = MyData.F0Embed->Shape().RawArray();
		if (Batch != BatchF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0Embed));
		if (Channel != ChannelF0Embed)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 embed channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0Embed));
		if (NumFrames != NumFramesF0Embed)
			_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = MyData.F0Embed->Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););
	}
	else
	{
		LogInfo(L"F0 embedding not found, generating f0 embedding with f0 tensor");
		if (MyData.F0.Null())
			_D_Dragonian_Lib_Throw_Exception("F0 could not be null");
		const auto [BatchF0, ChannelF0, NumFramesF0] = MyData.F0.Shape().RawArray();
		if (Batch != BatchF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 batch mismatch, expected: " + std::to_string(Batch) + ", got: " + std::to_string(BatchF0));
		if (Channel != ChannelF0)
			_D_Dragonian_Lib_Throw_Exception("Units and f0 channels mismatch, expected: " + std::to_string(Channel) + ", got: " + std::to_string(ChannelF0));

		if (NumFramesF0 != NumFrames)
			_D_Dragonian_Lib_Rethrow_Block(MyData.F0 = MyData.F0.Interpolate<Operators::InterpolateMode::Linear>(
				IDim(-1),
				{ IDim(NumFrames) }
			).Evaluate(););

		(MyData.F0 *= std::pow(2.f, Params.PitchOffset / 12.f)).Evaluate();

		_D_Dragonian_Lib_Rethrow_Block(MyData.F0Embed = GetF0Embed(MyData.F0, static_cast<Float32>(_MyF0Bin), _MyF0MelMax, _MyF0MelMin););
	}

	auto& HubertShape = _MyInputDims[0];
	auto& LengthShape = _MyInputDims[1];
	auto& F0Shape = _MyInputDims[2];
	
	const auto HubertAxis = HubertShape.Size();
	const auto LengthAxis = LengthShape.Size();
	const auto F0Axis = F0Shape.Size();

	MyData.OrtValues.clear();
	MyData.OrtValues.reserve(_MyInputCount);
	MyData.OrtValues.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			MyData.Units.Data(),
			MyData.Units.ElementCount(),
			MyData.Units.Shape().Data() + MyData.Units.Rank() - HubertAxis,
			HubertAxis
		)
	);
	MyData.OrtValues.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			MyData.UnitsLength->Data(),
			MyData.UnitsLength->ElementCount(),
			MyData.UnitsLength->Shape().Data() + MyData.UnitsLength->Rank() - LengthAxis,
			LengthAxis
		)
	);
	MyData.OrtValues.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			MyData.F0Embed->Data(),
			MyData.F0Embed->ElementCount(),
			MyData.F0Embed->Shape().Data() + MyData.F0Embed->Rank() - F0Axis,
			F0Axis
		)
	);
	if (HasSpeakerEmbedding())
	{
		auto& SpeakerIdShape = _MyInputDims[3];
		const auto SpeakerIdAxis = SpeakerIdShape.Size();
		MyData.OrtValues.emplace_back(
		   Ort::Value::CreateTensor(
			   *_MyMemoryInfo,
			   MyData.SpeakerId->Data(),
			   MyData.SpeakerId->ElementCount(),
			   MyData.SpeakerId->Shape().Data() + MyData.SpeakerId->Rank() - SpeakerIdAxis,
			   SpeakerIdAxis
		   )
	   );
	}

	return MyData;
}

_D_Dragonian_Lib_Lib_Singing_Voice_Conversion_End