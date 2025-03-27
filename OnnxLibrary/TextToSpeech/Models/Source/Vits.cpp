#include "../Vits.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_Lib_Text_To_Speech_Header

namespace Vits
{
	SpeakerEmbedding::SpeakerEmbedding(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger
	) : OnnxModelBase(_Environment, Params.ModelPaths.at(L"Embedding"), _Logger)
	{
		if (Params.Parameters.contains(L"GinChannel"))
			_MyGinChannel = std::stoll(Params.Parameters.at(L"GinChannel"));
		else
			LogInfo(L"hParameter \"GinChannel\" not found, use default value: 256!");

		if (Params.Parameters.contains(L"SpeakerCount"))
			_MyEmbeddingCount = std::stoll(Params.Parameters.at(L"SpeakerCount"));
		else
			LogInfo(L"hParameter \"SpeakerCount\" not found, use default value: 1!");

		if (_MyInputCount != 1)
			_D_Dragonian_Lib_Throw_Exception("Input count must be 1!");
		if (_MyOutputCount != 1)
			_D_Dragonian_Lib_Throw_Exception("Output count must be 1!");

		bool Found = false;
		for (const auto& i : _MyOutputDims[0])
		{
			if (i == _MyGinChannel)
			{
				Found = true;
				break;
			}
		}
		if (!Found)
			_D_Dragonian_Lib_Throw_Exception("Gin channel mismatch! expected: " + std::to_string(_MyGinChannel));

		Int64 SpeakerID = 0;
		Ort::Value InputTensor = Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			&SpeakerID,
			1,
			_MyInputDims.Front().Data(),
			_MyInputCount
		);

		_MyEmbedding = Tensor<Float, 2, Device::CPU>::New({ _MyEmbeddingCount, _MyGinChannel });
		for (Int64 j = 0; j < _MyEmbeddingCount; ++j)
		{
			const auto _MyEmb = _MyEmbedding.Data() + j * _MyGinChannel;
			*InputTensor.GetTensorMutableData<Int64>() = j;
			Ort::Value OutputTensor{ nullptr };
			_D_Dragonian_Lib_Rethrow_Block(
				_MyModel->Run(
					Ort::RunOptions{ nullptr },
					_MyInputNames.Data(),
					&InputTensor,
					_MyInputCount,
					_MyOutputNames.Data(),
					&OutputTensor,
					_MyOutputCount
				);
			);
			const auto* Emb = OutputTensor.GetTensorData<Float>();
			memcpy(_MyEmb, Emb, _MyGinChannel * sizeof(Float));
		}
	}

	Tensor<Float, 2, Device::CPU> SpeakerEmbedding::Forward(
		const Tensor<Float, 2, Device::CPU>& Input
	) const
	{
		const auto [Batch, SpeakerCount] = Input.Shape().RawArray();
		if (SpeakerCount != _MyEmbeddingCount)
			_D_Dragonian_Lib_Throw_Exception(
				"Speaker count mismatch! expected: " +
				std::to_string(_MyEmbeddingCount) +
				", got: " +
				std::to_string(SpeakerCount)
			);
		const auto ICont = Input.Continuous().Evaluate();

		auto Result = Tensor<Float, 2, Device::CPU>::New({ Batch, _MyGinChannel });
		for (Int64 i = 0; i < Batch; ++i)
		{
			auto CurBatch = Result[i];
			CurBatch = 0.f;
			for (Int64 j = 0; j < SpeakerCount; ++j)
			{
				const auto Factor = *(ICont.Data() + i * SpeakerCount + j);
				CurBatch += _MyEmbedding[j] * Factor;
			}
		}

		return Result.Evaluate();
	}

	Encoder::Encoder(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger
	) : OnnxModelBase(_Environment, Params.ModelPaths.at(L"Encoder"), _Logger)
	{
		if (Params.Parameters.contains(L"HasLength"))
			_HasLength = Params.Parameters.at(L"HasLength") == L"true" || Params.Parameters.at(L"HasLength") == L"True";
		else
			LogInfo(L"hParameter \"HasLength\" not found, use default value: true!");

		if (Params.Parameters.contains(L"HasEmotion"))
			_HasEmotion = Params.Parameters.at(L"HasEmotion") == L"true" || Params.Parameters.at(L"HasEmotion") == L"True";
		else
			LogInfo(L"hParameter \"HasEmotion\" not found, use default value: false!");

		if (Params.Parameters.contains(L"HasTone"))
			_HasTone = Params.Parameters.at(L"HasTone") == L"true" || Params.Parameters.at(L"HasTone") == L"True";
		else
			LogInfo(L"hParameter \"HasTone\" not found, use default value: false!");

		if (Params.Parameters.contains(L"HasLanguage"))
			_HasLanguage = Params.Parameters.at(L"HasLanguage") == L"true" || Params.Parameters.at(L"HasLanguage") == L"True";
		else
			LogInfo(L"hParameter \"HasLanguage\" not found, use default value: false!");

		if (Params.Parameters.contains(L"HasBert"))
			_HasBert = Params.Parameters.at(L"HasBert") == L"true" || Params.Parameters.at(L"HasBert") == L"True";
		else
			LogInfo(L"hParameter \"HasBert\" not found, use default value: false!");

		if (Params.Parameters.contains(L"HasClap"))
			_HasClap = Params.Parameters.at(L"HasClap") == L"true" || Params.Parameters.at(L"HasClap") == L"True";
		else
			LogInfo(L"hParameter \"HasClap\" not found, use default value: false!");

		if (Params.Parameters.contains(L"HasSpeaker"))
			_HasSpeaker = Params.Parameters.at(L"HasSpeaker") == L"true" || Params.Parameters.at(L"HasSpeaker") == L"True";
		else
			LogInfo(L"hParameter \"HasSpeaker\" not found, use default value: false!");

		if (Params.Parameters.contains(L"EncoderSpeaker"))
		{
			bool _MyEncoderSpeaker = Params.Parameters.at(L"EncoderSpeaker") == L"true" ||
				Params.Parameters.at(L"EncoderSpeaker") == L"True";
			_HasSpeaker = _HasSpeaker && _MyEncoderSpeaker;
		}
		else
		{
			_HasSpeaker = false;
			LogInfo(L"hParameter \"EncoderSpeaker\" not found, set \"HasSpeaker\" to value: \"false\"!");
		}

		if (Params.Parameters.contains(L"HasVQ"))
			_HasVQ = Params.Parameters.at(L"HasVQ") == L"true" || Params.Parameters.at(L"HasVQ") == L"True";
		else
			LogInfo(L"hParameter \"HasVQ\" not found, use default value: false!");

		if (_HasEmotion)
		{
			if (Params.Parameters.contains(L"EmotionDims"))
				_MyEmotionDims = std::stoll(Params.Parameters.at(L"EmotionDims"));
			else
				LogInfo(L"hParameter \"EmotionDims\" not found, use default value: 1024!");
		}

		if (_HasBert)
		{
			if (Params.Parameters.contains(L"BertDims"))
				_MyBertDims = std::stoll(Params.Parameters.at(L"BertDims"));
			else
				LogInfo(L"hParameter \"BertDims\" not found, use default value: 2048!");

			if (Params.Parameters.contains(L"BertCount"))
				_MyBertCount = std::stoll(Params.Parameters.at(L"BertCount"));
			else
				LogInfo(L"hParameter \"BertCount\" not found, use default value: 3!");
		}

		if (_HasClap)
		{
			if (Params.Parameters.contains(L"ClapDims"))
				_MyClapDims = std::stoll(Params.Parameters.at(L"ClapDims"));
			else
				LogInfo(L"hParameter \"ClapDims\" not found, use default value: 512!");
		}

		if (_HasSpeaker)
		{
			if (Params.Parameters.contains(L"GinChannel"))
				_MyGinChannel = std::stoll(Params.Parameters.at(L"GinChannel"));
			else
				LogInfo(L"hParameter \"GinChannel\" not found, use default value: 256!");
		}

		if (_HasVQ)
		{
			if (Params.Parameters.contains(L"VQCodebookSize"))
				_MyVQCodebookSize = std::stoll(Params.Parameters.at(L"VQCodebookSize"));
			else
				LogInfo(L"hParameter \"VQCodebookSize\" not found, use default value: 10!");
			if (Params.Parameters.contains(L"SpeakerCount"))
				_MySpeakerCount = std::stoll(Params.Parameters.at(L"SpeakerCount"));
			else
				LogInfo(L"hParameter \"SpeakerCount\" not found, use default value: 1!");
		}

		if (_MyOutputCount < 3 || _MyOutputCount > 4)
			_D_Dragonian_Lib_Throw_Exception("Output count must be 3 or 4!");

		Int64 Axis = 0;

		if (_HasLength)
		{
			bool Found = false;
			for (const auto& i : _MyInputDims[++Axis])
				if (i == 1)
				{
					Found = true;
					break;
				}
			if (!Found)
				_D_Dragonian_Lib_Throw_Exception("Length axis mismatch!");
		}

		if (_HasEmotion)
		{
			bool Found = false;
			for (const auto& i : _MyInputDims[++Axis])
				if (i == _MyEmotionDims)
				{
					Found = true;
					break;
				}
			if (!Found)
				_D_Dragonian_Lib_Throw_Exception("Emotion axis mismatch!");
		}

		if (_HasTone)
			++Axis;

		if (_HasLanguage)
			++Axis;

		if (_HasBert)
		{
			for (Int64 b = 0; b < _MyBertCount; ++b)
			{
				bool Found = false;
				for (const auto& i : _MyInputDims[++Axis])
					if (i == _MyBertDims)
					{
						Found = true;
						break;
					}
				if (!Found)
					_D_Dragonian_Lib_Throw_Exception("Bert axis mismatch!");
			}
		}

		if (_HasClap)
		{
			bool Found = false;
			for (const auto& i : _MyInputDims[++Axis])
				if (i == _MyClapDims)
				{
					Found = true;
					break;
				}
			if (!Found)
				_D_Dragonian_Lib_Throw_Exception("Clap axis mismatch!");
		}

		if (_HasSpeaker)
		{
			bool Found = false;
			for (const auto& i : _MyInputDims[++Axis])
				if (i == _MyGinChannel)
				{
					Found = true;
					break;
				}
			if (!Found)
				_D_Dragonian_Lib_Throw_Exception("Speaker axis mismatch!");
		}
	}

	Encoder::Encoded Encoder::Forward(
		const Tensor<Int64, 2, Device::CPU>& PhonemeIds,
		const std::optional<const Tensor<Float, 2, Device::CPU>>& SpeakerEmbedding,
		const std::optional<const Tensor<Float, 2, Device::CPU>>& Emotion,
		const std::optional<const Tensor<Int64, 2, Device::CPU>>& ToneIds,
		const std::optional<const Tensor<Int64, 2, Device::CPU>>& LanguageIds,
		const std::optional<const Tensor<Float, 4, Device::CPU>>& Bert,
		const std::optional<Tensor<Float, 2, Device::CPU>>& Clap,
		Int64 VQIndex,
		Int64 SpeakerIndex
	) const
	{
		if (PhonemeIds.Null())
			_D_Dragonian_Lib_Throw_Exception("Phoneme is required!");
		const auto [Batch, PhonemeCount] = PhonemeIds.Shape().RawArray();
		if (_HasEmotion)
		{
			if (!Emotion.has_value() || Emotion->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Emotion is required!"
				);
			if (Emotion->Shape(0) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Emotion batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(Emotion->Shape(0))
				);
			if (Emotion->Shape(0) != _MyEmotionDims)
				_D_Dragonian_Lib_Throw_Exception(
					"Emotion dims mismatch! required: " +
					std::to_string(_MyEmotionDims) +
					", got: " +
					std::to_string(Emotion->Shape(1))
				);
		}
		if (_HasTone)
		{
			if (!ToneIds.has_value() || ToneIds->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Tone is required! required"
				);
			if (ToneIds->Shape(0) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Tone batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(ToneIds->Shape(0))
				);
			if (ToneIds->Shape(1) != PhonemeCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Tone phoneme count mismatch! required: " +
					std::to_string(PhonemeCount) +
					", got: " +
					std::to_string(ToneIds->Shape(1))
				);
		}
		if (_HasLanguage)
		{
			if (!LanguageIds.has_value() || LanguageIds->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Language is required!"
				);
			if (LanguageIds->Shape(0) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Language batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(LanguageIds->Shape(0))
				);
			if (LanguageIds->Shape(1) != PhonemeCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Language phoneme count mismatch! required: " +
					std::to_string(PhonemeCount) +
					", got: " +
					std::to_string(LanguageIds->Shape(1))
				);
		}
		if (_HasBert)
		{
			if (!Bert.has_value() || Bert->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Bert is required!"
				);
			if (Bert->Shape(0) != _MyBertCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Bert count mismatch! required: " +
					std::to_string(_MyBertCount) +
					", got: " +
					std::to_string(Bert->Shape(0))
				);
			if (Bert->Shape(1) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Bert batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(Bert->Shape(1))
				);
			if (Bert->Shape(2) != PhonemeCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Bert phoneme count mismatch! required: " +
					std::to_string(PhonemeCount) +
					", got: " +
					std::to_string(Bert->Shape(2))
				);
			if (Bert->Shape(3) != _MyBertDims)
				_D_Dragonian_Lib_Throw_Exception(
					"Bert dims mismatch! required: " +
					std::to_string(_MyBertDims) +
					", got: " +
					std::to_string(Bert->Shape(3))
				);
		}
		if (_HasClap)
		{
			if (!Clap.has_value() || Clap->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Clap is required!"
				);
			if (Clap->Shape(0) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Clap batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(Clap->Shape(0))
				);
			if (Clap->Shape(1) != _MyClapDims)
				_D_Dragonian_Lib_Throw_Exception(
					"Clap dims mismatch! required: " +
					std::to_string(_MyClapDims) +
					", got: " +
					std::to_string(Clap->Shape(1))
				);
		}
		if (_HasSpeaker)
		{
			if (!SpeakerEmbedding.has_value() || SpeakerEmbedding->Null())
				_D_Dragonian_Lib_Throw_Exception(
					"Speaker embedding is required!"
				);
			if (SpeakerEmbedding->Shape(0) != Batch)
				_D_Dragonian_Lib_Throw_Exception(
					"Speaker embedding batch mismatch! required: " +
					std::to_string(Batch) +
					", got: " +
					std::to_string(SpeakerEmbedding->Shape(0))
				);
			if (SpeakerEmbedding->Shape(1) != _MyGinChannel)
				_D_Dragonian_Lib_Throw_Exception(
					"Speaker embedding dims mismatch! required: " +
					std::to_string(_MyGinChannel) +
					", got: " +
					std::to_string(SpeakerEmbedding->Shape(1))
				);
		}

		if (_HasVQ)
		{
			if (VQIndex < 0 || VQIndex >= _MyVQCodebookSize)
				_D_Dragonian_Lib_Throw_Exception(
					"VQ index out of range! required: [0, " +
					std::to_string(_MyVQCodebookSize) +
					"), got: " +
					std::to_string(VQIndex)
				);
			if (SpeakerIndex < 0 || SpeakerIndex >= _MySpeakerCount)
				_D_Dragonian_Lib_Throw_Exception(
					"Speaker index out of range! required: [0, " +
					std::to_string(_MySpeakerCount) +
					"), got: " +
					std::to_string(SpeakerIndex)
				);
		}

		InputTensorsType Inputs;
		Int64 Axis = 0;

		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*_MyMemoryInfo,
					PhonemeIds,
					_MyInputTypes[Axis],
					_MyInputDims[Axis],
					{ L"BatchSize", L"PhonemeLength" },
					"PhonemeIds",
					GetLoggerPtr()
				)
			);
		);

		if (_HasLength)
		{
			++Axis;
			Tensor<Int64, 2, Device::CPU> Lengths = Tensor<Int64, 2, Device::CPU>::ConstantOf(
				{ Batch, 1 }, PhonemeCount
			);
			_D_Dragonian_Lib_Rethrow_Block(
				Inputs.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						Lengths,
						_MyInputTypes[Axis],
						_MyInputDims[Axis],
						{ L"BatchSize", L"1" },
						"Lengths",
						GetLoggerPtr()
					)
				);
			);
		}

		if (_HasEmotion)
		{
			++Axis;
			_D_Dragonian_Lib_Rethrow_Block(
				Inputs.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						*Emotion,
						_MyInputTypes[Axis],
						_MyInputDims[Axis],
						{ L"BatchSize", L"EmotionDims" },
						"Emotion",
						GetLoggerPtr()
					)
				);
			);
		}

		if (_HasTone)
		{
			++Axis;
			_D_Dragonian_Lib_Rethrow_Block(
			   Inputs.Emplace(
				   CheckAndTryCreateValueFromTensor(
					   *_MyMemoryInfo,
					   *ToneIds,
					   _MyInputTypes[Axis],
					   _MyInputDims[Axis],
					   { L"BatchSize", L"PhonemeLength" },
					   "ToneIds",
					   GetLoggerPtr()
				   )
			   );
		   );
		}

		if (_HasLanguage)
		{
			++Axis;
			_D_Dragonian_Lib_Rethrow_Block(
			   Inputs.Emplace(
				   CheckAndTryCreateValueFromTensor(
					   *_MyMemoryInfo,
					   *LanguageIds,
					   _MyInputTypes[Axis],
					   _MyInputDims[Axis],
					   { L"BatchSize", L"PhonemeLength" },
					   "LanguageIds",
					   GetLoggerPtr()
				   )
			   );
		   );
		}

		if (_HasBert)
		{
			for (Int64 i = 0; i < _MyBertCount; ++i)
			{
				++Axis;
				auto BertTensor = (*Bert)[i];
				const auto BertName = "Bert-" + std::to_string(i);
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*_MyMemoryInfo,
							BertTensor,
							_MyInputTypes[Axis],
							_MyInputDims[Axis],
							{ L"BatchSize", L"PhonemeLength", L"BertDims" },
							BertName.c_str(),
							GetLoggerPtr()
						)
					);
				);
			}
		}

		if (_HasClap)
		{
			++Axis;
			if (_MyInputDims[Axis].Back() == 1)
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*_MyMemoryInfo,
							Clap->UnSqueeze(-1),
							_MyInputTypes[Axis],
							_MyInputDims[Axis],
							{ L"BatchSize", L"ClapDims", L"1" },
							"Clap",
							GetLoggerPtr()
						)
					);
				);
			else
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*_MyMemoryInfo,
							Clap->UnSqueeze(-1),
							_MyInputTypes[Axis],
							_MyInputDims[Axis],
							{ L"BatchSize", L"ClapDims" },
							"Clap",
							GetLoggerPtr()
						)
					);
				);
		}

		if (_HasSpeaker)
		{
			++Axis;
			if (_MyInputDims[Axis].Back() == 1)
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*_MyMemoryInfo,
							SpeakerEmbedding->UnSqueeze(-1),
							_MyInputTypes[Axis],
							_MyInputDims[Axis],
							{ L"BatchSize", L"GinChannel", L"1" },
							"SpeakerEmbedding",
							GetLoggerPtr()
						)
					);
				);
			else
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*_MyMemoryInfo,
							*SpeakerEmbedding,
							_MyInputTypes[Axis],
							_MyInputDims[Axis],
							{ L"BatchSize", L"GinChannel" },
							"SpeakerEmbedding",
							GetLoggerPtr()
						)
					);
				);
		}

		if (_HasVQ)
		{
			_D_Dragonian_Lib_Rethrow_Block(
				Inputs.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						Tensor<Int64, 2, Device::CPU>::ConstantOf({ Batch, 1 }, VQIndex),
						_MyInputTypes[++Axis],
						_MyInputDims[Axis],
						{ L"BatchSize", L"1" },
						"VQIndex",
						GetLoggerPtr()
					)
				);
			);
			_D_Dragonian_Lib_Rethrow_Block(
				Inputs.Emplace(
					CheckAndTryCreateValueFromTensor(
						*_MyMemoryInfo,
						Tensor<Int64, 2, Device::CPU>::ConstantOf({ Batch, 1 }, SpeakerIndex),
						_MyInputTypes[++Axis],
						_MyInputDims[Axis],
						{ L"BatchSize", L"1" },
						"SpeakerIndex",
						GetLoggerPtr()
					)
				);
			);
		}

		OrtTuple OutputTensors;
		_D_Dragonian_Lib_Rethrow_Block(
			OutputTensors = RunModel(
				Inputs
			);
		);

		Encoded Result;

		{
			const auto OutputXShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputXAxis = OutputXShape.size();
			Dimensions<3> XShape;
			XShape.AssignConstant(1, 0, 3 - OutputXAxis);
			XShape.Assign(OutputXShape.data(), 3 - OutputXAxis);
			Result.X = CreateTensorViewFromOrtValue<Float>(
				std::move(OutputTensors[0]),
				XShape
			);
		}

		{
			const auto OutputM_pShape = OutputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputM_pAxis = OutputM_pShape.size();
			Dimensions<3> M_pShape;
			M_pShape.AssignConstant(1, 0, 3 - OutputM_pAxis);
			M_pShape.Assign(OutputM_pShape.data(), 3 - OutputM_pAxis);
			Result.M_p = CreateTensorViewFromOrtValue<Float>(
				std::move(OutputTensors[1]),
				M_pShape
			);
		}

		{
			const auto OutputLogs_pShape = OutputTensors[2].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputLogs_pAxis = OutputLogs_pShape.size();
			Dimensions<3> Logs_pShape;
			Logs_pShape.AssignConstant(1, 0, 3 - OutputLogs_pAxis);
			Logs_pShape.Assign(OutputLogs_pShape.data(), 3 - OutputLogs_pAxis);
			Result.Logs_p = CreateTensorViewFromOrtValue<Float>(
				std::move(OutputTensors[2]),
				Logs_pShape
			);
		}

		{
			const auto OutputX_maskShape = OutputTensors[3].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputX_maskAxis = OutputX_maskShape.size();
			Dimensions<3> X_maskShape;
			X_maskShape.AssignConstant(1, 0, 3 - OutputX_maskAxis);
			X_maskShape.Assign(OutputX_maskShape.data(), 3 - OutputX_maskAxis);
			Result.X_mask = CreateTensorViewFromOrtValue<Float>(
				std::move(OutputTensors[3]),
				X_maskShape
			);
		}

		return Result;
	}

	DurationPredictor::DurationPredictor(
		const OnnxRuntimeEnvironment& _Environment,
		const HParams& Params,
		const std::shared_ptr<Logger>& _Logger
	) :
	_MyDP(
		_Environment,
		Params.ModelPaths.contains(L"DP") ? Params.ModelPaths.at(L"DP") : L"",
		_Logger,
		false
	),
	_MySDP(
		_Environment,
		Params.ModelPaths.contains(L"SDP") ? Params.ModelPaths.at(L"SDP") : L"",
		_Logger,
		false
	)
	{
		if (Params.Parameters.contains(L"HasSpeaker"))
			_HasSpeaker = Params.Parameters.at(L"HasSpeaker") == L"true" || Params.Parameters.at(L"HasSpeaker") == L"True";
		else
			_Logger->LogInfo(L"hParameter \"HasSpeaker\" not found, use default value: false!");

		if (_HasSpeaker)
		{
			if (Params.Parameters.contains(L"GinChannel"))
				_MyGinChannel = std::stoll(Params.Parameters.at(L"GinChannel"));
			else
				_Logger->LogInfo(L"hParameter \"GinChannel\" not found, use default value: 256!");
		}

		if (_MyDP)
		{
			if (_MyDP.GetOutputCount() != 1)
				_D_Dragonian_Lib_Throw_Exception("Output count must be 1!");
			if (_HasSpeaker)
			{
				const auto& GDims = _MyDP.GetInputDims().Back();
				bool Found = false;
				for (const auto& i : GDims)
				{
					if (i == _MyGinChannel)
					{
						Found = true;
						break;
					}
				}
				if (!Found)
					_D_Dragonian_Lib_Throw_Exception("Gin channel mismatch! expected: " + std::to_string(_MyGinChannel));
			}
		}
		if (_MySDP)
		{
			if (_MySDP.GetOutputCount() != 1)
				_D_Dragonian_Lib_Throw_Exception("Output count must be 1!");
			if (_HasSpeaker)
			{
				const auto& GDims = _MySDP.GetInputDims().Back();
				bool Found = false;
				for (const auto& i : GDims)
				{
					if (i == _MyGinChannel)
					{
						Found = true;
						break;
					}
				}
				if (!Found)
					_D_Dragonian_Lib_Throw_Exception("Gin channel mismatch! expected: " + std::to_string(_MyGinChannel));
			}

			if (Params.Parameters.contains(L"ZinDims"))
				_MyZinDims = std::stoll(Params.Parameters.at(L"ZinDims"));
			else
				_Logger->LogInfo(L"hParameter \"ZinDims\" not found, use default value: 2!");
		}
	}

	Tensor<Float32, 3, Device::CPU> DurationPredictor::Forward(
		const Tensor<Float32, 3, Device::CPU>& X,
		const Tensor<Float32, 3, Device::CPU>& X_Mask,
		const std::optional<const Tensor<Float32, 2, Device::CPU>>& SpeakerEmbedding,
		float DurationPredictorNoiseScale,
		float SdpRatio,
		Int64 Seed
	) const
	{
		if (!_MyDP && !_MySDP)
			_D_Dragonian_Lib_Throw_Exception("You must have a duration predictor");
		if (X.Null())
			_D_Dragonian_Lib_Throw_Exception("X is required!");
		if (X_Mask.Null())
			_D_Dragonian_Lib_Throw_Exception("X_Mask is required!");
		if (_HasSpeaker && (!SpeakerEmbedding.has_value() || SpeakerEmbedding->Null()))
			_D_Dragonian_Lib_Throw_Exception("Speaker embedding is required!");

		Tensor<Float32, 3, Device::CPU> SdpResult, DpResult;
		
		InputTensorsType Inputs;

		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					X,
					GetInputType(0),
					GetInputDims(0),
					{ L"BatchSize", L"X_Axis_1", L"X_Axis_2" },
					"X",
					GetLogger()
				)
			);
		);

		_D_Dragonian_Lib_Rethrow_Block(
			Inputs.Emplace(
				CheckAndTryCreateValueFromTensor(
					*GetMemoryInfo(),
					X_Mask,
					GetInputType(1),
					GetInputDims(1),
					{ L"BatchSize", L"PhonemeLength" },
					"X_Mask",
					GetLogger()
				)
			);
		);

		if (_HasSpeaker)
		{
			auto Dims = GetInputDims(114);
			if (Dims.Back() == 1)
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*GetMemoryInfo(),
							SpeakerEmbedding->UnSqueeze(-1),
							GetInputType(114),
							GetInputDims(114),
							{ L"BatchSize", L"GinChannel", L"1" },
							"SpeakerEmbedding",
							GetLogger()
						)
					);
				);
			else
				_D_Dragonian_Lib_Rethrow_Block(
					Inputs.Emplace(
						CheckAndTryCreateValueFromTensor(
							*GetMemoryInfo(),
							*SpeakerEmbedding,
							GetInputType(114),
							GetInputDims(114),
							{ L"BatchSize", L"GinChannel" },
							"SpeakerEmbedding",
							GetLogger()
						)
					);
				);
		}

		if (_MyDP)
		{
			OrtTuple OutputTensors;
			_D_Dragonian_Lib_Rethrow_Block(
				OutputTensors = _MyDP.RunModel(
					Inputs
				);
			);
			const auto OutputShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputAxis = OutputShape.size();
			Dimensions<3> Shape;
			Shape.AssignConstant(1, 0, 3 - OutputAxis);
			Shape.Assign(OutputShape.data(), 3 - OutputAxis);
			DpResult = CreateTensorViewFromOrtValue<Float32>(
				std::move(OutputTensors[0]),
				Shape
			);
		}

		if (_MySDP)
		{
			SetRandomSeed(Seed);
			auto Zin = Functional::Randn(
				Dimensions<3>{ X.Shape(0), _MyZinDims, X.Shape(2) }
			)* DurationPredictorNoiseScale;
			_D_Dragonian_Lib_Rethrow_Block(
				Inputs.Insert(
					2,
					CheckAndTryCreateValueFromTensor(
						*_MySDP.GetMemoryInfo(),
						Zin,
						_MySDP.GetInputTypes()[2],
						_MySDP.GetInputDims()[2],
						{ L"BatchSize", L"ZinDims", L"X_Axis_2" },
						"Zin",
						_MySDP.GetLoggerPtr()
					)
				);
			);
			OrtTuple OutputTensors;
			_D_Dragonian_Lib_Rethrow_Block(
				OutputTensors = _MySDP.RunModel(
					Inputs
				);
			);
			const auto OutputShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
			const auto OutputAxis = OutputShape.size();
			Dimensions<3> Shape;
			Shape.AssignConstant(1, 0, 3 - OutputAxis);
			Shape.Assign(OutputShape.data(), 3 - OutputAxis);
			SdpResult = CreateTensorViewFromOrtValue<Float32>(
				std::move(OutputTensors[0]),
				Shape
			);
		}

		if (_MyDP && _MySDP)
			return ((DpResult * (1.f - SdpRatio)) + (SdpResult * SdpRatio)).Evaluate();
		if (_MyDP)
			return DpResult;
		return SdpResult;
	}

}

/*
Vits::Vits(
	const ModelHParams& _Config,
	const ProgressCallback& _ProgressCallback,
	const DurationCallback& _DurationCallback,
	ExecutionProviders ExecutionProvider_,
	unsigned DeviceID_,
	unsigned ThreadCount_
) : TextToSpeech(_Config, ExecutionProvider_, DeviceID_, ThreadCount_),
VitsType(_Config.VitsConfig.VitsType), DefBertSize(_Config.DefBertSize), VQCodeBookSize(_Config.VQCodeBookSize),
BertCount(_Config.BertCount), UseTone(_Config.UseTone), UseBert(_Config.UseBert), UseLength(_Config.UseLength),
UseLanguage(_Config.UseLanguageIds), EncoderG(_Config.EncoderG), ReferenceBert(_Config.ReferenceBert),
UseVQ(_Config.UseVQ), UseClap(_Config.UseClap), Emotion2Id(_Config.Emotion2Id)
{
	if (UseLanguage && Language2ID.empty())
		_D_Dragonian_Lib_Throw_Exception("Language Symbol Could Not Be Empty!");

	ProgressCallbackFunction = _ProgressCallback;
	CustomDurationCallback = _DurationCallback;
	ModelSamplingRate = _Config.SamplingRate;

	//InputNames
	if (UseLength)
		EncoderInputNames.emplace_back("x_lengths");
	if (UseTone)
		EncoderInputNames.emplace_back("t");
	if (Emotion)
	{
		EmotionVector = EmotionLoader(_Config.EmotionFilePath);
		EncoderInputNames.emplace_back("emo");
	}
	if (UseLanguage)
		EncoderInputNames.emplace_back("language");

	if (UseBert)
	{
		if (BertCount > 0 && BertCount < static_cast<int64_t>(VistBertInputNames.size()))
		{
			for (int64_t i = 0; i < BertCount; ++i)
			{
				EncoderInputNames.emplace_back(VistBertInputNames[i]);
			}
		}
	}

	if (UseClap)
		EncoderInputNames.emplace_back("emo");

	sessionDec = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.DecoderPath.c_str(), *SessionOptions);
	sessionEnc_p = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.EncoderPath.c_str(), *SessionOptions);
	sessionFlow = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.FlowPath.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.SpeakerEmbedding.c_str(), 0) != -1)
		sessionEmb = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.SpeakerEmbedding.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.DurationPredictorPath.c_str(), 0) != -1)
		sessionDp = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.DurationPredictorPath.c_str(), *SessionOptions);

	if (_waccess(_Config.VitsConfig.SDurationPredictorPath.c_str(), 0) != -1)
		sessionSdp = std::make_shared<Ort::Session>(*OnnxEnv, _Config.VitsConfig.SDurationPredictorPath.c_str(), *SessionOptions);

	if (!sessionDp && !sessionSdp)
		_D_Dragonian_Lib_Throw_Exception("You must have a duration predictor");

	if (sessionEmb)
	{
		if (EncoderG) EncoderInputNames.emplace_back("g");
		SdpInputNames.emplace_back("g");
		DpInputNames.emplace_back("g");
		FlowInputNames.emplace_back("g");
		DecInputNames.emplace_back("g");
	}

	if (VitsType == "BertVits" && sessionEnc_p->GetInputCount() == EncoderInputNames.size() + 2)
	{
		EncoderInputNames.emplace_back("vqidx");
		EncoderInputNames.emplace_back("sid");
		UseVQ = true;
	}
}

DragonianLibSTL::Vector<float> Vits::GetEmotionVector(
	const DragonianLibSTL::Vector<std::wstring>& EmotionSymbol
) const
{
	if (EmotionSymbol.Empty())
		return { 1024, 0.f, TemplateLibrary::CPUAllocator() };
	DragonianLibSTL::Vector Result(1024, 0.f);
	uint64_t mul = 0;
	for (const auto& EmotionSymb : EmotionSymbol)
	{
		size_t emoId;
		auto EmoID = Emotion2Id.find(EmotionSymb);
		if (EmoID != Emotion2Id.end())
			emoId = EmoID->second;
		else
			emoId = _wtoi64(EmotionSymb.c_str());

		auto emoVec = EmotionVector[emoId];
		for (size_t i = 0; i < 1024; ++i)
			Result[i] += (emoVec[i] - Result[i]) / (float)(mul + 1ull);
		++mul;
	}
	return Result;
}

DragonianLibSTL::Vector<float> Vits::Inference(
	TTSInputData& InputData,
	const TTSParams& Params
) const
{
	std::mt19937_64 Generate(Params.Seed);
	std::normal_distribution FloatRandFn(0.f, 1.f);

	const auto PhonemeSize = InputData._PhonemesIds.Size();

	if (InputData._SpeakerMixIds.empty())
		_D_Dragonian_Lib_Throw_Exception("SpeakerMixIds Could Not Be Empty!");

	if (InputData._PhonemesIds.Empty())
		_D_Dragonian_Lib_Throw_Exception("PhonemesIds Could Not Be Empty!");
	if (AddBlank && PhonemeSize < 3)
		_D_Dragonian_Lib_Throw_Exception("PhonemesIds Size Must Be Greater Than 3!");

	if (UseTone && InputData._Tones.Empty())
		_D_Dragonian_Lib_Throw_Exception("Tones Could Not Be Empty!");
	if (InputData._Tones.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._Tones.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> Tones;
			Tones.Reserve(PhonemeSize);
			for (auto Tone : InputData._Tones)
			{
				Tones.EmplaceBack(Tone);
				Tones.EmplaceBack(Tone);
			}
			Tones.EmplaceBack(InputData._Tones.Back());
			InputData._Tones = std::move(Tones);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Tones Size Not Matched!");
	}

	if (UseLanguage && InputData._LanguageIds.Empty())
		InputData._LanguageIds = { PhonemeSize, Params.LanguageID, TemplateLibrary::CPUAllocator() };
	if (InputData._LanguageIds.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._LanguageIds.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> LanguageIds;
			LanguageIds.Reserve(PhonemeSize);
			for (auto LanguageId : InputData._LanguageIds)
			{
				LanguageIds.EmplaceBack(LanguageId);
				LanguageIds.EmplaceBack(LanguageId);
			}
			LanguageIds.EmplaceBack(InputData._LanguageIds.Back());
			InputData._LanguageIds = std::move(LanguageIds);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("LanguageIds Size Not Matched!");
	}

	if (!InputData._Durations.Empty() && InputData._Durations.Size() != PhonemeSize)
	{
		if (AddBlank && InputData._Durations.Size() * 2 + 1 == PhonemeSize)
		{
			DragonianLibSTL::Vector<int64_t> Durations;
			Durations.Reserve(PhonemeSize);
			for (auto Duration : InputData._Durations)
			{
				Durations.EmplaceBack(Duration);
				Durations.EmplaceBack(Duration);
			}
			Durations.EmplaceBack(InputData._Durations.Back());
			InputData._Durations = std::move(Durations);
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Durations Size Not Matched!");
	}

	if (Emotion)
	{
		if (InputData._Emotion.Empty())
			InputData._Emotion = GetEmotionVector(Params.EmotionPrompt);
		if (InputData._Emotion.Size() != 1024)
			_D_Dragonian_Lib_Throw_Exception("Emotion Size Not Matched!");
	}

	if (UseBert)
	{
		if (InputData._BertVec.Empty())
			_D_Dragonian_Lib_Throw_Exception("BertVec Could Not Be Empty!");
		if (InputData._BertVec.Size() != static_cast<size_t>(BertCount))
			_D_Dragonian_Lib_Throw_Exception("BertVec Size Not Matched!");

		size_t BertBufferSize = 0;
		for (const auto& BertVec : InputData._BertVec)
		{
			if (!BertBufferSize)
				BertBufferSize = BertVec.Size();
			else if (BertBufferSize != BertVec.Size())
				_D_Dragonian_Lib_Throw_Exception("BertVec Size Not Matched!");
		}
		if (BertBufferSize == 0)
			_D_Dragonian_Lib_Throw_Exception("BertVec Could Not Be Empty!");

		for (auto& BertVec : InputData._BertVec)
			if (BertVec.Empty())
				BertVec = { BertBufferSize, 0.f, TemplateLibrary::CPUAllocator() };

		auto BertSize = BertBufferSize / InputData._BertDims;
		if (BertSize != PhonemeSize)
		{
			if (PhonemeSize != InputData._Token2Phoneme.Size())
				InputData._Token2Phoneme = GetAligments(PhonemeSize, BertSize);
			const auto TargetBertSize = PhonemeSize * InputData._BertDims;
			for (auto& BertVec : InputData._BertVec)
			{
				DragonianLibSTL::Vector<float> BertData(TargetBertSize);
				if (BertVec.Size() == BertBufferSize)
				{
					for (size_t IndexOfSrcVector = 0; IndexOfSrcVector < PhonemeSize; ++IndexOfSrcVector)
						memcpy(
							BertData.Data() + IndexOfSrcVector * InputData._BertDims,
							BertVec.Data() + (InputData._Token2Phoneme[IndexOfSrcVector]) * InputData._BertDims,
							InputData._BertDims * sizeof(float)
						);
				}
				else
					memset(BertData.Data(), 0, TargetBertSize * sizeof(float));
				BertVec = std::move(BertData);
			}
		}
	}

	std::vector<Ort::Value> EncoderInputs, EncoderOutputs;

	const int64_t TextSeqShape[2] = { 1, (int64_t)PhonemeSize };
	EncoderInputs.emplace_back(
		Ort::Value::CreateTensor(
			*MemoryInfo,
			InputData._PhonemesIds.Data(),
			PhonemeSize,
			TextSeqShape,
			2
		)
	);

	auto Length = (int64_t)PhonemeSize;
	constexpr int64_t LengthShape[1] = { 1 };
	if (UseLength)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				&Length,
				1,
				LengthShape,
				1
			)
		);

	constexpr int64_t EmotionShape[1] = { 1024 };
	if (Emotion)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._Emotion.Data(),
				1024,
				EmotionShape,
				1
			)
		);

	if (UseTone)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._Tones.Data(),
				PhonemeSize,
				TextSeqShape,
				2
			)
		);

	if (UseLanguage)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._LanguageIds.Data(),
				PhonemeSize,
				TextSeqShape,
				2
			)
		);

	int64_t BertShape[2] = { (int64_t)PhonemeSize, InputData._BertDims };
	if (UseBert)
		for (auto BertVec : InputData._BertVec)
			EncoderInputs.emplace_back(
				Ort::Value::CreateTensor(
					*MemoryInfo,
					BertVec.Data(),
					BertVec.Size(),
					BertShape,
					2
				)
			);

	constexpr int64_t ClapShape[2] = { 512, 1 };
	if (UseClap)
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				InputData._ClapVec.Data(),
				512,
				ClapShape,
				2
			)
		);


	DragonianLibSTL::Vector<float> SpeakerVector;
	int64_t SpeakerShape[3] = { 1, 0, 1 };
	if (sessionEmb)
	{
		SpeakerVector = GetSpeakerEmbedding(InputData._SpeakerMixIds);
		SpeakerShape[1] = (int64_t)SpeakerVector.Size();
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				SpeakerVector.Data(),
				SpeakerVector.Size(),
				SpeakerShape,
				3
			)
		);
	}

	int64_t VQIndices[] = { Params.VQIndex };
	int64_t SidIndices[] = { Params.SpeakerID };
	if (UseVQ)
	{
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				VQIndices,
				1,
				LengthShape,
				1
			)
		);
		EncoderInputs.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				SidIndices,
				1,
				LengthShape,
				1
			)
		);
	}

	try
	{
		EncoderOutputs = sessionEnc_p->Run(Ort::RunOptions{ nullptr },
			EncoderInputNames.data(),
			EncoderInputs.data(),
			EncoderInputs.size(),
			EncoderOutputNames.data(),
			EncoderOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: enc_p\n") + e.what()));
	}

	std::vector<float>
		m_p(EncoderOutputs[1].GetTensorData<float>(), EncoderOutputs[1].GetTensorData<float>() + EncoderOutputs[1].GetTensorTypeAndShapeInfo().GetElementCount()),
		logs_p(EncoderOutputs[2].GetTensorData<float>(), EncoderOutputs[2].GetTensorData<float>() + EncoderOutputs[2].GetTensorTypeAndShapeInfo().GetElementCount()),
		x_mask(EncoderOutputs[3].GetTensorData<float>(), EncoderOutputs[3].GetTensorData<float>() + EncoderOutputs[3].GetTensorTypeAndShapeInfo().GetElementCount());

	const auto xshape = EncoderOutputs[0].GetTensorTypeAndShapeInfo().GetShape();

	std::vector w_ceil(PhonemeSize, 1.f);
	if (InputData._Durations.Empty())
	{
		const int64_t zinputShape[3] = { xshape[0],2,xshape[2] };
		const int64_t zinputCount = xshape[0] * xshape[2] * 2;
		std::vector<float> zinput(zinputCount);
		for (auto& it : zinput)
			it = FloatRandFn(Generate) * Params.DurationPredictorNoiseScale;
		std::vector<Ort::Value> DurationPredictorInput;
		DurationPredictorInput.emplace_back(std::move(EncoderOutputs[0]));
		DurationPredictorInput.emplace_back(std::move(EncoderOutputs[3]));
		DurationPredictorInput.emplace_back(
			Ort::Value::CreateTensor(
				*MemoryInfo,
				zinput.data(),
				zinputCount,
				zinputShape,
				3
			)
		);
		if (sessionEmb)
			DurationPredictorInput.emplace_back(
				Ort::Value::CreateTensor(
					*MemoryInfo,
					SpeakerVector.Data(),
					SpeakerVector.Size(),
					SpeakerShape,
					3
				)
			);
		if (sessionSdp)
		{
			std::vector<Ort::Value> StochasticDurationPredictorOutput;
			try
			{
				StochasticDurationPredictorOutput = sessionSdp->Run(Ort::RunOptions{ nullptr },
					SdpInputNames.data(),
					DurationPredictorInput.data(),
					DurationPredictorInput.size(),
					SdpOutputNames.data(),
					SdpOutputNames.size());
			}
			catch (Ort::Exception& e)
			{
				_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dp\n") + e.what()));
			}
			const auto w_data = StochasticDurationPredictorOutput[0].GetTensorData<float>();
			const auto w_data_length = StochasticDurationPredictorOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
			if (w_data_length != w_ceil.size())
				w_ceil.resize(w_data_length, 0.f);
			float SdpFactor = 1.f - Params.FactorDpSdp;
			if (sessionDp)
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i] * SdpFactor) * x_mask[i] * Params.LengthScale);
			else
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i]) * x_mask[i] * Params.LengthScale);
		}
		if (sessionDp)
		{
			std::vector<Ort::Value> DurationPredictorOutput;
			DurationPredictorInput.erase(DurationPredictorInput.begin() + 2);
			try
			{
				DurationPredictorOutput = sessionDp->Run(Ort::RunOptions{ nullptr },
					DpInputNames.data(),
					DurationPredictorInput.data(),
					DurationPredictorInput.size(),
					DpOutputNames.data(),
					DpOutputNames.size());
			}
			catch (Ort::Exception& e)
			{
				_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dp\n") + e.what()));
			}
			const auto w_data = DurationPredictorOutput[0].GetTensorData<float>();
			const auto w_data_length = DurationPredictorOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
			if (w_data_length != w_ceil.size())
				w_ceil.resize(w_data_length, 0.f);
			if (sessionSdp)
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] += ceil(exp(w_data[i] * Params.FactorDpSdp) * x_mask[i] * Params.LengthScale);
			else
				for (size_t i = 0; i < w_ceil.size(); ++i)
					w_ceil[i] = ceil(exp(w_data[i]) * x_mask[i] * Params.LengthScale);
			CustomDurationCallback(w_ceil.data(), w_ceil.data() + w_ceil.size());
		}
	}
	else
		for (size_t i = 0; i < w_ceil.size(); ++i)
			w_ceil[i] = float(InputData._Durations[i]);

	const auto maskSize = x_mask.size();
	float y_length_f = 0.0;
	int64_t y_length;
	for (size_t i = 0; i < w_ceil.size(); ++i)
		y_length_f += w_ceil[i];
	if (y_length_f < 1.0f)
		y_length = 1;
	else
		y_length = (int64_t)y_length_f;

	auto attn = generatePath(w_ceil.data(), y_length, maskSize);
	std::vector logVec(192, std::vector(y_length, 0.0f));
	std::vector mpVec(192, std::vector(y_length, 0.0f));
	std::vector<float> nlogs_pData(192 * y_length);
	for (size_t i = 0; i < static_cast<size_t>(y_length); ++i)
	{
		for (size_t j = 0; j < 192; ++j)
		{
			for (size_t k = 0; k < maskSize; k++)
			{
				if (attn[i][k])
				{
					mpVec[j][i] += m_p[j * maskSize + k];
					logVec[j][i] += logs_p[j * maskSize + k];
				}
			}
			nlogs_pData[j * y_length + i] = mpVec[j][i] + FloatRandFn(Generate) * exp(logVec[j][i]) * Params.NoiseScale;
		}
	}
	std::vector y_mask(y_length, 1.0f);
	const int64_t zshape[3] = { 1,192,y_length };
	const int64_t yshape[3] = { 1,1,y_length };

	std::vector<Ort::Value> FlowDecInputs, FlowDecOutputs;
	FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
		*MemoryInfo, nlogs_pData.data(), 192 * y_length, zshape, 3));
	FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
		*MemoryInfo, y_mask.data(), y_length, yshape, 3));
	if (sessionEmb)
		FlowDecInputs.push_back(Ort::Value::CreateTensor<float>(
			*MemoryInfo, SpeakerVector.Data(), SpeakerVector.Size(), SpeakerShape, 3));

	try
	{
		FlowDecOutputs = sessionFlow->Run(Ort::RunOptions{ nullptr },
			FlowInputNames.data(),
			FlowDecInputs.data(),
			FlowDecInputs.size(),
			FlowOutputNames.data(),
			FlowOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dec & flow\n") + e.what()));
	}
	FlowDecInputs[0] = std::move(FlowDecOutputs[0]);
	if (sessionEmb)
		FlowDecInputs[1] = std::move(FlowDecInputs[2]);
	FlowDecInputs.pop_back();
	try
	{

		FlowDecOutputs = sessionDec->Run(Ort::RunOptions{ nullptr },
			DecInputNames.data(),
			FlowDecInputs.data(),
			FlowDecInputs.size(),
			DecOutputNames.data(),
			DecOutputNames.size());
	}
	catch (Ort::Exception& e)
	{
		_D_Dragonian_Lib_Throw_Exception((std::string("Locate: dec & flow\n") + e.what()));
	}
	const auto shapeOut = FlowDecOutputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	const auto outData = FlowDecOutputs[0].GetTensorData<float>();
	return { outData, outData + shapeOut };
}

DragonianLibSTL::Vector<float> Vits::GetSpeakerEmbedding(const std::map<int64_t, float>& SpeakerMixIds) const
{
	DragonianLibSTL::Vector<float> SpeakerVector;

	constexpr int64_t LengthShape[1] = { 1 };
	for (const auto& SpeakerMixId : SpeakerMixIds)
	{
		std::vector<Ort::Value> EmbiddingInput;
		std::vector<Ort::Value> EmbiddingOutput;
		if (SpeakerMixId.first >= SpeakerCount || SpeakerMixId.second < 0.0001f || SpeakerMixId.first < 0)
			continue;
		int64_t Character[1] = { SpeakerMixId.first };
		EmbiddingInput.push_back(Ort::Value::CreateTensor(
			*MemoryInfo, Character, 1, LengthShape, 1));
		try
		{
			EmbiddingOutput = sessionEmb->Run(Ort::RunOptions{ nullptr },
				EmbiddingInputNames.data(),
				EmbiddingInput.data(),
				EmbiddingInput.size(),
				EmbiddingOutputNames.data(),
				EmbiddingOutputNames.size());
		}
		catch (Ort::Exception& e)
		{
			_D_Dragonian_Lib_Throw_Exception((std::string("Locate: emb\n") + e.what()));
		}
		const auto GOutData = EmbiddingOutput[0].GetTensorData<float>();
		const auto GOutCount = EmbiddingOutput[0].GetTensorTypeAndShapeInfo().GetElementCount();
		if (SpeakerVector.Empty())
		{
			SpeakerVector = { GOutData, GOutData + GOutCount };
			for (auto& idx : SpeakerVector)
				idx *= SpeakerMixId.second;
		}
		else
			for (size_t i = 0; i < GOutCount; ++i)
				SpeakerVector[i] += GOutData[i] * SpeakerMixId.second;
	}

	return SpeakerVector;
}*/

_D_Dragonian_Lib_Lib_Text_To_Speech_End