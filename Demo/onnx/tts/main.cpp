#include <iostream>

#include "OnnxLibrary/TextToSpeech/Models/Vits.hpp"

#ifndef DRAGONIANLIB_USE_SHARED_LIBS
#include "Libraries/AvCodec/AvCodec.h"
#include "OnnxLibrary/G2P/G2PW.hpp"
#include "OnnxLibrary/BertClap/Context.hpp"
#include "OnnxLibrary/TextToSpeech/Models/GPT-SoVits.hpp"
#include "Libraries/G2P/G2PModule.hpp"
#include "OnnxLibrary/UnitsEncoder/Register.hpp"
#include "OnnxLibrary/Vocoder/Register.hpp"

[[maybe_unused]] static auto InferG2PW(const std::wstring& Text)
{
	using namespace DragonianLib;
	static const auto Env = OnnxRuntime::CreateEnvironment({ Device::CPU, 0 });
	static G2P::CppPinYinConfigs Configs{
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/pypinyin_dict.json)",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/pypinyin_pinyin_dict.json)",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/bopomofo_to_pinyin_wo_tune_dict.json)",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/char_bopomofo_dict.json)"
	};
	static G2P::G2PWModelHParams gParams{
		&Configs,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/POLYPHONIC_CHARS.txt)",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/tokens.txt)",
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/G2P/G2PW.onnx)",
		&Env,
		&_D_Dragonian_Lib_Onnx_Runtime_Space GetDefaultLogger(),
		512
	};
	static G2P::G2PWModel G2PW(
		&gParams
	);

	G2P::CppPinYinParameters Parameters;
	Parameters.Style = G2P::CppPinYinParameters::TONE3;
	Parameters.NumberStyle = G2P::CppPinYinParameters::SPLITCHINESE;
	Parameters.Heteronym = false;
	Parameters.ReplaceASV = true;

	return G2PW.Convert(
		Text,
		"",
		&Parameters
	);
}

[[maybe_unused]] static void TestGptSoVits()
{
	using namespace DragonianLib;
	G2P::RegisterG2PModules(LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");
	auto Jap = G2P::New(L"BasicCleaner", LR"(C:\DataSpace\libsvc\PythonScript\SoVitsSvc4_0_SupportTensorRT\OnnxSoVits\G2P)");

	const auto Env = OnnxRuntime::CreateEnvironment({ Device::CPU, 0 });
	Env->SetExecutionMode(ORT_PARALLEL);
	Env->EnableMemPattern(false);

	/*OnnxRuntime::ContextModel::ContextModel Bert(
		Env,
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Bert.onnx)"
	);
	Dict::Tokenizer Tokenizer(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Vocab.json)",
		L"[CLS]",
		L"[SEP]",
		L"[EOS]",
		L"[UNK]"
	);*/
	OnnxRuntime::UnitsEncoder::HubertBase CnHubert(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\cnhubert.onnx)",
		Env,
		16000,
		768
	);

	Dict::IdsDict Text2Seq(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\symbols1.json)"
	);

	OnnxRuntime::Vocoder::VocoderBase BigVGan = OnnxRuntime::Vocoder::VocoderBase(
		LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\BigVGAN.onnx)",
		Env,
		24000,
		100
	);

	OnnxRuntime::Text2Speech::GptSoVits::T2SAR ARModel(
		Env,
		{
			{
				{
					L"Prompt",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\PromptProcessor.onnx)"
				},
			   {
					L"Decode",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\DecodeNextToken.onnx)"
			   }
			},
			0,
			{}
		}
	);

	OnnxRuntime::Text2Speech::GptSoVits::VQModel GSV(
		Env,
		{
			{
				{
					L"Vits",
				   LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\GptSoVits.onnx)"
				},
				{
					L"Extract",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\Extractor.onnx)"
				},
				{
					L"Cfm",
					LR"(D:\VSGIT\GPT-SoVITS-main\GPT_SoVITS\GPT-SoVITS-v3lora-20250228\GPT_SoVITS\onnx\GptSoVits_cfm.onnx)"
				}
			},
			32000,
			{}
		}
	);

	auto AudioInStream = AvCodec::OpenInputStream(
		LR"(C:\DataSpace\MediaProj\PlayList\ttt.wav)"
	);
	auto SrcAudio = AudioInStream.DecodeAll(
		44100
	).View(1, -1);

	std::wstring RefText = LR"(テレジアはもう戻らぬ旅路に就いたわ. 残された私は, "文明の存続"の中に存在するプログラムでしかない. アーミヤの思いがあったからこそ, 私はこの姿で現れたの. テレジアとは別人よ. たとえ私にかつての記憶が全てあったとしてもよ. )";
	std::wstring InputText = L"この会議室のことは覚えているわ. この席は, テレジアのために空けられているのかしら? いいえ……私は要らないわ. ";
	bool UV = false;

	auto [RefPhonemes, _] = Jap->Convert(
		RefText, "Japanese", &UV
	);
	auto [InputPhonemes, __] = Jap->Convert(
		InputText, "Japanese", &UV
	);

	auto RefPhonemeIds = Functional::FromVector(
		Text2Seq(RefPhonemes)
	).UnSqueeze(0);
	auto InputPhonemeIds = Functional::FromVector(
		Text2Seq(InputPhonemes)
	).UnSqueeze(0);

	auto SSL = CnHubert.Forward(
		SrcAudio.UnSqueeze(0), 44100
	);

	auto PhonemeIds = Functional::Cat(
		RefPhonemeIds, InputPhonemeIds, 1
	);
	auto BertFeature = Functional::Zeros(
		IDim(1, PhonemeIds.Size(1), 1024)
	);
	auto Prompt = GSV.ExtractLatent(
		SSL.Squeeze(0)
	).Squeeze(0);

	auto Ret = ARModel.Forward(
		PhonemeIds,
		Prompt,
		BertFeature,
		0.6f,
		0.6f,
		1.35f
	);

	auto Res = GSV.Forward(
		InputPhonemeIds,
		Ret,
		SrcAudio,
		44100,
		RefPhonemeIds,
		Prompt
	);

	auto AudioOutStream = AvCodec::OpenOutputStream(
		32000,
		LR"(C:\DataSpace\MediaProj\PlayList\Test.wav)"
	);
	AudioOutStream.EncodeAll(Res.GetCRng(), 32000);
}

#else

#endif

int main()
{
#ifndef DRAGONIANLIB_USE_SHARED_LIBS
	auto Enviroment = DragonianLib::OnnxRuntime::CreateOnnxRuntimeEnvironment({});

	DragonianLib::OnnxRuntime::Text2Speech::Vits::SynthesizerTrn Model{
		Enviroment,
		{
			{
				{L"Encoder", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_enc_p.onnx)"},
				{L"DP", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_dp.onnx)"},
				{L"SDP", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_sdp.onnx)"},
				{L"Flow", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_flow.onnx)"},
				{L"Decoder", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_dec.onnx)"},
				{L"Embedding", __DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/BertVits2.2PT/BertVits2.2PT_emb.onnx)"},
			},
			44100,
			{
				{L"SamplingRate",L"44100"},
				{L"HasLength",L"false"},
				{L"HasEmotion",L"false"},
				{L"HasTone",L"true"},
				{L"HasLanguage",L"true"},
				{L"HasBert",L"true"},
				{L"HasClap",L"true"},
				{L"HasSpeaker",L"true"},
				{L"EncoderSpeaker",L"true"},
				{L"HasVQ",L"false"},
				{L"EmotionDims",L"1"},
				{L"BertDims",L"1024"},
				{L"ClapDims",L"512"},
				{L"GinChannel",L"256"},
				{L"BertCount",L"3"},
				{L"VQCodebookSize",L"0"},
				{L"SpeakerCount",L"90"},
				{L"ZinDims",L"2"}
			}
		}
	};

	DragonianLib::OnnxRuntime::ContextModel::ContextModel ClapModel(
		Enviroment,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/Clap/clap-htsat-fused.onnx)"
	);
	DragonianLib::Dict::Tokenizer ClapTokenizer(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/Clap/clap-htsat-fused.json)"
	);

	DragonianLib::OnnxRuntime::ContextModel::ContextModel BertModel(
		Enviroment,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/Bert/chinese-roberta-wwm-ext-large/model.onnx)"
	);
	DragonianLib::Dict::Tokenizer BertTokenizer(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestModels/Bert/chinese-roberta-wwm-ext-large/Tokenizer.json)"
	);

	DragonianLib::Dict::IdsDict PhonemeDict(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/phoneme_dict.json)",
		L"UNK"
	);
	DragonianLib::Dict::Dict RepMapPh(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/opencpop-strict.json)"
	);
	DragonianLib::Dict::Dict RepMap(
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/PythonScript/Dict/RepMap.json)"
	);

	std::wstring Text = L"我是特蕾西娅，正在和普瑞塞斯打架";

	auto [_Phonemes, _Tones] = InferG2PW(Text);
	auto [Phonemes, InputSeq, Phoneme2Text, Tones] =
		[&](bool AddBlank)
		{
			auto NewPhs = decltype(_Phonemes)::NewEmpty();
			auto NewSeq = decltype(_Tones)::NewEmpty();
			auto NewP2T = decltype(_Tones)::NewEmpty();
			auto NewTones = decltype(_Tones)::NewEmpty();
			NewPhs.Reserve(_Phonemes.Size() * 3 + 2);
			NewSeq.Reserve(_Phonemes.Size() * 3 + 2);
			NewP2T.Reserve(_Phonemes.Size() * 3 + 2);
			NewTones.Reserve(_Phonemes.Size() * 3 + 2);
			auto Emplace = [&](const std::wstring& Token, size_t i)
				{
					NewPhs.EmplaceBack(Token);
					NewSeq.EmplaceBack(PhonemeDict[Token]);
					NewP2T.EmplaceBack(static_cast<DragonianLib::Int64>(i) + 1ll);
					NewTones.EmplaceBack(_Tones[i]);
					if (AddBlank)
					{
						NewPhs.EmplaceBack(L"_");
						NewSeq.EmplaceBack(PhonemeDict[L"_"]);
						NewP2T.EmplaceBack(1);
						NewTones.EmplaceBack(0);
					}
				};

			if (AddBlank)
			{
				NewPhs.EmplaceBack(L"_");
				NewSeq.EmplaceBack(PhonemeDict[L"_"]);
				NewP2T.EmplaceBack(1);
				NewTones.EmplaceBack(0);
			}
			for (size_t i = 0; i < _Phonemes.Size(); ++i)
			{
				auto RawPhoneme = _Phonemes[i];
				std::erase_if(
					RawPhoneme,
					[](wchar_t pr)
					{
						return std::ranges::contains(L"12345", pr);
					}
				);
				const auto& Tokens = RepMapPh.Search(RawPhoneme);
				if (&Tokens == &RepMapPh.GetUNK())
				{
					const auto& NewTokens = RepMap.Search(RawPhoneme);
					if (&NewTokens == &RepMap.GetUNK())
						Emplace(RawPhoneme, i);
					else
						for (auto& Token : NewTokens)
							Emplace(Token, i);
				}
				else
				{
					for (const auto& Phoneme : Tokens)
					{
						const auto& NewTokens = RepMap.Search(Phoneme);
						if (&NewTokens == &RepMap.GetUNK())
							Emplace(Phoneme, i);
						else
							for (auto& Token : NewTokens)
								Emplace(Token, i);
					}
				}
			}
			return std::make_tuple(
				std::move(NewPhs),
				std::move(NewSeq),
				std::move(NewP2T),
				std::move(NewTones)
			);
		}(true);
	int(L'\uFF01') == int(L'H');

	auto ClapTensor = ClapTokenizer({ ClapTokenizer.Tokenize(L"Happy") });
	auto ClapFeat = ClapModel.Forward(
		ClapTensor,
		DragonianLib::Functional::ZerosLike(ClapTensor),
		DragonianLib::Functional::OnesLike(ClapTensor)
	);

	auto BertTokens = BertTokenizer({ BertTokenizer.Tokenize(Text) });
	auto BertFeat = BertModel(
		BertTokens,
		DragonianLib::Functional::ZerosLike(BertTokens),
		DragonianLib::Functional::OnesLike(BertTokens),
		DragonianLib::Functional::FromVector(Phoneme2Text).UnSqueeze(0)
	).Padding({ DragonianLib::PadCount{0, 2} }, DragonianLib::PaddingType::Zero);
	
	auto Audio = Model.Forward(
		{},
		DragonianLib::Functional::FromVector(InputSeq).UnSqueeze(0),
		std::nullopt,
		std::nullopt,
		DragonianLib::Functional::FromVector(Tones).UnSqueeze(0),
		DragonianLib::Functional::Zeros<DragonianLib::Int64>(
			DragonianLib::Dimensions{ 1ll, static_cast<DragonianLib::Int64>(InputSeq.Size()) }
		),
		BertFeat.UnSqueeze(1),
		ClapFeat
	);

	auto AudioOutStream = DragonianLib::AvCodec::OpenOutputStream(
		44100,
		__DRAGONIANLIB_SOURCE_DIRECTORY LR"(/TestAudios/Test-TTS.wav)"
	);
	AudioOutStream.EncodeAll(
		Audio.GetCRng(), 44100
	);

#else

#endif
	return 0;
}