#include "TensorRT/SuperResolution/MoeSuperResolution.hpp"
#include "TextToSpeech/Modules/Models/Header/Vits.hpp"
#include <iostream>
#include "AvCodec/AvCodec.h"
#include "G2P/G2PModule.hpp"

void ShowProgressBar(size_t progress, size_t total) {
	int barWidth = 70;
	float progressRatio = static_cast<float>(progress) / float(total);
	int pos = static_cast<int>(float(barWidth) * progressRatio);

	std::cout << "\r";
	std::cout.flush();
	std::cout << "[";
	for (int i = 0; i < barWidth; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progressRatio * 100.0) << "%  ";
}

size_t TotalStep = 0;
void ProgressCb(size_t a, size_t)
{
	ShowProgressBar(a, TotalStep);
}

void ProgressCbS(size_t a, size_t b)
{
	ShowProgressBar(a, b);
}

template <typename _Function>
void WithTimer(_Function&& _Func)
{
	auto start = std::chrono::high_resolution_clock::now();
	_Func();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "\nTime: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
}

static void TrtSr()
{
	using namespace DragonianLib::TensorRTLib::SuperResolution;
	MoeSR Model{
		LR"(D:\VSGIT\hat_lite_seed0721.onnx)",
		4,
		DragonianLib::TensorRTLib::TrtConfig{
			LR"(D:\VSGIT\hat_lite_seed0721.trt)",
			{DragonianLib::TensorRTLib::DynaShapeSlice{
					"DynaArg0",
				   nvinfer1::Dims4(1, 3, 64, 64),
				   nvinfer1::Dims4(1, 3, 128, 128),
				   nvinfer1::Dims4(1, 3, 192, 192)
			}},
			0,
			true,
			false,
			true,
			false,
			nvinfer1::ILogger::Severity::kWARNING,
			4
		},
		ProgressCbS
	};

	DragonianLib::ImageVideo::GdiInit();

	DragonianLib::ImageVideo::Image Image(
		LR"(D:\VSGIT\CG000000.BMP)",
		192,
		192,
		16,
		0.f,
		false/*,
		LR"(D:\VSGIT\CG000002-DEB.png)"*/
	);
	/*Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TN.png)", 1, 100))
		std::cout << "1-Complete!\n";
	Image.Transpose();
	if (Image.MergeWrite(LR"(D:\VSGIT\CG000002-TNN.png)", 1, 100))
		std::cout << "2-Complete!\n";*/

	WithTimer([&]() {
		Model.Infer(Image, 1);
		});

	if (Image.MergeWrite(LR"(D:\VSGIT\CG000000114514.BMP)", 4, 100))
		std::cout << "Complete!\n";

	DragonianLib::ImageVideo::GdiClose();
}

static void TTS()
{
	DragonianLib::TextToSpeech::ContextModel BertModel(
		LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Bert\deberta-v2-large-japanese)",
		DragonianLib::TextToSpeech::LibTTSModule::ExecutionProviders::CPU,
		1,
		8
	);
	auto G2PModule = DragonianLib::G2P::GetG2P(
		DragonianLib::G2P::GetG2PModuleList().front(),
		LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\G2P)"
	);
	std::shared_ptr<DragonianLib::TextToSpeech::TextToSpeech> VitsModel = std::make_shared<DragonianLib::TextToSpeech::Vits>(
		DragonianLib::TextToSpeech::ModelHParams{
			{
				"BertVits2",
				LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_enc_p.onnx)",
			   LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_emb.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_sdp.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_dp.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_flow.onnx)",
				LR"(D:\VSGIT\MoeVoiceStudio - TTS\Build\Release\Models\BertVits - 66 Speaker\BertVits - 66 Speaker_dec.onnx)",
			},
			{},
			44100,
			1024,
			10,
			3,
			24,
			512,
			1024,
			true,
			true,
			true,
			false,
			true,
			true,
			false,
			false,
			false,
			L"",
			{},
			{{ L"\u5bab\u56ed\u85b0[\u65e5]", 0ll },
			{ L"\u7267\u4e4b\u539f\u7fd4\u5b50[\u65e5]", 1ll },
			{ L"\u67ab\u539f\u4e07\u53f6[\u4e2d]", 2ll },
			{ L"\u73ca\u745a\u5bab\u5fc3\u6d77[\u4e2d]", 3ll },
			{ L"\u5bb5\u5bab[\u4e2d]", 4ll },
			{ L"\u8fbe\u8fbe\u5229\u4e9a[\u4e2d]", 5ll },
			{ L"\u7eb3\u897f\u59b2[\u4e2d]", 6ll },
			{ L"\u7eee\u826f\u826f[\u4e2d]", 7ll },
			{ L"\u6e29\u8fea[\u4e2d]", 8ll },
			{ L"\u795e\u91cc\u7eeb\u4eba[\u4e2d]", 9ll },
			{ L"\u4f18\u83c8[\u4e2d]", 10ll },
			{ L"\u795e\u91cc\u7eeb\u534e[\u4e2d]", 11ll },
			{ L"\u8299\u5b81\u5a1c[\u4e2d]", 12ll },
			{ L"\u82ad\u82ad\u62c9[\u4e2d]", 13ll },
			{ L"\u949f\u79bb[\u4e2d]", 14ll },
			{ L"\u80e1\u6843[\u4e2d]", 15ll },
			{ L"\u8fea\u5a1c\u6cfd\u9edb[\u4e2d]", 16ll },
			{ L"\u523b\u6674[\u4e2d]", 17ll },
			{ L"\u53ef\u8389[\u4e2d]", 18ll },
			{ L"\u5a1c\u7ef4\u5a05[\u4e2d]", 19ll },
			{ L"\u9b48[\u4e2d]", 20ll },
			{ L"\u96f7\u7535\u5c06\u519b[\u4e2d]", 21ll },
			{ L"\u59ae\u9732[\u4e2d]", 22ll },
			{ L"\u514b\u62c9\u62c9[\u4e2d]", 23ll },
			{ L"\u5361\u8299\u5361[\u4e2d]", 24ll },
			{ L"\u7b26\u7384[\u4e2d]", 25ll },
			{ L"\u5a1c\u5854\u838e[\u4e2d]", 26ll },
			{ L"\u5e0c\u513f[\u4e2d]", 27ll },
			{ L"\u4e09\u6708\u4e03[\u4e2d]", 28ll },
			{ L"\u7d20\u88f3[\u4e2d]", 29ll },
			{ L"\u9ed1\u5854[\u4e2d]", 30ll },
			{ L"\u827e\u4e1d\u59b2[\u4e2d]", 31ll },
			{ L"\u5f00\u62d3\u8005(\u5973)[\u4e2d]", 32ll },
			{ L"\u5e03\u6d1b\u59ae\u5a05[\u4e2d]", 33ll },
			{ L"\u59ec\u5b50[\u4e2d]", 34ll },
			{ L"\u8fea\u5a1c\u6cfd\u9edb[\u65e5]", 35ll },
			{ L"\u4f18\u83c8[\u65e5]", 36ll },
			{ L"\u949f\u79bb[\u65e5]", 37ll },
			{ L"\u6e29\u8fea[\u65e5]", 38ll },
			{ L"\u5a1c\u7ef4\u5a05[\u65e5]", 39ll },
			{ L"\u795e\u91cc\u7eeb\u4eba[\u65e5]", 40ll },
			{ L"\u67ab\u539f\u4e07\u53f6[\u65e5]", 41ll },
			{ L"\u523b\u6674[\u65e5]", 42ll },
			{ L"\u96f7\u7535\u5c06\u519b[\u65e5]", 43ll },
			{ L"\u7eb3\u897f\u59b2[\u65e5]", 44ll },
			{ L"\u795e\u91cc\u7eeb\u534e[\u65e5]", 45ll },
			{ L"\u7eee\u826f\u826f[\u65e5]", 46ll },
			{ L"\u59ae\u9732[\u65e5]", 47ll },
			{ L"\u9b48[\u65e5]", 48ll },
			{ L"\u5bb5\u5bab[\u65e5]", 49ll },
			{ L"\u80e1\u6843[\u65e5]", 50ll },
			{ L"\u8fbe\u8fbe\u5229\u4e9a[\u65e5]", 51ll },
			{ L"\u8299\u5b81\u5a1c[\u65e5]", 52ll },
			{ L"\u73ca\u745a\u5bab\u5fc3\u6d77[\u65e5]", 53ll },
			{ L"\u82ad\u82ad\u62c9[\u65e5]", 54ll },
			{ L"\u53ef\u8389[\u65e5]", 55ll },
			{ L"\u7b26\u7384[\u65e5]", 56ll },
			{ L"\u5a1c\u5854\u838e[\u65e5]", 57ll },
			{ L"\u7d20\u88f3[\u65e5]", 58ll },
			{ L"\u4e09\u6708\u4e03[\u65e5]", 59ll },
			{ L"\u9ed1\u5854[\u65e5]", 60ll },
			{ L"\u5f00\u62d3\u8005(\u5973)[\u65e5]", 61ll },
			{ L"\u5e0c\u513f[\u65e5]", 62ll },
			{ L"\u5e03\u6d1b\u59ae\u5a05[\u65e5]", 63ll },
			{ L"\u59ec\u5b50[\u65e5]", 64ll },
			{ L"\u5361\u8299\u5361[\u65e5]", 65ll },
			{ L"\u514b\u62c9\u62c9[\u65e5]", 66ll },
			{ L"\u827e\u4e1d\u59b2[\u65e5]", 67ll },
			},
			{
				{ L"_", 0ll }, { L"AA", 1ll }, { L"E", 2ll }, { L"EE", 3ll }, { L"En", 4ll }, { L"N", 5ll },
				{ L"OO", 6ll }, { L"V", 7ll }, { L"a", 8ll }, { L"a:", 9ll }, { L"aa", 10ll }, { L"ae", 11ll },
				{ L"ah", 12ll }, { L"ai", 13ll }, { L"an", 14ll }, { L"ang", 15ll }, { L"ao", 16ll }, { L"aw", 17ll },
				{ L"ay", 18ll }, { L"b", 19ll }, { L"by", 20ll }, { L"c", 21ll }, { L"ch", 22ll }, { L"d", 23ll },
				{ L"dh", 24ll }, { L"dy", 25ll }, { L"e", 26ll }, { L"e:", 27ll }, { L"eh", 28ll }, { L"ei", 29ll },
				{ L"en", 30ll }, { L"eng", 31ll }, { L"er", 32ll }, { L"ey", 33ll }, { L"f", 34ll }, { L"g", 35ll }, { L"gy", 36ll }, { L"h", 37ll }, { L"hh", 38ll }, { L"hy", 39ll }, { L"i", 40ll }, { L"i0", 41ll },
				{ L"i:", 42ll }, { L"ia", 43ll }, { L"ian", 44ll }, { L"iang", 45ll }, { L"iao", 46ll }, { L"ie", 47ll },
				{ L"ih", 48ll }, { L"in", 49ll }, { L"ing", 50ll }, { L"iong", 51ll }, { L"ir", 52ll }, { L"iu", 53ll },
				{ L"iy", 54ll }, { L"j", 55ll }, { L"jh", 56ll }, { L"k", 57ll }, { L"ky", 58ll }, { L"l", 59ll },
				{ L"m", 60ll }, { L"my", 61ll }, { L"n", 62ll }, { L"ng", 63ll }, { L"ny", 64ll }, { L"o", 65ll },
				{ L"o:", 66ll }, { L"ong", 67ll }, { L"ou", 68ll }, { L"ow", 69ll }, { L"oy", 70ll }, { L"p", 71ll },
				{ L"py", 72ll }, { L"q", 73ll }, { L"r", 74ll }, { L"ry", 75ll }, { L"s", 76ll }, { L"sh", 77ll },
				{ L"t", 78ll }, { L"th", 79ll }, { L"ts", 80ll }, { L"ty", 81ll }, { L"u", 82ll }, { L"u:", 83ll },
				{ L"ua", 84ll }, { L"uai", 85ll }, { L"uan", 86ll }, { L"uang", 87ll }, { L"uh", 88ll }, { L"ui", 89ll },
				{ L"un", 90ll }, { L"uo", 91ll }, { L"uw", 92ll }, { L"v", 93ll }, { L"van", 94ll }, { L"ve", 95ll },
				{ L"vn", 96ll }, { L"w", 97ll }, { L"x", 98ll }, { L"y", 99ll }, { L"z", 100ll }, { L"zh", 101ll },
				{ L"zy", 102ll }, { L"!", 103ll }, { L"?", 104ll }, { L"\u2026", 105ll }, { L",", 106ll },
				{ L".", 107ll }, { L"'", 108ll }, { L"-", 109ll }, { L"SP", 110ll }, { L"UNK", 111ll }},
			{{L"ZH", 0}, {L"JP", 1}, {L"EN", 2} },
		},
		ProgressCb,
		[](float*, const float*){},
		DragonianLib::TextToSpeech::LibTTSModule::ExecutionProviders::CPU,
		1,
		8
	);

	auto InputText = L"こんにちは,世界ー..元気?!";

	DragonianLib::TextToSpeech::TTSInputData InputData;

	auto [Phonemes, Tones] = G2PModule->Convert(
		InputText,
		"Japanese",
		(void*)1ull
	);

	setlocale(LC_ALL, "ja_JP");
	printf("Phonemes: ");
	for (auto& Phoneme : Phonemes)
	{
		wprintf(L"%s ", Phoneme.c_str());
	}
	printf("\nTones: ");
	for (auto& Tone : Tones)
	{
		printf("%lld ", Tone);
		Tone += 6;
	}
	printf("\n");

	InputData.SetPhonemes(std::move(Phonemes));
	InputData._Tones = std::move(Tones);
	InputData._LanguageIds = { InputData._Tones.Size(), 1ll, DragonianLib::GetMemoryProvider(DragonianLib::Device::CPU) };
	auto [BertVec, BertDims] = BertModel.Inference(InputText);
	InputData._BertVec.EmplaceBack(BertVec.Size(), 0.f);
	//InputData._BertVec.EmplaceBack(BertVec.Size(), 0.f);
	InputData._BertVec.EmplaceBack(BertVec);
	InputData._BertVec.EmplaceBack(BertVec.Size(), 0.f);
	InputData._BertDims = BertDims;

	DragonianLib::TextToSpeech::TTSParams Params;
	Params.LanguageID = 1;
	auto Audio = VitsModel->Inference(
		InputData,
		Params,
		true
	);

	DragonianLib::WritePCMData(
		LR"(D:\VSGIT\test.wav)",
		Audio,
		44100
	);
}

int main()
{
	return 0;
}