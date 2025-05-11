#include "DefControl.hpp"
#include "Page/SidePage.h"
#include "Libraries/Stft/Stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

#include <numbers>
#include <User/Mui_GlobalStorage.h>
#include <Render/Sound/Mui_DirectSound.h>

#ifdef _WIN32
#include <gdiplus.h>
#pragma comment(lib, "Msimg32.lib")
#pragma comment(lib, "gdiplus.lib")
#else
#error "Only Windows is supported."
#endif

static std::wstring OfnHelper(const TCHAR* szFilter, HWND hwndOwner, const TCHAR* lpstrDefExt = TEXT("wav"))
{
	constexpr long MaxPath = 512 * 8;
#ifdef WIN32
	std::vector<TCHAR> szFileName(MaxPath);
	std::vector<TCHAR> szTitleName(MaxPath);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lpstrFile = szFileName.data();
	ofn.nMaxFile = MaxPath;
	ofn.lpstrFileTitle = szTitleName.data();
	ofn.nMaxFileTitle = MaxPath;
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = hwndOwner;
	//constexpr TCHAR szFilter[] = TEXT("Audio (*.wav;*.mp3;*.ogg;*.flac;*.aac)\0*.wav;*.mp3;*.ogg;*.flac;*.aac\0");
	ofn.lpstrFilter = szFilter;
	ofn.lpstrTitle = nullptr;
	ofn.lpstrDefExt = lpstrDefExt;
	ofn.Flags = OFN_HIDEREADONLY | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;
	if (GetOpenFileName(&ofn))
	{
		std::wstring preFix = szFileName.data();
		return preFix;
	}
	return L"";
#else
#endif
}

static std::vector<std::wstring> OfnHelperMult(const TCHAR* szFilter, HWND hwndOwner, const TCHAR* lpstrDefExt = TEXT("wav"))
{
	constexpr long MaxPath = 1024 * 512;
	std::vector<std::wstring> OFNLIST;
#ifdef WIN32
	std::vector<TCHAR> szFileName(MaxPath);
	std::vector<TCHAR> szTitleName(MaxPath);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lpstrFile = szFileName.data();
	ofn.nMaxFile = MaxPath;
	ofn.lpstrFileTitle = szTitleName.data();
	ofn.nMaxFileTitle = MaxPath;
	ofn.lStructSize = sizeof(OPENFILENAME);
	ofn.hwndOwner = hwndOwner;
	//constexpr TCHAR szFilter[] = TEXT("Audio (*.wav;*.mp3;*.ogg;*.flac;*.aac)\0*.wav;*.mp3;*.ogg;*.flac;*.aac\0");
	ofn.lpstrFilter = szFilter;
	ofn.lpstrTitle = nullptr;
	ofn.lpstrDefExt = lpstrDefExt;
	ofn.Flags = OFN_HIDEREADONLY | OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_ALLOWMULTISELECT | OFN_EXPLORER;
	if (GetOpenFileName(&ofn))
	{
		auto filePtr = szFileName.data();
		std::wstring preFix = filePtr;
		filePtr += preFix.length() + 1;
		if (!*filePtr)
			OFNLIST.emplace_back(preFix);
		else
		{
			preFix += L'\\';
			while (*filePtr != 0)
			{
				std::wstring thisPath(filePtr);
				OFNLIST.emplace_back(preFix + thisPath);
				filePtr += thisPath.length() + 1;
			}
		}
	}
	return OFNLIST;
#else
#endif
}

namespace WndControls
{
	using namespace SimpleF0Labeler;

	constexpr size_t MaxCacheCount = 10;
	constexpr size_t UndoRedoMaxCount = 100;

	struct Enviroment  // NOLINT(cppcoreguidelines-special-member-functions)
	{
		Enviroment()
		{
			if (!gdiplusToken) 
				GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
			DragonianLib::SetTaskPoolSize(4);
			DragonianLib::SetWorkerCount(8);
			DragonianLib::SetMaxTaskCountPerOperator(4);
		}
		~Enviroment()
		{
			if (gdiplusToken) Gdiplus::GdiplusShutdown(gdiplusToken);
		}

		Mui::Ctrl::UIListBox* AudioList = nullptr;
		CurveEditor* CurveEditor = nullptr;
		Waveform* CurvePlayer = nullptr;

		Mui::_m_color ColorModifiedInCache = Mui::Color::M_RED;
		Mui::_m_color ColorModified = Mui::Color::M_RGBA(255, 140, 0, 255);
		Mui::_m_color ColorNotModifiedInCache = Mui::Color::M_RGBA(0, 191, 255, 255);
		Mui::_m_color ColorNotModified = Mui::Color::M_White;
		Mui::XML::MuiXML* LanguageXml = nullptr;

		std::vector<std::wstring> AudioPaths;
		std::deque<std::pair<std::wstring, MyAudioData>> AudioCaches;
		std::mutex UndoMutex;

		Mui::_m_color SpecColorMap[256] = {};

	private:
		static inline Gdiplus::GdiplusStartupInput gdiplusStartupInput;
		static inline ULONG_PTR gdiplusToken = 0;
	} GlobalEnviroment;  // NOLINT(misc-use-internal-linkage)

	static DragonianLib::Int64 ColorMapSize = 0;

	static auto GetSamplingRate()
	{
		static SidePage* MSidePage = Mui::MObjStorage::GetObjInstance<SidePage*>();
		return MSidePage->GetSamplingRate();
	}

	static const DragonianLib::FunctionTransform::MFCCKernel& GetMelFn()
	{
		static DragonianLib::FunctionTransform::MFCCKernel MelKernel{
			SpecSamplingRate, 2048, SpecSamplingRate / 200, -1, 128, 20.f, 11025.f, DragonianLib::FunctionTransform::BlackmanHarrisWindow<double>(2048)
		};
		return MelKernel;
	}

	static void LoadColorMap()
	{
		static std::regex _Re(R"(\{[ ]?(.*)[ ]?,[ ]?(.*)[ ]?,[ ]?(.*)[ ]?\})");
		DragonianLib::FileStream _Stream(L"color_map.txt", L"r");
		while (ColorMapSize < 256)
		{
			std::string _Line = _Stream.ReadLine();
			if (_Line.empty())
				break;
			std::smatch _Mat;
			if (std::regex_match(_Line, _Mat, _Re))
			{
				unsigned char R = static_cast<unsigned char>(stoi(_Mat[1].str()));
				unsigned char G = static_cast<unsigned char>(stoi(_Mat[2].str()));
				unsigned char B = static_cast<unsigned char>(stoi(_Mat[3].str()));
				GlobalEnviroment.SpecColorMap[ColorMapSize++] = Mui::Color::M_RGB(R, G, B);
			}
			else
				++ColorMapSize;
		}
	}

	static void CvtSpec2ColorMap(
		const FloatTensor2D& Spec,
		ImageTensor& Image,
		ImageTensor& ImageLogView,
		bool UseLogSpec
	)
	{
		const auto [Frames, Bins] = Spec.Size().RawArray();

		auto SpecFlat = Spec.View(-1);
		if (UseLogSpec)
			SpecFlat = (SpecFlat + 1e-5f).Log10();

		//Min Max Normalize
		SpecFlat = DragonianLib::Functional::MinMaxNormalize(SpecFlat, 0).Evaluate();
		const auto FreqPerBin = static_cast<float>(GetMelFn().GetFreqPerBin());
		
		const auto SpecData = SpecFlat.Data();
		if (Image.Null() || !Image.IsContiguous())
			Image = DragonianLib::Functional::Empty<unsigned>(
				DragonianLib::Dimensions{ Bins, Frames }
			);
		if (ImageLogView.Null() || !ImageLogView.IsContiguous())
			ImageLogView = DragonianLib::Functional::Zeros<unsigned>(
				DragonianLib::Dimensions{ 1200, Frames }
			).Evaluate();
		const auto ImageData = Image.Data();
		const auto LogViewData = ImageLogView.Data();

		std::vector<std::vector<std::pair<int, float>>> Interp(Bins);

		for (INT y = 1; y < 1200; ++y)
		{
			const auto Freq = PitchLabel::PitchToF0(float(y - 1) * 0.1f);
			const auto FreqBin = Freq / FreqPerBin;
			const auto FreqFront = std::floor(FreqBin);
			const auto FreqArgT = FreqBin - FreqFront;
			const auto IntIndex = static_cast<int>(FreqFront);
			if (IntIndex < 0 || IntIndex >= Bins - 1)
				continue;
			Interp[IntIndex].emplace_back(y, FreqArgT);
		}
		
		for (INT x = 0; x < static_cast<INT>(Frames); ++x)
		{
			for (INT y = 0; y < static_cast<INT>(Bins); ++y)
			{
				const float Value = std::clamp(*(SpecData + x * Bins + y), 0.001f, 0.999f);
				ImageData[(Bins - y - 1) * Frames + x] = GlobalEnviroment.SpecColorMap[int(Value * 255.f)].argb;
				if (y < static_cast<INT>(Bins - 1))
				{
					const auto NextVal = *(SpecData + x * Bins + (y + 1));
					for (const auto& [idx, param] : Interp[y])
					{
						float CurValue = std::clamp(
							std::lerp(
								Value,
								NextVal,
								param
							),
							0.001f,
							0.999f
						);
						LogViewData[(1199 - idx) * Frames + x] = GlobalEnviroment.SpecColorMap[int(CurValue * 255.f)].argb;
					}
				}
			}
		}
		Image/*.Interpolate<DragonianLib::Operators::InterpolateMode::Nearest>(
				{0},
				DragonianLib::IScale(2.)
			)*/.Evaluate();
		ImageLogView/*.Interpolate<DragonianLib::Operators::InterpolateMode::Nearest>(
			{0},
			DragonianLib::IScale(2.)
		)*/.Evaluate();
	}

	static FloatTensor2D GetUpSampleRates(DragonianLib::Int64 SamplingRate)
	{
		static FloatTensor2D Upp(DragonianLib::Functional::Arange(1.f, 1001.f, 1.f).UnSqueeze(0).Evaluate());
		return Upp[{ std::nullopt, { 0, SamplingRate / 100 }}];
	}

	[[maybe_unused]] static FloatTensor2D SineGen(const FloatTensor2D& F0, DragonianLib::Int64 SamplingRate)
	{
		const auto SineSize = F0.Size(1) * SamplingRate / 100;
		const auto Freq = F0[0].Clone().UnSqueeze(-1);
		auto Audio = DragonianLib::Functional::Zeros(DragonianLib::Dimensions{ 1, SineSize }).Evaluate();
		auto Rad = Freq / (float)SamplingRate * GetUpSampleRates(SamplingRate);
		auto Rad2 = (Rad[{":", "-1:"}] + 0.5f) % 1.f - 0.5f;
		auto RadAcc = Rad2.CumSum(0) % 1.f;
		Rad[{"1:"}] += RadAcc[{":-1"}];
		Rad = Rad.View(1, -1);
		Rad.Evaluate();
		const auto F0Data = Rad.Data();
		const auto AudioData = Audio.Data() + 1;
		for (DragonianLib::SizeType F0Idx = 0; F0Idx < SineSize; ++F0Idx)
		{
			auto& SamplePoint = AudioData[F0Idx];
			SamplePoint = short(sin(F0Data[F0Idx] * 2.f * std::numbers::pi_v<float>) * 4000.f);
		}
		return Audio;
	}

	void InitCtrl(
		Mui::Ctrl::UIListBox* AudioList,
		CurveEditor* CurveEditor,
		Waveform* CurvePlayer
	)
	{
		GlobalEnviroment.AudioList = AudioList;
		GlobalEnviroment.CurveEditor = CurveEditor;
		GlobalEnviroment.CurvePlayer = CurvePlayer;
		GetMelFn();
		_D_Dragonian_Lib_Rethrow_Block(
			LoadColorMap();
		);
	}

	class ListItemC : public Mui::Ctrl::ListItem
	{
	public:
		void SetColor(Mui::_m_color color)
		{
			m_color = color;
		}
		void SetParent(Mui::Ctrl::UIListBox* Parent)
		{
			m_parent = Parent;
		}
	};

	static void SaveWithHistory(const std::wstring& Path, const FloatTensor2D& Data)
	{
		auto CurDir = std::filesystem::path(Path);
		auto LastDir = CurDir; 
		LastDir.replace_filename(LastDir.filename().string() + ".last");
		auto OldDir = LastDir;
		OldDir.replace_filename(OldDir.filename().string() + ".old");

		if (exists(CurDir))
		{
			if (exists(LastDir))
			{
				if (exists(OldDir))
					remove(OldDir);
				rename(LastDir, OldDir);
			}
			rename(CurDir, LastDir);
		}
		DragonianLib::Functional::NumpySave(Path, Data);
	}

	static std::wstring GetPath(const std::wstring& RawAudioPath, const wchar_t* Fol, const wchar_t* Ext)
	{
		auto Path = std::filesystem::path(RawAudioPath);
		std::wstring NewPath;
		for (auto& Stem : Path)
		{
			auto StemStr = Stem.wstring();
			std::ranges::replace(StemStr, L'\\', L'_');
			std::ranges::replace(StemStr, L'/', L'_');
			std::ranges::replace(StemStr, L':', L'_');
			NewPath += StemStr;
		}
		NewPath = std::filesystem::path(NewPath).replace_extension(Ext);
		NewPath = DragonianLib::GetCurrentFolder() + L"/User/" + Fol + NewPath;
		return NewPath;
	}

	static std::wstring GetF0Path(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"F0/", L".npy");
	}

	static std::wstring GetAudioPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Audio/", L".wav");
	}

	static std::wstring GetSpecPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Spec/", L".png");
	}

	static std::wstring GetMelPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Mel/", L".png");
	}

	static void EraseFront()
	{
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		if (!GlobalEnviroment.AudioCaches.empty())
		{
			const auto OffPtr = std::ranges::find(GlobalEnviroment.AudioPaths, GlobalEnviroment.AudioCaches.front().first);
			if (OffPtr != GlobalEnviroment.AudioPaths.end())
			{
				const auto Offset = static_cast<int>(std::distance(GlobalEnviroment.AudioPaths.begin(), OffPtr));
				if (GlobalEnviroment.AudioCaches.front().second.ModifyCount)
					dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[Offset].get())->SetColor(GlobalEnviroment.ColorModified);
				else
					dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[Offset].get())->SetColor(GlobalEnviroment.ColorNotModified);
			}
			GlobalEnviroment.AudioCaches.pop_front();
		}
	}

	static void EraseCache(int idx)
	{
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		const auto& Path = GlobalEnviroment.AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			GlobalEnviroment.AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != GlobalEnviroment.AudioCaches.end())
		{
			const auto OffPtr = std::ranges::find(GlobalEnviroment.AudioPaths, Iter->first);
			if (OffPtr != GlobalEnviroment.AudioPaths.end())
			{
				const auto Offset = static_cast<int>(std::distance(GlobalEnviroment.AudioPaths.begin(), OffPtr));
				if (Iter->second.ModifyCount)
					dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[Offset].get())->SetColor(GlobalEnviroment.ColorModified);
				else
					dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[Offset].get())->SetColor(GlobalEnviroment.ColorNotModified);
			}
			GlobalEnviroment.AudioCaches.erase(Iter);
		}
	}

	static MyAudioData& GetData(int idx, unsigned SamplingRate, bool UseLogSpec)
	{
		const auto& Path = GlobalEnviroment.AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			GlobalEnviroment.AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != GlobalEnviroment.AudioCaches.end())
		{
			if (std::cmp_not_equal(Iter->second.SamplingRate, SamplingRate))
			{
				Iter->second.Audio = DragonianLib::FunctionTransform::WindowedResample(
					Iter->second.Audio.UnSqueeze(0).Transpose(),
					Iter->second.SamplingRate,
					SamplingRate
				).Squeeze(0).Transpose().Contiguous().Evaluate();
				Iter->second.SamplingRate = SamplingRate;
			}
			return Iter->second;
		}
		auto AudioPath = GetAudioPath(Path);
		auto F0Path = GetF0Path(Path);
		FloatTensor2D Audio, F0, Spec, Mel;

		unsigned SourceSamplingRate;
		if (std::filesystem::exists(AudioPath))
		{
			std::tie(Audio, SourceSamplingRate) = DragonianLib::AvCodec::OpenInputStream(
				AudioPath
			).DecodeAudio(2);
			if (SourceSamplingRate != SamplingRate)
				Audio = DragonianLib::FunctionTransform::WindowedResample(
					Audio.UnSqueeze(0).Transpose(), SourceSamplingRate, SamplingRate
				).Squeeze(0).Transpose().Contiguous().Evaluate();
		}
		else if (std::filesystem::exists(Path))
		{
			std::tie(Audio, SourceSamplingRate) = DragonianLib::AvCodec::OpenInputStream(
				Path
			).DecodeAudio(2);
			if (SourceSamplingRate != SamplingRate)
				Audio = DragonianLib::FunctionTransform::WindowedResample(
					Audio.UnSqueeze(0).Transpose(), SourceSamplingRate, SamplingRate
				).Squeeze(0).Transpose().Contiguous().Evaluate();
			OpenOutputStream(
				SamplingRate,
				std::filesystem::path(AudioPath),
				DragonianLib::AvCodec::AvCodec::PCM_FORMAT_FLOAT32,
				2
			).EncodeAll(
				Audio.GetCRng(), SamplingRate, 2
			);
		}

		const auto HopSize = SamplingRate / 100;
		const auto AudioFrames = (DragonianLib::SizeType)ceil(double(Audio.Shape(0)) / double(HopSize)) + 1;
		bool Modified = false;
		if (std::filesystem::exists(F0Path + L".cache"))
		{
			F0 = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
				F0Path + L".cache"
			);
			if (F0.Size(0) != 10 || F0.Size(1) != AudioFrames)
			{
				if (std::filesystem::exists(F0Path))
					goto __F0CacheSizeMisMatch;
				goto __F0SizeMisMatch;
			}
			Modified = true;
		}
		else if (std::filesystem::exists(F0Path))
		{
		__F0CacheSizeMisMatch:
			F0 = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
				F0Path
			);
			if (F0.Size(0) != 10 || F0.Size(1) != AudioFrames)
				goto __F0SizeMisMatch;
		}
		else
		{
		__F0SizeMisMatch:
			const auto F0Shape = DragonianLib::Dimensions{ 10, AudioFrames };
			F0 = DragonianLib::Functional::Zeros(F0Shape);
		}

		Audio = Audio.Mean(-1).Evaluate().UnSqueeze(-1);
		auto SpecAudio =
			SamplingRate == SpecSamplingRate ?
			Audio.AutoView(1, 1, -2) :
			DragonianLib::FunctionTransform::WindowedResample(
				Audio.Transpose().UnSqueeze(0), SamplingRate, SpecSamplingRate
			).Evaluate();

		Spec = GetMelFn().GetStftKernel()(SpecAudio).AutoView(-2, -1);

		/*;

		Mel = GetMelFn()(
		   Spec.View(1, 1, Spec.Size(0), Spec.Size(1))
		   ).Squeeze(0).Squeeze(0).Evaluate();*/

		if (GlobalEnviroment.AudioCaches.size() >= MaxCacheCount)
			EraseFront();

		if (Modified)
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[idx].get())->SetColor(GlobalEnviroment.ColorModifiedInCache);
		else
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[idx].get())->SetColor(GlobalEnviroment.ColorNotModifiedInCache);

		return GlobalEnviroment.AudioCaches.emplace_back(
			Path,
			MyAudioData{
				SamplingRate,
				Audio.Pad(DragonianLib::PaddingCounts{DragonianLib::PadCount{0, 1}}, DragonianLib::PaddingType::Zero).Evaluate(),
				std::move(F0.Evaluate()),
				std::move(Spec.Evaluate()),
				std::move(Mel.Evaluate()),
				std::move(F0Path),
				UseLogSpec,
				Modified
			}
		).second;
	}

	static ptrdiff_t GetOffset(int idx)
	{
		if (idx < 0 || std::cmp_greater_equal(idx, GlobalEnviroment.AudioPaths.size()))
			return -1;
		const auto& Path = GlobalEnviroment.AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			GlobalEnviroment.AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != GlobalEnviroment.AudioCaches.end())
			return std::distance(GlobalEnviroment.AudioCaches.begin(), Iter);
		return -1;
	}

	static void IncModifyCount(const auto& CurSel, auto Offset)
	{
		const auto Prev = GlobalEnviroment.AudioCaches[Offset].second.ModifyCount;
		++GlobalEnviroment.AudioCaches[Offset].second.ModifyCount;
		if (Prev && !GlobalEnviroment.AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorNotModifiedInCache);
		else if (!Prev && GlobalEnviroment.AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorModifiedInCache);
		GlobalEnviroment.AudioList->UpdateLayout();
	}

	static void DecModifyCount(const auto& CurSel, auto Offset)
	{
		const auto Prev = GlobalEnviroment.AudioCaches[Offset].second.ModifyCount;
		--GlobalEnviroment.AudioCaches[Offset].second.ModifyCount;
		if (Prev && !GlobalEnviroment.AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorNotModifiedInCache);
		else if (!Prev && GlobalEnviroment.AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorModifiedInCache);
		GlobalEnviroment.AudioList->UpdateLayout();
	}

	static bool AllEqual(const float* a, const float* b, size_t size)
	{
		for (size_t i = 0; i < size; ++i)
			if (abs(a[i] - b[i]) > 0.001f)
				return false;
		return true;
	}

	void AppendUndo()
	{
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		const auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, GlobalEnviroment.AudioCaches.size()))
			return;
		GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.emplace_back(
			GlobalEnviroment.AudioCaches[CacheIdx].second.F0.Clone().Evaluate()
		);
	}

	void CheckUnchanged()
	{
		const auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, GlobalEnviroment.AudioCaches.size()))
			return;
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		const auto& CurF0 = GlobalEnviroment.AudioCaches[CacheIdx].second.F0;
		const auto& CurUndo = GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.back();
		const auto F0Size = CurF0.ElementCount();
		const auto UndoSize = CurUndo.ElementCount();
		if ((F0Size == UndoSize && AllEqual(CurF0.Data(), CurUndo.Data(), F0Size)))
			GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.pop_back();
		else
		{
			IncModifyCount(CurSel, CacheIdx);
			if (GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.size() > UndoRedoMaxCount)
				GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.pop_front();
			GlobalEnviroment.AudioCaches[CacheIdx].second.RedoList.clear();
		}
	}

	void ApplyAppendUndo()
	{
		const auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, GlobalEnviroment.AudioCaches.size()))
			return;
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		IncModifyCount(CurSel, CacheIdx);
		if (GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.size() > UndoRedoMaxCount)
			GlobalEnviroment.AudioCaches[CacheIdx].second.UndoList.pop_front();
		GlobalEnviroment.AudioCaches[CacheIdx].second.RedoList.clear();
	}

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges)
	{
		ApplyPitchShift(Ranges, Mui::MObjStorage::GetObjInstance<SidePage*>()->GetPitch());
	}

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges, float Pitch)
	{
		static const auto MaxFreq = PitchLabel::PitchToF0(119.9f);
		static const auto MinFreq = PitchLabel::PitchToF0(0);

		WndControls::AppendUndo();
		if (abs(Pitch) > 1e-5)
			for (auto& i : Ranges)
			{
				i = std::min(i * std::pow(2.f, Pitch / 12.f), MaxFreq);
				if (i < MinFreq) i = 0.f;
			}
		WndControls::CheckUnchanged();
		GlobalEnviroment.CurveEditor->UpDate();
	}

	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges)
	{
		ApplyCalc(
			Ranges,
			Mui::MObjStorage::GetObjInstance<SidePage*>()->GetAlpha(),
			Mui::MObjStorage::GetObjInstance<SidePage*>()->GetBeta()
		);
	}

	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges, float Alpha, float Beta)
	{
		static const auto MaxFreq = PitchLabel::PitchToF0(119.9f);
		static const auto MinFreq = PitchLabel::PitchToF0(0);

		WndControls::AppendUndo();
		if (abs(Alpha - 1.f) > 1e-6 || abs(Beta) > 1e-6)
			for (auto& i : Ranges)
			{
				i = std::min(Alpha * i + Beta, MaxFreq);
				if (i < MinFreq) i = 0.f;
			}
		WndControls::CheckUnchanged();
		GlobalEnviroment.CurveEditor->UpDate();
	}

	void SetPlayerPos(size_t Index)
	{
		GlobalEnviroment.CurvePlayer->SetPlayPos(Index);
	}

	size_t GetPcmSize()
	{
		return GlobalEnviroment.CurvePlayer->GetPCMSize();
	}

	void EmptyCache()
	{
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		GlobalEnviroment.AudioCaches.clear();
	}

	void InsertAudio(std::wstring Path)
	{
		const auto StdPath = std::filesystem::path(Path);
		if (std::ranges::contains(GlobalEnviroment.AudioPaths, Path) || !exists(StdPath))
			return;
		auto OldFileName = StdPath.stem().wstring();
		std::wstring FileName;
		size_t TextLength = 0;
		for (auto Ch : OldFileName)
		{
			if (TextLength > 15)
				break;
			FileName += Ch;
			TextLength += (Ch > 0xFF) ? 2 : 1;
		}
		if (TextLength < OldFileName.size())
			FileName += L"...";
		auto Item = std::make_shared<ListItemC>();
		Item->SetText(std::move(FileName));
		if (exists(std::filesystem::path(GetF0Path(Path) + L".cache")))
			Item->SetColor(GlobalEnviroment.ColorModified);
		else
			Item->SetColor(GlobalEnviroment.ColorNotModified);
		GlobalEnviroment.AudioPaths.emplace_back(std::move(Path));
		GlobalEnviroment.AudioList->Items.Add(Item);
	}

	void SetLanguageXML(Mui::XML::MuiXML* XmlUI)
	{
		GlobalEnviroment.LanguageXml = XmlUI;
	}

	void SetCurveEditorDataIdx(int AudioIdx, unsigned SamplingRate, bool UseLogSpec)
	{
		auto& AudioAndF0 = GetData(AudioIdx, SamplingRate, UseLogSpec);
		std::lock_guard lg(GlobalEnviroment.UndoMutex);
		GlobalEnviroment.CurveEditor->SetPlayLinePos(0);
		GlobalEnviroment.CurveEditor->SetCurveData(AudioAndF0.F0, AudioAndF0.Spec, AudioAndF0.LogSpec);
		GlobalEnviroment.CurvePlayer->SetPlayPos(0);
		GlobalEnviroment.CurvePlayer->SetAudioData(AudioAndF0.Audio, static_cast<DragonianLib::UInt>(AudioAndF0.SamplingRate));
	}

	void DeleteAudio(int Index)
	{
		EraseCache(Index);
		GlobalEnviroment.CurveEditor->SetPlayLinePos(0);
		GlobalEnviroment.CurveEditor->SetCurveData(std::nullopt, std::nullopt, std::nullopt);
		GlobalEnviroment.CurvePlayer->Clear();
		GlobalEnviroment.AudioList->Items.Remove(GlobalEnviroment.AudioList->Items[Index]);
		GlobalEnviroment.AudioList->SelectedItemIndex = -1;
		GlobalEnviroment.AudioPaths.erase(GlobalEnviroment.AudioPaths.begin() + Index);
	}

	static bool MoeURDo(
		std::deque<FloatTensor2D>& CurUndo,
		std::deque<FloatTensor2D>& CurRedo,
		ptrdiff_t Offset
	)
	{
		{
			std::unique_lock lg(GlobalEnviroment.UndoMutex);
			if (CurUndo.empty())
				return false;
		}
		GlobalEnviroment.CurveEditor->UPRButton();
		std::unique_lock lg(GlobalEnviroment.UndoMutex);
		auto UndoData = std::move(CurUndo.back());
		CurRedo.emplace_back(std::move(GlobalEnviroment.AudioCaches[Offset].second.F0));
		GlobalEnviroment.AudioCaches[Offset].second.F0 = std::move(UndoData.Evaluate());
		GlobalEnviroment.CurveEditor->ReSetCurveData(GlobalEnviroment.AudioCaches[Offset].second.F0, -1);
		GlobalEnviroment.CurveEditor->UpDate();

		if (CurRedo.size() > UndoRedoMaxCount)
			CurRedo.pop_front();
		CurUndo.pop_back();
		return true;
	}

	void PlayPause()
	{
		if (!GlobalEnviroment.CurvePlayer->IsPlay())
			SineGen();
		GlobalEnviroment.CurvePlayer->PlayPause();
	}

	void MoeVSUndo()
	{
		auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, GlobalEnviroment.AudioCaches.size()))
			return;
		if (MoeURDo(GlobalEnviroment.AudioCaches[Offset].second.UndoList, GlobalEnviroment.AudioCaches[Offset].second.RedoList, Offset))
			DecModifyCount(CurSel, Offset);
	}

	void MoeVSRedo()
	{
		auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, GlobalEnviroment.AudioCaches.size()))
			return;
		if (MoeURDo(GlobalEnviroment.AudioCaches[Offset].second.RedoList, GlobalEnviroment.AudioCaches[Offset].second.UndoList, Offset))
			IncModifyCount(CurSel, Offset);
	}

	void SaveAll()
	{
		for (int i = 0; std::cmp_less(i, GlobalEnviroment.AudioPaths.size()); ++i)
			SaveData(i);
	}

	void SaveData(int CurSel)
	{
		if (CurSel < 0)
			CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, GlobalEnviroment.AudioCaches.size()))
		{
			auto F0Path = GetF0Path(GlobalEnviroment.AudioPaths[CurSel]);
			std::filesystem::path CachePath = F0Path + L".cache";
			if (exists(CachePath))
			{
				auto F0 = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
					F0Path + L".cache"
				);
				SaveWithHistory(F0Path, F0);
				remove(CachePath);
				reinterpret_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorNotModified);
			}
			return;
		}
		if (GlobalEnviroment.AudioCaches[Offset].second.ModifyCount)
		{
			dynamic_cast<ListItemC*>(GlobalEnviroment.AudioList->Items[CurSel].get())->SetColor(GlobalEnviroment.ColorNotModifiedInCache);
			GlobalEnviroment.AudioList->UpdateLayout();
			GlobalEnviroment.AudioCaches[Offset].second.ModifyCount = 0;
			SaveWithHistory(
				GlobalEnviroment.AudioCaches[Offset].second.F0Path,
				GlobalEnviroment.AudioCaches[Offset].second.F0
			);
			std::filesystem::path CachePath = GlobalEnviroment.AudioCaches[Offset].second.F0Path + L".cache";
			if (exists(CachePath))
				remove(CachePath);
		}
	}

	void SineGen()
	{
		auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, GlobalEnviroment.AudioCaches.size()))
			return;
		auto& Audio = GlobalEnviroment.CurvePlayer->GetAudio();
		const auto& F0 = GlobalEnviroment.AudioCaches[Offset].second.F0;
		const auto SineSize = std::min(F0.Size(1) * GlobalEnviroment.CurvePlayer->GetSamplingRate() / 100, Audio.Size(0));
		const auto Freq = F0[0].Clone().UnSqueeze(-1);

		auto Rad = Freq / (float)GlobalEnviroment.CurvePlayer->GetSamplingRate() * GetUpSampleRates(GlobalEnviroment.CurvePlayer->GetSamplingRate());
		auto Rad2 = (Rad[{":", "-1:"}] + 0.5f) % 1.f - 0.5f;
		auto RadAcc = Rad2.CumSum(0) % 1.f;
		Rad[{"1:"}] += RadAcc[{":-1"}];
		Rad = Rad.View(1, -1);
		Rad.Evaluate();
		const auto F0Data = Rad.Data();
		const auto AudioData = Audio.Data() + 1;
		for (DragonianLib::SizeType F0Idx = 0; F0Idx < SineSize; ++F0Idx)
		{
			auto& SamplePoint = AudioData[F0Idx << 1];
			SamplePoint = short(sin(F0Data[F0Idx] * 2.f * std::numbers::pi_v<float>) * 8000.f);
		}
	}

	void LoadFiles(HWND hWnd)
	{
		auto Files = OfnHelperMult(
			TEXT("Audio Files (*.wav; *.mp3; *.ogg; *.flac)\0*.wav;*.mp3;*.ogg;*.flac\0All Files (*.*)\0*.*\0\0"),
			hWnd
		);
		for (auto& i : Files)
			InsertAudio(std::move(i));
	}

	void LoadF0(HWND hWnd)
	{
		const auto Path = OfnHelper(
			TEXT("Numpy Files (*.npy)\0*.npy\0\0"),
			hWnd,
			L"npy"
		);
		if (!Path.empty())
		{
			try
			{
				auto F0Tensor = DragonianLib::Functional::NumpyLoad<DragonianLib::Float64, 2>(
					Path
				).Cast<DragonianLib::Float32>().Evaluate();
				Write2Clipboard(
					F0Tensor.GetRng()
				);
			}
			catch (std::exception& e)
			{
				DragonianLib::GetDefaultLogger()->LogError(DragonianLib::UTF8ToWideString(e.what()));
			}
		}
	}

	void ReCalcSpec(bool _Log)
	{
		auto CurSel = GlobalEnviroment.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, GlobalEnviroment.AudioCaches.size()))
			return;
		auto& Audio = GlobalEnviroment.AudioCaches[Offset];
		CvtSpec2ColorMap(Audio.second.RawSpec, Audio.second.Spec, Audio.second.LogSpec, _Log);
	}

	std::wstring Localization(const std::wstring_view& Key)
	{
		return GlobalEnviroment.LanguageXml->GetStringValue(Key);
	}

	Mui::XML::MuiXML* GetUiXml()
	{
		return GlobalEnviroment.LanguageXml;
	}
}

SimpleF0Labeler::MyAudioData::~MyAudioData()
{
	SaveCache();
}

void SimpleF0Labeler::MyAudioData::SaveCache() const
{
	if (!F0Path.empty() && ModifyCount)
	{
		WndControls::SaveWithHistory(
			F0Path + L".cache",
			F0
		);
	}
}

void SimpleF0Labeler::MyAudioData::CalcSpec(bool _UseLogSpec)
{
	WndControls::CvtSpec2ColorMap(RawSpec, Spec, LogSpec, _UseLogSpec);
}

