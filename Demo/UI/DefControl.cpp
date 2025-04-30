#include "DefControl.hpp"
#include "Libraries/Stft/Stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

#include <Render/Sound/Mui_DirectSound.h>
#include <gdiplus.h>
#pragma comment(lib, "Msimg32.lib")
#pragma comment(lib, "gdiplus.lib")

#include <numbers>
#include <User/Mui_GlobalStorage.h>

#include "Page/SidePage.h"

namespace WndControls
{
	using namespace SimpleF0Labeler;

	const auto MaxFreq = SimpleF0Labeler::PitchLabel::PitchToF0(119.9f);
	const auto MinFreq = SimpleF0Labeler::PitchLabel::PitchToF0(0);

	constexpr auto ColorRed = Mui::Color::M_RED;
	const auto ColorOrigin = Mui::Color::M_RGBA(255, 140, 0, 255);
	const auto ColorSkyBlue = Mui::Color::M_RGBA(0, 191, 255, 255);
	constexpr auto ColorWhite = Mui::Color::M_White;

	static inline Mui::_m_color __SpecColorMap[]{
		Mui::Color::M_RGB(int(0.0f * 255), int(0.0f * 255),int(0.0f * 255)),       // black
		Mui::Color::M_RGB(int(0.f * 255), int(0.0f * 255),int(0.251f * 255)),     // dark blue
		Mui::Color::M_RGB(int(0.f * 255), int(0.0f * 255),int(0.502f * 255)),     // blue
		Mui::Color::M_RGB(int(0.251f * 255), int(0.0f * 255),int(0.502f * 255)),   // purple-blue
		Mui::Color::M_RGB(int(0.502f * 255),int(0.0f * 255),int(0.502f * 255)),   // purple
		Mui::Color::M_RGB(int(0.753f * 255),int(0.0f * 255),int(0.502f * 255)),   // magenta
		Mui::Color::M_RGB(int(1.0f * 255),int(0.0f * 255),int(0.0f * 255)),       // red
		Mui::Color::M_RGB(int(1.0f * 255),int(0.502f * 255),int(0.0f * 255)),     // orange
		Mui::Color::M_RGB(int(1.0f * 255),int(1.0f * 255),int(0.0f * 255)),       // yellow
		Mui::Color::M_RGB(int(1.0f * 255),int(1.0f * 255),int(0.502f * 255)),     // light yellow
	};

	static inline Mui::_m_color SpecColorMap[256];

	static inline std::vector<std::wstring> AudioPaths;
	static inline std::deque<std::pair<std::wstring, SimpleF0Labeler::MyAudioData>> AudioCaches;
	constexpr size_t MaxCacheCount = 10;

	static inline std::mutex UndoMutex;
	constexpr size_t UndoRedoMaxCount = 100;

	static inline Mui::XML::MuiXML* LanguageXml = nullptr;
	static inline MyControls LabelControls;

	static inline Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	static inline ULONG_PTR gdiplusToken;

	[[maybe_unused]] static inline DragonianLib::OnConstruct StartUPLabel{
		[]
		{
			GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
		}
	};

	[[maybe_unused]] static inline DragonianLib::SharedScopeExit OnLabelExit{
		[]
		{
			if (gdiplusToken)
				Gdiplus::GdiplusShutdown(gdiplusToken);
		}
	};

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

	static const DragonianLib::FunctionTransform::MFCCKernel& GetMelFn()
	{
		static DragonianLib::FunctionTransform::MFCCKernel MelKernel{
			SpecSamplingRate, 2048, SpecSamplingRate / 200, -1, 128, 20.f, 11025.f
		};
		return MelKernel;
	}

	static ImageTensor CvtSpec2ColorMap(
		const FloatTensor2D& Spec
	)
	{
		const auto [Frames, Bins] = Spec.Size().RawArray();

		const auto SpecFlat = (Spec.View(-1) + 1e-5f).Log10();

		//Min Max Normalize
		const auto Max = SpecFlat.ReduceMax(0);
		const auto Min = SpecFlat.ReduceMin(0);
		const auto SpecMin = Min.Evaluate().Item();
		const auto SpecMax = Max.Evaluate().Item();
		
		const auto SpecData = SpecFlat.Data();
		ImageTensor Image = DragonianLib::Functional::Empty<int>(
			DragonianLib::Dimensions{ Bins, Frames }
		);
		const auto ImageData = Image.Data();
		
		for (INT x = 0; x < static_cast<INT>(Frames); ++x)
		{
			for (INT y = 0; y < static_cast<INT>(Bins); ++y)
			{
				float Value = (*(SpecData + x * Bins + y) - SpecMin) / (SpecMax - SpecMin);
				Value = std::clamp(Value, 0.001f, 0.999f);

				const auto Color = __SpecColorMap[int(Value * 9.f)];
				ImageData[(Bins - y - 1) * Frames + x] = Color;
			}
		}

		return Image;
	}

	void InitCtrl(
		Mui::Ctrl::UIListBox* AudioList,
		CurveEditor* CurveEditor,
		Waveform* CurvePlayer
	)
	{
		LabelControls.AudioList = AudioList;
		LabelControls.CurveEditor = CurveEditor;
		LabelControls.CurvePlayer = CurvePlayer;
		GetMelFn();
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
		std::lock_guard lg(UndoMutex);
		if (!AudioCaches.empty())
		{
			const auto OffPtr = std::ranges::find(AudioPaths, AudioCaches.front().first);
			if (OffPtr != AudioPaths.end())
			{
				const auto Offset = static_cast<int>(std::distance(AudioPaths.begin(), OffPtr));
				if (AudioCaches.front().second.ModifyCount)
					dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[Offset].get())->SetColor(ColorOrigin);
				else
					dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[Offset].get())->SetColor(ColorWhite);
			}
			AudioCaches.pop_front();
		}
	}

	static void EraseCache(int idx)
	{
		std::lock_guard lg(UndoMutex);
		const auto& Path = AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != AudioCaches.end())
		{
			const auto OffPtr = std::ranges::find(AudioPaths, Iter->first);
			if (OffPtr != AudioPaths.end())
			{
				const auto Offset = static_cast<int>(std::distance(AudioPaths.begin(), OffPtr));
				if (Iter->second.ModifyCount)
					dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[Offset].get())->SetColor(ColorOrigin);
				else
					dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[Offset].get())->SetColor(ColorWhite);
			}
			AudioCaches.erase(Iter);
		}
	}

	static MyAudioData& GetData(int idx, unsigned SamplingRate)
	{
		const auto& Path = AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != AudioCaches.end())
			return Iter->second;
		auto AudioPath = GetAudioPath(Path);
		auto F0Path = GetF0Path(Path);
		FloatTensor2D Audio, F0;
		ImageTensor Spec, Mel;
		if (std::filesystem::exists(AudioPath))
			Audio = DragonianLib::AvCodec::OpenInputStream(
				AudioPath
			).DecodeAll(SamplingRate, 2);
		else if (std::filesystem::exists(Path))
		{
			Audio = DragonianLib::AvCodec::OpenInputStream(
				Path
			).DecodeAll(SamplingRate, 2);
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
		Audio = Audio.Mean(-1).Interpolate<DragonianLib::Operators::InterpolateMode::Linear>(
			DragonianLib::IDim(0),
			DragonianLib::IScale(48000. / double(SamplingRate))
		).Evaluate().UnSqueeze(-1);

		Spec = CvtSpec2ColorMap(GetMelFn().GetStftKernel()(
			Audio.View(1, 1, Audio.Size(0)).Interpolate<DragonianLib::Operators::InterpolateMode::Linear>(
				DragonianLib::IDim(-1),
				DragonianLib::IScale(double(SpecSamplingRate) / 48000.)
			)
			).Squeeze(0).Squeeze(0));

		/*;

		Mel = GetMelFn()(
		   Spec.View(1, 1, Spec.Size(0), Spec.Size(1))
		   ).Squeeze(0).Squeeze(0).Evaluate();*/

		if (AudioCaches.size() >= MaxCacheCount)
			EraseFront();

		if (Modified)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[idx].get())->SetColor(ColorRed);
		else
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[idx].get())->SetColor(ColorSkyBlue);

		return AudioCaches.emplace_back(
			Path,
			MyAudioData{
				SamplingRate,
				Audio.Pad(DragonianLib::PaddingCounts{DragonianLib::PadCount{0, 1}}, DragonianLib::PaddingType::Zero).Evaluate(),
				std::move(F0.Evaluate()),
				std::move(Spec.Evaluate()),
				std::move(Mel.Evaluate()),
				std::move(F0Path),
				Modified
			}
		).second;
	}

	static ptrdiff_t GetOffset(int idx)
	{
		if (idx < 0 || std::cmp_greater_equal(idx, AudioPaths.size()))
			return -1;
		const auto& Path = AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != AudioCaches.end())
			return std::distance(AudioCaches.begin(), Iter);
		return -1;
	}

	static void IncModifyCount(const auto& CurSel, auto Offset)
	{
		const auto Prev = AudioCaches[Offset].second.ModifyCount;
		++AudioCaches[Offset].second.ModifyCount;
		if (Prev && !AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorSkyBlue);
		else if (!Prev && AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorRed);
		LabelControls.AudioList->UpdateLayout();
	}

	static void DecModifyCount(const auto& CurSel, auto Offset)
	{
		const auto Prev = AudioCaches[Offset].second.ModifyCount;
		--AudioCaches[Offset].second.ModifyCount;
		if (Prev && !AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorSkyBlue);
		else if (!Prev && AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorRed);
		LabelControls.AudioList->UpdateLayout();
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
		std::lock_guard lg(UndoMutex);
		const auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, AudioCaches.size()))
			return;
		AudioCaches[CacheIdx].second.UndoList.emplace_back(
			AudioCaches[CacheIdx].second.F0.Clone().Evaluate()
		);
	}

	void CheckUnchanged()
	{
		const auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, AudioCaches.size()))
			return;
		std::lock_guard lg(UndoMutex);
		const auto& CurF0 = AudioCaches[CacheIdx].second.F0;
		const auto& CurUndo = AudioCaches[CacheIdx].second.UndoList.back();
		const auto F0Size = CurF0.ElementCount();
		const auto UndoSize = CurUndo.ElementCount();
		if ((F0Size == UndoSize && AllEqual(CurF0.Data(), CurUndo.Data(), F0Size)))
			AudioCaches[CacheIdx].second.UndoList.pop_back();
		else
		{
			IncModifyCount(CurSel, CacheIdx);
			if (AudioCaches[CacheIdx].second.UndoList.size() > UndoRedoMaxCount)
				AudioCaches[CacheIdx].second.UndoList.pop_front();
			AudioCaches[CacheIdx].second.RedoList.clear();
		}
	}

	void ApplyAppendUndo()
	{
		const auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || std::cmp_greater_equal(CacheIdx, AudioCaches.size()))
			return;
		std::lock_guard lg(UndoMutex);
		IncModifyCount(CurSel, CacheIdx);
		if (AudioCaches[CacheIdx].second.UndoList.size() > UndoRedoMaxCount)
			AudioCaches[CacheIdx].second.UndoList.pop_front();
		AudioCaches[CacheIdx].second.RedoList.clear();
	}

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges)
	{
		const auto Pitch = Mui::MObjStorage::GetObjInstance<SidePage*>()->GetPitch();
		if (abs(Pitch) > 1e-5)
			for (auto& i : Ranges)
			{
				i = std::min(i * std::pow(2.f, Pitch / 12.f), MaxFreq);
				if (i < MinFreq) i = 0.f;
			}
		LabelControls.CurveEditor->UpDate();
	}

	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges)
	{
		const auto Alpha = Mui::MObjStorage::GetObjInstance<SidePage*>()->GetAlpha();
		const auto Beta = Mui::MObjStorage::GetObjInstance<SidePage*>()->GetBeta();
		if (abs(Alpha - 1.f) > 1e-6 || abs(Beta) > 1e-6)
			for (auto& i : Ranges)
			{
				i = std::min(Alpha * i + Beta, MaxFreq);
				if (i < MinFreq) i = 0.f;
			}
		LabelControls.CurveEditor->UpDate();
	}

	void SetPlayerPos(size_t idx)
	{
		LabelControls.CurvePlayer->SetPlayPos(idx);
	}

	size_t GetPcmSize()
	{
		return LabelControls.CurvePlayer->GetPCMSize();
	}

	void EmptyCache()
	{
		std::lock_guard lg(UndoMutex);
		AudioCaches.clear();
	}

	void InsertAudio(std::wstring Path)
	{
		const auto StdPath = std::filesystem::path(Path);
		if (std::ranges::contains(AudioPaths, Path) || !exists(StdPath))
			return;
		auto FileName = StdPath.stem().wstring();
		FileName = (L"    " + FileName).substr(0, 50);
		auto Item = std::make_shared<ListItemC>();
		Item->SetText(std::move(FileName));

		if (exists(std::filesystem::path(GetF0Path(Path) + L".cache")))
			Item->SetColor(ColorOrigin);
		else
			Item->SetColor(ColorWhite);

		Item->SetParent(LabelControls.AudioList);
		AudioPaths.emplace_back(std::move(Path));
		LabelControls.AudioList->Items.Add(Item);
	}

	void SetLanguageXML(Mui::XML::MuiXML* xml)
	{
		LanguageXml = xml;
	}

	void SetCurveEditorDataIdx(int AudioIdx, unsigned SamplingRate)
	{
		auto& AudioAndF0 = GetData(AudioIdx, SamplingRate);
		std::lock_guard lg(UndoMutex);
		LabelControls.CurveEditor->SetPlayLinePos(0);
		LabelControls.CurveEditor->SetCurveData(AudioAndF0.F0, AudioAndF0.Spec);
		LabelControls.CurvePlayer->SetPlayPos(0);
		LabelControls.CurvePlayer->SetAudioData(AudioAndF0.Audio);
	}

	void DeleteAudio(int idx)
	{
		LabelControls.CurveEditor->SetPlayLinePos(0);
		LabelControls.CurveEditor->SetCurveData(std::nullopt, std::nullopt);
		LabelControls.CurvePlayer->Clear();
		LabelControls.AudioList->Items.Remove(LabelControls.AudioList->Items[idx]);
		LabelControls.AudioList->SelectedItemIndex = -1;

		AudioPaths.erase(AudioPaths.begin() + idx);
		EraseCache(idx);
	}

	static bool MoeURDo(
		std::deque<FloatTensor2D>& CurUndo,
		std::deque<FloatTensor2D>& CurRedo,
		ptrdiff_t Offset
	)
	{
		{
			std::unique_lock lg(UndoMutex);
			if (CurUndo.empty())
				return false;
		}
		LabelControls.CurveEditor->UPRButton();
		std::unique_lock lg(UndoMutex);
		auto UndoData = std::move(CurUndo.back());
		CurRedo.emplace_back(std::move(AudioCaches[Offset].second.F0));
		AudioCaches[Offset].second.F0 = std::move(UndoData.Evaluate());
		LabelControls.CurveEditor->ReSetCurveData(AudioCaches[Offset].second.F0, -1);
		LabelControls.CurveEditor->UpDate();

		if (CurRedo.size() > UndoRedoMaxCount)
			CurRedo.pop_front();
		CurUndo.pop_back();
		return true;
	}

	void PlayPause()
	{
		if (!LabelControls.CurvePlayer->IsPlay())
			SineGen();
		LabelControls.CurvePlayer->PlayPause();
	}

	void MoeVSUndo()
	{
		auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, AudioCaches.size()))
			return;
		if (MoeURDo(AudioCaches[Offset].second.UndoList, AudioCaches[Offset].second.RedoList, Offset))
			DecModifyCount(CurSel, Offset);
	}

	void MoeVSRedo()
	{
		auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, AudioCaches.size()))
			return;
		if (MoeURDo(AudioCaches[Offset].second.RedoList, AudioCaches[Offset].second.UndoList, Offset))
			IncModifyCount(CurSel, Offset);
	}

	void SaveAll()
	{
		for (int i = 0; std::cmp_less(i, AudioPaths.size()); ++i)
			SaveData(i);
	}

	void SaveData(int CurSel)
	{
		if (CurSel < 0)
			CurSel = LabelControls.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, AudioCaches.size()))
		{
			auto F0Path = GetF0Path(AudioPaths[CurSel]);
			std::filesystem::path CachePath = F0Path + L".cache";
			if (exists(CachePath))
			{
				auto F0 = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
					F0Path + L".cache"
				);
				SaveWithHistory(F0Path, F0);
				remove(CachePath);
				reinterpret_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorWhite);
			}
			return;
		}
		if (AudioCaches[Offset].second.ModifyCount)
		{
			dynamic_cast<ListItemC*>(LabelControls.AudioList->Items[CurSel].get())->SetColor(ColorSkyBlue);
			LabelControls.AudioList->UpdateLayout();
			AudioCaches[Offset].second.ModifyCount = 0;
			SaveWithHistory(
				AudioCaches[Offset].second.F0Path,
				AudioCaches[Offset].second.F0
			);
			std::filesystem::path CachePath = AudioCaches[Offset].second.F0Path + L".cache";
			if (exists(CachePath))
				remove(CachePath);
		}
	}

	static const FloatTensor2D& GetUpSampleRates()
	{
		static FloatTensor2D Upp(DragonianLib::Functional::Arange(1.f, 481.f, 1.f).UnSqueeze(0).Evaluate());
		return Upp;
	}

	void SineGen()
	{
		auto CurSel = LabelControls.AudioList->SelectedItemIndex;
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || std::cmp_greater_equal(Offset, AudioCaches.size()))
			return;
		auto& Audio = LabelControls.CurvePlayer->GetAudio();
		const auto& F0 = AudioCaches[Offset].second.F0;
		const auto SineSize = std::min(F0.Size(1) * 480, Audio.Size(0));
		const auto Freq = F0[0].Clone().UnSqueeze(-1);

		auto Rad = Freq / 48000.f * GetUpSampleRates();
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
			SamplePoint = short(sin(F0Data[F0Idx] * 2.f * std::numbers::pi_v<float>) * 4000.f);
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
}

std::wstring GetLocalizationString(const std::wstring_view& _Str)
{
	return WndControls::LanguageXml->GetStringValue(_Str);
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
