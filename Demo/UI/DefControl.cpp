#include "DefControl.hpp"
#include "Libraries/Stft/Stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

#include <Render/Sound/Mui_DirectSound.h>
#include <gdiplus.h>
#pragma comment(lib, "Msimg32.lib")
#pragma comment(lib, "gdiplus.lib")

#include "Page/SidePage.h"

namespace WndControls
{
	const auto MaxFreq = Mui::Ctrl::PitchLabel::PitchToF0(119.9f);
	const auto MinFreq = Mui::Ctrl::PitchLabel::PitchToF0(0);

	static inline Mui::_m_color __SpecColorMap[]{
		Mui::Color::M_RGB(int(0.0f * 255), int(0.0f * 255),int(0.0f * 255)),       // black
		Mui::Color::M_RGB(int(0.251f * 255), int(0.0f * 255),int(0.f * 255)),     // dark blue
		Mui::Color::M_RGB(int(0.502f * 255), int(0.0f * 255),int(0.f * 255)),     // blue
		Mui::Color::M_RGB(int(0.502f * 255), int(0.0f * 255),int(0.251f * 255)),   // purple-blue
		Mui::Color::M_RGB(int(0.502f * 255),int(0.0f * 255),int(0.502f * 255)),   // purple
		Mui::Color::M_RGB(int(0.502f * 255),int(0.0f * 255),int(0.753f * 255)),   // magenta
		Mui::Color::M_RGB(int(0.0f * 255),int(0.0f * 255),int(1.0f * 255)),       // red
		Mui::Color::M_RGB(int(0.0f * 255),int(0.502f * 255),int(1.0f * 255)),     // orange
		Mui::Color::M_RGB(int(0.0f * 255),int(1.0f * 255),int(1.0f * 255)),       // yellow
		Mui::Color::M_RGB(int(0.502f * 255),int(1.0f * 255),int(1.0f * 255)),     // light yellow
	};

	static inline std::vector<std::wstring> AudioPaths;
	static inline std::deque<std::pair<std::wstring, MyAudioData>> AudioCaches;
	constexpr size_t MaxCacheCount = 10;

	static inline std::mutex UndoMutex;
	static inline std::deque<std::deque<FloatTensor2D>> UndoList;
	static inline std::deque<std::deque<FloatTensor2D>> RedoList;
	constexpr size_t UndoRedoMaxCount = 100;

	static inline Mui::XML::MuiXML* LanguageXml = nullptr;
	static inline MyControls LabelControls;

	static inline Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	static inline ULONG_PTR gdiplusToken;

	static DragonianLib::OnStartUP StartUPLabel{
		[]
		{
			GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
		}
	};

	static DragonianLib::SharedScopeExit OnLabelExit{
		[]
		{
			if (gdiplusToken)
				Gdiplus::GdiplusShutdown(gdiplusToken);
		}
	};

	static std::pair<ImageTensor, ImageTensor> Write2Bmp(
		const std::wstring& Path,
		const FloatTensor2D& Spec
	)
	{
		const auto [Frames, Bins] = Spec.Size().RawArray();

		Gdiplus::Bitmap bitmap(static_cast<INT>(Frames), static_cast<INT>(Bins), PixelFormat24bppRGB);

		const auto SpecFlat = (Spec.View(-1) + 1e-5f).Log10();
		const auto Max = SpecFlat.ReduceMax(0);
		const auto Min = SpecFlat.ReduceMin(0);
		const auto SpecMin = Min.Evaluate().Item();
		const auto SpecMax = Max.Evaluate().Item();

		const auto SpecData = Spec.Data();

		for (INT y = 0; y < static_cast<INT>(Bins); ++y)
		{
			for (INT x = 0; x < static_cast<INT>(Frames); ++x)
			{
				float Value = (*(SpecData + x * Bins + y) - SpecMin) / (SpecMax - SpecMin);
				Value = std::clamp(Value, 0.001f, 0.999f);
				const auto Color = __SpecColorMap[int(Value * 9.f)];
				bitmap.SetPixel(x, static_cast<INT>(Bins) - y, Color);
			}
		}

		CLSID pngClsid;
		CLSIDFromString(L"{557CF406-1A04-11D3-9A73-0000F81EF32E}", &pngClsid); // PNG CLSID
		bitmap.Save(Path.c_str(), &pngClsid, nullptr);

		return {};
	}

	static const DragonianLib::FunctionTransform::MFCCKernel& GetMelFn()
	{
		static DragonianLib::FunctionTransform::MFCCKernel MelKernel{
			SpecSamplingRate, 8192, SpecSamplingRate / 400, -1, 128, 20.f, 11025.f
		};
		return MelKernel;
	}

	void InitCtrl(
		Mui::Ctrl::UIListBox* AudioList,
		Mui::Ctrl::CurveEditor* CurveEditor,
		Mui::Ctrl::Waveform* CurvePlayer
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
		return GetPath(RawAudioPath, L"Audio/", L".mp3");
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
		AudioCaches.pop_front();
		std::lock_guard lg(UndoMutex);
		UndoList.pop_front();
		RedoList.pop_front();
	}

	static void EraseCache(int idx)
	{
		const auto& Path = AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != AudioCaches.end())
		{
			auto Offset = std::distance(AudioCaches.begin(), Iter);
			AudioCaches.erase(Iter);
			std::lock_guard lg(UndoMutex);
			UndoList.erase(UndoList.begin() + Offset);
			RedoList.erase(RedoList.begin() + Offset);
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
		FloatTensor2D Audio, F0, Spec, Mel;
		if (std::filesystem::exists(AudioPath))
			Audio = DragonianLib::AvCodec::OpenInputStream(
				AudioPath
			).DecodeAll(SamplingRate, 2);
		else if (std::filesystem::exists(Path))
		{
			Audio = DragonianLib::AvCodec::OpenInputStream(
				Path
			).DecodeAll(SamplingRate, 2);
		}
		OpenOutputStream(
			SamplingRate,
			std::filesystem::path(AudioPath),
			DragonianLib::AvCodec::AvCodec::PCM_FORMAT_FLOAT32,
			2
		).EncodeAll(
			Audio.GetCRng(), SamplingRate, 2
		);
		const auto HopSize = SamplingRate / 100;
		const auto AudioFrames = (DragonianLib::SizeType)ceil(double(Audio.Shape(0)) / double(HopSize)) + 1;
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

		Spec = GetMelFn().GetStftKernel()(
			Audio.View(1, 1, Audio.Size(0)).Interpolate<DragonianLib::Operators::InterpolateMode::Linear>(
				DragonianLib::IDim(-1),
				DragonianLib::IScale(double(SpecSamplingRate) / 48000.)
			)
			).Squeeze(0).Squeeze(0)[{DragonianLib::Range{ 0, AudioFrames }}].Contiguous().Evaluate();

		Mel = GetMelFn()(
		   Spec.View(1, 1, Spec.Size(0), Spec.Size(1))
		   ).Squeeze(0).Squeeze(0).Evaluate();

		Write2Bmp(
			GetSpecPath(Path),
			Spec
		);

		if (AudioCaches.size() >= MaxCacheCount - 1)
			EraseFront();
		{
			std::lock_guard lg(UndoMutex);
			UndoList.emplace_back();
			RedoList.emplace_back();
		}
		return AudioCaches.emplace_back(
			Path,
			MyAudioData{
				SamplingRate,
				Audio.Pad(DragonianLib::PaddingCounts{DragonianLib::PadCount{0, 1}}, DragonianLib::PaddingType::Zero).Evaluate(),
				std::move(F0.Evaluate()),
				std::move(Spec.Evaluate()),
				std::move(Mel.Evaluate()),
				std::move(F0Path)
			}
		).second;
	}

	static ptrdiff_t GetOffset(int idx)
	{
		const auto& Path = AudioPaths[idx];
		auto Iter = std::ranges::find_if(
			AudioCaches, [&](const auto& pair) { return pair.first == Path; }
		);
		if (Iter != AudioCaches.end())
			return std::distance(AudioCaches.begin(), Iter);
		return -1;
	}

	static void IncModifyCount(auto CurSel, auto Offset)
	{
		const auto Prev = AudioCaches[Offset].second.ModifyCount;
		++AudioCaches[Offset].second.ModifyCount;
		if (Prev && !AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->GetItem(CurSel))->SetColor(Mui::Color::M_White);
		else if (!Prev && AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->GetItem(CurSel))->SetColor(Mui::Color::M_RED);
		LabelControls.AudioList->UpdateLayout();
	}

	static void DecModifyCount(auto CurSel, auto Offset)
	{
		const auto Prev = AudioCaches[Offset].second.ModifyCount;
		--AudioCaches[Offset].second.ModifyCount;
		if (Prev && !AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->GetItem(CurSel))->SetColor(Mui::Color::M_White);
		else if (!Prev && AudioCaches[Offset].second.ModifyCount)
			dynamic_cast<ListItemC*>(LabelControls.AudioList->GetItem(CurSel))->SetColor(Mui::Color::M_RED);
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
		const auto CurSel = LabelControls.AudioList->GetCurSelItem();
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || CacheIdx >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		std::lock_guard lg(UndoMutex);
		UndoList[CacheIdx].emplace_back(
			AudioCaches[CacheIdx].second.F0.Clone()
		);
	}

	void CheckUnchanged()
	{
		const auto CurSel = LabelControls.AudioList->GetCurSelItem();
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || CacheIdx >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		std::lock_guard lg(UndoMutex);
		const auto& CurF0 = AudioCaches[CacheIdx].second.F0;
		const auto& CurUndo = UndoList[CacheIdx].back();
		const auto F0Size = CurF0.ElementCount();
		const auto UndoSize = CurUndo.ElementCount();
		if ((F0Size == UndoSize && AllEqual(CurF0.Data(), CurUndo.Data(), F0Size)))
			UndoList[CacheIdx].pop_back();
		else
		{
			IncModifyCount(CurSel, CacheIdx);
			if (UndoList[CacheIdx].size() > UndoRedoMaxCount)
				UndoList[CacheIdx].pop_front();
			RedoList[CacheIdx].clear();
		}
	}

	void ApplyAppendUndo()
	{
		const auto CurSel = LabelControls.AudioList->GetCurSelItem();
		const auto CacheIdx = GetOffset(CurSel);
		if (CacheIdx == -1 || CacheIdx >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		std::lock_guard lg(UndoMutex);
		if (UndoList[CacheIdx].size() > UndoRedoMaxCount)
			UndoList[CacheIdx].pop_front();
		RedoList[CacheIdx].clear();
	}

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges)
	{
		const auto Pitch = dynamic_cast<UI::SidePage*>(UI::FindPage(L"sidepage"))->GetPitch();
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
		const auto Alpha = dynamic_cast<UI::SidePage*>(UI::FindPage(L"sidepage"))->GetAlpha();
		const auto Beta = dynamic_cast<UI::SidePage*>(UI::FindPage(L"sidepage"))->GetBeta();
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
		AudioCaches.clear();
		std::lock_guard lg(UndoMutex);
		UndoList.clear();
		RedoList.clear();
	}

	void InsertAudio(std::wstring Path)
	{
		if (std::ranges::contains(AudioPaths, Path))
			return;
		auto FileName = std::filesystem::path(Path).stem().wstring();
		FileName = L"    " + FileName.substr(0, 50);
		AudioPaths.emplace_back(std::move(Path));
		auto Item = new ListItemC;
		Item->SetText(std::move(FileName));
		LabelControls.AudioList->AddItem(Item, -1, true);
	}

	void SetLanguageXML(Mui::XML::MuiXML* xml)
	{
		LanguageXml = xml;
	}

	void SetCurveEditorDataIdx(int AudioIdx, unsigned SamplingRate)
	{
		auto& AudioAndF0 = GetData(AudioIdx, SamplingRate);
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
		LabelControls.AudioList->DeleteItem(idx);
		LabelControls.AudioList->SetCurSelItem(-1);

		AudioPaths.erase(AudioPaths.begin() + idx);
		EraseCache(idx);
	}

	static bool MoeURDo(
		std::deque<FloatTensor2D>& CurUndo,
		std::deque<FloatTensor2D>& CurRedo,
		ptrdiff_t Offset
	)
	{
		std::unique_lock lg(UndoMutex, std::try_to_lock);
		if (CurUndo.empty())
			return false;
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

	void MoeVSUndo()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		if (MoeURDo(UndoList[Offset], RedoList[Offset], Offset))
			DecModifyCount(CurSel, Offset);
	}

	void MoeVSRedo()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		if (MoeURDo(RedoList[Offset], UndoList[Offset], Offset))
			IncModifyCount(CurSel, Offset);
	}

	void SaveData()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		if (AudioCaches[Offset].second.ModifyCount)
		{
			dynamic_cast<ListItemC*>(LabelControls.AudioList->GetItem(CurSel))->SetColor(Mui::Color::M_White);
			LabelControls.AudioList->UpdateLayout();
			AudioCaches[Offset].second.ModifyCount = 0;
			SaveWithHistory(
				AudioCaches[Offset].second.F0Path,
				AudioCaches[Offset].second.F0
			);
		}
	}

	static const FloatTensor2D& GetUpSampleRates()
	{
		static FloatTensor2D Upp(DragonianLib::Functional::Arange(1.f, 481.f, 1.f).UnSqueeze(0).Evaluate());
		return Upp;
	}

	void SineGen()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
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
			SamplePoint = short(sin(F0Data[F0Idx] * 2.f * 3.1415926535f) * 4000.f);
		}
	}

	void LoadFiles(HWND hWnd)
	{
		auto Files = MoeGetOpenFiles(
			TEXT("Audio Files (*.wav; *.mp3; *.ogg; *.flac)|*.wav;*.mp3;*.ogg;*.flac|All Files (*.*)|*.*||"),
			hWnd
		);
		for (auto& i : Files)
			InsertAudio(std::move(i));
	}

	void LoadF0(HWND hWnd)
	{
		const auto Path = MoeGetOpenFile(
			TEXT("Numpy Files (*.wav)|*.npy||"),
			hWnd,
			L"npy"
		);
		if (!Path.empty())
		{
			try
			{
				auto F0Tensor = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
					Path
				);
				Mui::Ctrl::Write2Clipboard(
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

std::wstring MoeGetOpenFile(const TCHAR* szFilter, HWND hwndOwner, const TCHAR* lpstrDefExt)
{
	constexpr long MaxPath = 8000;
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

std::vector<std::wstring> MoeGetOpenFiles(const TCHAR* szFilter, HWND hwndOwner, const TCHAR* lpstrDefExt)
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

std::wstring GetLocalizationString(const std::wstring_view& _Str)
{
	return WndControls::LanguageXml->GetStringValue(_Str);
}

MyAudioData::~MyAudioData()
{
	if (!F0Path.empty())
		WndControls::SaveWithHistory(
			F0Path + L".cache",
			F0
		);
}