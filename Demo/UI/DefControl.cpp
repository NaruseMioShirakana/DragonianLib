#include "DefControl.hpp"
#include "Libraries/Stft/Stft.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

#include <Render/Sound/Mui_DirectSound.h>

#include "Page/SidePage.h"

namespace WndControls
{
	const auto MaxFreq = Mui::Ctrl::PitchLabel::PitchToF0(119.9f);
	const auto MinFreq = Mui::Ctrl::PitchLabel::PitchToF0(0);

	std::vector<std::wstring> AudioPaths;
	std::deque<std::pair<std::wstring, MyAudioData>> AudioCaches;
	size_t MaxCacheCount = 20;

	std::mutex UndoMutex;
	std::deque<std::deque<FloatTensor2D>> UndoList;
	std::deque<std::deque<FloatTensor2D>> RedoList;
	size_t UndoRedoMaxCount = 200;

	Mui::XML::MuiXML* LanguageXml = nullptr;
	MyControls LabelControls;

	const DragonianLib::FunctionTransform::MFCCKernel& GetMelFn()
	{
		static DragonianLib::FunctionTransform::MFCCKernel MelKernel{
			48000, 8192, 480, -1, 128, 20, 16000
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

	std::wstring GetPath(const std::wstring& RawAudioPath, const wchar_t* Fol, const wchar_t* Ext)
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
		NewPath += Ext;
		NewPath = DragonianLib::GetCurrentFolder() + L"/User/" + Fol + NewPath;
		return NewPath;
	}

	std::wstring GetF0Path(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"F0/", L".npy");
	}

	std::wstring GetAudioPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Audio/", L".wav");
	}

	std::wstring GetSpecPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Spec/", L".npy");
	}

	std::wstring GetMelPath(const std::wstring& RawAudioPath)
	{
		return GetPath(RawAudioPath, L"Mel/", L".npy");
	}

	static void SaveData(size_t idx)
	{
		if (idx < AudioCaches.size())
		{
			const auto& Path = AudioCaches[idx].first;
			DragonianLib::Functional::NumpySave(
				GetF0Path(Path),
				AudioCaches[idx].second.F0
			);
			DragonianLib::Functional::NumpySave(
				GetSpecPath(Path),
				AudioCaches[idx].second.Spec
			);
			DragonianLib::Functional::NumpySave(
				GetMelPath(Path),
				AudioCaches[idx].second.Mel
			);
		}
	}

	static void EraseFront()
	{
		SaveData(0);
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
			SaveData(Offset);
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
		auto SpecPath = GetSpecPath(Path);
		auto MelPath = GetMelPath(Path);
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
			auto Rng = Audio.GetCRng();
			OpenOutputStream(
				SamplingRate,
				AudioPath,
				DragonianLib::AvCodec::AvCodec::PCM_FORMAT_FLOAT32,
				2
			).EncodeAll(
				Rng, SamplingRate, 2
			);
		}
		if (std::filesystem::exists(F0Path))
		{
			F0 = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
				F0Path
			);
			if (F0.Size(0) != 10)
				F0 = F0.Padding(
					IArray(
						DragonianLib::PadCount(0, 10 - F0.Size(0))
					),
					DragonianLib::PaddingType::Zero
				);
		}
		else
		{
			const auto HopSize = SamplingRate / 100;
			const auto F0Size = (DragonianLib::SizeType)ceil(double(Audio.Shape(0)) / double(HopSize)) + 1;
			const auto F0Shape = DragonianLib::Dimensions{ 10, F0Size };
			F0 = DragonianLib::Functional::Zeros(F0Shape);
			DragonianLib::Functional::NumpySave(
				F0Path,
				F0
			);
		}
		Audio = Audio.Interpolate<DragonianLib::Operators::InterpolateMode::Linear>(
			DragonianLib::IDim(0),
			DragonianLib::IScale(48000. / double(SamplingRate))
		).Evaluate();
		if (std::filesystem::exists(SpecPath))
			Spec = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
				SpecPath
			);
		else
		{
			Spec = GetMelFn().GetStftKernel()(
				Audio.Mean(-1).UnSqueeze(0).UnSqueeze(0)
				).Squeeze(0).Squeeze(0).Evaluate();
			DragonianLib::Functional::NumpySave(
				SpecPath,
				Spec
			);
		}
		if (std::filesystem::exists(MelPath))
			Mel = DragonianLib::Functional::NumpyLoad<DragonianLib::Float32, 2>(
				MelPath
			);
		else
		{
			Mel = GetMelFn()(
			   Spec.View(1, 1, Spec.Size(0), Spec.Size(1))
			   ).Squeeze(0).Squeeze(0).Evaluate();
			DragonianLib::Functional::NumpySave(
				MelPath,
				Mel
			);
		}
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
				std::move(Audio),
				std::move(F0),
				std::move(Spec),
				std::move(Mel)
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
		for (size_t i = 0; i < AudioCaches.size(); ++i)
			SaveData(i);
		AudioCaches.clear();
		std::lock_guard lg(UndoMutex);
		UndoList.clear();
		RedoList.clear();
	}

	void InsertAudio(std::wstring Path)
	{
		auto FileName = std::filesystem::path(Path).stem().wstring();
		FileName = L"    " + FileName.substr(0, 50);
		AudioPaths.emplace_back(std::move(Path));
		auto Item = new Mui::Ctrl::ListItem;
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
		(AudioAndF0.F0 = 261.626f).Evaluate();
		LabelControls.CurveEditor->SetCurveData(AudioAndF0.F0);
		LabelControls.CurvePlayer->SetPlayPos(0);
		LabelControls.CurvePlayer->SetAudioData(AudioAndF0.Audio);
	}

	void DeleteAudio(int idx)
	{
		LabelControls.CurveEditor->SetPlayLinePos(0);
		LabelControls.CurveEditor->SetCurveData(std::nullopt);
		LabelControls.CurvePlayer->Clear();
		LabelControls.AudioList->DeleteItem(idx);
		LabelControls.AudioList->SetCurSelItem(-1);

		AudioPaths.erase(AudioPaths.begin() + idx);
		EraseCache(idx);
	}

	void MoeURDo(
		std::deque<FloatTensor2D>& CurUndo,
		std::deque<FloatTensor2D>& CurRedo,
		ptrdiff_t Offset
	)
	{
		std::unique_lock lg(UndoMutex, std::try_to_lock);
		if (CurUndo.empty())
			return;
		auto UndoData = std::move(CurUndo.back());
		CurRedo.emplace_back(std::move(AudioCaches[Offset].second.F0));
		AudioCaches[Offset].second.F0 = std::move(UndoData.Evaluate());
		LabelControls.CurveEditor->ReSetCurveData(AudioCaches[Offset].second.F0, -1);
		LabelControls.CurveEditor->UpDate();

		if (CurRedo.size() > UndoRedoMaxCount)
			CurRedo.pop_front();
		CurUndo.pop_back();
	}

	void MoeVSUndo()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		MoeURDo(UndoList[Offset], RedoList[Offset], Offset);
	}

	void MoeVSRedo()
	{
		auto CurSel = LabelControls.AudioList->GetCurSelItem();
		auto Offset = GetOffset(CurSel);
		if (Offset == -1 || Offset >= static_cast<ptrdiff_t>(UndoList.size()))
			return;
		MoeURDo(RedoList[Offset], UndoList[Offset], Offset);
	}

	void ExportSliceCurveData(HWND hwnd)
	{
		
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
	constexpr long MaxPath = 8000;
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

std::wstring MoeGetSaveFile(const TCHAR* szFilter, HWND hwndOwner, const TCHAR* lpstrDefExt)
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
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_OVERWRITEPROMPT;
	if (GetSaveFileName(&ofn))
	{
		std::wstring preFix = szFileName.data();
		return preFix;
	}
	return L"";
#else
#endif
}

std::wstring GetLocalizationString(const std::wstring_view& _Str)
{
	return WndControls::LanguageXml->GetStringValue(_Str);
}