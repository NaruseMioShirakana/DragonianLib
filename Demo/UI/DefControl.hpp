#pragma once
#include <Mui.h>
#include "MainWindow.h"

#define MoeMessageBox(title, text, cap, hwnd) MessageBoxW(hwnd, GetLocalizationString(text).c_str(), GetLocalizationString(title).c_str(), (cap))

#define MoeMessageBoxQ(title, text, cap, hwnd) MessageBoxW(hwnd, (text), (title), (cap))

#define MoeMessageBoxC(title, text, cap) MoeMessageBox(title, text, cap, hwnd)

#define MoeMessageBoxAskC(title, text) MoeMessageBox(title, text, MB_YESNO | MB_ICONASTERISK, hwnd)

#define MoeMessageBoxAsk(title, text, hwnd) MoeMessageBox(title, text, MB_YESNO | MB_ICONASTERISK, (hwnd))

#define MoeGetHwnd (HWND)this->m_parent->GetParentWin()->GetWindowHandle()

std::wstring GetLocalizationString(const std::wstring_view& _Str);

namespace WndControls
{
	constexpr DragonianLib::Int64 SpecSamplingRate = 16000;

	struct MyControls
	{
		Mui::Ctrl::UIListBox* AudioList = nullptr;
		Mui::Ctrl::CurveEditor* CurveEditor = nullptr;
		Mui::Ctrl::Waveform* CurvePlayer = nullptr;
	};

	void InitCtrl(
		Mui::Ctrl::UIListBox* AudioList = nullptr,
		Mui::Ctrl::CurveEditor* CurveEditor = nullptr,
		Mui::Ctrl::Waveform* CurvePlayer = nullptr
	);

	void AppendUndo();

	void CheckUnchanged();

	void ApplyAppendUndo();

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges);

	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges);

	void SetPlayerPos(size_t idx);

	size_t GetPcmSize();

	void EmptyCache();

	void InsertAudio(std::wstring Path);

	void SetLanguageXML(Mui::XML::MuiXML* xml);

	void SetCurveEditorDataIdx(int AudioIdx, unsigned SamplingRate);

	void DeleteAudio(int idx);

	void PlayPause();

	void MoeVSUndo();

	void MoeVSRedo();

	void SaveAll();

	void SaveData(int CurSel = -1);

	void SineGen();

	void LoadFiles(HWND hWnd);

	void LoadF0(HWND hWnd);
}