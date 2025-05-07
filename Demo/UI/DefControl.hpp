#pragma once
#include <Mui.h>
#include "Extend/Waveform.h"
#include "Extend/CurveEditor.h"
#include "MainWindow.h"

#define MoeMessageBox(title, text, cap, hwnd) MessageBoxW(hwnd, GetLocalizationString(text).c_str(), GetLocalizationString(title).c_str(), (cap))

namespace WndControls
{
	constexpr DragonianLib::Int64 SpecSamplingRate = 16000;

	void InitCtrl(
		Mui::Ctrl::UIListBox* AudioList = nullptr,
		SimpleF0Labeler::CurveEditor* CurveEditor = nullptr,
		SimpleF0Labeler::Waveform* CurvePlayer = nullptr
	);

	std::wstring Localization(const std::wstring_view& Key);

	Mui::XML::MuiXML* GetUiXml();

	void AppendUndo();

	void CheckUnchanged();

	void ApplyAppendUndo();

	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges);
	void ApplyPitchShift(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges, float Shift);

	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges);
	void ApplyCalc(const DragonianLib::TemplateLibrary::MutableRanges<float>& Ranges, float Alpha, float Beta);

	void SetPlayerPos(size_t Index);

	size_t GetPcmSize();

	void EmptyCache();

	void InsertAudio(std::wstring Path);

	void SetLanguageXML(Mui::XML::MuiXML* XmlUI);

	void SetCurveEditorDataIdx(int AudioIdx, unsigned SamplingRate, bool UseLogSpec);

	void DeleteAudio(int Index);

	void PlayPause();

	void MoeVSUndo();

	void MoeVSRedo();

	void SaveAll();

	void SaveData(int CurSel = -1);

	void SineGen();

	void LoadFiles(HWND hWnd);

	void LoadF0(HWND hWnd);

	void ReCalcSpec(bool _Log);
}