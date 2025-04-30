#pragma once
#include "../MainWindow.h"
#include "../Extend/Waveform.h"
#include "../Extend/CurveEditor.h"

namespace SimpleF0Labeler
{
	//UI主页面
	class MainPage : public Mui::UIPage
	{
	public:
		MainPage(
			Mui::UIPage* root
		) : UIPage(root)
		{

		}

	protected:
		friend class MainWindow;
		bool EventProc(
			Mui::XML::PropName event,
			Mui::Ctrl::UIControl* control,
			std::any param
		) override;

	private:
		Mui::Ctrl::UIControl* OnLoadPageContent(
			Mui::Ctrl::UIControl* parent,
			Mui::XML::MuiXML* ui
		) override;

	protected:
		Waveform* m_wave = nullptr;
		CurveEditor* m_editor = nullptr;
		Mui::Ctrl::UIListBox* m_list = nullptr;
	};
}