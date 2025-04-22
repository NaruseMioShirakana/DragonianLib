#pragma once
#include "../MainWindow.h"

namespace UI
{
	using namespace Mui;

	//UI主页面
	class MainPage : public Page
	{
	public:
		MainPage(Ctrl::UIControl* parent, XML::MuiXML* ui);

		bool EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param) override;
	private:
		Ctrl::UIControl* m_sidepage = nullptr;
	};
}