#pragma once
#include "../MainWindow.h"

namespace UI
{
	using namespace Mui;

	//参数侧边栏
	class SidePage : public Page
	{
	public:
		SidePage(Ctrl::UIControl* parent, XML::MuiXML* ui);
		~SidePage() override { delete m_anicls; m_page = nullptr; }
		bool EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param) override;
		void Show(bool show);

		float GetAlpha() const;
		float GetBeta() const;
		float GetPitch() const;

	private:
		MAnimation* m_anicls = nullptr;
		bool m_isani = false;
		Ctrl::UIEditBox* m_alpha = nullptr;
		Ctrl::UIEditBox* m_beta = nullptr;
		Ctrl::UIEditBox* m_pitch = nullptr;
	};
}