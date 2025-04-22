#pragma once
#include "../MainWindow.h"

namespace UI
{
	using namespace Mui;

	//设置 关于页面
	class SettingPage : public Page
	{
	public:
		SettingPage(Ctrl::UIControl* parent, XML::MuiXML* ui);
		~SettingPage() override { delete m_anicls; }

		bool EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param) override;

		void Show(bool show);

	private:
		MAnimation* m_anicls = nullptr;
		bool m_isani = false;
	public:
		Ctrl::UIComBox* ExecutionProviderList = nullptr;
		Ctrl::UIComBox* DeviceIDList = nullptr;
		Ctrl::UIComBox* NumThreadList = nullptr;
		Ctrl::UIEditBox* SlicerThresholdEditBox = nullptr;
		Ctrl::UIEditBox* SlicerMinLengthEditBox = nullptr;
		Ctrl::UIEditBox* SlicerWindowLengthEditBox = nullptr;
		Ctrl::UIEditBox* SlicerHopSizeEditBox = nullptr;
		Ctrl::UIEditBox* VocoderHopSizeEditBox = nullptr;
		Ctrl::UIEditBox* VocoderMelBinsEditBox = nullptr;
		Ctrl::UIComBox* SamplerList = nullptr;
		Ctrl::UIComBox* F0ExtractorList = nullptr;
		Ctrl::UIComBox* ReflowSamplerList = nullptr;
	};
}