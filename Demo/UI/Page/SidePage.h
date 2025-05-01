#pragma once
#include "../MainWindow.h"

namespace SimpleF0Labeler
{
	class SidePage : public Mui::UIPage  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		SidePage(
			Mui::UIPage* parent
		) : UIPage(parent), _MyAnimation(std::make_shared<Mui::MAnimation>())
		{

		}

		void Show();
		float GetAlpha() const;
		float GetBeta() const;
		float GetPitch() const;
		int64_t GetSamplingRate() const;
		bool IsUsingLogView() const;
		bool IsUsingLogSpec() const;

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
		bool _IsAnimation = false;
		std::shared_ptr<Mui::MAnimation> _MyAnimation = nullptr;
		Mui::Ctrl::UIControl* _MyPageContent = nullptr;
		Mui::Ctrl::UIEditBox* _MySamplingRateEditBox = nullptr;
		Mui::Ctrl::UISwitch* _MyLogViewCheckBox = nullptr;
		Mui::Ctrl::UISwitch* _MyLogSpecCheckBox = nullptr;
		Mui::Ctrl::UIEditBox* _MyAlphaEditBox = nullptr;
		Mui::Ctrl::UIEditBox* _MyBetaEditBox = nullptr;
		Mui::Ctrl::UIEditBox* _MyPitchEditBox = nullptr;
		bool _IsShow = true;
	};
}