#include "SidePage.h"
#include "../DefControl.hpp"

namespace SimpleF0Labeler
{
	void SidePage::Show()
	{
		if (_IsAnimation)
			return;

		bool show = !_IsShow;

		if (show)
			_MyPageContent->SetPos(-265, 0, false);
		else
			_MyPageContent->SetPos(265, 0, false);

		_MyPageContent->SetVisible(true);

		//设置播放器的动画状态 在动画过程中不要重采样视图 不然会造成卡顿
		auto Player = (Waveform*)_MyPageContent->GetParent()->FindChildren(L"EditorPlayer");
		Player->SetAniFlag(true);

		_IsAnimation = true;
		_MyAnimation->CreateTask(
			[this, show, Player](const Mui::MAnimation::Calculator* calc, float percent)
			{
				const int x = calc->calc(Mui::MAnimation::Quintic_Out, show ? -265 : 0, show ? 0 : -265);
				_MyPageContent->SetPos(x, 0);
				_MyPageContent->GetParent()->UpdateLayout();
				if (percent == 100.f)
				{
					if (!show)
						_MyPageContent->SetVisible(false);
					_IsAnimation = false;
					Player->SetAniFlag(false);
				}
				return _IsAnimation;
			},
			300
		);

		_IsShow = show;
	}

	bool SidePage::EventProc(
		Mui::XML::PropName event,
		Mui::Ctrl::UIControl* control,
		std::any param
	)
	{
		const auto Name = control->GetName();

		if (event == Mui::Ctrl::Events::Control_LostFocus.Event() ||
			(event == Mui::Ctrl::Events::Key_Down.Event() && GetKeyState(VK_RETURN) & 0x8000))
		{
			if (Name == L"SidePageSamplingRate")
			{
				auto Control = dynamic_cast<Mui::Ctrl::UIEditBox*>(control);
				const auto val = _wcstoi64(Control->Text.Get().cstr(), nullptr, 10);
				if (val < 8000) Control->Text.Set(L"8000");
				if (val > 96000) Control->Text.Set(L"96000");
				return true;
			}
			if (Name == L"SidePageAlpha" || Name == L"SidePageBeta" || Name == L"SidePagePitchShift")
			{
				auto Control = dynamic_cast<Mui::Ctrl::UIEditBox*>(control);
				std::wstring CurText = Control->Text.Get().cstr();
				if (CurText.back() == L'.') CurText += L"0";
				Control->Text.Set(std::to_wstring(wcstof(CurText.c_str(), nullptr)));
			}
		}

		return false;
	}

	Mui::Ctrl::UIControl* SidePage::OnLoadPageContent(
		Mui::Ctrl::UIControl* parent,
		Mui::XML::MuiXML* ui
	)
	{
		const std::wstring xml = LR"(
			<PropGroup ID="SidePageLineI" Frame="0,5,265,20" Align="LinearH" AutoSize="false" />
			<PropGroup ID="SidePageLineII" Size="133,100%" Align="LinearH" AutoSize="false" />
			<PropGroup ID="SizePageEditBox" Inset="2,2,2,2" Size="100,20" />
			<PropGroup ID="SizePageCheckBox" Inset="2,2,2,2" Frame="11,5,110,110" />

			<UIControl Name="_SidePage" Size="100%,100%">
				<UIControl AutoSize="false" BgColor="#MenuFrame" Frame="0,11,100%,1" />

				<UILabel FontSize="16" Text="#SidePageParameters" Pos="11,11" />
				<UIControl AutoSize="false" Frame="0,10,265,100%" Align="LinearH">
					<UIControl AutoSize="false" Size="100%,100%" Align="LinearV">
						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="#SidePageSamplingRate" /></UIControl>
							<UIEditBox Prop="SizePageEditBox" Text="48000" Name="SidePageSamplingRate" Number="true" />
						</UIControl>
						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="#SidePageUseLogView" /></UIControl>
							<UICheckBox Prop="SizePageCheckBox" Name="SidePageUseLogView" Selected="true" />
						</UIControl>

						<UIControl AutoSize="false" BgColor="#MenuFrame" Frame="0,11,100%,1" />
						<UILabel Pos="11,11" Text="#SidePageArgAlphaBeta" />

						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="α" /></UIControl>
							<UIEditBox Prop="SizePageEditBox" Text="1.000000" Name="SidePageAlpha" />
						</UIControl>
						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="β" /></UIControl>
							<UIEditBox Prop="SizePageEditBox" Text="0.000000" Name="SidePageBeta" />
						</UIControl>

						<UIControl AutoSize="false" BgColor="#MenuFrame" Frame="0,11,100%,1" />
						<UILabel Pos="11,11" Text="#SidePageArgPitchShift" />

						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="#SidePagePitchShift" /></UIControl>
							<UIEditBox Prop="SizePageEditBox" Text="0.000000" Name="SidePagePitchShift" />
						</UIControl>

						<UIControl AutoSize="false" BgColor="#MenuFrame" Frame="0,11,100%,1" />
					</UIControl>
				</UIControl>
			</UIControl>
		)";
		_MyPageContent = parent->Child(L"SidePage");
		if (!ui->CreateUIFromXML(_MyPageContent, xml))
		{
			__debugbreak();
		}
		_MySamplingRateEditBox = _MyPageContent->FindChildren<Mui::Ctrl::UIEditBox>(L"SidePageSamplingRate");
		_MyLogViewCheckBox = _MyPageContent->FindChildren<Mui::Ctrl::UICheckBox>(L"SidePageUseLogView");
		_MyAlphaEditBox = _MyPageContent->FindChildren<Mui::Ctrl::UIEditBox>(L"SidePageAlpha");
		_MyBetaEditBox = _MyPageContent->FindChildren<Mui::Ctrl::UIEditBox>(L"SidePageBeta");
		_MyPitchEditBox = _MyPageContent->FindChildren<Mui::Ctrl::UIEditBox>(L"SidePagePitchShift");

		return _MyPageContent->Child(L"_SidePage");
	}

	float SidePage::GetAlpha() const
	{
		return wcstof(_MyAlphaEditBox->Text.Get().cstr(), nullptr);
	}

	float SidePage::GetBeta() const
	{
		return wcstof(_MyBetaEditBox->Text.Get().cstr(), nullptr);
	}

	float SidePage::GetPitch() const
	{
		return wcstof(_MyPitchEditBox->Text.Get().cstr(), nullptr);
	}

	int64_t SidePage::GetSamplingRate() const
	{
		return _wcstoi64(_MySamplingRateEditBox->Text.Get().cstr(), nullptr, 10);
	}

	bool SidePage::IsUsingLogView() const
	{
		return _MyLogViewCheckBox->Selected;
	}

}
