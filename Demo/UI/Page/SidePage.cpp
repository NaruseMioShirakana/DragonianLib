#include "SidePage.h"
#include "../DefControl.hpp"

namespace SimpleF0Labeler
{
	void SidePage::Show()
	{
		if (_IsAnimation)
			return;

		_IsAnimation = true;
		auto Player = (Waveform*)_MyPageContent->GetParent()->FindChildren(L"EditorPlayer");
		Player->SetAniFlag(true);

		if (_IsShow)
			_MyPageContent->SetPos(-265, 0, false);
		else
			_MyPageContent->SetPos(0, 0, false);

		_MyPageContent->SetVisible(!_IsShow);
		_MyPageContent->GetParent()->UpdateLayout();

		Player->SetAniFlag(false);
		_IsAnimation = false;

		_IsShow = !_IsShow;
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
			<PropGroup ID="SizePageEditBox" Inset="2,2,2,2" Frame="11,0,100,20" />
			<PropGroup ID="SizePageCheckBox" Inset="2,2,2,2" Frame="11,2,30,2f" />

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
							<UISwitch Prop="SizePageCheckBox" Name="SidePageUseLogView" SwitchOn="true" />
						</UIControl>
						<UIControl Prop="SidePageLineI">
							<UIControl Prop="SidePageLineII"><UILabel Pos="11,2" Text="#SidePageUseLogSpec" /></UIControl>
							<UISwitch Prop="SizePageCheckBox" Name="SidePageUseLogSpec" SwitchOn="true" />
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
		_MyLogViewCheckBox = _MyPageContent->FindChildren<Mui::Ctrl::UISwitch>(L"SidePageUseLogView");
		_MyLogSpecCheckBox = _MyPageContent->FindChildren<Mui::Ctrl::UISwitch>(L"SidePageUseLogSpec");
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
		return _MyLogViewCheckBox->SwitchOn;
	}

	bool SidePage::IsUsingLogSpec() const
	{
		return _MyLogSpecCheckBox->SwitchOn;
	}

}
