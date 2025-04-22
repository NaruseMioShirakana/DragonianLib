#include "SidePage.h"
#include "../DefControl.hpp"

namespace UI
{
	SidePage::SidePage(Ctrl::UIControl* parent, XML::MuiXML* ui) : Page(parent, ui)
	{
		const std::wstring xml = LR"(
		<PropGroup id="sidecombox" fontSize="12" style="comboxDark" textAlign="5" fontColor="#textColor"
		listStyle="listDark" itemStyle="comitemDark" itemHeight="25" autoSize="false" menuHeight="200"
		iFontColor="#textColor" iTextAlign="5" styleV="scrollDark" button="false" barWidth="6" inset="1,1,1,1"
		dropIcon="icon_drop_light" size="243,26" />
		<PropGroup id="sideicobtn" fontSize="12" style="buttonDark" textAlign="5" fontColor="#textColor"
		autoSize="false" offset="0,0,5,5" iconSize="16,16" frame="11,5,243,26" />
		<PropGroup id="sidepm_l1" frame="0,5,265,20" align="LinearH" autoSize="false" />
		<PropGroup id="sidepm_l2" size="133,100%" align="LinearH" autoSize="false" />
		<PropGroup id="side_edit" inset="2,2,2,2" frame="11,0,110,20" />
		<PropGroup id="side_check" inset="2,2,2,2" frame="11,5,110,110" />

		<UIControl autoSize="false" bgColor="#menuframe" frame="0,11,100%,1" />

		<UILabel fontSize="16" text="参数" pos="11,11" />
		<UIControl autoSize="false" frame="0,10,265,100%" align="LinearH">
			<UIControl autoSize="false" size="100%,100%" align="LinearV">
				<PropGroup id="side_edit" inset="2,2,2,2" size="100,20" />
				<UIControl prop="sidepm_l1"><UIControl prop="sidepm_l2">
					<UILabel pos="11,2" text="采样率" /></UIControl>
				<UIEditBox prop="side_edit" text="48000" name="side_sampling_rate" number="true" /></UIControl>
				<UIControl prop="sidepm_l1"><UIControl prop="sidepm_l2">
					<UILabel pos="11,2" text="使用对数视图" /></UIControl>
				<UICheckBox prop="side_check" name="side_use_log_view" isSel="true" /></UIControl>

				<UIControl autoSize="false" bgColor="#menuframe" frame="0,11,100%,1" />
				<UILabel pos="11,11" text="使用公式 f0 = (α * f0) + β 计算" />

				<UIControl prop="sidepm_l1"><UIControl prop="sidepm_l2">
					<UILabel pos="11,2" text="α" /></UIControl>
				<UIEditBox prop="side_edit" text="1.000000" name="side_alpha_value" /></UIControl>
				<UIControl prop="sidepm_l1"><UIControl prop="sidepm_l2">
					<UILabel pos="11,2" text="β" /></UIControl>
				<UIEditBox prop="side_edit" text="0.000000" name="side_beta_value" /></UIControl>

				<UIControl autoSize="false" bgColor="#menuframe" frame="0,11,100%,1" />
				<UILabel pos="11,11" text="使用音高偏移公式计算" />
				<UIControl prop="sidepm_l1"><UIControl prop="sidepm_l2">
					<UILabel pos="11,2" text="音高偏移" /></UIControl>
				<UIEditBox prop="side_edit" text="0.000000" name="side_pitch_value" /></UIControl>

				<UIControl autoSize="false" bgColor="#menuframe" frame="0,11,100%,1" />
			</UIControl>
		</UIControl>

		)";
		m_page = parent->Child(L"sidepage");
		if (!ui->CreateUIFromXML(m_page, xml))
		{
			__debugbreak();
		}

		m_anicls = new MAnimation(parent->GetParentWin());
		m_alpha = m_page->FindChildren<Ctrl::UIEditBox>(L"side_alpha_value");
		m_beta = m_page->FindChildren<Ctrl::UIEditBox>(L"side_beta_value");
		m_pitch = m_page->FindChildren<Ctrl::UIEditBox>(L"side_pitch_value");
	}

	std::vector<std::wstring> VitsSpeakerName, DiffSpeakerName;

	bool SidePage::EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param)
	{
		const auto Name = control->GetName();
		
		if ((event == Event_Focus_False || (event == Event_Key_Down && GetKeyState(VK_RETURN) & 0x8000)))
		{
			if (Name == L"side_sampling_rate")
			{
				auto ctrl = dynamic_cast<Ctrl::UIEditBox*>(control);
				const auto val = _wcstoi64(ctrl->GetCurText().c_str(), nullptr, 10);
				if (val < 8000)ctrl->SetCurText(L"8000");
				if (val > 96000)ctrl->SetCurText(L"96000");
				return true;
			}
			if (Name == L"side_alpha_value" || Name == L"side_beta_value" || Name == L"side_pitch_value")
			{
				auto ctrl = dynamic_cast<Ctrl::UIEditBox*>(control);
				auto text = ctrl->GetCurText();
				if (text.back() == L'.')
					text += L"0";
				const auto val = wcstof(ctrl->GetCurText().c_str(), nullptr);
				ctrl->SetCurText(std::to_wstring(val));
			}
		}

		return false;
	}

	void SidePage::Show(bool show)
	{
		if (m_isani)
			return;

		if (show)
			m_page->SetPos(-265, 0, false);
		else
			m_page->SetPos(265, 0, false);

		m_page->SetVisible(true);

		//设置播放器的动画状态 在动画过程中不要重采样视图 不然会造成卡顿
		auto curve_player = (Ctrl::Waveform*)m_page->GetParent()->FindChildren(L"curve_player");
		curve_player->SetAniFlag(true);

		auto effect = MAnimation::Quintic_Out;
		auto anifun = [effect, this, show, curve_player](const MAnimation::MathsCalc* calc, float percent)
		{
			const int x = calc->calc(effect, show ? -265 : 0, show ? 0 : -265);
			m_page->SetPos(x, 0);
			m_page->GetParent()->UpdateLayout();
			if (percent == 100.f)
			{
				if(!show)
					m_page->SetVisible(false);
				m_isani = false;
				curve_player->SetAniFlag(false);
			}
			return m_isani;
		};
		m_isani = true;
		m_anicls->CreateTask(anifun, 300);
	}

	float SidePage::GetAlpha() const
	{
		return wcstof(m_alpha->GetCurText().c_str(), nullptr);
	}

	float SidePage::GetBeta() const
	{
		return wcstof(m_beta->GetCurText().c_str(), nullptr);
	}

	float SidePage::GetPitch() const
	{
		return wcstof(m_pitch->GetCurText().c_str(), nullptr);
	}
}
