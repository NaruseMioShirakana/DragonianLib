#include <regex>
#include "SettingsPage.h"
#include "../DefControl.hpp"
namespace UI
{
	SettingPage::SettingPage(Ctrl::UIControl* parent, XML::MuiXML* ui) : Page(parent, ui)
	{
		const std::wstring xml = LR"(
		<PropGroup id="sett_module" bgColor="#settpage_modbg" autoSize="false" frameRound="6.f" frame="32,8,32f,46" />

		<UIControl bgColor="#dlgBgLayer" size="100%,100%" autoSize="false" name="setting_page" visible="false">
			<UIControl autoSize="false" frame="0,25,100%,100%" align="LinearV" name="settpage_content" bgColor="#windbgColor">
				<UIControl pos="32,32" autoSize="true">
				<UIImgBox autoSize="false" img="icon_setts_light" size="32,32" />
				<UILabel pos="15,5" fontSize="18" text="#settpage_title" />
				</UIControl>
				<UILabel pos="32,32" fontSize="14" text="#settpage_title1" />
				<UIControl prop="sett_module" align="LinearH">
					<UILabel pos="20,15" text="#settpage_title1_t1" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_title1_t2" name="settpage_t1_edit1" />
					<UILabel pos="20,15" text="#settpage_title1_t3" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_title1_t4" name="settpage_t1_edit2" />
					<UILabel pos="20,15" text="#settpage_title1_t5" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_title1_t6" name="settpage_t1_edit3" />
					<UILabel pos="20,15" text="#settpage_title1_t7" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_title1_t8" name="settpage_t1_edit4" />
				</UIControl>
				<UILabel pos="32,15" fontSize="14" text="#settpage_title2" />
				<UIControl prop="sett_module" align="LinearH">
					<UILabel pos="20,15" text="#settpage_title2_t1" />	
					<UIComBox prop="sidecombox" frame="10,10,75,25" name="settpage_t2_com1" menuHeight="100" />
					<UILabel pos="20,15" text="#settpage_title2_t2" />	
					<UIComBox prop="sidecombox" frame="10,10,300,25" name="settpage_t2_com2" menuHeight="100" />
					<UILabel pos="20,15" text="#settpage_title2_t3" />	
					<UIComBox prop="sidecombox" frame="10,10,65,25" name="settpage_t2_com3" />
				</UIControl>
				<UILabel pos="32,15" fontSize="14" text="#settpage_title3" />
				<UIControl prop="sett_module" align="LinearH">
					<UILabel pos="20,15" text="#settpage_title3_t1" />	
					<UIComBox prop="sidecombox" frame="10,10,115,25" name="settpage_t3_com1" menuHeight="100" />
					<UILabel pos="20,15" text="#settpage_title3_t2" />	
					<UIComBox prop="sidecombox" frame="10,10,115,25" name="settpage_t3_com2" menuHeight="100" />
					<UILabel pos="20,15" text="#settpage_title3_t5" />	
					<UIComBox prop="sidecombox" frame="10,10,115,25" name="settpage_t3_com3" menuHeight="100" />
					<UILabel pos="20,15" text="#settpage_title3_t3" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_t3_ed1" name="settpage_t3_ed1" />
					<UILabel pos="20,15" text="#settpage_title3_t4" />	
					<UIEditBox frame="10,10,134,25" text="#settpage_t3_ed2" name="settpage_t3_ed2" />
				</UIControl>
				<UIControl bgColor="#menuframe" autoSize="false" frame="32,32,32f,1" />
				<UIControl pos="32,28" autoSize="true">
				<UIImgBox autoSize="false" img="icon_about_light" size="32,32" />
				<UILabel pos="15,5" fontSize="18" text="#settpage_title4" />
				</UIControl>
				<UIControl align="LinearH" autoSize="true">
					<UIControl align="LinearV" pos="32,0" autoSize="true">
						<UILabel fontSize="14" pos="0,15" text="#settpage_title4_t1" hyperlink="true"
						url="https://github.com/NaruseMioShirakana/MoeVoiceStudio/" fontUnderline="true" />
						<UILabel fontSize="14" pos="0,5" text="#settpage_title4_t2" hyperlink="true" 
						url="https://github.com/NaruseMioShirakana/MoeVoiceStudio/" fontUnderline="true"  />
						<UILabel fontSize="14" pos="0,15" text="#settpage_title4_t3" hyperlink="true" 
						url="https://space.bilibili.com/108592413/" fontUnderline="true"  />
						<UILabel fontSize="14" pos="0,20" text="#settpage_title4_l" />
					</UIControl>
					<UIControl align="LinearV" pos="32,0" autoSize="true">
						<UILabel fontSize="14" pos="0,15" text="#settpage_title4_t4" />
						<UILabel fontSize="14" pos="0,5" text=" " />
						<UILabel fontSize="14" pos="0,15" text="#settpage_title4_t5" hyperlink="true" 
						url="https://space.bilibili.com/87195798" fontUnderline="true"  />
					</UIControl>
					<UIImgBox autoSize="false" frame="10,0,360,164" img="logo_miaoui" />
				</UIControl>
				<UIControl align="LinearVB" frame="0,0,100%,100%">	
					<UIButton frame="32,20,80,30" text="#settpage_ret" name="settpage_ret" />
				</UIControl>
			</UIControl>
		</UIControl>
		)";
		if (!ui->CreateUIFromXML(parent, xml))
		{
			__debugbreak();
		}
	}

	bool SettingPage::EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param)
	{
		if (MUIEVENT(Event_Mouse_LClick, L"settpage_ret"))
		{
			return true;
		}
		return false;
	}

	void SettingPage::Show(bool show)
	{
		if (m_isani)
			return;

		auto content = m_page->Child(L"settpage_content");

		if (show)
		{
			content->SetPos(0, -150, false);
			m_page->SetAlpha(0, false);
		}
		else
		{
			content->SetPos(0, 25, false);
			m_page->SetAlpha(255, false);
		}
		m_page->SetVisible(true);

		auto effect = MAnimation::Exponential_Out;
		auto anifun = [effect, this, show, content](const MAnimation::MathsCalc* calc, float percent)
		{
			const int y = calc->calc(effect, show ? -150 : 25, show ? 25 : -150);
			const auto alpha = (_m_byte)Helper::M_Clamp(0,255, calc->calc(MAnimation::Default, show ? 0 : 255, show ? 255 : 0));
			content->SetPos(0, y, false);
			m_page->SetAlpha(alpha);
			m_page->GetParent()->UpdateLayout();
			if (percent == 100.f)
			{
				if (!show)
					m_page->SetVisible(false);
				m_isani = false;
			}
			return m_isani;
		};
		m_isani = true;
		m_anicls->CreateTask(anifun, 300);
	}
}
