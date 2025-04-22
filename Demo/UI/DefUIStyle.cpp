#include "Label.h"
#include "DefUIStyle.h"
#include "MainWindow.h"
#include "DefControl.hpp"
#include "Libraries/MJson/MJson.h"
#include "Libraries/Util/StringPreprocess.h"

namespace UI
{
	void CreateDefaultStyle(Mui::XML::MuiXML* xmlUI)
	{
		auto mgr = xmlUI->Mgr();
		//加载样式列表
		xmlUI->Mgr()->LoadStyleList();

		//添加文本列表
		std::vector<std::pair<std::wstring_view, std::wstring>> strList =
		{
			{ L"app_title", m_wndTitle + L" - " + App::m_version },
			{ L"side_pm1_v", L"0.3000" }, { L"side_pm2_v", L"0.8000" }, { L"side_pm3_v", L"0.0000" }, { L"side_pm4_v", L"0.0000" },
			{ L"side_pm5_v", L"52608" },{ L"side_pm6_v", L"0.0" }, { L"side_pm7_v", L"2" },{ L"side_pm8_v", L"1" },{ L"side_pm9_v", L"100" }, { L"side_pm10_v", L"44100" }, { L"side_pm11_v", L"320" }, { L"side_pm12_v", L"2" },
			{ L"settpage_title1_t2", L"30.00" }, { L"settpage_title1_t4", L"3.0" }, { L"settpage_title1_t6", L"2048" },
			{ L"settpage_title1_t8", L"512" },
			{ L"settpage_title4_t1",  m_wndTitle + L" v" + App::m_version },
			{ L"settpage_title4_t2",  m_wndTitle + L"Core v" + App::m_versionCore }, { L"settpage_title4_t3", L"Developer: 纳鲁塞-缪-希娜卡纳 (MoeVSCore)"},
			{ L"settpage_title4_t4", std::wstring(L"MiaoUI ") + Mui::MInfo::MuiEngineVer }, { L"settpage_title4_t5", L"Developer: Maplespe (MiaoUI)"},
			{ L"settpage_title4_l", L"本软件是免费开源软件 使用即代表你同意《用户协议》和《免责声明》" }
		};
		for (auto& str : strList)
			xmlUI->AddStringList(str.first, str.second);

		const auto LanguageJson = DragonianLib::MJson::MJsonDocument(
			DragonianLib::WideStringToUTF8(DragonianLib::GetCurrentFolder() + L"/lang.json").c_str()
		);
		const auto LanguageDict = LanguageJson.GetMemberArray();
		for (auto& lang_member : LanguageDict)
		{
			if (lang_member.first.empty() || !lang_member.second.IsString() || lang_member.second.Empty())
				continue;
			xmlUI->AddStringList(DragonianLib::UTF8ToWideString(lang_member.first), DragonianLib::UTF8ToWideString(lang_member.second.GetString()));
		}

		//添加默认字体样式
		Mui::Ctrl::UILabel::Attribute fontstyle;
		fontstyle.fontColor = Mui::Helper::M_GetAttribValueColor(xmlUI->GetStringValue(L"textColor"));
		xmlUI->AddFontStyle(L"fontstyle", fontstyle);

		//全局控件的默认样式
		std::wstring xml = LR"(
		<DefPropGroup control="UILabel" fontColor="#textColor" />
		<DefPropGroup control="UIButton" style="buttonDark" fontColor="#textColor" autoSize="false" textAlign="5" prop="ani" />
		<DefPropGroup control="UIListBox" style="listDark" itemStyle="itemDark" autoSize="false" iFontColor="#textColor"
		iTextAlign="4" styleV="scroll" button="false" barWidth="6" inset="2,2,2,2" lineSpace="2" />
		<DefPropGroup control="UICheckBox" style="checkboxDark" prop="ani" />
		<DefPropGroup control="UISlider" trackInset="0,5,0,5" autoSize="false" style="strack" btnStyle="sbutton" />
		<DefPropGroup control="UIEditBox" autoSize="false" style="editDark" inset="5,5,5,5" caretColor="#textColor" fontStyle="fontstyle" />
		<DefPropGroup control="UIProgBar" autoSize="false" style="progressDark" />
		)";

		xmlUI->AddDefPropGroup(xml);

		//菜单栏按钮样式
		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" color="120,120,120,255" value="3.f" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" color="100,100,100,255" value="3.f" />
		</part>
		<part />
		)";
		mgr->AddGeometryStyle(L"menubtn", xml);

		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" color="120,120,120,150" value="3.f" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" color="150,150,150,150" value="3.f" />
		</part>
		<part />
		)";
		mgr->AddGeometryStyle(L"menubtnDark", xml);

		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" color="220,220,220,150" value="6.f" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" color="200,200,200,150" value="6.f" />
		</part>
		<part />
		)";
		mgr->AddGeometryStyle(L"menuitem", xml);

		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" color="100,100,100,150" value="6.f" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" color="120,120,120,150" value="6.f" />
		</part>
		<part />
		)";
		mgr->AddGeometryStyle(L"menuitemDark", xml);

		//滚动条
		xml = LR"(
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="135,135,135,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="150,150,150,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="140,140,140,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="160,160,160,255" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="255,255,255,100" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="200,200,200,100" />
		</part>
		<part />
		<part />
		)";
		mgr->AddGeometryStyle(L"scroll", xml);

		xml = LR"(
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="135,135,135,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="150,150,150,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="140,140,140,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="160,160,160,255" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="150,150,150,100" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="100,100,100,100" />
		</part>
		<part />
		<part />
		)";
		mgr->AddGeometryStyle(L"scrollDark", xml);

		xml = LR"(
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="135,135,135,100" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="150,150,150,100" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="140,140,140,100" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="3.0" color="160,160,160,100" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="150,150,150,20" />
		</part>
		<part>
			<fill_rect rc="0,0,0,0" color="100,100,100,20" />
		</part>
		<part />
		<part />
		)";
		mgr->AddGeometryStyle(L"scrollWavDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="243,243,243,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="243,243,243,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="243,243,243,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="243,243,243,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"list", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"listDark", xml);

		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="230,230,230,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="235,235,235,255" />
		</part>
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="235,235,235,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="240,240,240,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="235,235,235,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="200,200,200,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="180,180,180,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"item", xml);

		xml = LR"(
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="115,115,115,255" />
		</part>
		<part />
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="115,115,115,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="130,130,130,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="115,115,115,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="92,183,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="100,100,100,255" />
			<fill_round rc="0,5,l-5,5" value="2.0" color="80,80,80,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"itemDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="60,60,60,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"comboxDark", xml);

		xml = LR"(
		<part>
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="84,164,227,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
		</part>
		<part>
			<draw_round rc="0,0,0,0" value="6.0" color="84,164,227,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="0,116,181,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
		</part>
		)";
		mgr->AddGeometryStyle(L"comitem", xml);

		xml = LR"(
		<part>
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="100,100,100,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="150,150,150,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="84,164,227,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="120,120,120,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
		</part>
		<part>
			<draw_round rc="0,0,0,0" value="6.0" color="84,164,227,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="150,150,150,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="0,116,181,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="150,150,150,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="120,120,120,160" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
		</part>
		)";
		mgr->AddGeometryStyle(L"comitemDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="235,235,235,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="225,225,225,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="245,245,245,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"buttonLight", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="50,50,50,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="80,80,80,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="60,60,60,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="50,50,50,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="80,80,80,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="50,50,50,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="80,80,80,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="50,50,50,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="60,60,60,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"buttonDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="235,235,235,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="225,225,225,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="245,245,245,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="245,245,245,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="225,225,225,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
			<fill_ellipse rc="5,5,5,5" color="190,190,190,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="245,245,245,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"checkbox", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
			<fill_ellipse rc="5,5,5,5" color="190,190,190,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="80,80,80,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="100,100,100,255" />
			<fill_ellipse rc="5,5,5,5" color="175,175,175,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"checkboxDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="120,195,255,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="170,170,170,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="200,200,200,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="130,130,130,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"strack", xml);

		xml = LR"(
		<part>
			<fill_ellipse rc="0,0,0,0" color="250,250,250,255" />
			<draw_ellipse rc="0,0,0,0" color="220,220,220,255" width="1" />
			<fill_ellipse rc="6,6,6,6" color="92,183,255,255" />
		</part>
		<part>
			<fill_ellipse rc="0,0,0,0" color="235,235,235,255" />
			<draw_ellipse rc="0,0,0,0" color="200,200,200,255" width="1" />
			<fill_ellipse rc="6,6,6,6" color="120,195,255,255" />
		</part>
		<part>
			<fill_ellipse rc="0,0,0,0" color="225,225,225,255" />
			<draw_ellipse rc="0,0,0,0" color="200,200,200,255" width="1" />
			<fill_ellipse rc="6,6,6,6" color="120,195,255,255" />
		</part>
		<part>
			<fill_ellipse rc="0,0,0,0" color="220,220,220,255" />
			<draw_ellipse rc="0,0,0,0" color="190,190,190,255" width="1" />
			<fill_ellipse rc="6,6,6,6" color="130,130,130,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"sbutton", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="130,130,130,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="245,245,245,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="154,185,233,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="255,255,255,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="230,230,230,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="154,185,233,255" />
			<fill_rect rc="4,b2,4,1" color="154,185,233,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="220,220,220,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="200,200,200,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"edit", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="50,50,50,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="100,100,100,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="120,120,120,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="45,45,45,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="120,120,120,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="84,110,143,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="55,55,55,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="120,120,120,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="84,110,143,255" />
			<fill_rect rc="4,b2,4,1" color="84,110,143,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="6.0" color="40,40,40,255" />
			<draw_round rc="0,0,0,0" value="6.0" color="190,190,190,255" width="1" />
			<fill_rect rc="5,b1,5,0" color="150,150,150,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"editDark", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="250,250,250,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="230,230,230,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="184,217,251,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="170,170,170,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="200,200,200,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="130,130,130,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"progress", xml);

		xml = LR"(
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="130,130,130,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="150,150,150,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="184,217,251,255" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="120,120,120,255" />
			<draw_round rc="0,0,0,0" value="5.0" color="100,100,100,255" width="1" />
		</part>
		<part>
			<fill_round rc="0,0,0,0" value="5.0" color="130,130,130,255" />
		</part>
		)";
		mgr->AddGeometryStyle(L"progressDark", xml);
		WndControls::SetLanguageXML(xmlUI);
	}
}