#include "DefUIStyle.h"
#include "MainWindow.h"
#include "DefControl.hpp"
#include "Libraries/MJson/MJson.h"
#include "Libraries/Util/StringPreprocess.h"

namespace SimpleF0Labeler
{
	std::wstring WindowTitle = L"SimpleLabeler";

	void CreateDefaultStyle(Mui::XML::MuiXML* xmlUI)
	{
		xmlUI->LoadDefaultStyle(true);
		xmlUI->Mgr()->LoadStyleList();
		xmlUI->AddStringList(L"MainWindowTitle", WindowTitle);
		auto mgr = xmlUI->Mgr();

		const auto LocalizationDocument = DragonianLib::MJson::MJsonDocument(
			DragonianLib::WideStringToUTF8(
				DragonianLib::GetCurrentFolder() +
				L"/localization.json"
			).c_str()
		);
		const auto LocalizationDict = LocalizationDocument.GetMemberArray();
		for (auto& [Key, Value] : LocalizationDict)
		{
			if (Key.empty() || !Value.IsString() || Value.Empty())
				continue;
			xmlUI->AddStringList(
				DragonianLib::UTF8ToWideString(Key),
				DragonianLib::UTF8ToWideString(Value.GetString())
			);
		}

		//全局控件的默认样式
		std::wstring xml = LR"(
		<DefPropGroup Control="UILabel" FontColor="#TextColor" />
		<DefPropGroup Control="UIButton" Style="buttonDark" FontColor="#TextColor" AutoSize="false" TextAlign="5" Prop="ani" />
		<DefPropGroup Control="UIListBox" Style="listDark" ItemStyle="itemDark" AutoSize="false" IFontColor="#TextColor"
		ITextAlign="4" StyleV="scroll" Button="false" BarWidth="6" Inset="2,2,2,2" LineSpace="2" />
		<DefPropGroup Control="UICheckBox" Style="checkboxDark" Prop="ani" />
		<DefPropGroup Control="UISlider" TrackInset="0,5,0,5" AutoSize="false" Style="strack" BtnStyle="sbutton" />
		<DefPropGroup Control="UIEditBox" AutoSize="false" Style="editDark" Inset="5,5,5,5" CaretColor="#TextColor" fontStyle="fontstyle" />
		<DefPropGroup Control="UIProgBar" AutoSize="false" Style="progressDark" />
		)";

		xmlUI->AddDefPropGroup(xml, true);

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