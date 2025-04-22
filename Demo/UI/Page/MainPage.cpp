#include "MainPage.h"
#include "SidePage.h"
#include "SettingsPage.h"
#include "../MainWindow.h"
#include "../DefControl.hpp"

namespace UI
{
	std::wstring formatTime(float time)
	{
		const auto minutes = int(time) / 60;
		const auto seconds = int(time) % 60;
		std::wostringstream oss;
		oss << std::setfill(L'0') << std::setw(2) << minutes << ":" << std::setfill(L'0') << std::setw(2) << seconds;
		return oss.str();
	}

	MainPage::MainPage(Ctrl::UIControl* parent, XML::MuiXML* ui) : Page(parent, ui)
	{
		const std::wstring xml = LR"(
		<PropGroup id="menubtn" autoSize="false" style="menubtnDark" frame="5,0,40,20" textAlign="5" />
		<PropGroup id="combox" fontSize="12" style="comboxDark" textAlign="5" fontColor="#textColor"
		listStyle="listDark" itemStyle="comitemDark" itemHeight="25" autoSize="false" menuHeight="200"
		iFontColor="#textColor" iTextAlign="5" styleV="scrollDark" button="false" barWidth="6" inset="1,1,1,1"
		dropIcon="icon_drop_light" />

		<UIControl autoSize="false" size="100%,100%" name="mainPage" align="Absolute">
			<UIControl autoSize="false" size="100%,20" align="LinearH">
				<UIButton frame="5,0,60,20" text="导入文件" name="import_file" />
				<UIButton frame="5,0,80,20" text="导入文件夹" name="import_dir" />
				<UIButton frame="5,0,80,20" text="保存为文件" name="save_file" />
				<UIButton frame="5,0,60,20" text="保存所有" name="save_all" />
			</UIControl>
			<UIControl autoSize="false" frame="0,25,100%,1" bgColor="#menuline" />
			<UIControl autoSize="false" frame="0,25,100%,40" align="LinearHL">
				<IconButton style="menubtn" icon="icon_more_light" autoSize="false" frame="9,9,25,25"
				iconSize="20,20" offset="0,0,2,2" prop="ani" name="slidebar_show" />
			</UIControl>
			<UIControl autoSize="false" frame="162,70,100%,100%" align="Absolute">
				<UIControl autoSize="false" size="100%,101%" align="LinearHL" pos="0,0">
					<UIControl autoSize="false" frame="0,0,265,101%" bgColor="#windbgColor" frameWidth="1"
					frameColor="#menuframe" name="sidepage" align="LinearV" visible="true" />
					<UIControl autoSize="false" align="LinearVR" size="100%,100%">
						<UIControl autoSize="false" frame="8,5,8f,135f" align="Absolute">
		                    <UIControl autoSize="false" frame="0,0,100%,100%" align="LinearVBR">
		                        <UIImgBox name="curve_background" alpha="70" autoSize="false" frame="0,0,100%,100%" imgStyle="2"  />
		                    </UIControl>
		                    <CurveEditor frameColor="#menuframe" frameWidth="1" autoSize="false" frame="0,0,100%,100%" fontColor="#textColor"
		                    styleH="scrollWavDark" styleV="scrollDark" button="false" name="curve_editor" />
						</UIControl>
						<UIProgBar frame="8,1,8f,6" value="30" name="infer_prog" visible="false" />
						<Waveform frame="8,1,8f,43f" frameColor="#menuframe" frameWidth="1" frameRound="2.f" autoSize="false"
						name="test" preHeight="0" showLine="false" name="curve_player" />
						<UIControl autoSize="false" size="100%,100%" align="LinearHL">
							<UISlider frame="10,8,130,15" value="80" name="editor_vol" />
							<UIImgBox autoSize="false" frame="10,8,17,17" img="icon_audio_light" />
							<UIControl autoSize="false" size="100%,100%">
								<UILabel pos="8,8" text="00:00\00:00" name="editor_time" />
							</UIControl>
						</UIControl>
					</UIControl>
				</UIControl>
				<UINavBar autoSize="false" frame="40,15,200,25" name="navbar_editor" fontHoverColor="#textColor" barColor="0,0,0,0" fontSize="14"
				barAnitime="0" />
			</UIControl>

			<UIImgBox pos="6,40" size="20,20" img="icon_list_dark" autoSize="false" />
			<UILabel text="#voicelist" fontSize="14" pos="35,41" fontColor="196,158,166,255" />

			<UIListBox frame="6,70,150,6f" name="audio_list" />
			<UIControl frame="162,25,1,100%" bgColor="#menuframe" autoSize="false" />
			<UIControl frame="162,70,100%,1" bgColor="#menuframe" autoSize="false" />
			<UINavBar pos="180,38" fontHoverColor="#textColor" barColor="92,183,255,255" fontSize="14" name="editernav" />
		</UIControl>
		)";

		if(!ui->CreateUIFromXML(parent, xml))
		{
			__debugbreak();
		}

		m_editor = parent->Child<Ctrl::CurveEditor>(L"curve_editor");
		m_wave = parent->Child<Ctrl::Waveform>(L"curve_player");
		auto timelabel = parent->Child<Ctrl::UILabel>(L"editor_time");
		auto callback = [this, timelabel](float pre, float tm)
		{
			_m_size ps = 0;
			if (const auto data = m_editor->GetCurveData(); !data.Null())
				ps = size_t((float)data.Size(1) * pre);
			m_editor->SetPlayLinePos(ps);
			const std::wstring str = formatTime(tm) + L"\\" + formatTime(m_wave->GetDataDuration());
			timelabel->SetAttribute(L"text", str);
		};
		m_wave->SetAudioPlayer(m_player);
		m_wave->SetPlayCallback(callback);
		m_list = parent->Child<Ctrl::UIListBox>(L"audio_list");
		m_page = parent->Child(L"mainPage");
		m_sidepage = m_page->Child(L"sidepage");

		WndControls::InitCtrl(
			m_list,
			m_editor,
			m_wave
		);

		if (auto Directory = std::filesystem::path(DragonianLib::GetCurrentFolder() + L"/User/F0/"); !exists(Directory))
			create_directories(Directory);
		if (auto Directory = std::filesystem::path(DragonianLib::GetCurrentFolder() + L"/User/Audio/"); !exists(Directory))
			create_directories(Directory);
		if (auto Directory = std::filesystem::path(DragonianLib::GetCurrentFolder() + L"/User/Spec/"); !exists(Directory))
			create_directories(Directory);
		if (auto Directory = std::filesystem::path(DragonianLib::GetCurrentFolder() + L"/User/Mel/"); !exists(Directory))
			create_directories(Directory);
		//m_editor->SetShowPitch(false);
		WndControls::InsertAudio(LR"(C:\DataSpace\MediaProj\PlayList\Echoism-Vocal.wav)");
	}

	bool MainPage::EventProc(UINotifyEvent event, Ctrl::UIControl* control, _m_param param)
	{
		if(MUIEVENT(Event_Mouse_LClick, L"slidebar_show"))
			dynamic_cast<SidePage*>(FindPage(L"sidepage"))->Show(!m_sidepage->IsVisible());
		else if (MUIEVENT(Event_Slider_Change, L"editor_vol"))
			m_wave->SetVolume((_m_byte)param);
		else if (MUIEVENT(Event_ListBox_ItemChanged, L"audio_list"))
			WndControls::SetCurveEditorDataIdx(
				m_list->GetCurSelItem(),
				(unsigned)_wcstoi64(
					m_sidepage->FindChildren<Ctrl::UIEditBox>(L"side_sampling_rate")->GetCurText().c_str(),
					nullptr, 10
				)
			);
		else if (MUIEVENT(Event_Mouse_LUp, L"side_use_log_view"))
			m_editor->SetShowPitch(dynamic_cast<Ctrl::UICheckBox*>(control)->GetSel());
		else if (event == Event_Key_Down)
		{
			if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState(0x5a) & 0x8000)
				WndControls::MoeVSUndo();
			else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState(0x59) & 0x8000)
				WndControls::MoeVSRedo();
			else if (GetKeyState(VK_SPACE) & 0x8000)
				 m_wave->PlayPause();
			return false;
		}
		else
			return false;
		return true;
	}
}