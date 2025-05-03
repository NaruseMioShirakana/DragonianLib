#include "MainPage.h"

#include <User/Mui_GlobalStorage.h>

#include "SidePage.h"
#include "../MainWindow.h"
#include "../DefControl.hpp"
#include "Libraries/F0Extractor/F0ExtractorManager.hpp"

namespace SimpleF0Labeler
{
	static std::wstring FormatTime(float time)
	{
		const auto minutes = int(time) / 60;
		const auto seconds = int(time) % 60;
		std::wostringstream oss;
		oss << std::setfill(L'0') << std::setw(2) << minutes << ":" << std::setfill(L'0') << std::setw(2) << seconds;
		return oss.str();
	}

	bool MainPage::EventProc(
		Mui::XML::PropName event,
		Mui::Ctrl::UIControl* control,
		std::any param
	)
	{
		if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Clicked.Event(), L"ShowSideBar"))
			Mui::MObjStorage::GetObjInstance<SidePage*>()->Show();
		else if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Clicked.Event(), L"ImportAudio"))
			WndControls::LoadFiles((HWND)Parent()->UI()->GetOwnerWnd()->GetWindowHandle());
		else if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Clicked.Event(), L"ImportF0"))
			WndControls::LoadF0((HWND)Parent()->UI()->GetOwnerWnd()->GetWindowHandle());
		else if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Clicked.Event(), L"SaveAll"))
			WndControls::SaveAll();
		else if (MUIEVENT(Mui::Ctrl::Events::Slider_Changed.Event(), L"EditorVolume"))
			m_wave->SetVolume(static_cast<Mui::_m_byte>(std::any_cast<int>(param)));
		else if (MUIEVENT(Mui::Ctrl::Events::ListBox_ItemChanged.Event(), L"AudioList"))
			WndControls::SetCurveEditorDataIdx(
				m_list->SelectedItemIndex,
				static_cast<unsigned>(Mui::MObjStorage::GetObjInstance<SidePage*>()->GetSamplingRate()),
				Mui::MObjStorage::GetObjInstance<SidePage*>()->IsUsingLogSpec()
			);
		else if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Up.Event(), L"SidePageUseLogView"))
			m_editor->SetShowPitch(Mui::MObjStorage::GetObjInstance<SidePage*>()->IsUsingLogView());
		else if (MUIEVENT(Mui::Ctrl::Events::Mouse_LButton_Up.Event(), L"SidePageUseLogSpec"))
			WndControls::ReCalcSpec(Mui::MObjStorage::GetObjInstance<SidePage*>()->IsUsingLogSpec());
		else if (event == Mui::Ctrl::Events::Key_Down.Event())
		{
			if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('Z') & 0x8000)
				WndControls::MoeVSUndo();
			else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('Y') & 0x8000)
				WndControls::MoeVSRedo();
			else if (GetKeyState(VK_SPACE) & 0x8000)
				WndControls::PlayPause();
			else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('S') & 0x8000)
				WndControls::SaveData();
			else
				return false;
		}
		else
			return false;
		return true;
	}

	Mui::Ctrl::UIControl* MainPage::OnLoadPageContent(Mui::Ctrl::UIControl* parent, Mui::XML::MuiXML* ui)
	{
		const std::wstring MainPageXml = LR"(
		<UIControl AutoSize="false" Size="100%,100%" Name="mainPage" Align="Absolute">
			<UIControl AutoSize="false" Frame="0,0,100%,1" BgColor="#Menuline" />
			<UIControl AutoSize="false" Frame="0,0,100%,31" Align="LinearHL">
				<UIIconBtn AutoSize="false" Frame="3,3,23,23" IconSize="20,20" IconOffset="0,0,2,2" Prop="ani" Name="ShowSideBar" />
			</UIControl>
			<UIControl AutoSize="false" Frame="162,30,100%,100%" Align="Absolute">
				<UIControl AutoSize="false" Size="100%,101%" Align="LinearHL" Pos="0,0">
					<UIControl AutoSize="false" Frame="0,0,265,101%" BgColor="#WindowBackGroundColor" FrameWidth="1" FrameColor="#MenuFrame" Name="SidePage" Align="LinearV" Visible="true" />
					<UIControl AutoSize="false" Align="LinearVR" Size="100%,100%">
						<UIControl AutoSize="false" Frame="8,5,8f,135f" Align="Absolute">
		                    <CurveEditor FrameColor="#MenuFrame" FrameWidth="1" AutoSize="false" Frame="0,0,100%,100%" FontColor="#TextColor" Button="false" Name="F0Editor"/>
						</UIControl>
						<Waveform Frame="8,1,8f,43f" FrameColor="#MenuFrame" FrameWidth="1" FrameRound="2.f" AutoSize="false" PreHeight="0" ShowLine="false" Name="EditorPlayer" />
						<UIControl AutoSize="false" Size="100%,100%" Align="LinearHL">
							<UISlider Frame="10,8,130,15" Value="80" Name="EditorVolume" />
							<UIImgBox AutoSize="false" Frame="10,8,17,17" />
							<UIControl AutoSize="false" Size="100%,100%">
								<UILabel Pos="8,8" Text="00:00\00:00" Name="EditorTime" />
							</UIControl>
						</UIControl>
					</UIControl>
				</UIControl>
			</UIControl>

			<UIImgBox Pos="6,5" Size="20,20" AutoSize="false" />
			<UILabel Text="#AudioList" FontSize="14" Pos="35,6" FontColor="196,158,166,255" />

			<UIListBox Frame="6,30,150,6f" Name="AudioList" />
			<UIControl Frame="162,0,1,100%" BgColor="#MenuFrame" AutoSize="false" />
			<UIControl Frame="162,30,100%,1" BgColor="#MenuFrame" AutoSize="false" />
		
		</UIControl>
		)";

		if (!ui->CreateUIFromXML(parent, MainPageXml))
			__debugbreak();

		m_editor = parent->Child<CurveEditor>(L"F0Editor");
		m_wave = parent->Child<Waveform>(L"EditorPlayer");
		auto timelabel = parent->Child<Mui::Ctrl::UILabel>(L"EditorTime");
		auto callback = [this, timelabel](float pre, float tm)
			{
				Mui::_m_size ps = 0;
				if (const auto& data = m_editor->GetCurveData(); !data.Null())
					ps = size_t(round((float)data.Size(1) * pre));
				m_editor->SetPlayLinePos(ps);
				const std::wstring str = FormatTime(tm) + L"\\" + FormatTime(m_wave->GetDataDuration());
				timelabel->SetAttribute(L"text", str);
			};
		m_wave->SetAudioPlayer(Mui::MObjStorage::GetObj<Mui::Render::MDS_AudioPlayer>().get());
		m_wave->SetPlayCallback(callback);
		m_list = parent->Child<Mui::Ctrl::UIListBox>(L"AudioList");

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
		return parent->Child(L"mainPage");
	}

}
