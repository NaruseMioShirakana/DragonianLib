#include "Label.h"
#include "MainWindow.h"
#include "DefUIStyle.h"
#include "DefControl.hpp"
#include "User/Mui_GlobalStorage.h"
#include "Render/Sound/Mui_DirectSound.h"
#include "Extend/CurveEditor.h"
#include "Extend/Waveform.h"
#include "Page/MainPage.h"
#include "Page/SidePage.h"

namespace SimpleF0Labeler
{
	const std::wstring WindowTitle = L"SimpleLabeler";
	constexpr Mui::UISize WindowSize = { 1280, 768 };

	bool CreateMainWindow()
	{
		auto Manager = Mui::Window::MWindowManager::InitFromCurrentThread();

		Mui::Window::MWindowCtx* WindowContext = Manager->CreateWindowCtxEx(
			{
				Mui::UIRect(0, 0, WindowSize.width, WindowSize.height),
				(int)Mui::Window::MWindowType::NoTitleBar,
				WindowTitle,
				true,
				true,
				0,
				0
			},
			Mui::Window::MWindowManager::ThreadMode::UniqueAll
		);
		if (!WindowContext)
			return false;

		/*WindowContext->Base()->GetResourceMgr()->AddResourcePath(
			DragonianLib::GetCurrentFolder() + L"\\MVSResource.dmres",
			L"12345678"
		);*/

#ifdef _WIN32
		{
			const auto hWnd = (HWND)WindowContext->Base()->GetWindowHandle();
			const LONG_PTR currentStyle = GetWindowLongPtrW(
				hWnd, 
				GWL_STYLE
			);
			SetWindowLongPtrW(
				hWnd,
				GWL_STYLE,
				currentStyle | WS_SIZEBOX
			);
			SetWindowPos(
				hWnd,
				nullptr,
				0, 0, 0, 0,
				SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
			);
		}

#else
#error __TODO__
#endif

		auto MyPlayerContext = std::make_shared<Mui::Render::MDS_AudioPlayer>(WindowContext->Base());
		
		//初始化窗口
		if (!MyPlayerContext->InitAudioPlayer() || !Mui::MObjStorage::AddObj(MyPlayerContext) || !WindowContext->InitWindow(InitWindow, false))
		{
			MessageBoxW(nullptr, L"初始化窗口失败!", L"error", MB_ICONERROR);
			return false;
		}
		
		const auto WindowBase = WindowContext->Base();
		WindowBase->SetMinimSize(WindowSize);
		WindowBase->SetCacheMode(true);
		WindowBase->CenterWindow();
		WindowBase->ShowDebugRect(true);
		WindowBase->ShowWindow(true);

		Manager->EventLoop();

		return true;
	}

	// ReSharper disable once CppParameterMayBeConstPtrOrRef
	bool InitWindow(
		Mui::Window::MWindowCtx* Context,
		Mui::Ctrl::UIControl* /*Root*/,
		Mui::XML::MuiXML* UI
	)
	{
		CreateDefaultStyle(UI);

		auto RootPage = Context->GetRootPage();
		if (!RootPage || !Mui::MObjStorage::AddObj(RootPage, L"ROOTPAGE")) return false;

		Waveform::Register();
		CurveEditor::Register();

		const auto MyMainWindow = Mui::MakeUIPage<MainWindow>(RootPage);
		if (!MyMainWindow || !Mui::MObjStorage::AddObj(MyMainWindow)) return false;

		auto MyMainPage = Mui::MakeUIPage<MainPage>(MyMainWindow);
		if (!MyMainPage || !Mui::MObjStorage::AddObj(MyMainPage)) return false;

		auto MySidePage = Mui::MakeUIPage<SidePage>(MyMainPage);
		if (!MySidePage || !Mui::MObjStorage::AddObj(MySidePage)) return false;
		

		return true;
	}

	MainWindow::~MainWindow()
	{
		Mui::MObjStorage::DelAllObj();
	}

	bool MainWindow::UIEventProc(
		Mui::XML::PropName event,
		Mui::Ctrl::UIControl* control,
		const std::any& param
	)
	{
		if (EventProc(event, control, param))
			return true;
		if (auto Obj = Mui::MObjStorage::GetObjInstance<MainPage*>(); Obj && Obj->EventProc(event, control, param))
			return true;
		if (auto Obj = Mui::MObjStorage::GetObjInstance<SidePage*>(); Obj && Obj->EventProc(event, control, param))
			return true;
		return false;
	}

	bool MainWindow::EventProc(
		Mui::XML::PropName event,
		Mui::Ctrl::UIControl* control,
		std::any param
	)
	{
#ifdef _WIN32
		const HWND hWnd = (HWND)_MyRoot->GetParentWin()->GetWindowHandle();
		switch (event)
		{
		case Mui::Ctrl::Events::Mouse_LButton_Up:
		{
			if (control->GetName() == L"MainClose")
			{
				const HWND& hwnd = hWnd;
				const int ret = MessageBoxW(
					hwnd,
					WndControls::Localization(L"MainClose::Desc").c_str(),
					WndControls::Localization(L"MainClose::Title").c_str(),
					MB_YESNO | MB_ICONASTERISK
				);
				if (ret == IDNO)
					return true;
				::PostMessageW(hWnd, WM_CLOSE, 0, 0);
				return true;
			}
			if (control->GetName() == L"MainMinSize")
			{
				::ShowWindow(hWnd, SW_MINIMIZE);
				return true;
			}
			if (control->GetName() == L"MainMaxSize")
			{
				if (IsZoomed(hWnd))
					::ShowWindow(hWnd, SW_RESTORE);
				else
					::ShowWindow(hWnd, SW_MAXIMIZE);
				return true;
			}
			break;
		}
		case Mui::Ctrl::Events::Mouse_LButton_DoubleClicked:
		{
			if (control->GetName() == L"MainTitleBar" || control->GetName() == L"MainLabel")
			{
				if (IsZoomed(hWnd))
					::ShowWindow(hWnd, SW_RESTORE);
				else
					::ShowWindow(hWnd, SW_MAXIMIZE);
			}
			break;
		}
		case Mui::Ctrl::Events::Mouse_LButton_Down:
		{
			if (control->GetName() == L"MainTitleBar" || control->GetName() == L"MainLabel")
			{
				::SendMessageW(hWnd, WM_SYSCOMMAND, SC_MOVE | HTCAPTION, 0);
				return true;
			}
			break;
		}
		default:
			return false;
		}
#else
#error __TODO__
#endif
		return false;
	}

	Mui::Ctrl::UIControl* MainWindow::OnLoadPageContent(
		Mui::Ctrl::UIControl* parent,
		Mui::XML::MuiXML* ui
	)
	{
		const std::wstring xml = LR"(
			<PropGroup ID="ani" Animate="true" AnimateAlphaType="false" />
			<UIControl BgColor="#WindowBackGroundColor" Size="100%,100%" AutoSize = "false" />
			<UIControl Name="MyMainPage" AutoSize="false" Frame="0,35,100%,100%" Align="Absolute" />
			<UIControl AutoSize="false" Size="100%,35" Name="MainTitleBar" Align="Absolute">
				<UIControl AutoSize="false" Pos="0,0" Size="100%,30" Name="MainWindowTitle" Align="Center">
					<UILabel FontSize="13" Pos="0,0" Text="#MainWindowTitle" Name="MainLabel" />
				</UIControl>
				<UIControl AutoSize="false" Size="100%,30" Name="MainTitleBar" Align="LinearHL">
					<!--ButtonStyle="StyleExitButton"-->
					<UIButton Name="MainClose" Size="33,30" />
					<!--ButtonStyle="StyleMaximumButton"-->
					<UIButton Name="MainMaxSize" Size="33,30" />
					<!--ButtonStyle="StyleMinimumButton"-->
					<UIButton Name="MainMinSize" Size="33,30" />
				</UIControl>
				<UIControl AutoSize="true" Size="50%,30" Align="LinearH">
					<UIButton Frame="0,0,100,30" Text="#ImportAudio" Name="ImportAudio" />
					<UIButton Frame="0,0,100,30" Text="#ImportF0" Name="ImportF0" />
					<UIButton Frame="0,0,100,30" Text="#SaveAll" Name="SaveAll" />
				</UIControl>
			</UIControl>
		)";

		if (!ui->CreateUIFromXML(parent, xml))
			__debugbreak();
		auto ImportAudio = parent->FindChildren<Mui::Ctrl::UIButton>(L"ImportAudio");
		auto ImportF0 = parent->FindChildren<Mui::Ctrl::UIButton>(L"ImportF0");
		auto SaveAll = parent->FindChildren<Mui::Ctrl::UIButton>(L"SaveAll");
		ImportAudio->SetSize(ImportAudio->GetTextMetric().width, ImportAudio->GetSize().height, true);
		ImportF0->SetSize(ImportF0->GetTextMetric().width, ImportF0->GetSize().height, true);
		SaveAll->SetSize(SaveAll->GetTextMetric().width, SaveAll->GetSize().height, true);
		_MyRoot = parent->Child(L"MyMainPage");
		return _MyRoot;
	}

}
