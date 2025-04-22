#include "Label.h"
#include "MainWindow.h"

#include "DefUIStyle.h"
#include "DefControl.hpp"

#include "Page/MainPage.h"
#include "Page/SidePage.h"
#include "Page/SettingsPage.h"

#include "Render/Sound/Mui_DirectSound.h"


namespace UI
{
	using namespace Mui;

	const UISize m_wndSize = { 1280, 768 };
	const std::wstring m_wndTitle = L"Label";
	Render::MAudioPlayer* m_player = nullptr;

	MainWindow* m_window = nullptr;
	std::vector<Page*> m_pageList;
	std::vector<Menu*> m_menuList;
	Menu* m_curShowMenu = nullptr;

	bool CreateMainWindow(MiaoUI& engine, std::vector<std::wstring> cmdList)
	{
		if (m_window)
			return true;

		const auto ctx = engine.CreateWindowCtx({ 0, 0, m_wndSize.width, m_wndSize.height }, NoTitleBar, m_wndTitle + L" - " + App::m_version, true, true);
		if (!ctx)
			return false;

		Ctrl::IconButton::Register();
		Ctrl::Waveform::Register();
		Ctrl::CurveEditor::Register();

#ifdef _WIN32
		//为窗口增加可调整大小样式
		const auto hWnd = (HWND)ctx->Base()->GetWindowHandle();
		const LONG_PTR currentStyle = GetWindowLongPtrW(hWnd, GWL_STYLE);
		SetWindowLongPtrW(hWnd, GWL_STYLE, currentStyle | WS_SIZEBOX);
		SetWindowPos(hWnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
#endif

		m_window = new MainWindow(ctx);
		m_window->m_cmdList = std::move(cmdList);

#ifdef _WIN32
		m_player = new Render::MDS_AudioPlayer(ctx->Base());
#else
#error __TODO__
#endif

		auto callback = [ObjectPtr = m_window](auto&& PH1, auto&& PH2, auto&& PH3)
		{
			return ObjectPtr->InitWindow(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
			                      std::forward<decltype(PH3)>(PH3));
		};
		std::wstring err;
		if(!m_player->InitAudioPlayer(err) || !ctx->InitWindow(callback, false))
		{
			delete m_window;
			delete m_player;
			m_window = nullptr;
			return false;
		}

		auto events = [ObjectPtr = m_window](auto&& PH1, auto&& PH2, auto&& PH3, auto&& PH4)
		{
			return ObjectPtr->EventCallback(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
				std::forward<decltype(PH3)>(PH3), std::forward<decltype(PH4)>(PH4));
		};
		ctx->SetEventCallback(events);

		auto source = [ObjectPtr = m_window](auto&& PH1, auto&& PH2, auto&& PH3, auto&& PH4)
		{
			return ObjectPtr->EventSource(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
			                       std::forward<decltype(PH3)>(PH3), std::forward<decltype(PH4)>(PH4));
		};
		ctx->SetEventSourceCallback(source);

		const auto base = ctx->Base();
		base->SetMinimSize(m_wndSize);
		base->SetResMode(true);
		base->CenterWindow();
		//base->ShowDebugRect(true);
		base->ShowWindow(true);
		return true;
	}

	void MainEventLoop()
	{
		m_window->m_window->EventLoop();
		delete m_window;
		m_window = nullptr;
	}

	Page* FindPage(std::wstring_view name)
	{
		for(const auto& page : m_pageList)
		{
			if (page->m_page->GetName() == name)
				return page;
		}
		return nullptr;
	}

	MainWindow::~MainWindow()
	{
		for (const auto& page : m_pageList)
			delete page;
		for (const auto& menu : m_menuList)
			delete menu;
		m_pageList.clear();
		delete m_window;
	}

	bool MainWindow::InitWindow(MWindowCtx* ctx, Ctrl::UIControl* root, XML::MuiXML* xmlUI)
	{
		CreateDefaultStyle(xmlUI);

		const std::wstring xml = LR"(
		<PropGroup id="ani" animate="true" aniAlphaType="false" />

		<UIControl bgColor="#windbgColor" size="100%,100%" autoSize="false" />
		<UIControl name="content" autoSize="false" frame="0,35,100%,100%" align="Absolute" />
		<UIControl autoSize="false" size="100%,35" name="titlebar" align="Absolute">
			<UIControl autoSize="false" size="100%,30" name="titlebar" align="LinearHL">
				<UIButton style="style_btnclose_dark" name="title_close" size="33,30" />
				<UIButton style="style_btnmax_dark" name="title_maxsize" size="33,30" />
				<UIButton style="style_btnmin_dark" name="title_minsize" size="33,30" />
			</UIControl>
		<UILabel fontSize="13" pos="12,12" text="#app_title" name="title_label" />
		</UIControl>
		)";

		if(!xmlUI->CreateUIFromXML(root, xml))
		{
			_M_OutErrorDbg_(L"无效的XML代码 创建UI失败!", false);
			return false;
		}

		root = root->Child(L"content");
		CreatePageList(root, xmlUI);
		CreateMenuList(root, xmlUI);

		//const HWND hWnd = (HWND)m_window->Base()->GetWindowHandle();
		//HICON hIcon = LoadIcon(GetModuleHandle(nullptr), MAKEINTRESOURCE(IDI_ICON1));
		//SendMessageW(hWnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
		//SendMessageW(hWnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);

		return true;
	}

	bool MainWindow::EventCallback(MWindowCtx*, UINotifyEvent event, Ctrl::UIControl* control, _m_param param)
	{
		if (TitleBarEvent(event, control) || MenuEvent(event, control, param))
			return true;

		//分发消息给页面
		for(const auto& page : m_pageList)
		{
			if (page->EventProc(event, control, param))
				return true;
		}

		return false;
	}

	_m_result MainWindow::EventSource(const MWindowCtx* ctx, const MWndDefEventSource& defcallback, MEventCodeEnum msg, _m_param param)
		const {
		if(msg == M_WND_SIZE)
		{
			const auto base = m_window->Base();
			//切换最大化和还原的按钮图标
			const auto btn = ctx->Base()->GetRootControl()->Child<Ctrl::UIButton>(L"title_maxsize");
			if (btn->GetUserData() == 1 && !base->IsMaximize())
			{
				static UIStyle* maxstyle = ctx->XML()->Mgr()->FindStyle(L"style_btnmax_dark");
				btn->SetUserData(0);
				btn->SetAttributeSrc(L"style", maxstyle);
			}
			else if (btn->GetUserData() == 0 && base->IsMaximize())
			{
				static UIStyle* maxstyle = ctx->XML()->Mgr()->FindStyle(L"style_btnres_dark");
				btn->SetUserData(1);
				btn->SetAttributeSrc(L"style", maxstyle);
			}
		}
		return defcallback(msg, param);
	}

	void MainWindow::CreatePageList(Ctrl::UIControl* root, XML::MuiXML* xmlUI)
	{
		m_pageList.push_back(new MainPage(root, xmlUI));
		m_pageList.push_back(new SidePage(root, xmlUI));
		m_pageList.push_back(new SettingPage(root, xmlUI));
	}

	void MainWindow::CreateMenuList(Ctrl::UIControl* root, XML::MuiXML* xmlUI)
	{
		const std::wstring xml = LR"(
		<PropGroup id="menu" autoSize="true" bgColor="#menuColor" frameWidth="1"
		frameColor="#menuframe" frameRound="6.f" dsbFontColor="#menuDsbitem" iFontColor="#textColor" 
		itemStyle="menuitemDark" lineColor="#menuline" iconOffset="5,4" textOffset="30,5" inset="5,5,5,5" />
		)";
		if (!xmlUI->CreateUIFromXML(root, xml))
		{
			__debugbreak();
		}
	}

	bool MainWindow::TitleBarEvent(UINotifyEvent event, const Ctrl::UIControl* control) const
	{
#ifdef _WIN32
		const HWND hWnd = (HWND)m_window->Base()->GetWindowHandle();
		switch (event)
		{
		case Event_Mouse_LClick:
		{
			if (_MNAME(L"title_close"))
			{
				const HWND& hwnd = hWnd;
				const int ret = MoeMessageBoxAskC(L"msgbox_11.title",L"msgbox_11.desc");
				if (ret == IDNO)
					return true;
				::SendMessageW(hWnd, WM_CLOSE, 0, 0);
				return true;
			}
			if (_MNAME(L"title_minsize"))
			{
				::ShowWindow(hWnd, SW_MINIMIZE);
				return true;
			}
			if (_MNAME(L"title_maxsize"))
			{
				if (IsZoomed(hWnd))
					::ShowWindow(hWnd, SW_RESTORE);
				else
					::ShowWindow(hWnd, SW_MAXIMIZE);
				return true;
			}
			break;
		}
		case Event_Mouse_LDoubleClicked:
		{
			if (_MNAME(L"titlebar") || _MNAME(L"title_label"))
			{
				if (IsZoomed(hWnd))
					::ShowWindow(hWnd, SW_RESTORE);
				else
					::ShowWindow(hWnd, SW_MAXIMIZE);
			}
			break;
		}
		case Event_Mouse_LDown:
		{
			if (_MNAME(L"titlebar") || _MNAME(L"title_label"))
			{
				::SendMessageW(hWnd, WM_SYSCOMMAND, SC_MOVE | HTCAPTION, 0);
				return true;
			}
			break;
		}
		default:
			return false;
		}
#endif
		return false;
	}

	bool MainWindow::MenuEvent(UINotifyEvent event, const Ctrl::UIControl* control, _m_param param)
	{
		//失去焦点 隐藏菜单
		if (event == Event_Focus_False && m_curShowMenu)
		{
			//判断不是点击的菜单本体或者是被禁用的菜单项目
			const auto ctrl = (Ctrl::UIControl*)param;
			if (ctrl && ctrl->GetParent() == m_curShowMenu->m_menu)
				return false;

			m_curShowMenu->Show(false);
			m_curShowMenu = nullptr;
		}
		//按钮被点击 查找是否是点击的菜单栏的按钮
		else if(event == Event_Mouse_LClick)
		{
			//匹配对应的菜单
			for(const auto& menu : m_menuList)
			{
				if (menu->GetBtnName() != control->GetName())
					continue;

				m_curShowMenu = menu;
				menu->Show(true);
				return true;
			}
		}
		else if (event == Event_Menu_ItemLClick)
		{
			for (const auto& menu : m_menuList)
			{
				if (menu->GetMenuName() == control->GetName() && menu->EventProc((int)param))
				{
					m_curShowMenu->Show(false);
					m_curShowMenu = nullptr;
					return true;
				}
			}
			m_curShowMenu->Show(false);
			m_curShowMenu = nullptr;
		}
		return false;
	}

	void Menu::Show(bool show)
	{
		if(m_isani || m_menu->IsVisible() == show)
		{
			return;
			//m_isani = false;
			//m_ani->StopTask(m_task, true);
		}
		auto anifun = [this, show](const MAnimation::MathsCalc* calc, float percent)
		{
			const auto alpha = (_m_byte)Helper::M_Clamp(0, 255, calc->calc(MAnimation::Default, show ? 0 : 255, show ? 255 : 0));

			m_menu->SetAlpha(alpha);
			m_menu->UpdateDisplay();

			if (percent == 100.f)
			{
				m_isani = false;
				if(!show)
					m_menu->SetVisible(false);
			}
			return m_isani;
		};
		if (show)
		{
			m_menu->SetAlpha(0, false);
			m_menu->SetVisible(true);
			m_menu->UpdateLayout();
		}
		m_isani = true;
		m_ani->CreateTask(anifun, 150);
	}
}