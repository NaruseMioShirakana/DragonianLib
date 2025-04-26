#pragma once
#include "Framework.h"
#include "Extend/Waveform.h"
#include "Extend/CurveEditor.h"
#include <Render/Sound/Mui_SoundDef.h>

namespace UI
{
	extern const Mui::UISize m_wndSize;
	extern const std::wstring m_wndTitle;
	extern Mui::Render::MAudioPlayer* m_player;

	extern bool CreateMainWindow(Mui::MiaoUI& engine, const std::vector<std::wstring>& cmdList);

	extern void MainEventLoop();

	class Page  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		Page(Mui::Ctrl::UIControl* parent, Mui::XML::MuiXML* ui)
		: m_parent(parent), m_ui(ui) {}

		virtual ~Page()
		{
			if (m_page && m_parent)
				m_parent->RemoveChildren(m_page);
			delete m_page;
		}

		virtual bool EventProc(Mui::MEventCodeEnum event, Mui::Ctrl::UIControl* control, Mui::_m_param param) = 0;

	protected:
		Mui::Ctrl::UIControl* m_parent = nullptr;
		Mui::Ctrl::UIControl* m_page = nullptr;
		Mui::XML::MuiXML* m_ui = nullptr;

		Mui::Ctrl::UIListBox* m_list = nullptr;
		Mui::Ctrl::CurveEditor* m_editor = nullptr;
		Mui::Ctrl::Waveform* m_wave = nullptr;

		friend Page* FindPage(std::wstring_view name);
	};

	extern Page* FindPage(std::wstring_view name);

	class Menu  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		Menu(Mui::Ctrl::UIControl* parent, Mui::XML::MuiXML* ui)
			: m_parent(parent), m_ui(ui)
		{
			m_ani = new Mui::MAnimation(m_parent->GetParentWin());
		}

		virtual std::wstring GetBtnName() = 0;

		virtual std::wstring GetMenuName() = 0;

		virtual ~Menu()
		{
			if (m_menu && m_parent)
				m_parent->RemoveChildren(m_menu);
			delete m_menu;
			delete m_ani;
		}

		virtual bool EventProc(int index) = 0;

		void Show(bool show);

	protected:
		Mui::Ctrl::UIControl* m_parent = nullptr;
		Mui::Ctrl::UIControl* m_menu = nullptr;
		Mui::XML::MuiXML* m_ui = nullptr;
		Mui::MAnimation* m_ani = nullptr;
		Mui::MAnimation::TaskID m_task = 0;
		bool m_isani = false;

		friend class MainWindow;
	};

	class MainWindow  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		MainWindow(Mui::MWindowCtx* ctx) : m_window(ctx) {}
		~MainWindow();

		bool InitWindow(Mui::MWindowCtx* ctx, Mui::Ctrl::UIControl* root, Mui::XML::MuiXML* xmlUI);

		bool EventCallback(Mui::MWindowCtx*, Mui::UINotifyEvent, Mui::Ctrl::UIControl*, Mui::_m_param);

		Mui::_m_result EventSource(const Mui::MWindowCtx*, const Mui::MWndDefEventSource&, Mui::MEventCodeEnum, Mui::_m_param) const;

	private:
		static void CreatePageList(Mui::Ctrl::UIControl*, Mui::XML::MuiXML*);
		static void CreateMenuList(Mui::Ctrl::UIControl*, Mui::XML::MuiXML*);
		bool TitleBarEvent(Mui::UINotifyEvent, const Mui::Ctrl::UIControl*) const;
		static bool MenuEvent(Mui::UINotifyEvent, const Mui::Ctrl::UIControl*, Mui::_m_param);

		Mui::MWindowCtx* m_window = nullptr;
		std::vector<std::wstring> m_cmdList;

		friend bool CreateMainWindow(Mui::MiaoUI&, const std::vector<std::wstring>&);
		friend void MainEventLoop();
	};
}