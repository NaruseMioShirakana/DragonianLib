#pragma once
#include <Render/Sound/Mui_DirectSound.h>

#include "Framework.h"
#include <User/Mui_Page.h>
#include <Render/Sound/Mui_SoundDef.h>
#include "Page/SidePage.h"

namespace SimpleF0Labeler
{
	bool CreateMainWindow();

	bool InitWindow(
		Mui::Window::MWindowCtx* Context,
		Mui::Ctrl::UIControl* Root,
		Mui::XML::MuiXML* UI
	);

	class MainWindow : public Mui::UIPage  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		MainWindow(
			Mui::UIPage* root
		) : UIPage(root)
		{

		}
		~MainWindow() override;

		Mui::Ctrl::UIControl* GetRoot() const
		{
			return _MyRoot;
		}

		bool UIEventProc(
			Mui::XML::PropName event,
			Mui::Ctrl::UIControl* control,
			const std::any& param
		);

	private:
		Mui::Ctrl::UIControl* OnLoadPageContent(
			Mui::Ctrl::UIControl* parent,
			Mui::XML::MuiXML* ui
		) override;

		bool EventProc(
			Mui::XML::PropName event,
			Mui::Ctrl::UIControl* control,
			std::any param
		) override;

	protected:
		Mui::Ctrl::UIControl* _MyRoot = nullptr;
		std::shared_ptr<Mui::Render::MDS_AudioPlayer> _MyPlayerContext = nullptr;
	};

}
