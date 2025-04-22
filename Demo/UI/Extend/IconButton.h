#pragma once
#include "../framework.h"

namespace Mui::Ctrl 
{
	//图标按钮扩展控件
	class IconButton : public UIButton
	{
	public:
		M_DEF_CTRL(L"IconButton")
		{
			{ CtrlMgr::AttribType::UIResource, L"icon" },
		},
		UIButton::M_PTRATTRIB
		M_DEF_CTRL_END

		IconButton(UIControl* parent, Attribute attrib) : UIButton(parent, std::move(attrib))
		{
			UILabel::OffsetDraw = true;
		}
		~IconButton() override;

		void SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw = true) override;

	protected:
		void OnLoadResource(MRenderCmd* render, bool recreate) override;
		void OnPaintProc(MPCPaintParam param) override;

	private:
		MBitmap* m_icon = nullptr;
		UIResource m_iconRes;
		UIPoint textOffset;
		UIPoint iconOffset;
		UISize iconSize;
	};
}