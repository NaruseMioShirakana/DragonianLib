#include "IconButton.h"

namespace Mui::Ctrl
{
	void IconButton::Register()
	{
		static auto method = [](UIControl* parent)
		{
			return new IconButton(parent, Attribute());
		};
		M_REGISTER_CTRL(method);
	}

	IconButton::~IconButton()
	{
		MSafeRelease(m_icon);
		m_iconRes.Release();
	}

	void IconButton::SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw)
	{
		using namespace Helper;

		if (attribName == L"offset")
		{
			std::vector<int> value;
			M_GetAttribValueInt(attrib, value, 4);
			textOffset = UIPoint(value[0], value[1]);
			iconOffset = UIPoint(value[2], value[3]);
		}
		else if (attribName == L"iconSize")
		{
			std::vector<int> value;
			M_GetAttribValueInt(attrib, value, 2);
			iconSize = UISize(value[0], value[1]);
		}
		else if (attribName == L"icon")
		{
			MSafeRelease(m_icon);
			m_iconRes.Release();
			if (UIResource* icon = (UIResource*)M_StoULong64(attrib))
			{
				if (icon->data && icon->size)
					m_icon = m_render->CreateBitmap(*icon);
				if(m_icon)
				{
					m_iconRes = { new _m_byte[icon->size], icon->size };
					memcpy(m_iconRes.data, icon->data, icon->size);
				}
			}
		}
		else 
		{
			UIButton::SetAttribute(attribName, attrib, draw);
			return;
		}
		m_cacheUpdate = true;
		if (draw)
			UpdateDisplay();
	}

	void IconButton::OnLoadResource(MRenderCmd* render, bool recreate)
	{
		UIButton::OnLoadResource(render, recreate);
		MSafeRelease(m_icon);
		if (m_iconRes.data)
			m_icon = m_render->CreateBitmap(m_iconRes);
	}

	void IconButton::OnPaintProc(MPCPaintParam param)
	{
		//文字偏移
		auto scale = GetRectScale().scale();
		UILabel::OffsetDrawRc = param->destRect->ToRect();
		UILabel::OffsetDrawRc.left += _scale_to(textOffset.x, scale.cx);
		UILabel::OffsetDrawRc.top += _scale_to(textOffset.y, scale.cy);

		UIButton::OnPaintProc(param);

		if (m_icon)
		{
			UIRect dest = *param->destRect;
			dest.left += _scale_to(iconOffset.x, scale.cx);
			dest.top += _scale_to(iconOffset.y, scale.cy);

			dest.right = dest.left + _scale_to(iconSize.width, scale.cx);
			dest.bottom = dest.top + _scale_to(iconSize.height, scale.cy);

			_m_byte alpha = param->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst;
			if(!IsEnabled())
			{
				if (alpha > 120)
					alpha -= 120;
				else
					alpha = 0;
			}
			param->render->DrawBitmap(m_icon, alpha, dest);
		}
	}
}
