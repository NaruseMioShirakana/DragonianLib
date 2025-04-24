#include "CurveEditor.h"
#include <random>
#include "../DefControl.hpp"

#define append_jump_discontinuity do{ \
jump_discontinuity.emplace_back(\
	D2D1::Point2F\
	(\
		x_prev, \
		(float)viewRect.top + ((float)fullHeight - value_prev * (float)fullHeight) - offsetY\
		), \
	D2D1::Point2F\
	(\
		x, \
		(float)viewRect.top + ((float)fullHeight - value * (float)fullHeight) - offsetY\
		)\
);}while(0)

namespace Mui::Ctrl
{
	const auto MaxFreq = PitchLabel::PitchToF0(119.9f);
	const auto MinFreq = PitchLabel::PitchToF0(0);

	class MD2DBrush final : MBrush_D2D
	{
	public:
		static ID2D1SolidColorBrush* GetBrush(MBrush* brush)
		{
			return static_cast<MD2DBrush*>(brush)->m_brush;
		}
	};

	void CurveEditor::Register()
	{
		static auto method = [](UIControl* parent)
		{
			return new CurveEditor(parent);
		};
		M_REGISTER_CTRL(method);
	}

	CurveEditor::CurveEditor(UIControl* parent) : UIScroll(Attribute())
	{
		parent->AddChildren(this);
		m_anicls = new MAnimation(UIControl::GetParentWin());

		ScrollCallBack callback = [this](auto&& PH1, auto&& PH2, auto&& PH3)
		{
			OnScrollView(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
				std::forward<decltype(PH3)>(PH3));
		};
		SetCallback(callback);
		UIScroll::SetAttributeSrc(L"horizontal", true, false);
		SetAttributeSrc(L"barWidth", m_preHeight, false);

		auto attrib = Attribute();
		attrib.callback = [this](auto&& PH1, auto&& PH2, auto&& PH3)
		{
			OnScrollView(std::forward<decltype(PH1)>(PH1), std::forward<decltype(PH2)>(PH2),
				std::forward<decltype(PH3)>(PH3));
		};
		attrib.barWidth = m_barWidth;
		attrib.vertical = true;
		attrib.button = false;
		m_sidebar = new UIScroll(this, attrib);
		static_cast<CurveEditor*>(m_sidebar)->m_ALLWheel = false;
		m_sidebar->AutoSize(false, false);
		PosSizeUnit uint;
		uint.x_w = Percentage;
		uint.y_h = FillMinus;
		m_sidebar->SetSizeUint(uint, false);
		m_sidebar->SetSize(100, m_preHeight, false);
		m_sidebar->SetMsgFilter(true);
		HMIDIOUT MidioutH;
		MidiOutOpen = midiOutOpen(&MidioutH, 0, 0, 0, CALLBACK_NULL) == MMSYSERR_NOERROR;
		if (MidiOutOpen)
			MidiOutHandle = MidioutH;
		else
			MidiOutHandle = nullptr;
	}

	CurveEditor::~CurveEditor()
	{
		MSafeRelease(m_font);
		MSafeRelease(m_brush_m);
		MSafeRelease(m_pen);
		if(MidiOutOpen && MidiOutHandle)
		{
			midiOutClose(HMIDIOUT(MidiOutHandle));
			MidiOutOpen = false;
		}
	}

	void CurveEditor::SetCurveData(const FloatTensor2D& data, const FloatTensor2D& spec)
	{
		std::lock_guard lock(mx);
		curve_idx = 0;
		if (data.HasValue())
			m_f0data = data.View();
		else
			m_f0data = std::nullopt;
		if (spec.HasValue())
			m_specData = spec.View();
		else
			m_specData = std::nullopt;
		selected_f0_begin = selected_f0_end = nullptr;
		m_viewScaleH = 1.f;
		CalcRangeViewH();
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::ReSetCurveData(const FloatTensor2D& data, int64_t idx)
	{
		std::lock_guard lock(mx);
		const auto Range = GetSelectedRange();
		const auto begin_pos = std::distance(m_f0data.Data() + curve_idx * m_f0data.Size(1), Range.begin());
		const auto end_pos = std::distance(m_f0data.Data() + curve_idx * m_f0data.Size(1), Range.end());

		if (data.HasValue() && data.IsContiguous())
			if (data.Size(1) == m_f0data.Size(1))
				m_f0data = data.View();
		if (idx < 0)
			idx = curve_idx;
		if (idx == curve_idx)
		{
			if (!m_f0data.Null())
			{
				if (!Range.Null())
				{
					selected_f0_begin = m_f0data.Data() + begin_pos + idx * m_f0data.Size(1);
					selected_f0_end = m_f0data.Data() + end_pos + idx * m_f0data.Size(1);
				}
			}
		}
		else
		{
			if (idx < m_f0data.Size(0))
				curve_idx = idx;
			selected_f0_begin = selected_f0_end = nullptr;
		}
	}

	void CurveEditor::SetCurveIndex(int64_t idx)
	{
		std::lock_guard lock(mx);
		if (idx != curve_idx)
		{
			curve_idx = idx;
			selected_f0_begin = selected_f0_end = nullptr;
			UpDate();
		}
	}

	void CurveEditor::SetShowPitch(bool show)
	{
		m_showPitch = show;
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::SetPlayLinePos(_m_size offset)
	{
		m_plineOffset = offset;
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::UpDate()
	{
		SetDragValue(false, 0);
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	DragonianLib::TemplateLibrary::MutableRanges<float> CurveEditor::GetSelectedRange() const
	{
		auto begin = selected_f0_begin < selected_f0_end ? selected_f0_begin : selected_f0_end;
		auto end = selected_f0_begin < selected_f0_end ? selected_f0_end : selected_f0_begin;
		return { begin, end };
	}

	void CurveEditor::SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw)
	{
		if (attribName == L"fontColor")
		{
			m_fontColor = Helper::M_GetAttribValueColor(attrib);
		}
		else if (attribName == L"lineColor")
		{
			m_lineColor = Helper::M_GetAttribValueColor(attrib);
		}
		else if (attribName == L"curveColor")
		{
			m_curveColor = Helper::M_GetAttribValueColor(attrib);
		}
		else if (attribName == L"preHeight")
		{
			m_preHeight = Helper::M_StoInt(attrib);
			UIScroll::SetAttributeSrc(L"barWidth", m_preHeight);
		}
		else if(attribName == L"fontName")
		{
			m_font->SetFontName(attrib.data());
		}
		else if(attribName == L"fontSize")
		{
			_m_scale scale = GetRectScale().scale();
			m_fontSize = Helper::M_StoInt(attrib.data());
			const float fontSize = Helper::M_MIN(scale.cx, scale.cy) * (float)m_fontSize;
			m_font->SetFontSize((_m_uint)fontSize, std::make_pair(0u, (_m_uint)m_font->GetText().length()));
		}
		else
		{
			if (attribName == L"styleV")
				m_sidebar->SetAttribute(attribName, attrib, false);
			if(attribName != L"styleV")
				UIScroll::SetAttribute(attribName, attrib, draw);
			return;
		}
		m_cacheUpdate = true;
		if (draw)
			UpdateDisplay();
	}

	std::wstring CurveEditor::GetAttribute(std::wstring_view attribName)
	{
		if(attribName == L"fontColor")
			return Color::M_RGBA_STR(m_fontColor);
		if (attribName == L"lineColor")
			return Color::M_RGBA_STR(m_lineColor);
		if (attribName == L"curveColor")
			return Color::M_RGBA_STR(m_curveColor);
		if (attribName == L"preHeight")
			return std::to_wstring(m_preHeight);
		if (attribName == L"fontName")
			return m_fontName;
		if (attribName == L"fontSize")
			return std::to_wstring(m_fontSize);
		return UIScroll::GetAttribute(attribName);
	}

	void CurveEditor::OnScale(_m_scale scale)
	{
		UIScroll::OnScale(scale);
		const float fontSize = Helper::M_MIN(scale.cx, scale.cy) * (float)m_fontSize;
		m_font->SetFontSize((_m_uint)fontSize, std::make_pair(0u, (_m_uint)m_font->GetText().length()));
	}

	void CurveEditor::OnLoadResource(MRenderCmd* render, bool recreate)
	{
		UIScroll::OnLoadResource(render, recreate);

		MSafeRelease(m_pen);
		MSafeRelease(m_brush_m);
		MSafeRelease(m_font);
		m_pen = render->CreatePen(1, m_lineColor);
		m_brush_m = render->CreateBrush(m_fontColor);

		_m_scale scale = GetRectScale().scale();
		const float fontSize = Helper::M_MIN(scale.cx, scale.cy) * (float)m_fontSize;
		m_font = render->CreateFonts(L"", m_fontName, (_m_uint)fontSize);
	}

	void CurveEditor::PlaySoundPitch(const UIPoint& point)
	{
		if (MidiOutOpen && MidiOutHandle && m_showPitch)
		{
			UIPoint pt = point;
			pt.y -= UINodeBase::m_data.Frame.top;
			const auto attribY = m_sidebar->GetAttribute();
			//视图偏移位置Y
			const auto scale = GetRectScale().scale();
			const int space = _scale_to(5, scale.cy);
			const int viewHeight = m_viewRect.GetHeight();
			const int fontHeight = m_font->GetMetrics().bottom;
			const int fullHeight = int((float)viewHeight * m_viewScaleV);
			float offsetY = 0.f;
			if (attribY.dragValue.height != 0)
				offsetY = (float)attribY.dragValue.height / (float)attribY.range.height;
			offsetY = (float)(fullHeight - viewHeight) * (offsetY);
			if (attribY.range.height == m_sidebar->Frame().GetHeight())
				offsetY = 0.f;
			float value = ((float)(pt.y - space - fontHeight) + offsetY) / (float)fullHeight;
			value = Helper::M_Clamp(0.f, 120.f, 119.9f - value * 120.f);
			auto Pitch = DWORD(round(value));
			if (Pitch > 0x7f) Pitch = 0x7f;
			if(Pitch != LastMidiPitch)
			{
				constexpr int volume = 0x7f;
				constexpr int instrumet = 0x90;
				const DWORD msg = (volume << 16) + (Pitch << 8) + instrumet;
				constexpr DWORD MIDOOUT_MIDIKEYRELEASE = (0XFF << 8) + instrumet;
				midiOutShortMsg(HMIDIOUT(MidiOutHandle), MIDOOUT_MIDIKEYRELEASE);
				midiOutShortMsg(HMIDIOUT(MidiOutHandle), msg);
				LastMidiPitch = Pitch;
			}
		}
	}

	void CurveEditor::OnPaintProc(MPCPaintParam param)
	{
		std::lock_guard lock(mx);

		int width = param->destRect->GetWidth();
		int height = param->destRect->GetHeight();
		if(width != m_size.width || height != m_size.height)
		{
			m_size = { width, height };
			CalcViewRect();
			CalcRangeViewH();
			CalcRangeViewV();
		}

		_m_scale scale = GetRectScale().scale();

		if (m_insel)
		{
			int space = _scale_to(5, scale.cy);
			int fontHeight = m_font->GetMetrics().bottom;
			int preHeight = _scale_to(m_preHeight, scale.cy);
			int fontCenter = fontHeight / 2;
			const auto bot = param->destRect->bottom - preHeight - space - fontCenter;
			const auto top = param->destRect->top + space;
			m_brush_m->SetColor(m_fontColor);
			m_brush_m->SetOpacity(50);

			int c_width = m_viewRect.GetWidth();
			c_width = int((float)c_width * m_viewScaleH);
			float stepX = (float)c_width / float(m_f0data.Size(1) - 1);
			const auto left = _scale_to(m_viewRect.left, GetRectScale().scale().cx);
			const auto cur_x_off = CalcViewHOffset();
			const auto data = m_f0data.Data() + curve_idx * m_f0data.Size(1);
			const auto x_off_beg = selected_f0_begin - data;
			const auto x_off_end = selected_f0_end - data;
			auto m_l_x_pos = std::max(int((x_off_beg * stepX) - cur_x_off), 0) + left;
			auto m_c_x_pos = std::max(int((x_off_end * stepX) - cur_x_off), 0) + left;

			param->render->FillRectangle(
				UIRect(std::min(m_c_x_pos, m_l_x_pos), top,
					std::abs(m_c_x_pos - m_l_x_pos), bot - top),
				m_brush_m
			);
		}

		DrawLabel(scale, param);

		if(m_f0data.HasValue())
			DrawCurve(scale, param, m_f0data);

		DrawPlayeLine(scale, param);

		UIScroll::OnPaintProc(param);
	}

	bool CurveEditor::OnMouseWheel(_m_uint flag, short delta, const UIPoint& point)
	{
		if (UIScroll::OnMouseWheel(flag, delta, point))
			return true;
		UIRect barFrame = m_sidebar->Frame();
		barFrame.left = barFrame.right - _scale_to(m_barWidth, GetRectScale().scale().cx);
		if (Helper::M_IsPtInRect(barFrame, point))
			return true;

		const auto& attrib = UIScroll::GetAttribute();

		int viewWidth = m_viewRect.GetWidth();
		int fullWidth = int((float)viewWidth * m_viewScaleH);

		auto getXoffset = [&](int index)
			{
				//x offset
				float offsetX = 0.f;
				if (attrib.dragValue.width != 0)
					offsetX = (float)attrib.dragValue.width / (float)attrib.range.width;
				offsetX = (float)(fullWidth - viewWidth) * (offsetX);
				offsetX = ((float)(index)+offsetX) / (float)fullWidth;
				offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

				return  size_t(offsetX * (float)m_f0data.Size(1));
			};

		const UIPoint pt = {
			std::max(point.x - UINodeBase::m_data.Frame.left, 0),
			std::max(point.y - UINodeBase::m_data.Frame.top, 0)
		};

		if (m_insel)
		{
			auto x = pt.x - m_viewRect.left;
			x -= m_viewRect.left;
			if (x < 0) x = 0;
			auto xoff = getXoffset(x);
			selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
		}

		float delta_ = (float)delta / static_cast<float>(WHEEL_DELTA);
#ifdef _WIN32
		if(GetKeyState(VK_LCONTROL) & 0x8000)
#else
#erro __TODO__
#endif // _WIN32
		{
			if (!m_f0data.HasValue())
				return false;
			const auto max_ScaleH = float(m_f0data.Size(1)) / 10.f;
			const auto ScaleH = m_viewScaleH * powf(4.2f, delta_ * 0.1f);
			m_viewScaleH = Helper::M_MAX(Helper::M_MIN(max_ScaleH, ScaleH) * 1.f, 1.f); //最大20000%最小100%
			std::lock_guard lock(mx);
			if (m_viewScaleH >= max_ScaleH || m_viewScaleH <= 1.f) CalcRangeViewH();
			else CalcRangeViewH(point, delta);
		}
		else if(GetKeyState(VK_LMENU) & 0x8000)
		{
			m_viewScaleV *= powf(2.2f, delta_ * 0.1f);
			m_viewScaleV = Helper::M_MAX(Helper::M_MIN(200.f, m_viewScaleV) * 1.f, 1.f); //最大1000%最小100%
			CalcRangeViewV();
		}
		else if (GetKeyState(VK_LSHIFT) & 0x8000)
		{
			const int range = GetRange(true);
			int step = int(float(range) / (m_viewScaleH * 3));
			if (step == 0) step = 1;
			const int curval = GetDragValue(true);
			int val = curval + int(-delta_ * float(step));
			if (val > range) val = range;
			if (val < 0) val = 0;
			SetDragValue(true, val);
		}
		else
		{
			const int range = m_sidebar->GetRange(false);
			int step = int(float(range) / (3 * m_viewScaleV));
			if (step == 0) step = 1;
			const int curval = m_sidebar->GetDragValue(false);
			int val = curval + int(-delta_ * float(step));
			if (val > range) val = range;
			if (val < 0) val = 0;
			m_sidebar->SetDragValue(false, val);
			SetDragValue(false, 0);
		}

		return false;
	}

	bool CurveEditor::OnMouseMove(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnMouseMove(flag, point))
			return true;

		const auto& attrib = UIScroll::GetAttribute();

		const int viewWidth = m_viewRect.GetWidth();
		const int fullWidth = int((float)viewWidth * m_viewScaleH);

		auto getXoffset = [&](int index)
			{
				//x offset
				float offsetX = 0.f;
				if (attrib.dragValue.width != 0)
					offsetX = (float)attrib.dragValue.width / (float)attrib.range.width;
				offsetX = (float)(fullWidth - viewWidth) * (offsetX);
				offsetX = ((float)(index)+offsetX) / (float)fullWidth;
				offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

				return size_t(offsetX * (float)m_f0data.Size(1));
			};

		const UIPoint pt = {
			std::max(point.x - UINodeBase::m_data.Frame.left, 0),
			std::max(point.y - UINodeBase::m_data.Frame.top, 0)
		};

		if (m_lisdown && GetKeyState(VK_TAB) & 0x8000)
			PlaySoundPitch(point);

		if (time_in_sel && m_f0data.HasValue())
		{
			const auto x = pt.x - m_viewRect.left;
			auto xof = getXoffset(x);
			xof = xof * WndControls::GetPcmSize() / m_f0data.Size(1);
			WndControls::SetPlayerPos(xof);
		}

		if (m_insel)
		{
			auto x = pt.x - m_viewRect.left;
			if (x < 0) x = 0;
			auto xoff = getXoffset(x);
			selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
		}

		else if (m_isdown && m_f0data.HasValue())
		{
			std::lock_guard lock(mx);
			auto y = pt.y, x = pt.x;
#ifdef _WIN32
			if (GetKeyState(VK_LSHIFT) & 0x8000)
				y = m_lastPos.y;
#endif

			auto attribY = m_sidebar->GetAttribute();
			//视图偏移位置Y
			auto scale = GetRectScale().scale();
			int space = _scale_to(5, scale.cy);
			int viewHeight = m_viewRect.GetHeight();
			int fontHeight = m_font->GetMetrics().bottom;
			int fullHeight = int((float)viewHeight * m_viewScaleV);

			float offsetY = 0.f;
			if (attribY.dragValue.height != 0)
				offsetY = (float)attribY.dragValue.height / (float)attribY.range.height;
			offsetY = (float)(fullHeight - viewHeight) * (offsetY);
			if (attribY.range.height == m_sidebar->Frame().GetHeight())
				offsetY = 0.f;

			float value = ((float)(y - space - fontHeight) + offsetY) / (float)fullHeight;

			if (m_showPitch)
			{
				value = Helper::M_Clamp(0.f, 120.f, 120.f - value * 120.f) - .1f;
				if (value < .0f)
					value = 0.f;
				else
					value = PitchLabel::PitchToF0(value);
			}
			else
				value = Helper::M_Clamp(0.f, 1.f, 1.f - value) * MaxFreq;

			x -= m_viewRect.left;
			if (x < 0) x = 0;
			if (m_lastPos.x < 0) m_lastPos.x = 0;
			int begin = m_lastPos.x < x ? m_lastPos.x : x;
			int end = x > m_lastPos.x ? x : m_lastPos.x;

			if (end <= begin)
				end = begin + 3;

			auto xbegin = getXoffset(begin);
			auto xend = getXoffset(end);
			const auto m_f0data_data = m_f0data.Data() + curve_idx * m_f0data.Size(1);
			auto count = xend - xbegin;

			const auto range_sel = GetSelectedRange();

			for (size_t i = 0; i < count; ++i)
				if (xbegin + i < static_cast<size_t>(m_f0data.Size(1)))
					if (auto cur_data_ptr = &m_f0data_data[xbegin + i]; range_sel.Contains(cur_data_ptr))
						*cur_data_ptr = value;

			if (count == 0 && xbegin < static_cast<size_t>(m_f0data.Size(1)))
				if (auto cur_data_ptr = &m_f0data_data[xbegin]; range_sel.Contains(cur_data_ptr))
					*cur_data_ptr = value;

			m_lastPos = { x, y };
		}
		m_cacheUpdate = true;
		UpdateDisplay();

		return false;
	}

	bool CurveEditor::OnRButtonDown(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnRButtonDown(flag, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - UINodeBase::m_data.Frame.left, 0),
			std::max(point.y - UINodeBase::m_data.Frame.top, 0)
		};

		if (Helper::M_IsPtInRect(m_viewRect, pt))
		{
			WndControls::AppendUndo();
			m_isdown = true;
			m_lastPos = { pt.x - m_viewRect.left, pt.y };
		}
		return false;
	}

	bool CurveEditor::OnRButtonUp(_m_uint flag, const UIPoint& point)
	{
		if(m_isdown && m_f0data.HasValue())
		{
			WndControls::CheckUnchanged();
			m_isdown = false;
		}

		return UIScroll::OnRButtonUp(flag, point);
	}

	bool CurveEditor::OnLButtonDown(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnLButtonDown(flag, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - UINodeBase::m_data.Frame.left, 0),
			std::max(point.y - UINodeBase::m_data.Frame.top, 0)
		};

		const auto& attrib = UIScroll::GetAttribute();

		const int viewWidth = m_viewRect.GetWidth();
		const int fullWidth = int((float)viewWidth * m_viewScaleH);

		auto getXoffset = [&](int index)
			{
				//x offset
				float offsetX = 0.f;
				if (attrib.dragValue.width != 0)
					offsetX = (float)attrib.dragValue.width / (float)attrib.range.width;
				offsetX = (float)(fullWidth - viewWidth) * (offsetX);
				offsetX = ((float)(index)+offsetX) / (float)fullWidth;
				offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

				return  size_t(offsetX * (float)m_f0data.Size(1));
			};

		if (GetKeyState(VK_TAB) & 0x8000)
		{
			if (Helper::M_IsPtInRect(m_viewRect, pt))
			{
				m_lisdown = true;
				m_llastPos = pt;
				if (m_showPitch)
					PlaySoundPitch(point);
			}
			return true;
		}
		else if (GetKeyState(VK_LSHIFT) & 0x8000)
		{
			if (m_f0data.HasValue())
			{
				m_insel = true;
				auto x = pt.x;
				x -= m_viewRect.left;
				if (x < 0) x = 0;
				auto xoff = getXoffset(x);
				selected_f0_begin = selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
			}
			return true;
		}
		else if (GetKeyState(VK_LCONTROL) & 0x8000)
		{
			in_move = true;
			m_llast_movePos = pt;
		}
		else
			time_in_sel = true;
		return false;
	}

	bool CurveEditor::OnLButtonUp(_m_uint flag, const UIPoint& point)
	{
		const auto& attrib = UIScroll::GetAttribute();
		int viewWidth = m_viewRect.GetWidth();
		int fullWidth = int((float)viewWidth * m_viewScaleH);

		auto getXoffset = [&](int index)
			{
				//x offset
				float offsetX = 0.f;
				if (attrib.dragValue.width != 0)
					offsetX = (float)attrib.dragValue.width / (float)attrib.range.width;
				offsetX = (float)(fullWidth - viewWidth) * (offsetX);
				offsetX = ((float)(index)+offsetX) / (float)fullWidth;
				offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

				return  size_t(offsetX * (float)m_f0data.Size(1));
			};

		if (m_f0data.HasValue())
		{
			if (m_insel && selected_f0_begin == selected_f0_end)
				selected_f0_begin = selected_f0_end = nullptr;
			if (time_in_sel)
			{
				auto x = point.x;
				x -= UINodeBase::m_data.Frame.left;
				x -= m_viewRect.left;
				auto xof = getXoffset(x);
				xof = xof * WndControls::GetPcmSize() / m_f0data.Size(1);
				WndControls::SetPlayerPos(xof);
			}
		}
		m_insel = false;
		m_lisdown = false;
		time_in_sel = false;
		in_move = false;
		constexpr DWORD MIDOOUT_MIDIKEYRELEASE = (0XFF << 8) + 0x90;
		midiOutShortMsg(HMIDIOUT(MidiOutHandle), MIDOOUT_MIDIKEYRELEASE);
		LastMidiPitch = 256;
		m_cacheUpdate = true;
		UpdateDisplay();

		return UIScroll::OnLButtonUp(flag, point);
	}

	void Write2Clipboard(const DragonianLib::TemplateLibrary::MutableRanges<float>& Range)
	{
		if (OpenClipboard(nullptr))
		{
			EmptyClipboard();
			std::string Buffer;
			Buffer.reserve(Range.Size() * 13);
			for (const auto& i : Range)
				Buffer += std::to_string(i) + ',';
			HGLOBAL clipbuffer = GlobalAlloc(GMEM_DDESHARE, Buffer.size() + 1);
			char* buffer = (char*)GlobalLock(clipbuffer);
			memcpy(buffer, Buffer.data(), Buffer.size() + 1);
			GlobalUnlock(clipbuffer);
			SetClipboardData(CF_TEXT, clipbuffer);
			CloseClipboard();
		}
	}

	bool CurveEditor::OnLButtonDoubleClicked(_m_uint flag, const UIPoint& point)
	{
		selected_f0_begin = selected_f0_end = nullptr;
		return UIScroll::OnLButtonDoubleClicked(flag, point);
	}

	char VK_Nums[]{ L'1',L'2',L'3',L'4',L'5',L'6',L'7',L'8',L'9' };

	bool CurveEditor::OnWindowMessage(MEventCodeEnum code, _m_param wParam, _m_param lParam)
	{
		auto Range = GetSelectedRange();
		bool Exec = !Range.Null() && Range.Size();
		if (code == M_WND_KEYDOWN)
		{
			if (GetKeyState(VK_LCONTROL) & 0x8000)
			{
				if (GetKeyState('C') & 0x8000)
				{
					if (Exec)
						Write2Clipboard(Range);
					return true;
				}
				else if (GetKeyState('X') & 0x8000)
				{
					if (Exec)
					{
						Write2Clipboard(Range);
						WndControls::AppendUndo();
						for (auto& i : Range)
							i = 0.f;
						WndControls::ApplyAppendUndo();
					}
					return true;
				}
				else if (GetKeyState('V') & 0x8000)
				{
					auto DataBegin = m_f0data.Data() + curve_idx * m_f0data.Size(1) + m_plineOffset;
					const auto DataEnd = m_f0data.Data() + (curve_idx + 1) * m_f0data.Size(1);
					if (DataBegin < DataEnd)
					{
						std::wstring Str;
						if (OpenClipboard(nullptr))
						{
							HGLOBAL hMem = GetClipboardData(CF_TEXT);
							if (hMem != nullptr)
							{
								if (const auto lpStr = (char*)::GlobalLock(hMem))
									Str = DragonianLib::UTF8ToWideString(lpStr);
								if (hMem != nullptr)
									GlobalUnlock(hMem);
							}
							CloseClipboard();
						}
						if (!Str.empty())
						{
							WndControls::AppendUndo();
							auto iter = std::regex_token_iterator{
								Str.cbegin(), Str.cend(),
								DragonianLib::PreDefinedRegex::RealRegex,
								0
							};
							for (
								auto end = decltype(iter)();
								DataBegin < DataEnd && iter != end;
								++iter, ++DataBegin)
							{
								*DataBegin = std::min(MaxFreq, wcstof(iter->str().c_str(), nullptr));
								if (*DataBegin < MinFreq)
									*DataBegin = 0.f;
							}

							WndControls::CheckUnchanged();
						}
					}
					return true;
				}
				else if (GetKeyState('W') & 0x8000)
				{
					if (Exec)
					{
						WndControls::AppendUndo();
						WndControls::ApplyPitchShift(GetSelectedRange());
						WndControls::CheckUnchanged();
					}
					return true;
				}
				else if (GetKeyState('E') & 0x8000)
				{
					if (Exec)
					{
						WndControls::AppendUndo();
						WndControls::ApplyCalc(GetSelectedRange());
						WndControls::CheckUnchanged();
					}
					return true;
				}
				else if (GetKeyState('A') & 0x8000)
				{
					selected_f0_begin = m_f0data.Data() + curve_idx * m_f0data.Size(1);
					selected_f0_end = m_f0data.Data() + (curve_idx + 1) * m_f0data.Size(1);
					return true;
				}
				else if (GetKeyState('0') & 0x8000)
					SetCurveIndex(9);
				else
					for (auto k : VK_Nums)
						if (GetKeyState(k) & 0x8000)
							SetCurveIndex(k - '1');
			}
			else if (GetKeyState(VK_DELETE) & 0x8000)
			{
				if (Exec)
				{
					WndControls::AppendUndo();
					for (auto& i : Range)
						i = 0.f;
					WndControls::ApplyAppendUndo();
				}
				return true;
			}
		}

		return UIScroll::OnWindowMessage(code, wParam, lParam);
	}

	bool CurveEditor::OnMouseExited(_m_uint flag, const UIPoint& point)
	{
		UIScroll::OnMouseExited(flag, point);
		if (m_f0data.HasValue() && m_insel)
			if (selected_f0_begin == selected_f0_end)
				selected_f0_begin = selected_f0_end = nullptr;
		m_insel = false;
		time_in_sel = false;

		if (m_isdown && m_f0data.HasValue())
		{
			WndControls::CheckUnchanged();
			m_isdown = false;
		}

		in_move = false;
		return false;
	}

	void CurveEditor::OnScrollView(UIScroll*, int dragValue, bool horizontal)
	{
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::DrawLabel(_m_scale scale, MPCPaintParam param)
	{
		m_font->SetText(L"C0");

		int height = param->destRect->GetHeight();

		if(m_curHeight != height)
		{
			m_curHeight = height;
			CalcRangeViewV();
		}

		int space = _scale_to(5, scale.cy);
		int fontWidth = 60;
		int fontHeight = m_font->GetMetrics().bottom;
		int preHeight = _scale_to(m_preHeight, scale.cy);
		int sidebarWidth = _scale_to(m_barWidth, scale.cx);
		//曲线视图高度
		int viewHeight = m_viewRect.GetHeight();
		int fullHeight = int((float)viewHeight * m_viewScaleV);
		int fontCenter = fontHeight / 2;
		//x轴基线y
		int baseLineY = height - preHeight - space - fontCenter;

		m_brush_m->SetColor(m_fontColor);
		m_brush_m->SetOpacity(param->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst);

		int lineWidth = _scale_to(10, scale.cx);
		int lineHeight = _scale_to(1, scale.cx);

		//计算基础位置
		UIRect dst = *param->destRect;

		//计算视图偏移位置
		float offsetY = 0.f;
		const auto& attribY = m_sidebar->GetAttribute();
		if (attribY.dragValue.height != 0)
			offsetY = (float)attribY.dragValue.height / (float)attribY.range.height;
		offsetY = (float)(fullHeight - viewHeight) * (offsetY);
		if (attribY.range.height == m_sidebar->Frame().GetHeight())
			offsetY = 0.f;

		int gHeight = fullHeight / 11;
		bool detail = gHeight / (fontHeight + space) > 6;
		bool detailFull = gHeight / (fontHeight + space) > 11;
		bool detailFullFull = gHeight / (fontHeight + space) > 75;
		bool detailFullFullFull = gHeight / (fontHeight + space) > 150;

		int gridCount = 120 * 10;

		std::vector<UIRect> rootRect(gridCount);
		ArrangeRect({ 0,0, fontWidth, fullHeight }, rootRect);

		gridCount++;
		float pitch = 0.f;
		const auto ppreHeight = preHeight / 4;
		const auto label_baseLineY = baseLineY + 3 - fontCenter;
		const auto other_baseLineY = baseLineY - 3;
		int last_draw = 0;
		for (int i = 0; i < gridCount; ++i)
		{
			//pitch的最后一个刻度
			UIRect rc;
			if(i == 1200)
			{
				rc = rootRect[i - 1];
				rc.Offset(0, -rc.GetHeight());
			}
			else
				rc = rootRect[i];
			rc.Offset(param->destRect->left + space, param->destRect->top + space + fontHeight);
			rc.Offset(0, (int)-offsetY);

			m_brush_m->SetColor(m_lineColor);
			//短刻度
			bool draw;
			bool drawWord;
			if (m_showPitch)
				drawWord = draw = (int)(pitch - 1.f) % 120 == 0 || (detail && (int)(pitch - 1.f) % 20 == 0) || (detailFull && (int)(pitch - 1.f) % 10 == 0) || detailFullFull;
			else
			{
				draw = !(i % 120) ||
					(detail && !(i % 80)) ||
					(detailFull && !(i % 40)) ||
					(detailFullFull && !(i % 20)) ||
					(detailFullFullFull && !(i % 10));
				drawWord = i == 0 || i == 1200 || (draw && rootRect[last_draw].bottom - rootRect[i].bottom > 20);
			}

			if (drawWord)
				last_draw = i;

			if (drawWord)
			{
				UIRect _rc = rc;
				_rc.left = rc.right;
				_rc.right = _rc.left + lineWidth; //param->destRect->right - space - sidebarWidth;
				_rc.bottom = rc.bottom + 1;
				_rc.top = _rc.bottom - 3;
				if (_rc.bottom < other_baseLineY && _rc.top >ppreHeight)
					param->render->FillRectangle(_rc, m_brush_m);
			}

			if (draw)
			{
				UIRect _rc = rc;
				_rc.left = rc.right;
				_rc.right = param->destRect->right - space - sidebarWidth;
				_rc.bottom = rc.bottom;
				_rc.top = _rc.bottom - 1;
				if (_rc.bottom < other_baseLineY && _rc.top >ppreHeight)
				{
					m_brush_m->SetColor(Color::M_RGBA(Color::M_GetRValue(m_lineColor), Color::M_GetGValue(m_lineColor),
						Color::M_GetBValue(m_lineColor), (i - 1) % 10 ? 40 : 100));
					param->render->FillRectangle(_rc, m_brush_m);
				}
			}

			if (drawWord)
			{
				UIRect _rc = rc;
				_rc.Offset(-space, rc.GetHeight() - fontCenter);
				if (_rc.top < label_baseLineY && _rc.bottom > ppreHeight)
				{
					m_brush_m->SetColor(m_fontColor);
					if (m_showPitch)
					{
						if (abs(pitch) < 1e-5)
							m_font->SetText(L"[UV]");
						else
							m_font->SetText(PitchLabel::PitchToLabel(pitch - 1.f));
					}
					else
					{
						if (abs(pitch) < 1e-5)
							m_font->SetText(L"[UV]");
						else
							m_font->SetText(std::to_wstring(PitchLabel::PitchToF0((pitch - 1.f) / 10.f)).substr(0, 5) + L"HZ");
					}
					param->render->DrawTextLayout(m_font, _rc, m_brush_m, TextAlign_Right);
				}
			}
			pitch += 1.f;
		}

		//y轴线
		m_brush_m->SetColor(m_lineColor);
		dst.top = param->destRect->top + space;
		dst.bottom = param->destRect->bottom - preHeight - space - fontCenter;
		dst.left += space + fontWidth;
		//const int dstl = dst.left;
		dst.right = dst.left + lineHeight;
		param->render->FillRectangle(dst, m_brush_m);
		//x轴线
		dst.top = param->destRect->bottom - preHeight - space - fontCenter;
		dst.bottom = dst.top + lineHeight;
		dst.right = param->destRect->right - space - sidebarWidth;
		param->render->FillRectangle(dst, m_brush_m);
		DrawXLabel(scale, param);
	}

	void CurveEditor::DrawXLabel(_m_scale scale, MPCPaintParam param)
	{
		
	}

	void CurveEditor::DrawCurve(_m_scale scale, MPCPaintParam param, const FloatTensor2D& data)
	{
		if (!data.HasValue())
			return;

		UIRect viewRect = m_viewRect;
		viewRect.Offset(param->destRect->left, param->destRect->top);
		int viewHeight = viewRect.GetHeight();
		int viewWidth = viewRect.GetWidth();
		int fullHeight = int((float)viewHeight * m_viewScaleV);

		//重设裁剪区
		param->render->PopClipRect();
		param->render->PushClipRect(viewRect);

#ifdef _WIN32
		auto render = param->render->GetBase<MRender_D2D>();

		auto context = static_cast<ID2D1DeviceContext*>(render->Get());

		context->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

		ID2D1Factory* factory = nullptr;
		context->GetFactory(&factory);

		if (param->cacheCanvas)
		{
			UIRect subrect = param->render->GetCanvas()->GetSubRect();
			viewRect.Offset(subrect.left, subrect.top);
		}

		int width = viewRect.GetWidth();
		width = int((float)width * m_viewScaleH);

		auto attribY = m_sidebar->GetAttribute();
		//视图偏移位置Y
		float offsetY = 0.f;
		if (attribY.dragValue.height != 0)
			offsetY = (float)attribY.dragValue.height / (float)attribY.range.height;
		offsetY = (float)(fullHeight - viewHeight) * (offsetY);
		if (attribY.range.height == m_sidebar->Frame().GetHeight())
			offsetY = 0.f;

		//计算偏移index
		auto attribX = UIScroll::GetAttribute();
		float offsetX = CalcViewHOffset();

		//视图偏移位置X
		if (attribX.dragValue.width != 0)
			offsetX = (float)attribX.dragValue.width / (float)attribX.range.width;
		offsetX = (float)(width - viewWidth) * (offsetX);

		const auto DataPointer = data.Data();
		const auto [Batch, Frames] = data.Size().RawArray();

		float stepX = (float)width / float(Frames - 1);
		float rStepX = 1.f / stepX;
		
		const auto PointRange = 2.f * (float)viewRect.GetWidth() / (float(Frames) / m_viewScaleH);
		const float DrawRangeLeft = (float)viewRect.left - PointRange;
		const float DrawRangeRight = (float)viewRect.right + PointRange;

		constexpr float jump_dis_val = 0.1f;

		m_brush_m->SetColor(m_curveColor);

		auto drawPart = [&factory, this, rStepX, &viewRect, fullHeight, offsetY, DrawRangeLeft, DrawRangeRight, context, &scale](const float* BeginPos, const float* EndPos, float LineWidth, float PosX, ID2D1StrokeStyle* pt_style = nullptr)
			{
				bool figure_begin = false;
				ID2D1PathGeometry* geometry = nullptr;
				factory->CreatePathGeometry(&geometry);

				ID2D1GeometrySink* sink = nullptr;
				geometry->Open(&sink);

				float curIndex = 0.f;
				float size = float(EndPos - BeginPos);

				std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>> jump_discontinuity;
				jump_discontinuity.reserve(size_t(size));
				float x = 0.f;
				float x_prev = 0.f;
				const float stride = std::max(1.f, rStepX);
				for (; int(curIndex) < int(size); curIndex += stride)
				{
					x = (curIndex) / rStepX + PosX;
					//x -= offsetX;

					if (x < DrawRangeLeft)
					{
						x_prev = x;
						continue;
					}
					if (x > DrawRangeRight)
						break;

					const float indexPrev = std::max(0.f, curIndex - rStepX);
					float value = BeginPos[int(curIndex)];
					float value_prev = BeginPos[int(indexPrev)];
					if (value >= MinFreq)
					{
						if (m_showPitch)
							value = (PitchLabel::F0ToPitch(value) + .1f) / 120.f;
						else
							value = value / MaxFreq;
					}
					else
						value = 0.f;
					if (value_prev >= MinFreq)
					{
						if (m_showPitch)
							value_prev = (PitchLabel::F0ToPitch(value_prev) + .1f) / 120.f;
						else
							value_prev = value_prev / MaxFreq;
					}
					else
						value_prev = 0.f;

					float y = (float)viewRect.top + ((float)fullHeight - value * (float)fullHeight);
					y -= offsetY;

					if (abs(value - value_prev) > jump_dis_val)
					{
						if (figure_begin)
						{
							sink->EndFigure(D2D1_FIGURE_END_OPEN);
							append_jump_discontinuity;
							sink->BeginFigure(D2D1::Point2F(x, y), D2D1_FIGURE_BEGIN_FILLED);
						}
						x_prev = x;
						continue;
					}

					if (!figure_begin)
					{
						sink->BeginFigure(D2D1::Point2F(x, y), D2D1_FIGURE_BEGIN_FILLED);
						figure_begin = true;
						if (abs(value_prev) < 1e-5)
							append_jump_discontinuity;
					}
					else
						sink->AddLine(D2D1::Point2F(x, y));

					//如果绘制内容已超出视图区域 结束绘制
					if (x > (float)viewRect.right)
						break;
					x_prev = x;
				}

				if (figure_begin)
					sink->EndFigure(D2D1_FIGURE_END_OPEN);
				sink->Close();

				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m), _scale_to(LineWidth, Helper::M_MIN(scale.cx, scale.cy)), pt_style);

				sink->Release();
				geometry->Release();

				//绘制跳变区域
				factory->CreatePathGeometry(&geometry);
				geometry->Open(&sink);

				for (auto i : jump_discontinuity)
				{
					sink->BeginFigure(i.first, D2D1_FIGURE_BEGIN_FILLED);
					sink->AddLine(i.second);
					sink->EndFigure(D2D1_FIGURE_END_OPEN);
				}

				sink->Close();
				auto AlphaRes = m_brush_m->GetOpacity();
				if (AlphaRes < 2) AlphaRes = 2;
				m_brush_m->SetOpacity(AlphaRes / 2);

				ID2D1StrokeStyle* dashed_style;
				D2D1_STROKE_STYLE_PROPERTIES dashed_properties = D2D1::StrokeStyleProperties();
				dashed_properties.dashStyle = D2D1_DASH_STYLE_DASH;
				factory->CreateStrokeStyle(dashed_properties, nullptr, 0, &dashed_style);
				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m), _scale_to(LineWidth, Helper::M_MIN(scale.cx, scale.cy)), dashed_style);
				m_brush_m->SetOpacity(AlphaRes);

				dashed_style->Release();
				sink->Release();
				geometry->Release();

				return x;
			};

		ID2D1StrokeStyle* MinorStyle = nullptr;
		D2D1_STROKE_STYLE_PROPERTIES MinroProperties = D2D1::StrokeStyleProperties();
		MinroProperties.dashStyle = D2D1_DASH_STYLE_DASH;
		factory->CreateStrokeStyle(MinroProperties, nullptr, 0, &MinorStyle);
		for (int64_t curveIdx = Batch - 1; curveIdx >= 0; --curveIdx)
		{
			ID2D1StrokeStyle* Style = curveIdx == curve_idx ? nullptr : MinorStyle;
			float LineWidth = curveIdx == curve_idx ? 1.f : 0.5;
			const float* CurvePointer = DataPointer + curveIdx * Frames;
			const float* CurveEnd = CurvePointer + Frames;
			auto SelectedRange = GetSelectedRange().RawConst();
			DragonianLib::TemplateLibrary::ConstantRanges<float> Ranges[3];
			if (SelectedRange.Null() || !SelectedRange.Size() || curveIdx != curve_idx)
				Ranges[0] = { CurvePointer, CurveEnd };
			else
			{
				if (SelectedRange.Begin() > CurvePointer)
					Ranges[0] = { CurvePointer, SelectedRange.Begin() };
				Ranges[1] = SelectedRange;
				if (SelectedRange.End() < CurveEnd)
					Ranges[2] = { SelectedRange.End(), CurveEnd };
			}
			float x = (float)viewRect.left - offsetX;
			if (!Ranges[0].Null() && Ranges[0].Size())
				x = drawPart(
					Ranges[0].begin(),
					std::min(Ranges[0].end() + 1, CurveEnd),
					LineWidth, x, Style
				);
			if (!Ranges[1].Null() && Ranges[1].Size())
				x = drawPart(
					Ranges[1].begin(),
					std::min(Ranges[1].end() + 1, CurveEnd),
					LineWidth + 1.5f, x, Style
				);
			if (!Ranges[2].Null() && Ranges[2].Size())
				x = drawPart(
					Ranges[2].begin(),
					Ranges[2].end(),
					LineWidth, x, Style
				);
		}
		MinorStyle->Release();
#endif
		param->render->PopClipRect();
		param->render->PushClipRect(*param->destRect);

#ifdef _WIN32
		const auto PreViewData = curve_idx * m_f0data.Size(1) + m_f0data.Data();

		int preHeight = _scale_to(m_preHeight, scale.cy);
		viewRect = *param->destRect;
		viewRect.top = param->destRect->bottom - preHeight;
		if (param->cacheCanvas)
		{
			UIRect subrect = param->render->GetCanvas()->GetSubRect();
			viewRect.Offset(subrect.left, subrect.top);
		}
		auto drawPreview = [factory, this, &param, &viewRect, PreViewData, context, &scale, Frames, preHeight]()
			{
				ID2D1PathGeometry* geometry = nullptr;

				ID2D1GeometrySink* sink = nullptr;
				factory->CreatePathGeometry(&geometry);
				geometry->Open(&sink);

				float stepX = (float)param->destRect->GetWidth() / float(Frames - 1);
				float rStepX = 1.f / stepX;
				const float stride = std::max(1.f, rStepX);

				float max_val = 0.f, min_val = 50000.f;
				for (float curIndex = 0.f, size = float(Frames); int(curIndex) < int(size); curIndex += stride)
				{
					const auto i = PreViewData[int(curIndex)];
					if (i > max_val) max_val = i;
					if (i < min_val && i > 0.001f) min_val = i;
				}
				if (m_showPitch)
				{
					max_val = PitchLabel::F0ToPitch(max_val) / 120.f;
					min_val = PitchLabel::F0ToPitch(min_val) / 120.f;
				}
				else
				{
					max_val /= MaxFreq;
					min_val /= MaxFreq;
				}
				max_val = Helper::M_MIN(max_val + 0.02f, 1.f);
				min_val = Helper::M_MAX(min_val - 0.02f, 0.f);
				const auto range_pc = (max_val - min_val) / 1.f;

				sink->BeginFigure(D2D1::Point2F((float)viewRect.left, (float)viewRect.bottom), D2D1_FIGURE_BEGIN_FILLED);

				for (float curIndex = 0.f, size = float(Frames); int(curIndex) < int(size); curIndex += stride)
				{
					float value = PreViewData[int(curIndex)];
					if (value >= MinFreq)
					{
						if (m_showPitch)
							value = PitchLabel::F0ToPitch(value) / 120.f;
						else
							value = value / MaxFreq;
					}
					else
						value = 0.f;
					float x = (float)viewRect.left + curIndex / rStepX;
					float y = (float)viewRect.top + ((float)preHeight - ((value - min_val) / range_pc) * (float)preHeight);

					sink->AddLine(D2D1::Point2F(x, y));
				}

				sink->EndFigure(D2D1_FIGURE_END_OPEN);
				sink->Close();
				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m), (float)_scale_to(0.5, scale.cx));

				sink->Release();
				geometry->Release();
			};
		drawPreview();
#endif
	}

	void CurveEditor::DrawPlayeLine(_m_scale scale, MPCPaintParam param)
	{
		if (m_plineOffset == 0)
			return;

		int fullWidth = int((float)m_viewRect.GetWidth() * m_viewScaleH);
		float offset = (float)m_plineOffset / (float)m_f0data.Size(1);
		offset = offset * (float)fullWidth;
		UIRect rc = m_viewRect;
		rc.left += (int)offset;
		rc.right = rc.left + _scale_to(2, scale.cx);
		rc.Offset(param->destRect->left, param->destRect->top);

		//视图偏移位置X
		auto attribX = UIScroll::GetAttribute();
		offset = 0.f;
		if (attribX.dragValue.width != 0)
			offset = (float)attribX.dragValue.width / (float)attribX.range.width;
		offset = (float)(fullWidth - m_viewRect.GetWidth()) * offset;
		rc.Offset((int)-offset, 0);

		UIRect viewRect = m_viewRect;
		viewRect.Offset(param->destRect->left, param->destRect->top);
		if(Helper::M_IsRectCross(viewRect, rc))
		{
			m_brush_m->SetColor(Color::M_RED);
			param->render->FillRectangle(rc, m_brush_m);
		}

		m_brush_m->SetColor(m_lineColor);

		//预览图指针
		offset = (float)m_plineOffset / (float)m_f0data.Size(1);
		offset = offset * (float)param->destRect->GetWidth();
		rc = *param->destRect;
		rc.top = rc.bottom - _scale_to(m_preHeight, scale.cy);
		rc.left += (int)offset;
		rc.right = rc.left + _scale_to(2, scale.cx);
		param->render->FillRectangle(rc, m_brush_m);
	}

	void CurveEditor::CalcRangeViewV()
	{
		int height = m_sidebar->Frame().GetHeight();

		const auto& attrib = m_sidebar->GetAttribute();

		int range = _scale_to(height, m_viewScaleV);
		int drag = 0;
		if (attrib.dragValue.height != 0)
			drag = int((float)attrib.dragValue.height / (float)attrib.range.height * (float)range);

		if (m_viewScaleV == 1.f)
			drag = 0;

		m_sidebar->SetAttributeSrc(L"rangeV", range, false);
		m_sidebar->SetAttributeSrc(L"dragValueV", drag, false);
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::CalcRangeViewH()
	{
		int width = UINodeBase::m_data.Frame.GetWidth();

		const auto& attrib = UIScroll::GetAttribute();

		int range = _scale_to(width, m_viewScaleH);
		int drag = 0;
		if (attrib.dragValue.width != 0)
			drag = int((float)attrib.dragValue.width / (float)attrib.range.width * (float)range);

		if (m_viewScaleH == 1.f)
			drag = 0;

		UIScroll::SetAttributeSrc(L"rangeH", range, false);
		UIScroll::SetAttributeSrc(L"dragValueH", drag, true);

		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::CalcRangeViewH(const UIPoint& point, short delta)
	{
		const int width = UINodeBase::m_data.Frame.GetWidth();
		const int LWidth = point.x - UINodeBase::m_data.Frame.left;
		const auto& attrib = UIScroll::GetAttribute();
		const int range = _scale_to(width, m_viewScaleH);
		int drag = int((int64_t)attrib.dragValue.width * (int64_t)range / (int64_t)attrib.range.width);
		if (delta > 0) drag += (LWidth - width / 2) / 2;
		if (drag < 0) drag = 0;
		if (drag > range) drag = range;

		UIScroll::SetAttributeSrc(L"rangeH", range, false);
		UIScroll::SetAttributeSrc(L"dragValueH", drag, true);

		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void CurveEditor::CalcViewRect()
	{
		m_font->SetText(L"C0");

		const auto [cx, cy] = GetRectScale().scale();
		const int height = UINodeBase::m_data.Frame.GetHeight();
		const int space = _scale_to(5, cy);
		const int preHeight = _scale_to(m_preHeight, cy);
		const int sidebarWidth = _scale_to(m_barWidth, cx);
		const int fontHeight = m_font->GetMetrics().bottom;
		const int fontCenter = fontHeight / 2;
		constexpr int fontWidth = 60;
		//曲线视图高度
		UIRect viewRect = 
		{
			fontWidth + space,
			space + fontHeight,
			0,
			height - preHeight - space - fontCenter - space - fontHeight
		};
		viewRect.right = UINodeBase::m_data.Frame.GetWidth() - sidebarWidth - space;
		m_viewRect = viewRect;
	}

	float CurveEditor::CalcViewHOffset(int drag)
	{
		int viewWidth = m_viewRect.GetWidth();
		const auto& attrib = UIScroll::GetAttribute();

		if (attrib.dragValue.width == 0 && drag == 0)
			return 0.f;

		if (drag == 0)
			drag = attrib.dragValue.width;

		float newRange = (float)viewWidth * m_viewScaleH;
		float newDrag = (float)drag / (float)attrib.range.width * newRange;

		float percentage = newDrag / newRange;
		float value = (newRange - (float)viewWidth) * percentage;
		return value / newRange;
	}

	void CurveEditor::ArrangeRect(const UIRect& rect, std::vector<UIRect>& dst) const
	{
		const int numRects = (int)dst.size();
		const int rectWidth = rect.GetWidth();

		if (m_showPitch)
		{
			float rectHeight = (float)rect.GetHeight() / (float)numRects;
			float lastBottom = 0.f;
			for (int i = 0; i < numRects; ++i) 
			{

				dst[i].right = rectWidth;
				dst[i].top = (int)lastBottom;
				dst[i].bottom = (int)round(lastBottom + rectHeight);
			
				lastBottom = lastBottom + rectHeight;
			}
			std::ranges::reverse(dst);
		}
		else
		{
			const auto RectHeight = (float)rect.GetHeight();
			float Pitch = 0.f;
			dst[0].right = rectWidth;
			dst[0].top = static_cast<int>((MaxFreq - MinFreq) / MaxFreq * RectHeight);
			dst[0].bottom = static_cast<int>(RectHeight);
			for (int i = 1; i < numRects; ++i)
			{
				const auto CurFreq = MaxFreq - PitchLabel::PitchToF0(Pitch);
				const auto NextFreq = MaxFreq - PitchLabel::PitchToF0(Pitch + 0.1f);
				const auto Offset = static_cast<int>(CurFreq / MaxFreq * RectHeight);
				const auto Next = static_cast<int>(NextFreq / MaxFreq * RectHeight);
				dst[i].right = rectWidth;
				dst[i].top = Next;
				dst[i].bottom = Offset;

				Pitch += 0.1f;
			}
		}
	}
}