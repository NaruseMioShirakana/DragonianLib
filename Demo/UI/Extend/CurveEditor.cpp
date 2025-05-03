#include "CurveEditor.h"
#include <random>
#include "../DefControl.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

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

namespace SimpleF0Labeler
{
	using namespace Mui;

	static bool IsInRect(const UIRect& Rect, const UIPoint& Point)
	{
		return Point.x >= Rect.left && Point.x <= Rect.right &&
			Point.y >= Rect.top && Point.y <= Rect.bottom;
	}

	const auto MaxFreq = PitchLabel::PitchToF0(119.9f);
	const auto MinFreq = PitchLabel::PitchToF0(0);

	class MD2DBrush final : Render::MBrush_D2D
	{
	public:
		static ID2D1SolidColorBrush* GetBrush(MBrush* brush)
		{
			return static_cast<MD2DBrush*>(brush)->m_brush;
		}
	};

	static _m_color GetColor(const std::wstring& Symbol)
	{
		const auto Color = WndControls::Localization(Symbol);
		static const std::wregex Reg(LR"([ ]?([0-9]+)[ ]?,[ ]?([0-9]+)[ ]?,[ ]?([0-9]+)[ ]?,[ ]?([0-9]+)[ ]?)");
		std::wsmatch Mat;
		if (std::regex_match(Color, Mat, Reg))
			return Color::M_RGBA(
				(_m_byte)std::wcstol(Mat[1].str().c_str(), nullptr, 10),
				(_m_byte)std::wcstol(Mat[2].str().c_str(), nullptr, 10),
				(_m_byte)std::wcstol(Mat[3].str().c_str(), nullptr, 10),
				(_m_byte)std::wcstol(Mat[4].str().c_str(), nullptr, 10)
			);
		throw MError("Illegal color text!");
	}

	void CurveEditor::Register()
	{
		Mui::XML::RegisterControl(
			ClassName,
			[](UIControl* parent) { return new CurveEditor(parent); }
		);
	}

	CurveEditor::CurveEditor(UIControl* parent)
	{
		parent->AddChildren(this);
		m_anicls = std::make_shared<MAnimation>(UIControl::GetParentWin());

		Horizontal = true;
		BarWidth = m_preHeight;

		Callback = [this](UIScroll* PH1, int PH2, bool PH3) {OnScrollView(PH1, PH2, PH3); };

		m_sidebar = new UIScroll(this);
		m_sidebar->Vertical = true;
		m_sidebar->Button = false;
		m_sidebar->BarWidth = m_barWidth;
		m_sidebar->SetVisible(true, true);

		m_sidebar->AutoSize(false, false);
		PosSizeUnit uint;
		uint.x_w = Percentage;
		uint.y_h = FillMinus;
		m_sidebar->SetSizeUnit(uint, false);
		m_sidebar->SetSize(100, m_preHeight, false);
		m_sidebar->SetMsgFilter(true);
		HMIDIOUT MidioutH;
		MidiOutOpen = midiOutOpen(&MidioutH, 0, 0, 0, CALLBACK_NULL) == MMSYSERR_NOERROR;
		if (MidiOutOpen)
			MidiOutHandle = MidioutH;
		else
			MidiOutHandle = nullptr;
		m_backColor = GetColor(L"WindowBackGroundColor");
		for (int i = 0; i < 10; ++i)
			m_curveColor[i] = GetColor(L"CurveColor" + std::to_wstring(i));
	}

	CurveEditor::~CurveEditor()
	{
		if(MidiOutOpen && MidiOutHandle)
		{
			midiOutClose(HMIDIOUT(MidiOutHandle));
			MidiOutOpen = false;
		}
		selected_f0_begin = selected_f0_end = nullptr;
		SetCurveData(std::nullopt, std::nullopt, std::nullopt);
	}

	void CurveEditor::SetCurveData(const FloatTensor2D& data, const ImageTensor& spec, const ImageTensor& spec_logview)
	{
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
			if (spec_logview.HasValue())
				m_specLogView = spec_logview.View();
			else
				m_specLogView = std::nullopt;
			selected_f0_begin = selected_f0_end = nullptr;
		}
		m_viewScaleH = 1.f;
		CalcRangeViewH();
		UpDate();
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
		{
			std::lock_guard lock(mx);
			if (idx != curve_idx)
			{
				curve_idx = idx;
				selected_f0_begin = selected_f0_end = nullptr;
			}
		}
		UpDate();
	}

	void CurveEditor::SetShowPitch(bool show)
	{
		m_showPitch = show;
		UpDate();
	}

	void CurveEditor::SetPlayLinePos(_m_size offset)
	{
		if (m_f0data.HasValue())
			offset = std::min(offset, static_cast<size_t>(m_f0data.Size(1) - 1));
		m_plineOffset = offset;
		UpDate();
	}

	void CurveEditor::UpDate()
	{
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	DragonianLib::TemplateLibrary::MutableRanges<float> CurveEditor::GetSelectedRange() const
	{
		auto begin = selected_f0_begin < selected_f0_end ? selected_f0_begin : selected_f0_end;
		auto end = selected_f0_begin < selected_f0_end ? selected_f0_end : selected_f0_begin;
		return { begin, end };
	}

	void CurveEditor::UPRButton()
	{
		if (m_isdown && m_f0data.HasValue())
		{
			m_isdown = false;
			WndControls::CheckUnchanged();
		}
	}

	bool CurveEditor::SetAttribute(Mui::XML::PropName attribName, std::wstring_view attrib, bool draw)
	{
		if (attribName == L"FontColor")
		{
			m_fontColor = Helper::M_GetAttribValueColor(attrib);
		}
		else if (attribName == L"LineColor")
		{
			m_lineColor = Helper::M_GetAttribValueColor(attrib);
		}
		else if (attribName == L"PreHeight")
		{
			m_preHeight = Helper::M_StoInt(attrib);
			BarWidth = m_preHeight;
		}
		else if (attribName == L"FontName")
		{
			m_font->SetFontName({ attrib.data(), attrib.size() });
		}
		else if (attribName == L"FontSize")
		{
			_m_scale scale = GetRectScale().scale();
			std::wstring attrstr = { attrib.data(), attrib.size() };
			m_fontSize = Helper::M_StoInt(attrstr);
			const float fontSize = Helper::M_MIN(scale.cx, scale.cy) * (float)m_fontSize;
			m_font->SetFontSize((_m_uint)fontSize, std::make_pair(0u, (_m_uint)m_font->GetText().length()));
		}
		else
		{
			if (attribName == L"StyleV")
				return m_sidebar->SetAttribute(attribName, attrib, false);
			return UIScroll::SetAttribute(attribName, attrib, draw);
		}
		if (draw)
			UpDate();
		return true;
	}

	std::wstring CurveEditor::GetAttribute(Mui::XML::PropName attribName)
	{
		if (attribName == L"FontColor")
			return Color::M_RGBA_STR(m_fontColor);
		if (attribName == L"LineColor")
			return Color::M_RGBA_STR(m_lineColor);
		if (attribName == L"PreHeight")
			return std::to_wstring(m_preHeight);
		if (attribName == L"FontName")
			return m_fontName;
		if (attribName == L"FontSize")
			return std::to_wstring(m_fontSize);
		return UIScroll::GetAttribute(attribName);
	}

	void CurveEditor::OnScale(_m_scale scale)
	{
		UIScroll::OnScale(scale);
		const float fontSize = Helper::M_MIN(scale.cx, scale.cy) * (float)m_fontSize;
		m_font->SetFontSize((_m_uint)fontSize, std::make_pair(0u, (_m_uint)m_font->GetText().length()));
	}

	void CurveEditor::OnLoadResource(Render::MRenderCmd* render, bool recreate)
	{
		UIScroll::OnLoadResource(render, recreate);

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
			pt.y -= static_cast<int>(UINodeBase::m_data.Frame.top);
			//视图偏移位置Y
			const auto scale = GetRectScale().scale();
			const int space = _scale_to(5, scale.cy);
			const int viewHeight = m_viewRect.GetHeight();
			const int fontHeight = m_font->GetMetrics().bottom;
			const int fullHeight = int((float)viewHeight * m_viewScaleV);
			float offsetY = 0.f;
			if (m_sidebar->DragValue.Get().y != 0)
				offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
			offsetY = (float)(fullHeight - viewHeight) * (offsetY);
			if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
				offsetY = 0.f;
			float value = ((float)(pt.y - space - fontHeight) + offsetY) / (float)fullHeight;
			value = Helper::M_Clamp(0.f, 120.f, 119.9f - value * 120.f);
			auto Pitch = DWORD(round(value));
			Pitch = std::min(Pitch, DWORD(0x7f));
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

		const int space = _scale_to(5, scale.cy);
		const int fontHeight = m_font->GetMetrics().bottom;
		const int preHeight = _scale_to(m_preHeight, scale.cy);
		const int fontCenter = fontHeight / 2;
		const auto bot = param->destRect->bottom - preHeight - space - fontCenter;
		const auto top = param->destRect->top + space;

		if (m_specData.HasValue())
			DrawSpec(scale, param);
		
		if (const auto Range = GetSelectedRange(); !Range.Null() && Range.Size())
		{
			m_brush_m->SetColor(m_fontColor);
			m_brush_m->SetOpacity(25);
			
			const auto X_POS = std::max(CalcXPosWithPtr(param, Range.begin()), m_viewRect.left);
			const auto X_L_POS = std::max(CalcXPosWithPtr(param, Range.end()), m_viewRect.left);
			if (X_POS < X_L_POS)
				param->render->FillRectangle(
					UIRect{ X_POS, top, X_L_POS - X_POS, bot - top }.ToRect(),
					m_brush_m
				);
		}

		auto Rect = UINodeBase::m_data.Frame.ToRectT<int>();
		Rect.left += m_viewRect.left;
		UIPoint CursorPoint;
		if (GetCursorPos((LPPOINT)&CursorPoint) && ScreenToClient((HWND)GetParentWin()->GetWindowHandle(), (LPPOINT)&CursorPoint) &&
			IsInRect(Rect, CursorPoint))
		{
			CursorPoint.x -= static_cast<int>(UINodeBase::m_data.Frame.left);
			CursorPoint.y -= static_cast<int>(UINodeBase::m_data.Frame.top);
			m_brush_m->SetColor(m_fontColor);
			m_brush_m->SetOpacity(255);
			const auto x = std::max(0, CursorPoint.x);
			param->render->FillRectangle(
				UIRect{ x, top, _scale_to(2, scale.cx), bot - top }.ToRect(),
				m_brush_m
			);
			if (m_f0data.HasValue())
			{
				const auto xFpOff = GetFpOffset(static_cast<float>(x - m_viewRect.left));
				const auto xOff = (size_t)std::round(xFpOff);
				const auto xOffCeil = (size_t)std::ceil(xFpOff);
				const auto xOffFloor = (size_t)std::floor(xFpOff);
				const auto xOffCeilVal = m_f0data.Data() + xOffCeil + curve_idx * m_f0data.Size(1);
				const auto xOffFloorVal = m_f0data.Data() + xOffFloor + curve_idx * m_f0data.Size(1);
				//interp
				const auto yVal = (*xOffCeilVal - *xOffFloorVal) * (xFpOff - static_cast<float>(xOffFloor)) + *xOffFloorVal;
				m_font->SetText(std::to_wstring(xOff));
				param->render->DrawTextLayout(
					m_font,
					UIRect{
						x - _scale_to(3, scale.cx) - m_font->GetMetrics().GetWidth(),
						top,
						m_font->GetMetrics().GetWidth(),
						fontHeight
					}.ToRect(),
					m_brush_m,
					TextAlign_Right
				);
				m_font->SetText(std::to_wstring(yVal).substr(0, 7) + L"HZ");
				param->render->DrawTextLayout(
					m_font,
					UIRect{
						x + space,
						top,
						m_font->GetMetrics().GetWidth(),
						fontHeight
					}.ToRect(),
					m_brush_m,
					TextAlign_Left
				);

				int viewHeight = m_viewRect.GetHeight();
				int fullHeight = int((float)viewHeight * m_viewScaleV);
				float offsetY = 0.f;
				if (m_sidebar->DragValue.Get().y != 0)
					offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
				offsetY = (float)(fullHeight - viewHeight) * (offsetY);
				if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
					offsetY = 0.f;
				float value = ((float)(CursorPoint.y - space - fontHeight) + offsetY) / (float)fullHeight;

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

				m_font->SetText(std::to_wstring(value).substr(0, 7) + L"HZ");
				param->render->DrawTextLayout(
					m_font,
					UIRect{
						x + space,
						bot - fontHeight,
						m_font->GetMetrics().GetWidth(),
						fontHeight
					}.ToRect(),
					m_brush_m,
					TextAlign_Left
				);
			}
		}

		DrawLabel(scale, param);

		if (m_f0data.HasValue())
			DrawCurve(scale, param);

		DrawPlayeLine(scale, param);

		UIScroll::OnPaintProc(param);
	}

	bool CurveEditor::OnMouseWheel(_m_uint flag, short delta, const UIPoint& point)
	{
		if (UIScroll::OnMouseWheel(flag, delta, point))
			return true;
		UIRect barFrame = m_sidebar->Frame();
		barFrame.left = barFrame.right - _scale_to(m_barWidth, GetRectScale().scale().cx);
		if (IsInRect(barFrame, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - static_cast<int>(UINodeBase::m_data.Frame.left), 0),
			std::max(point.y - static_cast<int>(UINodeBase::m_data.Frame.top), 0)
		};

		if (m_end_insel || m_begin_insel)
		{
			auto x = pt.x - m_viewRect.left;
			x -= m_viewRect.left;
			x = std::max(0, x);
			auto xoff = GetXOffset(static_cast<float>(x));
			if (m_end_insel)
				selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
			else if (m_begin_insel)
				selected_f0_begin = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
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
			const auto max_ScaleH = float(m_f0data.Size(1)) / 50.f;
			const auto ScaleH = m_viewScaleH * powf(4.2f, delta_ * 0.1f);
			m_viewScaleH = Helper::M_MAX(Helper::M_MIN(max_ScaleH, ScaleH) * 1.f, 1.f); //最大20000%最小100%
			if (m_viewScaleH >= max_ScaleH || m_viewScaleH <= 1.f) CalcRangeViewH();
			else CalcRangeViewH(point, delta);
		}
		else if(GetKeyState(VK_LMENU) & 0x8000)
		{
			m_viewScaleV *= powf(2.2f, delta_ * 0.1f);
			m_viewScaleV = Helper::M_MAX(Helper::M_MIN(50.f, m_viewScaleV) * 1.f, 1.f); //最大1000%最小100%
			CalcRangeViewV();
		}
		else if (GetKeyState(VK_LSHIFT) & 0x8000)
		{
			const int range = Range.Get().width;
			int step = int(float(range) / (m_viewScaleH * 3));
			if (step == 0) step = 1;
			const int curval = DragValue.Get().x;
			int val = curval + int(-delta_ * float(step));
			val = std::clamp(val, 0, range);
			DragValue.Set().value->x = val;
		}
		else
		{
			const int range = m_sidebar->Range.Get().height;
			int step = int(float(range) / (3 * m_viewScaleV));
			if (step == 0) step = 1;
			const int curval = m_sidebar->DragValue.Get().y;
			int val = curval + int(-delta_ * float(step));
			val = std::clamp(val, 0, range);
			m_sidebar->DragValue.Set().value->y = val;
			DragValue.Set().value->y = val;
		}
		UpDate();
		return false;
	}

	bool CurveEditor::OnMouseMove(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnMouseMove(flag, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - static_cast<int>(UINodeBase::m_data.Frame.left), 0),
			std::max(point.y - static_cast<int>(UINodeBase::m_data.Frame.top), 0)
		};

		if (m_lisdown && GetKeyState(VK_TAB) & 0x8000)
			PlaySoundPitch(point);

		
#ifdef _WIN32
		if (const auto Range = GetSelectedRange(); GetKeyState(VK_LSHIFT) & 0x8000 && !Range.Null() && Range.Size())
		{
			const auto Begin = m_f0data.Data() + curve_idx * m_f0data.Size(1);
			const auto XPOS = static_cast<ptrdiff_t>(GetXOffset(static_cast<float>(pt.x - m_viewRect.left)));
			const auto BeginIdx = std::distance(Begin, Range.begin());
			const auto EndIdx = std::distance(Begin, Range.end());

			const auto curMinFrameFp = GetFpOffset(0.f);
			const auto curMaxFrameFp = GetFpOffset(float(m_viewRect.GetWidth()));
			const auto curPixPerFrame = float(m_viewRect.GetWidth()) / (curMaxFrameFp - curMinFrameFp);

			const auto FrameCount = std::max(
				int64_t(10.f / curPixPerFrame),
				1ll
			);
			if (abs(BeginIdx - XPOS) < FrameCount || abs(EndIdx - XPOS) < FrameCount)
				SetCursor(IDC_SIZEWE);
			else
				goto III_ARR;
		}
		else
		{
			III_ARR:
			SetCursor(IDC_ARROW);
		}
#endif

		if (time_in_sel && m_f0data.HasValue())
		{
			const auto x = pt.x - m_viewRect.left;
			auto xof = GetXOffset(static_cast<float>(x));
			xof = xof * WndControls::GetPcmSize() / (m_f0data.Size(1) - 1);
			WndControls::SetPlayerPos(xof);
		}

		if (m_end_insel || m_begin_insel)
		{
			auto x = pt.x - m_viewRect.left;
			x = std::max(0, x);
			auto xoff = GetXOffset(static_cast<float>(x));
			if (m_end_insel)
				selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
			else if (m_begin_insel)
				selected_f0_begin = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
		}

		else if (m_isdown && m_f0data.HasValue())
		{
			std::lock_guard lock(mx);
			auto y = pt.y, x = pt.x;
#ifdef _WIN32
			if (GetKeyState(VK_LSHIFT) & 0x8000)
				y = m_lastPos.y;
#endif

			//视图偏移位置Y
			auto scale = GetRectScale().scale();
			int space = _scale_to(5, scale.cy);
			int viewHeight = m_viewRect.GetHeight();
			int fontHeight = m_font->GetMetrics().bottom;
			int fullHeight = int((float)viewHeight * m_viewScaleV);

			float offsetY = 0.f;
			if (m_sidebar->DragValue.Get().y != 0)
				offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
			offsetY = (float)(fullHeight - viewHeight) * (offsetY);
			if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
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
			x = std::max(0, x);
			m_lastPos.x = std::max(0, m_lastPos.x);
			int begin = m_lastPos.x < x ? m_lastPos.x : x;
			int end = x > m_lastPos.x ? x : m_lastPos.x;

			if (end <= begin)
				end = begin + 3;

			auto xbegin = GetXOffset(static_cast<float>(begin));
			auto xend = GetXOffset(static_cast<float>(end));
			const auto m_f0data_data = m_f0data.Data() + curve_idx * m_f0data.Size(1);
			auto count = xend - xbegin;

			const auto range_sel = GetSelectedRange();

			for (size_t i = 0; i < count; ++i)
				if (xbegin + i < static_cast<size_t>(m_f0data.Size(1)))
					if (auto cur_data_ptr = &m_f0data_data[xbegin + i]; range_sel.Contains(cur_data_ptr, true))
						*cur_data_ptr = value;

			if (count == 0 && std::cmp_less(xbegin, m_f0data.Size(1)))
				if (auto cur_data_ptr = &m_f0data_data[xbegin]; range_sel.Contains(cur_data_ptr, true))
					*cur_data_ptr = value;

			m_lastPos = { x, y };
		}
		UpDate();
		return false;
	}

	bool CurveEditor::OnRButtonDown(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnRButtonDown(flag, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - static_cast<int>(UINodeBase::m_data.Frame.left), 0),
			std::max(point.y - static_cast<int>(UINodeBase::m_data.Frame.top), 0)
		};

		if (m_f0data.HasValue() && IsInRect(m_viewRect, pt))
		{
			WndControls::AppendUndo();
			m_isdown = true;
			m_lastPos = { pt.x - m_viewRect.left, pt.y };
		}
		return false;
	}

	bool CurveEditor::OnRButtonUp(_m_uint flag, const UIPoint& point)
	{
		UPRButton();

		return UIScroll::OnRButtonUp(flag, point);
	}

	bool CurveEditor::OnLButtonDown(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnLButtonDown(flag, point))
			return true;

		const UIPoint pt = {
			std::max(point.x - static_cast<int>(UINodeBase::m_data.Frame.left), 0),
			std::max(point.y - static_cast<int>(UINodeBase::m_data.Frame.top), 0)
		};

		if (GetKeyState(VK_TAB) & 0x8000)
		{
			if (IsInRect(m_viewRect, pt))
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
				auto x = static_cast<float>(pt.x);
				x -= static_cast<float>(m_viewRect.left);
				x = std::max(0.f, x);
				auto xoff = static_cast<ptrdiff_t>(GetXOffset(x));
				if (const auto Range = GetSelectedRange(); !Range.Null() && Range.Size())
				{
					const auto begin = m_f0data.Data() + curve_idx * m_f0data.Size(1);
					const auto beginIdx = std::distance(begin, Range.begin());
					const auto endIdx = std::distance(begin, Range.end());

					const auto curMinFrameFp = GetFpOffset(0.f);
					const auto curMaxFrameFp = GetFpOffset(float(m_viewRect.GetWidth()));
					const auto curPixPerFrame = float(m_viewRect.GetWidth()) / (curMaxFrameFp - curMinFrameFp);

					const auto frameCount = std::max(
						int64_t(10.f / curPixPerFrame),
						1ll
					);
					if (abs(beginIdx - xoff) < frameCount)
						m_begin_insel = true;
					else if (abs(endIdx - xoff) < frameCount)
						m_end_insel = true;
				}
				else
				{
					m_end_insel = true;
					selected_f0_begin = selected_f0_end = m_f0data.Data() + xoff + curve_idx * m_f0data.Size(1);
				}
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
		if (m_f0data.HasValue())
		{
			if ((m_end_insel || m_begin_insel) && selected_f0_begin == selected_f0_end)
				selected_f0_begin = selected_f0_end = nullptr;
			if (time_in_sel)
			{
				auto x = static_cast<float>(point.x);
				x -= UINodeBase::m_data.Frame.left;
				x -= static_cast<float>(m_viewRect.left);
				auto xof = GetXOffset(x);
				xof = xof * WndControls::GetPcmSize() / (m_f0data.Size(1) - 1);
				WndControls::SetPlayerPos(xof);
			}
		}
		m_end_insel = m_begin_insel = false;
		m_lisdown = false;
		time_in_sel = false;
		in_move = false;
		constexpr DWORD MIDOOUT_MIDIKEYRELEASE = (0XFF << 8) + 0x90;
		midiOutShortMsg(HMIDIOUT(MidiOutHandle), MIDOOUT_MIDIKEYRELEASE);
		LastMidiPitch = 256;
		UpDate();
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

	static char VK_Nums[]{ L'1',L'2',L'3',L'4',L'5',L'6',L'7',L'8',L'9' };

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
				}
				else if (GetKeyState('W') & 0x8000)
				{
					if (Exec)
					{
						WndControls::AppendUndo();
						WndControls::ApplyPitchShift(GetSelectedRange());
						WndControls::CheckUnchanged();
					}
				}
				else if (GetKeyState('E') & 0x8000)
				{
					if (Exec)
					{
						WndControls::AppendUndo();
						WndControls::ApplyCalc(GetSelectedRange());
						WndControls::CheckUnchanged();
					}
				}
				else if (GetKeyState('A') & 0x8000)
				{
					selected_f0_begin = m_f0data.Data() + curve_idx * m_f0data.Size(1);
					selected_f0_end = m_f0data.Data() + (curve_idx + 1) * m_f0data.Size(1);
				}
				else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('Z') & 0x8000)
					WndControls::MoeVSUndo();
				else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('Y') & 0x8000)
					WndControls::MoeVSRedo();
				else if (GetKeyState(VK_SPACE) & 0x8000)
					WndControls::PlayPause();
				else if (GetKeyState(VK_LCONTROL) & 0x8000 && GetKeyState('S') & 0x8000)
					WndControls::SaveData();
				else if (GetKeyState('0') & 0x8000)
					SetCurveIndex(9);
				else
				{
					return std::ranges::any_of(
						VK_Nums,
						[&](char k) {
							if (GetKeyState(k) & 0x8000)
							{
								SetCurveIndex(k - '1');
								return true;
							}
							return false;
						}
					);
				}
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
			}
			else
			{
				goto EndLabel;
			}
			UpDate();
			return true;
		}

		EndLabel:
		return UIScroll::OnWindowMessage(code, wParam, lParam);
	}

	bool CurveEditor::OnSetCursor(Mui::_m_param hCur, Mui::_m_param lParam)
	{
#ifdef _WIN32
		::SetCursor((HCURSOR)hCur);
#endif
		return true;
	}

	bool CurveEditor::OnMouseExited(_m_uint flag, const UIPoint& point)
	{
		UIScroll::OnMouseExited(flag, point);
		if (m_f0data.HasValue() && (m_end_insel || m_begin_insel))
			if (selected_f0_begin == selected_f0_end)
				selected_f0_begin = selected_f0_end = nullptr;
		m_end_insel = m_begin_insel = false;
		time_in_sel = false;

		UPRButton();

		in_move = false;
		return false;
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
		int fontWidth = _scale_to(60, scale.cx);
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
		if (m_sidebar->DragValue.Get().y != 0)
			offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
		offsetY = (float)(fullHeight - viewHeight) * (offsetY);
		if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
			offsetY = 0.f;

		int gHeight = fullHeight / 11;
		bool detail = gHeight / (fontHeight + space) > 6;
		bool detailFull = gHeight / (fontHeight + space) > 11;
		bool detailFullFull = gHeight / (fontHeight + space) > 75;
		bool detailFullFullFull = gHeight / (fontHeight + space) > 150;

		int gridCount = 120 * 10;

		std::vector<UIRect> rootRect(gridCount);
		ArrangeRect({ 0,0, fontWidth, fullHeight }, rootRect);

		++gridCount;
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
					param->render->FillRectangle(_rc.ToRect(), m_brush_m);
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
					param->render->FillRectangle(_rc.ToRect(), m_brush_m);
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
					param->render->DrawTextLayout(m_font, _rc.ToRect(), m_brush_m, TextAlign_Right);
				}
			}
			pitch += 1.f;
		}

		//y轴线
		m_brush_m->SetColor(m_lineColor);
		dst.top = param->destRect->top + space + fontHeight - 1;
		dst.bottom = param->destRect->bottom - preHeight - space - fontCenter;
		dst.left += space + fontWidth;
		//const int dstl = dst.left;
		dst.right = dst.left + lineHeight;
		param->render->FillRectangle(dst.ToRect(), m_brush_m);
		//x轴线
		dst.top = param->destRect->top + space + fontHeight - 1;
		dst.bottom = dst.top + lineHeight;
		dst.right = param->destRect->right - space - sidebarWidth;
		param->render->FillRectangle(dst.ToRect(), m_brush_m);

		dst.top = param->destRect->bottom - preHeight - space - fontCenter;
		dst.bottom = dst.top + lineHeight;
		
		param->render->FillRectangle(dst.ToRect(), m_brush_m);
	}

	void CurveEditor::DrawSpec(_m_scale scale, MPCPaintParam param)
	{
		constexpr float MaxSpec = 8000.f;
		const auto F0Frames = m_f0data.Size(1);
		const auto SpecFrames = m_specData.Size(1);
		const auto SpecBins = m_specData.Size(0);
		const auto FreqPerBin = MaxSpec / static_cast<float>(SpecBins - 1);
		const auto FrameScale = static_cast<float>(SpecFrames) / static_cast<float>(F0Frames);

		const auto space = _scale_to(5, scale.cy);
		const auto viewWidth = m_viewRect.GetWidth();
		const auto viewHeight = m_viewRect.GetHeight();
		const auto fontHeight = m_font->GetMetrics().bottom;
		const auto fullHeight = int((float)viewHeight * m_viewScaleV);
		const auto fontWidth = _scale_to(60, scale.cx);
		const auto preHeight = _scale_to(m_preHeight, scale.cy);
		const auto sidebarWidth = _scale_to(m_barWidth, scale.cx);
		const auto fontCenter = fontHeight / 2;

		const auto curMinFrameFp = GetFpOffset(0.f) * FrameScale;
		const auto curMaxFrameFp = GetFpOffset(float(viewWidth)) * FrameScale;
		const auto curPixPerFrame = float(viewWidth) / (curMaxFrameFp - curMinFrameFp);
		const auto curCenterFactorX = abs(curMinFrameFp) < 1.f ? 0 : int(0.5f * curPixPerFrame);
		const auto curFrameStride = (int64_t)round(std::max(1.f / curPixPerFrame, 1.f));

		const auto curMinFrameBin = curMinFrameFp;
		const auto curMaxFrameBin = std::min(curMaxFrameFp + (curCenterFactorX ? 1.f : 0.f), float(SpecFrames));

		const auto curMinBinFactorX = int((curMinFrameBin - floor(curMinFrameBin)) * curPixPerFrame);
		const auto curMaxBinFactorX = int((ceil(curMaxFrameBin) - curMaxFrameBin + (curCenterFactorX ? 1.f : 0.f)) * curPixPerFrame);

		float offsetY = 0.f;
		if (m_sidebar->DragValue.Get().y != 0)
			offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
		offsetY = (float)(fullHeight - viewHeight) * (offsetY);
		if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
			offsetY = 0.f;

		if (!m_specData.HasValue())
			return;
		if (m_showPitch)
		{
			

		}
		else
		{
			const auto curMaxFreq = MaxFreq * (1.f - offsetY / (float)fullHeight);
			const auto curMinFreq = MaxFreq * (1.f - (offsetY + float(viewHeight)) / (float)fullHeight);
			const auto curPixPerFreq = float(viewHeight) / (curMaxFreq - curMinFreq);

			if (curMinFreq > MaxSpec) return;
			const auto curMinBinFp = std::min(curMaxFreq, MaxSpec) / FreqPerBin;
			const auto curMaxBinFp = std::max(0.f, curMinFreq) / FreqPerBin;
			const auto curPixPerBin = float(viewHeight) / (curMinBinFp - curMaxBinFp);
			const auto curCenterFactorY = int(FreqPerBin / 2 * curPixPerFreq);

			const auto curMinBin = (float)SpecBins - curMinBinFp - (curCenterFactorY ? 1.f : 0.f);
			const auto curMaxBin = (float)SpecBins - curMaxBinFp;

			const auto curMinBinFactorY = int((curMinBin - floor(curMinBin) + (curCenterFactorY ? 1.f : 0.f)) * curPixPerBin);
			const auto curMaxBinFactorY = int((ceil(curMaxBin) - curMaxBin) * curPixPerBin);

			auto SpecData = m_specData.Slice(
				{
					DragonianLib::Range{ (int64_t)floor(curMinBin), (int64_t)ceil(curMaxBin) },
					DragonianLib::Range{ (int64_t)floor(curMinFrameBin), curFrameStride, (int64_t)ceil(curMaxFrameBin) }
				}
			).Contiguous().Evaluate();
			const auto [NewBins, NewFrames] = SpecData.Size().RawArray();
			Render::MBitmapPtr bitmap = param->render->CreateBitmap(
				static_cast<_m_uint>(NewFrames),
				static_cast<_m_uint>(NewBins),
				SpecData.Data(),
				static_cast<_m_uint>(SpecData.ElementCount()) * 4,
				static_cast<_m_uint>(NewFrames) * 4
				);

			//计算绘制区域
			UIRect dst = *param->destRect;
			dst.left += fontWidth + space;
			dst.right -= sidebarWidth + space;
			dst.top += space + fontHeight;
			dst.bottom -= preHeight + space + fontCenter;

			//频谱的频率上限为8000hz，但是编辑器上限为C10对应的频率，对齐频谱和编辑器的频率
			if (curMaxFreq > MaxSpec) dst.top += int((curMaxFreq - MaxSpec) * curPixPerFreq);

			//对由于缩放导致的频谱区域前后存在的小于1Bin的空隙进行处理
			dst.top -= curMinBinFactorY;
			dst.bottom += curMaxBinFactorY;
			dst.left -= curMinBinFactorX;
			dst.right += curMaxBinFactorX;

			//确保每一个Bin的数据都在中心点
			dst.top += curCenterFactorY;
			dst.bottom += curCenterFactorY;
			dst.left -= curCenterFactorX;
			dst.right -= curCenterFactorX;

			param->render->DrawBitmap(
				bitmap,
				255,
				dst.ToRect()
			);
		}
		m_brush_m->SetColor(m_backColor);
		UIRect dst = *param->destRect;
		dst.right = dst.left + fontWidth + space;
		param->render->FillRectangle(
			dst.ToRect(),
			m_brush_m
		);

		dst = *param->destRect;
		dst.left = dst.right - sidebarWidth - space;
		param->render->FillRectangle(
			dst.ToRect(),
			m_brush_m
		);

		dst = *param->destRect;
		dst.bottom = dst.top + space + fontHeight;
		param->render->FillRectangle(
			dst.ToRect(),
			m_brush_m
		);

		dst = *param->destRect;
		dst.top = dst.bottom - (preHeight + space + fontCenter);
		param->render->FillRectangle(
			dst.ToRect(),
			m_brush_m
		);
	}

	void CurveEditor::DrawCurve(_m_scale scale, MPCPaintParam param)
	{
		if (!m_f0data.HasValue())
			return;

		UIRect viewRect = m_viewRect;
		viewRect.Offset(param->destRect->left, param->destRect->top);
		int viewHeight = viewRect.GetHeight();
		int viewWidth = viewRect.GetWidth();
		int fullHeight = int((float)viewHeight * m_viewScaleV);

		param->render->PopClipRect();
		param->render->PushClipRect(viewRect.ToRect());

#ifdef _WIN32
		auto render = param->render->GetBase<Render::MRender_D2D>();

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

		float offsetY = 0.f;
		if (m_sidebar->DragValue.Get().y != 0)
			offsetY = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height;
		offsetY = (float)(fullHeight - viewHeight) * (offsetY);
		if (m_sidebar->Range.Get().height == m_sidebar->Frame().GetHeight())
			offsetY = 0.f;

		float offsetX = CalcViewHOffset();
		if (DragValue.Get().x != 0)
			offsetX = (float)DragValue.Get().x / (float)Range.Get().width;
		offsetX = (float)(width - viewWidth) * (offsetX);

		const auto DataPointer = m_f0data.Data();
		const auto [Batch, Frames] = m_f0data.Size().RawArray();

		float stepX = (float)width / float(Frames - 1);
		float rStepX = 1.f / stepX;
		
		const auto PointRange = 2.f * (float)viewRect.GetWidth() / (float(Frames) / m_viewScaleH);
		const float DrawRangeLeft = (float)viewRect.left - PointRange;
		const float DrawRangeRight = (float)viewRect.right + PointRange;

		constexpr float jump_dis_val = 0.1f;

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

					if (x > (float)viewRect.right)
						break;
					x_prev = x;
				}

				if (figure_begin)
					sink->EndFigure(D2D1_FIGURE_END_OPEN);
				sink->Close();

				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m.get()), _scale_to(LineWidth, Helper::M_MIN(scale.cx, scale.cy)), pt_style);

				sink->Release();
				geometry->Release();

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
				AlphaRes = std::max(2ui8, AlphaRes);
				m_brush_m->SetOpacity(AlphaRes / 2);

				ID2D1StrokeStyle* dashed_style;
				D2D1_STROKE_STYLE_PROPERTIES dashed_properties = D2D1::StrokeStyleProperties();
				dashed_properties.dashStyle = D2D1_DASH_STYLE_DASH;
				factory->CreateStrokeStyle(dashed_properties, nullptr, 0, &dashed_style);
				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m.get()), _scale_to(LineWidth, Helper::M_MIN(scale.cx, scale.cy)), dashed_style);
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
			m_brush_m->SetColor(m_curveColor[curveIdx]);
			ID2D1StrokeStyle* Style = curveIdx == curve_idx ? nullptr : MinorStyle;
			float LineWidth = curveIdx == curve_idx ? 1.f : 0.5f;
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
				drawPart(
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
					max_val = std::max(max_val, i);
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
				context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m.get()), (float)_scale_to(0.5, scale.cx));

				sink->Release();
				geometry->Release();
			};
		drawPreview();
#endif
	}

	void CurveEditor::DrawPlayeLine(_m_scale scale, MPCPaintParam param) const
	{
		if (m_f0data.Null())
			return;

		int fullWidth = int((float)m_viewRect.GetWidth() * m_viewScaleH);
		float offset = (float)m_plineOffset / (float)(m_f0data.Size(1) - 1);
		offset = offset * (float)fullWidth;
		UIRect rc = m_viewRect;
		rc.left += (int)offset;
		rc.right = rc.left + _scale_to(2, scale.cx);
		rc.Offset(param->destRect->left, param->destRect->top);

		offset = 0.f;
		if (DragValue.Get().x != 0)
			offset = (float)DragValue.Get().x / (float)Range.Get().width;
		offset = (float)(fullWidth - m_viewRect.GetWidth()) * offset;
		rc.Offset((int)-offset, 0);

		UIRect viewRect = m_viewRect;
		viewRect.Offset(param->destRect->left, param->destRect->top);
		if (rc.right > viewRect.left && rc.left < viewRect.right)
		{
			m_brush_m->SetColor(Color::M_RED);
			param->render->FillRectangle(rc.ToRect(), m_brush_m);

			m_font->SetText(std::to_wstring(m_plineOffset));
			const auto fontHeight = m_font->GetMetrics().bottom;
			const int space = _scale_to(5, scale.cy);
			const int preHeight = _scale_to(m_preHeight, scale.cy);
			const int fontCenter = fontHeight / 2;
			const auto bot = param->destRect->bottom - preHeight - space - fontCenter;
			param->render->DrawTextLayout(
				m_font,
				UIRect{
					rc.left - _scale_to(2, scale.cx) - m_font->GetMetrics().GetWidth(),
					bot,
					m_font->GetMetrics().GetWidth(),
					fontHeight
				}.ToRect(),
				m_brush_m,
				TextAlign_Right
			);
			m_font->SetText(
				std::to_wstring(
					(m_f0data.Data() + m_f0data.Size(1) * curve_idx)[m_plineOffset]
				).substr(0, 7) + L"HZ"
			);
			param->render->DrawTextLayout(
				m_font,
				UIRect{
					rc.right + _scale_to(2, scale.cx),
					bot,
					m_font->GetMetrics().GetWidth(),
					fontHeight
				}.ToRect(),
				m_brush_m,
				TextAlign_Left
			);
		}

		m_brush_m->SetColor(m_lineColor);

		//预览图指针
		offset = (float)m_plineOffset / (float)(m_f0data.Size(1) - 1);
		offset = offset * (float)param->destRect->GetWidth();
		rc = *param->destRect;
		rc.top = rc.bottom - _scale_to(m_preHeight, scale.cy);
		rc.left += (int)offset;
		rc.right = rc.left + _scale_to(2, scale.cx);
		param->render->FillRectangle(rc.ToRect(), m_brush_m);
	}

	void CurveEditor::CalcRangeViewV()
	{
		float height = static_cast<float>(m_sidebar->Frame().GetHeight());
		float range = _scale_to(height, m_viewScaleV);
		float drag = 0;
		if (m_sidebar->DragValue.Get().y != 0) 
			drag = (float)m_sidebar->DragValue.Get().y / (float)m_sidebar->Range.Get().height * range;
		if (m_viewScaleV == 1.f) 
			drag = 0;
		drag = std::clamp(drag, 0.f, range);

		m_sidebar->Range.Set().value->height = (int)round(range);
		Range.Set().value->height = (int)round(range);
		m_sidebar->DragValue.Set().value->y = (int)round(drag);
		DragValue.Set().value->y = (int)round(drag);
	}

	void CurveEditor::CalcRangeViewH()
	{
		float width = UINodeBase::m_data.Frame.GetWidth();

		float range = _scale_to(width, m_viewScaleH);
		float drag = 0;
		if (DragValue.Get().x != 0) drag = (float)DragValue.Get().x / (float)Range.Get().width * range;
		if (m_viewScaleH == 1.f) drag = 0.f;
		drag = std::clamp(drag, 0.f, range);

		Range.Set().value->width = (int)round(range);
		DragValue.Set().value->x = (int)round(drag);
	}

	void CurveEditor::CalcRangeViewH(const UIPoint& point, short delta)
	{
		const float width = UINodeBase::m_data.Frame.GetWidth();
		const float LWidth = (float)point.x - UINodeBase::m_data.Frame.left;

		const float range = _scale_to(width, m_viewScaleH);
		float drag = (float)DragValue.Get().x * range / (float)Range.Get().width;
		if (delta > 0) drag += (LWidth - width / 2) / 2;
		drag = std::clamp(drag, 0.f, range);

		Range.Set().value->width = (int)round(range);
		DragValue.Set().value->x = (int)round(drag);
	}

	void CurveEditor::CalcViewRect()
	{
		m_font->SetText(L"C0");

		const auto [cx, cy] = GetRectScale().scale();
		const float height = UINodeBase::m_data.Frame.GetHeight();
		const float space = _scale_to(5.f, cy);
		const float preHeight = _scale_to(static_cast<float>(m_preHeight), cy);
		const float sidebarWidth = _scale_to(static_cast<float>(m_barWidth), cx);
		const float fontHeight = (float)m_font->GetMetrics().bottom;
		const float fontCenter = fontHeight / 2.f;
		const float fontWidth = _scale_to(60.f, cx);
		//曲线视图高度
		UIRect viewRect = 
		{
			(int)round(fontWidth + space),
			(int)round(space + fontHeight),
			0,
			(int)round(height - preHeight - space - fontCenter - space - fontHeight)
		};
		viewRect.right = (int)round(UINodeBase::m_data.Frame.GetWidth() - sidebarWidth - space);
		m_viewRect = viewRect;
	}

	float CurveEditor::CalcViewHOffset(int drag) const
	{
		int viewWidth = m_viewRect.GetWidth();

		if (DragValue.Get().x == 0 && drag == 0)
			return 0.f;

		if (drag == 0)
			drag = DragValue.Get().x;

		float newRange = (float)viewWidth * m_viewScaleH;
		float newDrag = (float)drag / (float)Range.Get().width * newRange;

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

	int CurveEditor::CalcXPosWithPtr(MPCPaintParam param, const float* Ptr)
	{
		int fullWidth = int((float)m_viewRect.GetWidth() * m_viewScaleH);
		float offset = (float)(Ptr - m_f0data.Data() + curve_idx * m_f0data.Size(1)) / (float)(m_f0data.Size(1) - 1);
		offset = offset * (float)fullWidth;
		const auto result = m_viewRect.left + int(offset) + param->destRect->left;

		offset = 0.f;
		if (DragValue.Get().x != 0)
			offset = (float)DragValue.Get().x / (float)Range.Get().width;
		offset = (float)(fullWidth - m_viewRect.GetWidth()) * offset;
		return result - (int)offset;
	}

	size_t CurveEditor::GetXOffset(float PointX) const
	{
		float viewWidth = (float)m_viewRect.GetWidth();
		float fullWidth = viewWidth * float(m_viewScaleH);

		float offsetX = 0.f;
		if (DragValue.Get().x != 0)
			offsetX = (float)DragValue.Get().x / (float)Range.Get().width;
		offsetX = (fullWidth - viewWidth) * (offsetX);
		offsetX = (PointX + offsetX) / fullWidth;
		offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

		return size_t(round(offsetX * (float)(m_f0data.Size(1) - 1)));
	}

	float CurveEditor::GetFpOffset(float PointX) const
	{
		float viewWidth = (float)m_viewRect.GetWidth();
		float fullWidth = viewWidth * float(m_viewScaleH);

		float offsetX = 0.f;
		if (DragValue.Get().x != 0)
			offsetX = (float)DragValue.Get().x / (float)Range.Get().width;
		offsetX = (fullWidth - viewWidth) * (offsetX);
		offsetX = (PointX + offsetX) / fullWidth;
		offsetX = Helper::M_Clamp(0.f, 1.f, offsetX);

		return offsetX * (float)(m_f0data.Size(1) - 1);
	}

	void CurveEditor::OnScrollView(UIScroll*, int dragValue, bool horizontal)
	{
		UpDate();
	}

}