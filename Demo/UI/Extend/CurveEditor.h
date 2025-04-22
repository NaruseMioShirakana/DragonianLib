#pragma once
#include "../framework.h"
#include "../DataStruct.h"
#include <Render/Sound/Mui_SoundDef.h>

namespace Mui::Ctrl
{
	//曲线编辑器
	class CurveEditor : public UIScroll
	{
	public:
		M_DEF_CTRL(L"CurveEditor")
		UIScroll::M_PTRATTRIB
		M_DEF_CTRL_END

		CurveEditor(UIControl* parent);
		~CurveEditor() override;

		void SetCurveData(const FloatTensor2D& data);

		void ReSetCurveData(const FloatTensor2D& data, int64_t idx);

		const FloatTensor2D& GetCurveData() const { return m_f0data; }

		void SetShowPitch(bool show);

		void SetPlayLinePos(_m_size offset);

		_m_size GetPlayLinePos() const
		{
			return m_plineOffset;
		}

		void SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw = true) override;

		std::wstring GetAttribute(std::wstring_view attribName) override;

		void UpDate();

		DragonianLib::TemplateLibrary::MutableRanges<float> GetSelectedRange() const;

	protected:
		void OnScale(_m_scale scale) override;
		void OnLoadResource(MRenderCmd* render, bool recreate) override;
		void OnPaintProc(MPCPaintParam param) override;

		bool OnMouseWheel(_m_uint flag, short delta, const UIPoint& point) override;
		bool OnMouseMove(_m_uint flag, const UIPoint& point) override;
		bool OnRButtonDown(_m_uint flag, const UIPoint& point) override;
		bool OnRButtonUp(_m_uint flag, const UIPoint& point) override;
		bool OnMouseExited(_m_uint flag, const UIPoint& point) override;
		bool OnLButtonDown(_m_uint flag, const UIPoint& point) override;
		bool OnLButtonUp(_m_uint flag, const UIPoint& point) override;
		bool OnLButtonDoubleClicked(_m_uint flag, const UIPoint& point) override;
		bool OnWindowMessage(MEventCodeEnum code, _m_param wParam, _m_param lParam) override;

	private:
		void OnScrollView(UIScroll*, int dragValue, bool horizontal);

		void DrawLabel(_m_scale scale, MPCPaintParam param);

		void DrawXLabel(_m_scale scale, MPCPaintParam param);

		void DrawCurve(_m_scale scale, MPCPaintParam param, const FloatTensor2D& data);

		void DrawPlayeLine(_m_scale scale, MPCPaintParam param);

		void CalcRangeViewV();

		void CalcRangeViewH(const UIPoint& point, short delta);

		void CalcRangeViewH();

		void CalcViewRect();

		float CalcViewHOffset(int drag = 0);

		void PlaySoundPitch(const UIPoint& point);

		std::mutex mx;

		void ArrangeRect(const UIRect& rect, std::vector<UIRect>& dst) const;

		MBrush* m_brush_m = nullptr;
		MPen* m_pen = nullptr;
		MFont* m_font = nullptr;

		_m_color m_lineColor = Color::M_White;
		_m_color m_curveColor = Color::M_RGBA(86, 179, 231, 255);
		_m_color m_fontColor = Color::M_Black;
		_m_uint m_fontSize = 12;
		std::wstring m_fontName = M_DEF_SYSTEM_FONTNAME;

		UISize m_size;

		void* MidiOutHandle = nullptr;
		bool MidiOutOpen = false;
		DWORD LastMidiPitch = 256;
		
		int m_preHeight = 40;
		int m_barWidth = 10;
		int m_curHeight = 0;
		bool m_showPitch = true;

		float m_viewScaleV = 1.f;
		float m_viewScaleH = 1.f;
		UIRect m_viewRect;

		UIPoint m_lastPos;
		bool m_isdown = false;

		//int m_l_x_pos, m_c_x_pos;
		bool m_insel = false;
		//float last_x_offset = 0.f;

		_m_size m_plineOffset = 0;

		FloatTensor2D m_f0data;

		bool m_lisdown = false;
		UIPoint m_llastPos;

		UIScroll* m_sidebar = nullptr;

		int64_t curve_idx = 0;

		bool time_in_sel = false;
		bool in_move = false;
		UIPoint m_llast_movePos;

		float* selected_f0_begin = nullptr;
		float* selected_f0_end = nullptr;
	};
}