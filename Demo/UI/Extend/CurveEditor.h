#pragma once
#include "../framework.h"
#include <Render/Sound/Mui_SoundDef.h>

namespace SimpleF0Labeler
{
	class CurveEditor : public Mui::Ctrl::UIScroll  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		static constexpr Mui::XML::PropName ClassName{ L"CurveEditor" };
		Mui::XML::PropName GetClsName() const override { return ClassName; }
		static void Register();

		CurveEditor(UIControl* parent);
		~CurveEditor() override;

		void SetCurveData(const FloatTensor2D& data, const ImageTensor& spec, const ImageTensor& spec_logview);

		void ReSetCurveData(const FloatTensor2D& data, int64_t idx);

		void SetCurveIndex(int64_t idx);

		const FloatTensor2D& GetCurveData() const { return m_f0data; }

		void SetShowPitch(bool show);

		void SetPlayLinePos(Mui::_m_size offset);

		Mui::_m_size GetPlayLinePos() const
		{
			return m_plineOffset;
		}

		bool SetAttribute(Mui::XML::PropName attribName, std::wstring_view attrib, bool draw) override;

		std::wstring GetAttribute(Mui::XML::PropName attribName) override;

		void UpDate();

		DragonianLib::TemplateLibrary::MutableRanges<float> GetSelectedRange() const;

		void UPRButton();

	protected:
		void OnScale(Mui::_m_scale scale) override;
		void OnLoadResource(Mui::Render::MRenderCmd* render, bool recreate) override;
		void OnPaintProc(MPCPaintParam param) override;

		bool OnMouseWheel(Mui::_m_uint flag, short delta, const Mui::UIPoint& point) override;
		bool OnMouseMove(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnRButtonDown(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnRButtonUp(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnMouseExited(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnLButtonDown(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnLButtonUp(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnLButtonDoubleClicked(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnWindowMessage(Mui::MEventCodeEnum code, Mui::_m_param wParam, Mui::_m_param lParam) override;

	private:
		void DrawLabel(Mui::_m_scale scale, MPCPaintParam param);

		void DrawSpec(Mui::_m_scale scale, MPCPaintParam param);

		void DrawCurve(Mui::_m_scale scale, MPCPaintParam param);

		void DrawPlayeLine(Mui::_m_scale scale, MPCPaintParam param) const;

		void CalcRangeViewV();

		void CalcRangeViewH(const Mui::UIPoint& point, short delta);

		void CalcRangeViewH();

		void CalcViewRect();

		float CalcViewHOffset(int drag = 0) const;

		void PlaySoundPitch(const Mui::UIPoint& point);

		void ArrangeRect(const Mui::UIRect& rect, std::vector<Mui::UIRect>& dst) const;

		int CalcXPosWithPtr(MPCPaintParam param, const float* Ptr);

		size_t GetXOffset(float PointX) const;

		float GetFpOffset(float PointX) const;

		void OnScrollView(UIScroll*, int dragValue, bool horizontal);

		std::mutex mx;

		Mui::Render::MBrushPtr m_brush_m = nullptr;
		Mui::Render::MFontPtr m_font = nullptr;

		Mui::_m_color m_lineColor = Mui::Color::M_White;
		Mui::_m_color m_curveColor[10];
		Mui::_m_color m_fontColor = Mui::Color::M_Black;
		Mui::_m_color m_backColor = Mui::Color::M_Black;
		Mui::_m_uint m_fontSize = 12;
		std::wstring m_fontName = Mui::M_DEF_SYSTEM_FONTNAME;

		Mui::UISize m_size;

		void* MidiOutHandle = nullptr;
		bool MidiOutOpen = false;
		DWORD LastMidiPitch = 256;
		
		int m_preHeight = 40;
		int m_barWidth = 10;
		int m_curHeight = 0;
		bool m_showPitch = true;

		float m_viewScaleV = 1.f;
		float m_viewScaleH = 1.f;
		//float m_viewScaleV_last = 0.f;
		//float m_viewScaleH_last = 0.f;
		Mui::UIRect m_viewRect;

		//int m_dragValueV_last = -114;
		//int m_dragValueH_last = -514;

		Mui::UIPoint m_lastPos;
		bool m_isdown = false;

		//int m_l_x_pos, m_c_x_pos;
		bool m_insel = false;
		//float last_x_offset = 0.f;

		Mui::_m_size m_plineOffset = 0;

		FloatTensor2D m_f0data;
		ImageTensor m_specData;
		ImageTensor m_specLogView;

		bool m_lisdown = false;
		Mui::UIPoint m_llastPos;

		UIScroll* m_sidebar = nullptr;

		int64_t curve_idx = 0;

		bool time_in_sel = false;
		bool in_move = false;
		Mui::UIPoint m_llast_movePos;

		float* selected_f0_begin = nullptr;
		float* selected_f0_end = nullptr;
	};

	void Write2Clipboard(const DragonianLib::TemplateLibrary::MutableRanges<float>& Range);
}