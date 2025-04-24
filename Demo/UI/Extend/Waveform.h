#pragma once
#include "../framework.h"
#include "../DataStruct.h"
#include <Render/Sound/Mui_SoundDef.h>

namespace Mui::Ctrl
{
	//波形图显示控件
	class Waveform : public UIScroll
	{
	public:
		M_DEF_CTRL(L"Waveform")
		UIScroll::M_PTRATTRIB
		M_DEF_CTRL_END

		Waveform(UIControl* parent);
		~Waveform() override;

		void SetAudioPlayer(MAudioPlayer* player);
		void SetAudioData(const FloatTensor2D& data);

		Int16Tensor2D& GetAudio()
		{
			return m_audio;
		}

		size_t GetPCMSize() const;

		void SetPtrOffset(_m_ptrv offset);
		void SetPlayCallback(std::function<void(float, float)> callback);

		float GetDataDuration() const;

		void SetPlayPos(_m_ptrv offset);
		_m_ptrv GetPlayPos() const
		{
			return m_ptrOffset;
		}

		void Clear();

		void Play() const;
		void Pause() const;
		void PlayPause();

		bool IsPlay() const;

		void SetVolume(_m_byte vol) const;

		void SetAniFlag(bool ani);
		void SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw = true) override;
		std::wstring GetAttribute(std::wstring_view attribName) override;

		bool _m_has_f0 = false;

	protected:
		void OnLoadResource(MRenderCmd* render, bool recreate) override;
		void OnPaintProc(MPCPaintParam param) override;

		//bool OnMouseWheel(_m_uint flag, short delta, const UIPoint& point) override;
		bool OnSetCursor(_m_param hCur, _m_param lParam) override;

		bool OnLButtonDown(_m_uint flag, const UIPoint& point) override;
		bool OnMouseMove(_m_uint flag, const UIPoint& point) override;
		bool OnLButtonUp(_m_uint flag, const UIPoint& point) override;
		bool OnMouseExited(_m_uint flag, const UIPoint& point) override;

		//bool OnWindowMessage(MEventCodeEnum code, _m_param wParam, _m_param lParam) override;
		_m_size GetOffset(int x);
		void SetPlayPosWithX(int x);

	private:
		void OnScrollView(UIScroll*, int dragValue, bool horizontal);

		void Resample(int width, bool pre = false);

		_m_color m_ptrColor = Color::M_White;
		_m_color m_wavColor = Color::M_RGBA(53, 192, 242, 255);
		_m_color m_lineColor = Color::M_GREEN;
		_m_ptrv m_ptrOffset = 0;
		Int16Tensor2D m_audio;
		std::vector<std::pair<short, short>> m_overviewData;
		std::vector<std::pair<short, short>> m_previewData;

		int lastRange = 0;

		std::function<void(float, float)> m_callback;

		int m_width = 0;
		int m_preHeight = 100;
		float m_viewScale = 1.f;
		float m_lastScale = 1.f;

		std::mutex mx;

		std::pair<_m_long64, _m_long64> m_lineRange = { 0, 0 };

		MAudioPlayer* m_player = nullptr;
		MAudioTrack* m_track = nullptr;
		void* m_audioData = nullptr;

		bool m_isDown = false;
		//bool m_isMove = false;
		bool m_isPause = true;
		bool m_isani = false;

		MBrush* m_brush_m = nullptr;
		MPen* m_pen = nullptr;
	};
}