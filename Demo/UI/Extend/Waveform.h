#pragma once
#include "../framework.h"
#include <Render/Sound/Mui_SoundDef.h>

namespace SimpleF0Labeler
{
	class Waveform : public Mui::Ctrl::UIScroll  // NOLINT(cppcoreguidelines-special-member-functions)
	{
	public:
		static constexpr Mui::XML::PropName ClassName{ L"Waveform" };
		Mui::XML::PropName GetClsName() const override { return ClassName; }
		static void Register();

		Waveform(UIControl* parent);
		~Waveform() override;
		void Update();

		void SetAudioPlayer(Mui::Render::MAudioPlayer* Player);

		size_t GetPCMSize() const;
		float GetDataDuration() const;
		void Clear();
		Int16Tensor2D& GetAudio();
		void SetAudioData(const FloatTensor2D& AudioData);

		void Play() const;
		void Pause() const;
		bool IsPlay() const;
		size_t GetPlayPos() const;
		void SetVolume(Mui::_m_byte Volume) const;
		void PlayPause();
		void SetPlayPos(size_t Offset);
		void SetPtrOffset(size_t Offset);
		void SetPlayCallback(std::function<void(float, float)> callback);

		void SetAniFlag(bool IsAnimate);

		bool SetAttribute(Mui::XML::PropName AttributeName, std::wstring_view Value, bool Draw) override;
		std::wstring GetAttribute(Mui::XML::PropName AttributeName) override;

	protected:
		void OnLoadResource(Mui::Render::MRenderCmd* render, bool recreate) override;
		void OnPaintProc(MPCPaintParam param) override;

		bool OnSetCursor(Mui::_m_param hCur, Mui::_m_param lParam) override;

		bool OnLButtonDown(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnMouseMove(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnLButtonUp(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnMouseExited(Mui::_m_uint flag, const Mui::UIPoint& point) override;
		bool OnMouseEntered(Mui::_m_uint flag, const Mui::UIPoint& point) override;

		size_t GetOffset(int x) const;
		void SetPlayPosWithX(int x);

	private:
		void Resample(float width);

		std::mutex mx;

		Mui::Render::MBrushPtr _MyBrush = nullptr;
		Mui::Render::MPenPtr _MyPen = nullptr;
		bool _IsAnimate = false;

		bool _IsPause = true;
		size_t _MyPtrOffset = 0;
		Int16Tensor2D _MyInt16Audio;
		std::shared_ptr<void> _MyAudioData = nullptr;
		Mui::Render::MAudioTrack* _MyTrack = nullptr;
		Mui::Render::MAudioPlayer* _MyPlayer = nullptr;
		std::function<void(float, float)> _MyCallback;

		Mui::_m_color _MyPtrColor = Mui::Color::M_White;
		Mui::_m_color _MyWavColor = Mui::Color::M_RGBA(53, 192, 242, 255);
		Mui::_m_color _MyLineColor = Mui::Color::M_GREEN;
		
		Int16Tensor2D _MyResampledData;

		int _MyWidth = 0;

		bool _MyLBIsDown = false;
		//bool m_isMove = false;
	};
}