#include "Waveform.h"
#include "../DefControl.hpp"
#include "TensorLib/Include/Base/Tensor/Functional.h"

namespace SimpleF0Labeler
{
	class MD2DBrush final : Mui::Render::MBrush_D2D
	{
	public:
		static ID2D1SolidColorBrush* GetBrush(MBrush* brush)
		{
			return static_cast<MD2DBrush*>(brush)->m_brush;
		}
	};

	void Waveform::Register()
	{
		Mui::XML::RegisterControl(
			ClassName,
			[](UIControl* parent) { return new Waveform(parent); }
		);
	}

	Waveform::Waveform(UIControl* parent)
	{
		parent->AddChildren(this);
		m_anicls = std::make_shared<Mui::MAnimation>(UIControl::GetParentWin());

		UIScroll::Horizontal = true;
		UIScroll::BarWidth = 0;

		_MyAudioData = std::make_shared<PCMAudio>();
		_MyAudioData->SetCallback(
			[this](size_t Pos) { this->SetPtrOffset(Pos); }
		);
	}

	Waveform::~Waveform()
	{
		Clear();
	}

	void Waveform::Update()
	{
		m_cacheUpdate = true;
		UpdateDisplay();
	}


	void Waveform::SetAudioPlayer(Mui::Render::MAudioPlayer* Player)
	{
		_MyPlayer = Player;
		_MyTrack = _MyPlayer->CreateTrack();
	}


	size_t Waveform::GetPCMSize() const
	{
		return _MyAudioData->PCMGetDataSize() / sizeof(std::int16_t);
	}

	float Waveform::GetDataDuration() const
	{
		if (_MyPlayer && !_MyInt16Audio.Null())
			return _MyAudioData->GetDuration();
		return 0.0f;
	}

	void Waveform::Clear()
	{
		if (_MyPlayer && _MyTrack)
			_MyPlayer->StopTrack(_MyTrack);
		_MyInt16Audio.Clear();
		_MyAudioData->SetData(std::nullopt);
		_MyResampledData.Clear();
		_MyPtrOffset = 0;
		_IsPause = true;
		UIScroll::DragValue.Set().value->x = 0;
		UIScroll::Range.Set().value->width = 0;
		Update();
	}

	Int16Tensor2D& Waveform::GetAudio()
	{
		return _MyInt16Audio;
	}

	void Waveform::SetAudioData(const FloatTensor2D& AudioData, DragonianLib::UInt SamplingRate)
	{
		Clear();
		_MyInt16Audio = (AudioData * 32767.f).Cast<short>().Evaluate();
		_MyAudioData->SetData(_MyInt16Audio);
		_MyAudioData->SetSamplingRate(SamplingRate);
		if (_MyPlayer && _MyTrack)
			_MyPlayer->SetTrackSound(_MyTrack, static_cast<Mui::Render::MAudio*>(_MyAudioData.get()));
		_MyTrack->GetSamplerate();
		Update();
	}

	void Waveform::Play() const
	{
		if (_MyPlayer && _MyTrack && _MyInt16Audio.HasValue())
			_MyPlayer->PlayTrack(_MyTrack);
	}

	void Waveform::Pause() const
	{
		if (_MyPlayer && _MyTrack && _MyInt16Audio.HasValue())
			_MyPlayer->PauseTrack(_MyTrack);
	}

	bool Waveform::IsPlay() const
	{
		return !_IsPause;
	}

	size_t Waveform::GetPlayPos() const
	{
		return _MyPtrOffset;
	}

	void Waveform::SetVolume(Mui::_m_byte Volume) const
	{
		if (_MyPlayer && _MyTrack)
			_MyPlayer->SetTrackVolume(_MyTrack, Volume);
	}

	void Waveform::PlayPause()
	{
		if (_MyPlayer && _MyTrack && _MyInt16Audio.HasValue())
		{
			if (_IsPause)
			{
				_IsPause = false;
				_MyPlayer->PlayTrack(_MyTrack);
			}
			else
			{
				_IsPause = true;
				_MyPlayer->PauseTrack(_MyTrack);
			}
		}
	}

	void Waveform::SetPlayPos(size_t Offset)
	{
		SetPtrOffset(Offset);
		Offset *= sizeof(std::int16_t);

		//计算时间
		auto MyAudio = _MyAudioData.get();
		auto Duration = Offset / MyAudio->GetBlockAlign();
		double SampleOffset = (double)Duration / (double)MyAudio->GetSamplerate();
		_MyPlayer->SetTrackPlaybackPos(_MyTrack, (float)SampleOffset);
	}

	void Waveform::SetPtrOffset(size_t Offset)
	{
		_MyPtrOffset = Offset;
		if (_MyCallback)
		{
			double PreSize = (double)_MyPtrOffset / (double)_MyInt16Audio.ElementCount();
			auto MyAudio = _MyAudioData.get();
			auto Duration = _MyPtrOffset * sizeof(std::int16_t) / MyAudio->GetBlockAlign();
			double SampleIdx = (double)Duration / (double)MyAudio->GetSamplerate();
			_MyCallback((float)PreSize, (float)SampleIdx);
		}
		
		Update();
	}

	void Waveform::SetPlayCallback(std::function<void(float, float)> callback)
	{
		_MyCallback = std::move(callback);
	}


	void Waveform::SetAniFlag(bool IsAnimate)
	{
		_IsAnimate = IsAnimate;
		Resample(UINodeBase::m_data.Frame.GetWidth());
		Update();
	}

	bool Waveform::SetAttribute(Mui::XML::PropName AttributeName, std::wstring_view Value, bool Draw)
	{
		if (AttributeName == L"PtrColor")
			_MyPtrColor = Mui::Helper::M_GetAttribValueColor(Value);
		else if (AttributeName == L"WavColor")
			_MyWavColor = Mui::Helper::M_GetAttribValueColor(Value);
		else if (AttributeName == L"LineColor")
			_MyLineColor = Mui::Helper::M_GetAttribValueColor(Value);
		else
			return UIScroll::SetAttribute(AttributeName, Value, Draw);
		if (Draw) Update();
		return true;
	}

	std::wstring Waveform::GetAttribute(Mui::XML::PropName AttributeName)
	{
		if (AttributeName == L"PtrColor")
			return Mui::Color::M_RGBA_STR(_MyPtrColor);
		if (AttributeName == L"WavColor")
			return std::to_wstring(_MyWavColor);
		if (AttributeName == L"LineColor")
			return std::to_wstring(_MyLineColor);

		return UIScroll::GetAttribute(AttributeName);
	}


	void Waveform::OnLoadResource(Mui::Render::MRenderCmd* render, bool recreate)
	{
		UIScroll::OnLoadResource(render, recreate);
		_MyPen = render->CreatePen(1, _MyWavColor);
		_MyBrush = render->CreateBrush(_MyPtrColor);
	}

	void Waveform::OnPaintProc(MPCPaintParam Params)
	{
		UIScroll::OnPaintProc(Params);

		if (_MyInt16Audio.Null() || std::wstring_view(Params->render->GetRenderName()) != L"D2D")
			return;

		int MyWidth = Params->destRect->GetWidth();
		int MyHeight = Params->destRect->GetHeight();

		_MyBrush->SetColor(Mui::Color::M_RGBA(53, 192, 242, 255));

		if (_MyWidth != MyWidth || _MyResampledData.Empty())
		{
			Resample(static_cast<float>(MyWidth));
			_MyWidth = MyWidth;
		}

		auto MyScale = GetRectScale().scale();

		int OffsetY = Mui::_scale_to(20, MyScale.cy);
		MyHeight -= OffsetY;

		float DestX = 0.f;
		float DestY = 0.f;

#ifdef _WIN32
		auto Render = Params->render->GetBase<Mui::Render::MRender_D2D>();

		auto DeviceContext = static_cast<ID2D1DeviceContext*>(Render->Get());

		ID2D1Factory* Factory = nullptr;
		DeviceContext->GetFactory(&Factory);

		if(Params->cacheCanvas)
		{
			Mui::UIRect Sub = Params->render->GetCanvas()->GetSubRect();
			DestX += (float)Sub.left;
			DestY += (float)Sub.top;
		}
#endif

		DestX += (float)Params->destRect->left;
		DestY += (float)Params->destRect->top + (float)Mui::_scale_to(10, MyScale.cy);

		_MyPen->SetColor(_MyWavColor);
		_MyPen->SetOpacity(Params->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst);

		{
			const float center = DestY + (float)MyHeight / 2.f;
			float scaleY = (float)MyHeight / 65536.f;

			ID2D1PathGeometry* geometry;
			Factory->CreatePathGeometry(&geometry);

			ID2D1GeometrySink* sink = nullptr;
			geometry->Open(&sink);

			sink->BeginFigure(D2D1::Point2F(DestX, center), D2D1_FIGURE_BEGIN_FILLED);

			float offset = (float)_MyResampledData.Size(0) / (float)MyWidth;
			const auto MData = reinterpret_cast<const std::pair<short, short>*>(_MyResampledData.Data());
			for (int i = 0; i < MyWidth; ++i)
			{
				const float x = DestX + (float)i;
				float y = center - float(MData[int((float)i * offset)].first) * scaleY;

				sink->AddLine(D2D1::Point2F(x, y));

				y = center - float(MData[int((float)i * offset)].second) * scaleY;
				sink->AddLine(D2D1::Point2F(x, y));
			}

			sink->EndFigure(D2D1_FIGURE_END_OPEN);
			sink->Close();
			sink->Release();

			DeviceContext->DrawGeometry(geometry, MD2DBrush::GetBrush(_MyBrush.get()));

			geometry->Release();
		};

		_MyBrush->SetOpacity(Params->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst);

		auto ptrOff = _MyPtrOffset;
		auto audioSize = (double)_MyInt16Audio.ElementCount();
		if (ptrOff != 0)
			ptrOff = Mui::_m_size((double)ptrOff / audioSize * (double)MyWidth);
		if (std::cmp_less_equal(ptrOff, MyWidth))
		{
			_MyBrush->SetColor(_MyPtrColor);
			Params->render->FillRectangle(
				Mui::UIRect{ (int)ptrOff, 0, Mui::_scale_to(2, MyScale.cx), Params->destRect->GetHeight() }.ToRect(),
				_MyBrush
			);
		}
	}

	bool Waveform::OnSetCursor(Mui::_m_param hCur, Mui::_m_param lParam)
	{
#ifdef _WIN32
		::SetCursor((HCURSOR)hCur);
#endif
		return true;
	}

	bool Waveform::OnLButtonDown(Mui::_m_uint flag, const Mui::UIPoint& point)
	{
		if (UIScroll::OnLButtonDown(flag, point))
			return true;

		if (!_MyLBIsDown)
		{
			_MyLBIsDown = true;
			if (!_MyPlayer || !_MyTrack || _MyInt16Audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			_MyPlayer->PauseTrack(_MyTrack);
			_IsPause = true;
			int x = (int)round(static_cast<float>(point.x) - UINodeBase::m_data.Frame.left);
			auto poffset = GetOffset(x);
			SetPtrOffset(poffset);
			return true;
		}
		return false;
	}

	bool Waveform::OnMouseMove(Mui::_m_uint flag, const Mui::UIPoint& point)
	{
		if (UIScroll::OnMouseMove(flag, point))
			return true;

		if (_MyLBIsDown)
		{
			if (!_MyPlayer || !_MyTrack || _MyInt16Audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			int x = (int)round(static_cast<float>(point.x) - UINodeBase::m_data.Frame.left);
			auto poffset = GetOffset(x);
			SetPtrOffset(poffset);
			return true;
		}
		return false;
	}

	bool Waveform::OnLButtonUp(Mui::_m_uint flag, const Mui::UIPoint& point)
	{
		if (UIScroll::OnLButtonUp(flag, point))
			return true;

		if (_MyLBIsDown)
		{
			_MyLBIsDown = false;
			if (!_MyPlayer || !_MyTrack || _MyInt16Audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			int x = (int)round(static_cast<float>(point.x) - UINodeBase::m_data.Frame.left);
			SetPlayPosWithX(x);
		}
		return true;
	}

	bool Waveform::OnMouseExited(Mui::_m_uint flag, const Mui::UIPoint& point)
	{
#ifdef _WIN32
		SetCursor(IDC_ARROW);
#endif
		if (_MyLBIsDown)
		{
			_MyLBIsDown = false;
			if (!_MyPlayer || !_MyTrack || _MyInt16Audio.Null())
				return UIControl::OnMouseExited(flag, point);
			int x = (int)round(static_cast<float>(point.x) - UINodeBase::m_data.Frame.left);
			SetPlayPosWithX(x);
		}
		return UIScroll::OnMouseExited(flag, point);
	}

	bool Waveform::OnMouseEntered(Mui::_m_uint flag, const Mui::UIPoint& point)
	{
#ifdef _WIN32
		SetCursor(IDC_IBEAM);
#endif
		return UIScroll::OnMouseEntered(flag, point);
	}

	size_t Waveform::GetOffset(int x) const
	{
		return size_t(
			(double)Mui::Helper::M_MIN(
				(float)x / UINodeBase::m_data.Frame.GetWidth(),
				1.f
			) * (double)_MyInt16Audio.ElementCount()
		);
	}

	void Waveform::SetPlayPosWithX(int x)
	{
		const auto PtrOffset = GetOffset(x);
		const auto MOffset = PtrOffset * sizeof(std::int16_t);

		auto MyAudio = _MyAudioData.get();
		auto Duration = MOffset / MyAudio->GetBlockAlign();
		double SampleOffset = (double)Duration / (double)MyAudio->GetSamplerate();

		_MyPlayer->SetTrackPlaybackPos(_MyTrack, (float)SampleOffset);
		SetPtrOffset(PtrOffset);
	}

	void Waveform::Resample(float width)
	{
		std::lock_guard lock(mx);
		if (_MyInt16Audio.Null() || _IsAnimate) return;
		const auto SampleCount = _MyInt16Audio.Size(0);
		const auto WindowCount = static_cast<int64_t>(width);
		const auto WindowSize = SampleCount / WindowCount;
		const auto NewCount = WindowCount * WindowSize;
		const auto Slice = _MyInt16Audio.Slice(
			{ DragonianLib::Range(0, NewCount) }
		).GatherAxis<1>(0).ReShape(WindowCount, WindowSize);
		_MyResampledData = DragonianLib::Functional::Stack(
			Slice.ReduceMax(1),
			Slice.ReduceMin(1),
			-1
		).Evaluate();
	}
}
