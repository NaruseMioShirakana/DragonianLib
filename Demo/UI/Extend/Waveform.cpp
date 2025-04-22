#include "Waveform.h"
#include "../DefControl.hpp"

namespace Mui::Ctrl
{
	class WAVAudio : public MAudio
	{
	public:

		//获取音频时长(秒)
		float GetDuration() override
		{
			auto pers = source.ElementCount() * sizeof(std::int16_t) / 4;
			double sc = (double)pers / 48000.0;
			return (float)sc;
		}

		//获取音频位率(bits)
		_m_uint GetBitrate() override { return 16; }

		//获取音频比特率(kbps)
		_m_uint GetBitPerSecond() override { return _m_uint(source.ElementCount() / 4); }

		//获取音频采样率(hz)
		_m_uint GetSamplerate() override { return 48000; }

		//获取音频声道数
		_m_uint GetChannel() override { return 2; }

		//获取字节对齐数
		_m_uint GetBlockAlign() override { return 4; }

		//获取PCM数据尺寸
		_m_size PCMGetDataSize() override { return source.ElementCount() * sizeof(std::int16_t); }

		//读取PCM数据
		_m_size PCMReadData(_m_size begin, _m_size size, _m_byte* dst) override
		{
			if (source.Null())
				return 0;
			auto maxSize = source.ElementCount() * sizeof(std::int16_t);
			if (begin >= maxSize)
				return 0;

			if (begin + size >= maxSize)
				size = maxSize - begin - 1;

			auto offset = source.Data() + begin / sizeof(std::int16_t);

			memcpy(dst, offset, size);

			control->SetPtrOffset((begin + size) / sizeof(std::int16_t));

			return size;
		}

		Int16Tensor2D source;
		Waveform* control = nullptr;
	};

	void Waveform::Register()
	{
		static auto method = [](UIControl* parent)
		{
			return new Waveform(parent);
		};
		M_REGISTER_CTRL(method);
	}

	class MD2DBrush final : MBrush_D2D
	{
	public:
		static ID2D1SolidColorBrush* GetBrush(MBrush* brush)
		{
			return static_cast<MD2DBrush*>(brush)->m_brush;
		}
	};

	Waveform::Waveform(UIControl* parent)
		: UIScroll(Attribute())
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

		m_audioData = new WAVAudio();
		static_cast<WAVAudio*>(m_audioData)->control = this;
		lastRange = UINodeBase::m_data.Frame.GetWidth();
	}

	Waveform::~Waveform()
	{
		MSafeRelease(m_pen);
		MSafeRelease(m_brush_m);
		Clear();
		delete (WAVAudio*)m_audioData;
	}

	void Waveform::SetAudioPlayer(MAudioPlayer* player)
	{
		m_player = player;
		m_track = m_player->CreateTrack();
	}

	void Waveform::SetAudioData(const FloatTensor2D& data)
	{
		m_audio = (data * 32767.f).Cast<short>().Evaluate();
		static_cast<WAVAudio*>(m_audioData)->source = m_audio.View();
		if (m_player && m_track)
			m_player->SetTrackSound(m_track, static_cast<MAudio*>(m_audioData));
		m_cacheUpdate = true;
	}

	size_t Waveform::GetPCMSize() const
	{
		return static_cast<WAVAudio*>(m_audioData)->PCMGetDataSize() / sizeof(std::int16_t);
	}

	void Waveform::SetPtrOffset(_m_ptrv offset)
	{
		if(m_callback)
		{
			double pre = (double)offset / (double)m_audio.ElementCount();
			auto maudio = static_cast<WAVAudio*>(m_audioData);
			auto pers = offset * sizeof(std::int16_t) / maudio->GetBlockAlign();
			double sc = (double)pers / (double)maudio->GetSamplerate();
			m_callback((float)pre, (float)sc);
		}
		m_ptrOffset = offset;
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void Waveform::SetPlayCallback(std::function<void(float, float)> callback)
	{
		m_callback = std::move(callback);
	}

	float Waveform::GetDataDuration() const
	{
		if (m_player && !m_audio.Null())
			return static_cast<WAVAudio*>(m_audioData)->GetDuration();
		return 0.0f;
	}

	void Waveform::SetPlayPos(_m_ptrv offset)
	{
		SetPtrOffset(offset);
		offset *= sizeof(std::int16_t);
		auto lastpos = m_player->GetTrackPlaybackPos(m_track);

		//计算时间
		auto maudio = static_cast<WAVAudio*>(m_audioData);
		auto pers = offset / maudio->GetBlockAlign();
		double sc = (double)pers / (double)maudio->GetSamplerate();
		m_player->SetTrackPlaybackPos(m_track, (float)sc);

		//播放器播放完毕会自动暂停 重新播放
		if ((int)lastpos == (int)static_cast<WAVAudio*>(m_audioData)->GetDuration())
			m_player->PlayTrack(m_track);
	}

	void Waveform::Clear()
	{
		if (m_player && m_track)
			m_player->StopTrack(m_track);
		m_audio.Clear();
		static_cast<WAVAudio*>(m_audioData)->source.Clear();
		m_previewData.clear();
		m_overviewData.clear();
		m_ptrOffset = 0;
		m_viewScale = 1.f;
		m_lastScale = 1.f;
		m_lineRange = { 0, 0 };
		m_isPause = true;
		SetAttributeSrc(L"dragValueH", 0, false);
		SetAttributeSrc(L"rangeH", 0, false);
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void Waveform::Play() const
	{
		if (m_player && m_track)
			m_player->PlayTrack(m_track);
	}

	void Waveform::Pause() const
	{
		if (m_player && m_track)
			m_player->PauseTrack(m_track);
	}

	void Waveform::PlayPause()
	{
		if (m_player && m_track)
		{
			if (m_isPause)
			{
				m_isPause = false;
				m_player->PlayTrack(m_track);
			}
			else
			{
				m_isPause = true;
				m_player->PauseTrack(m_track);
			}
		}
	}

	void Waveform::SetVolume(_m_byte vol) const
	{
		if (m_player && m_track)
			m_player->SetTrackVolume(m_track, vol);
	}

	void Waveform::SetAniFlag(bool ani)
	{
		m_isani = ani;
		Resample(UINodeBase::m_data.Frame.GetWidth(), true);
		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void Waveform::SetAttribute(std::wstring_view attribName, std::wstring_view attrib, bool draw)
	{
		if (attribName == L"ptrColor")
			m_ptrColor = Helper::M_GetAttribValueColor(attrib);
		else if (attribName == L"wavColor")
			m_wavColor = Helper::M_GetAttribValueColor(attrib);
		else if (attribName == L"lineColor")
			m_lineColor = Helper::M_GetAttribValueColor(attrib);
		else if (attribName == L"preHeight")
		{
			m_preHeight = Helper::M_StoInt(attrib);
			UIScroll::SetAttributeSrc(L"horizontal", m_preHeight != 0, false);
			UIScroll::SetAttributeSrc(L"barWidth", m_preHeight);
		}
		else
		{
			UIScroll::SetAttribute(attribName, attrib, draw);
			return;
		}
		m_cacheUpdate = true;
		if (draw)
			UpdateDisplay();
	}

	std::wstring Waveform::GetAttribute(std::wstring_view attribName)
	{
		if (attribName == L"ptrColor")
			return Color::M_RGBA_STR(m_ptrColor);
		if (attribName == L"wavColor")
			return std::to_wstring(m_wavColor);
		if (attribName == L"lineColor")
			return std::to_wstring(m_lineColor);
		if (attribName == L"preHeight")
			return std::to_wstring(m_preHeight);

		return UIScroll::GetAttribute(attribName);
	}

	void Waveform::OnLoadResource(MRenderCmd* render, bool recreate)
	{
		UIScroll::OnLoadResource(render, recreate);

		MSafeRelease(m_pen);
		MSafeRelease(m_brush_m);
		m_pen = render->CreatePen(1, m_wavColor);
		m_brush_m = render->CreateBrush(m_ptrColor);
	}

	void Waveform::OnPaintProc(MPCPaintParam param)
	{
		if (m_audio.Null())
		{
			UIScroll::OnPaintProc(param);
			return;
		}
		int width = param->destRect->GetWidth();
		int height = param->destRect->GetHeight();

		m_brush_m->SetColor(Color::M_RGBA(53, 192, 242, 255));

		if (m_width != width || m_overviewData.empty() || m_previewData.empty())
		{
			Resample(width, true);
			m_width = width;
		}

		auto scale = GetRectScale().scale();
		int preHeight = _scale_to(m_preHeight, scale.cy);

		int offsetY = preHeight + _scale_to(20, scale.cy);
		height -= offsetY;

		if (std::wstring_view(param->render->GetRenderName()) != L"D2D")
		{
			UIScroll::OnPaintProc(param);
			return;
		}

		float dstX = 0.f;
		float dstY = 0.f;

#ifdef _WIN32

		auto render = param->render->GetBase<MRender_D2D>();

		auto context = static_cast<ID2D1DeviceContext*>(render->Get());

		//context->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

		ID2D1Factory* factory = nullptr;
		context->GetFactory(&factory);

		//因为直接调用的D2D接口 没有走界面库接口 缓存画布是Atlas 需要偏移矩形
		if(param->cacheCanvas)
		{
			UIRect subrect = param->render->GetCanvas()->GetSubRect();
			dstX += (float)subrect.left;
			dstY += (float)subrect.top;
		}
#endif

		dstX += (float)param->destRect->left;
		dstY += (float)param->destRect->top + (float)_scale_to(10, scale.cy);

		auto drawWaveform = [
#ifdef _WIN32
			render, factory, context,
#else
			&param,
#endif
			&width, this
		]
		(float dx, float dy, int height, const auto& data)
		{
			const float center = dy + (float)height / 2.f;
			float scaleY = (float)height / 65536.0f;//SHORT_MAX

			//Windows下使用D2D渲染器可以用几何形实现高质量的曲线效果
#ifdef _WIN32
			//创建路径几何图形
			ID2D1PathGeometry* geometry;
			factory->CreatePathGeometry(&geometry);

			ID2D1GeometrySink* sink = nullptr;
			geometry->Open(&sink);

			sink->BeginFigure(D2D1::Point2F(dx, center), D2D1_FIGURE_BEGIN_FILLED);

			float offset = (float)data.size() / (float)width;
			for (int i = 0; i < width; ++i)
			{
				const float x = dx + (float)i;
				float y = center - float(data[int((float)i * offset)].first) * scaleY;

				sink->AddLine(D2D1::Point2F(x, y));

				y = center - float(data[int((float)i * offset)].second) * scaleY;
				sink->AddLine(D2D1::Point2F(x, y));
			}

			sink->EndFigure(D2D1_FIGURE_END_OPEN);
			sink->Close();
			sink->Release();

			context->DrawGeometry(geometry, MD2DBrush::GetBrush(m_brush_m));

			geometry->Release();

			//对于其他平台 OpenGL渲染器 使用传统的DrawLine 可实现近似效果 在不缩放的情况下效果一样
#else
			for (int i = 0; i < width; i++)
			{
				const float x = (float)i;
				float y = center - float(data[i].first) * scaleY;
				param->render->DrawLine({ (int)x, (int)center }, { (int)x, (int)y }, m_pen);

				y = center - float(data[i].second) * scaleY;
				param->render->DrawLine({ (int)x, (int)center }, { (int)x, (int)y }, m_pen);
			}

			param->render->DrawLine({ (int)dx, (int)center }, { width, (int)center }, m_pen);
#endif
		};

		m_pen->SetColor(m_wavColor);
		m_pen->SetOpacity(param->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst);

		auto add = _scale_to(10, scale.cy);
		drawWaveform(dstX, dstY, height, m_overviewData);
		if (m_preHeight != 0)
		{
			dstY += (float)height + (float)add;
			height = _scale_to(m_preHeight, scale.cy) - add / 2;
			drawWaveform(dstX, dstY, height, m_previewData);
		}

		m_brush_m->SetColor(Color::M_RED);
		m_brush_m->SetOpacity(param->cacheCanvas ? 255 : UINodeBase::m_data.AlphaDst);

		//音频播放ptr线
		auto aptr_x = (double)m_ptrOffset / (double)m_audio.ElementCount() * (double)width;
		int aptr_y = param->destRect->bottom - preHeight + 1;

		UIRect ptrRect((int)aptr_x, aptr_y, _scale_to(2, scale.cx), preHeight);

		param->render->FillRectangle(ptrRect, m_brush_m);

		//音频ptr在缩放视图中的线
		auto& sattrib = UIScroll::GetAttribute();
		float soffset = 0;
		if(sattrib.dragValue.width != 0)
			soffset = (float)CalcOffsetDragValue(true, sattrib.dragValue.width, width) / (float)sattrib.range.width;
		_m_size poffset = _m_size((double)soffset * (double)m_audio.ElementCount());
		if (m_ptrOffset >= poffset)
		{
			poffset = m_ptrOffset - poffset;

			auto audioSize = (double)m_audio.ElementCount() / (double)m_viewScale;
			if (poffset != 0)
				poffset = _m_size((double)poffset / audioSize * (double)width);

			if ((int)poffset <= width)
			{
				m_brush_m->SetColor(m_ptrColor);
				aptr_x = (int)poffset;
				aptr_y = 0;
				ptrRect = UIRect((int)aptr_x, aptr_y, _scale_to(2, scale.cx), param->destRect->GetHeight() - preHeight);
				param->render->FillRectangle(ptrRect, m_brush_m);
			}
		}

		UIScroll::OnPaintProc(param);
	}

	/*bool Waveform::OnMouseWheel(_m_uint flag, short delta, const UIPoint& point)
	{
		if (!UIScroll::OnMouseWheel(flag, delta, point) && m_preHeight != 0)
		{
			float delta_scale = 0.03f;
			if (GetKeyState(VK_LCONTROL) & 0x8000)
				delta_scale = 0.1f;
			float delta_ = (float)delta / static_cast<float>(WHEEL_DELTA);
			m_viewScale *= powf(4.2f, delta_ * delta_scale);
			m_viewScale = Helper::M_MAX(Helper::M_MIN(500.f, m_viewScale) * 1.f, 1.f); //最大50000%最小100%

			//m_viewScale != m_lastScale
			if (abs(m_viewScale - m_lastScale) > std::numeric_limits<float>::epsilon())
				m_lastScale = m_viewScale;
			return true;
		}
		return false;
	}*/

	bool Waveform::OnSetCursor(_m_param hCur, _m_param lParam)
	{
#ifdef _WIN32
		::SetCursor((HCURSOR)hCur);
#endif
		return true;
	}

	bool Waveform::OnLButtonDown(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnLButtonDown(flag, point))
			return true;

		if (!m_isDown)
		{
			m_isDown = true;
			if (!m_player || !m_track || m_audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			m_player->PauseTrack(m_track);
			m_isPause = true;
			int x = point.x - UINodeBase::m_data.Frame.left;
			auto poffset = GetOffset(x);
			SetPtrOffset(poffset);
			m_cacheUpdate = true;
			UpdateDisplay();
			return true;
		}
		return false;
	}

	bool Waveform::OnMouseMove(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnMouseMove(flag, point))
			return true;

		if (m_isDown)
		{
			if (!m_player || !m_track || m_audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			int x = point.x - UINodeBase::m_data.Frame.left;
			auto poffset = GetOffset(x);
			SetPtrOffset(poffset);
			m_cacheUpdate = true;
			UpdateDisplay();
			return true;
		}
		return false;
	}

	bool Waveform::OnLButtonUp(_m_uint flag, const UIPoint& point)
	{
		if (UIScroll::OnLButtonUp(flag, point))
			return true;

		if (m_isDown)
		{
			m_isDown = false;
			if (!m_player || !m_track || m_audio.Null())
				return UIControl::OnLButtonDown(flag, point);
			int x = point.x - UINodeBase::m_data.Frame.left;
			SetPlayPosWithX(x);
		}
		return true;
	}

	bool Waveform::OnMouseExited(_m_uint flag, const UIPoint& point)
	{
		SetCursor(IDC_ARROW);
		if (m_isDown)
		{
			m_isDown = false;
			if (!m_player || !m_track || m_audio.Null())
				return UIControl::OnMouseExited(flag, point);
			int x = point.x - UINodeBase::m_data.Frame.left;
			SetPlayPosWithX(x);
		}
		return UIScroll::OnMouseExited(flag, point);
	}

	_m_size Waveform::GetOffset(int x)
	{
		//计算点击位置在缩放视图中对应的音频offset
		_m_size poffset;

		auto& sattrib = UIScroll::GetAttribute();
		float soffset;
		if (sattrib.dragValue.width != 0)
		{
			int dragOffset = CalcOffsetDragValue(true, sattrib.dragValue.width, UINodeBase::m_data.Frame.GetWidth());
			soffset = Helper::M_MIN((float)dragOffset / (float)sattrib.range.width, 1.f);
			double audioSize = (double)m_audio.ElementCount() / (double)m_viewScale;
			audioSize *= (double)Helper::M_MIN((float)x / (float)UINodeBase::m_data.Frame.GetWidth(), 1.f);
			poffset = _m_size((double)soffset * (double)m_audio.ElementCount()) + (_m_size)audioSize;
		}
		else
		{
			soffset = Helper::M_MIN((float)x / (float)UINodeBase::m_data.Frame.GetWidth(), 1.f);
			//计算缩放视图后的可视size
			double audioSize = (double)m_audio.ElementCount() / (double)m_viewScale;
			poffset = _m_size((double)soffset * audioSize);
		}
		return poffset;
	}

	void Waveform::SetPlayPosWithX(int x)
	{
		auto poffset = GetOffset(x);
		SetPtrOffset(poffset);
		poffset *= sizeof(std::int16_t);
		auto lastpos = m_player->GetTrackPlaybackPos(m_track);

		//计算时间
		auto maudio = static_cast<WAVAudio*>(m_audioData);
		auto pers = poffset / maudio->GetBlockAlign();
		double sc = (double)pers / (double)maudio->GetSamplerate();
		m_player->SetTrackPlaybackPos(m_track, (float)sc);

		//播放器播放完毕会自动暂停 重新播放
		if ((int)lastpos == (int)static_cast<WAVAudio*>(m_audioData)->GetDuration())
			m_player->PlayTrack(m_track);

		m_cacheUpdate = true;
		UpdateDisplay();
	}

	void Waveform::OnScrollView(UIScroll*, int dragValue, bool horizontal)
	{
		if (!horizontal || m_preHeight == 0) return;
		Resample(UINodeBase::m_data.Frame.GetWidth());
	}

	void Waveform::Resample(int width, bool pre)
	{
		if(m_audio.Null() || m_isani) return;

		m_overviewData.resize(width);

		const auto& attrib = UIScroll::GetAttribute();

		_m_size audioSize = m_audio.ElementCount();
		_m_size maxSize = m_audio.ElementCount();

		audioSize = _m_size((double)audioSize / (double)m_viewScale);

		_m_size N = audioSize / width;
		if (N == 0)
			N = 1;
		float soffset = 0.f;
		if(attrib.dragValue.width != 0)
			soffset = (float)CalcOffsetDragValue(true, attrib.dragValue.width, width) / (float)attrib.range.width;

		_m_size offset = _m_size((double)soffset * (double)m_audio.ElementCount());

		auto audio_data = m_audio.Data();
		for (_m_size i = 0; i < audioSize; i += N)
		{
			_m_size index = offset + i;
			if (index >= maxSize)
				break;

			_m_size endIndex = Helper::M_MIN(i + N, audioSize);
			short maxSample = audio_data[index];
			short minSample = maxSample;
			for (size_t j = i + 1; j < endIndex; ++j)
			{
				auto _index = offset + j;
				if(_index >= maxSize)
					continue;
				if (audio_data[_index] > maxSample)
					maxSample = audio_data[_index];
				if (audio_data[_index] < minSample)
					minSample = audio_data[_index];
			}
			auto sindex = i / N;
			if (sindex < (size_t)width)
				m_overviewData[sindex] = std::make_pair(maxSample, minSample);
		}

		//刷新预览图
		if (!pre)
			return;

		m_previewData.resize(width);
		if(soffset == 0.f && audioSize == maxSize)
		{
			m_previewData = m_overviewData;
			return;
		}

		for (_m_size i = 0; i < maxSize; i += N)
		{
			_m_size endIndex = Helper::M_MIN(i + N, maxSize);
			short maxSample = audio_data[i];
			short minSample = maxSample;
			for (size_t j = i + 1; j < endIndex; ++j)
			{
				if (audio_data[j] > maxSample)
					maxSample = audio_data[j];
				if (audio_data[j] < minSample)
					minSample = audio_data[j];
			}
			auto sindex = i / N;
			if (sindex < (size_t)width)
				m_previewData[sindex] = std::make_pair(maxSample, minSample);
		}
	}
}