#pragma once
#include <Mui.h>
#include <numbers>
#include <Render/Sound/Mui_SoundDef.h>
#include "Libraries/AvCodec/AvCodec.h"

namespace SimpleF0Labeler
{
	using FloatTensor1D = DragonianLib::Tensor<DragonianLib::Float32, 1, DragonianLib::Device::CPU>;
	using FloatTensor2D = DragonianLib::Tensor<DragonianLib::Float32, 2, DragonianLib::Device::CPU>;
	using Int16Tensor2D = DragonianLib::Tensor<DragonianLib::Int16, 2, DragonianLib::Device::CPU>;
	using ImageTensor = DragonianLib::Tensor<DragonianLib::UInt32, 2, DragonianLib::Device::CPU>;

	class PCMAudio : public Mui::Render::MAudio
	{
	public:
		PCMAudio() = default;
		PCMAudio(
			DragonianLib::UInt SamplingRate,
			const Int16Tensor2D& AudioData,
			void(*Callback)(size_t)
		) : _MySamplingRate(SamplingRate), _MySource(AudioData.View()), _MyCallback(Callback)
		{

		}

		void SetData(const Int16Tensor2D& AudioData) { _MySource = AudioData.View(); }
		void SetSamplingRate(DragonianLib::UInt SamplingRate) { _MySamplingRate = SamplingRate; }
		void SetCallback(std::function<void(size_t)> Callback) { _MyCallback = std::move(Callback); }

		float GetDuration() override
		{
			auto NSamples = _MySource.Size(0);
			return (float)((double)NSamples / double(_MySamplingRate));
		}

		//获取音频位率(bits)
		DragonianLib::UInt GetBitrate() override { return 16; }

		//获取音频比特率(kbps)
		DragonianLib::UInt GetBitPerSecond() override { return GetBitrate() * GetSamplerate() * GetChannel(); }

		//获取音频采样率(hz)
		DragonianLib::UInt GetSamplerate() override { return _MySamplingRate; }

		//获取音频声道数
		DragonianLib::UInt GetChannel() override { return 2; }

		//获取字节对齐数
		DragonianLib::UInt GetBlockAlign() override { return 4; }

		//获取PCM数据尺寸
		DragonianLib::StdSize PCMGetDataSize() override { return _MySource.ElementCount() * sizeof(DragonianLib::Int16); }

		//读取PCM数据
		DragonianLib::StdSize PCMReadData(
			DragonianLib::StdSize Begin,
			DragonianLib::StdSize Size,
			DragonianLib::Byte* Dest
		) override
		{
			if (_MySource.Null())
				return 0;
			auto maxSize = _MySource.ElementCount() * sizeof(std::int16_t);
			if (Begin >= maxSize)
				return 0;

			if (Begin + Size >= maxSize)
				Size = maxSize - Begin - 1;

			auto offset = _MySource.Data() + Begin / sizeof(std::int16_t);

			memcpy(Dest, offset, Size);

			_MyCallback((Begin) / sizeof(std::int16_t));

			return Size;
		}

		operator bool() const
		{
			return _MySource.HasValue();
		}

	private:
		DragonianLib::UInt _MySamplingRate = 0;
		Int16Tensor2D _MySource = std::nullopt;
		std::function<void(size_t)> _MyCallback = nullptr;
	};

	class PitchLabel
	{
	public:
		static std::wstring PitchToLabel(float PitchValue)
		{
			auto Pitch = int(round(PitchValue));
			if (Pitch < 0 || Pitch > 1200)
				return L"Null";
			const int Keys = Pitch / 10;
			Pitch = Pitch % 10;
			if (Pitch == 0)
				return Labels[Keys % 12] + std::to_wstring(Keys / 12);
			return Labels[Keys % 12] + std::to_wstring(Keys / 12) + L',' + std::to_wstring(Pitch * 10) + L'\'';
		}

		static float F0ToPitch(float Freq)
		{
			return (12 * log(float(Freq) / CenterC) / std::numbers::ln2_v<float>) + CenterCPitch;
		}

		static float PitchToF0(float PitchValue)
		{
			return (powf(2.f, (PitchValue - CenterCPitch) / 12.f)) * CenterC;
		}

	private:
		constexpr static float CenterC = 261.626f;
		constexpr static float CenterCPitch = 4.f * 12.f;
		constexpr static const wchar_t* Labels[12] = {
			L"C", L"C#", L"D", L"D#", L"E", L"F",
			L"F#", L"G", L"G#", L"A", L"A#", L"B"
		};
	};

	struct MyAudioData
	{
		MyAudioData(
			DragonianLib::Int64 _SamplingRate,
			FloatTensor2D _Audio,
			FloatTensor2D _F0,
			FloatTensor2D _RawSpec,
			FloatTensor2D _RawMel,
			std::wstring _F0Path,
			bool _UseLogSpec,
			bool _Modified
		) :
			SamplingRate(_SamplingRate),
			Audio(std::move(_Audio)),
			F0(std::move(_F0)),
			RawSpec(std::move(_RawSpec)),
			RawMel(std::move(_RawMel)),
			F0Path(std::move(_F0Path))
		{
			if (_Modified)
				ModifyCount = 1;
			CalcSpec(_UseLogSpec);
		}

		void CalcSpec(bool _UseLogSpec);

		DragonianLib::Int64 SamplingRate;
		FloatTensor2D Audio;
		FloatTensor2D F0;
		FloatTensor2D RawSpec;
		FloatTensor2D RawMel;
		ImageTensor Spec;
		ImageTensor Mel;
		ImageTensor LogSpec;
		ImageTensor LogMel;
		std::wstring F0Path;
		int64_t ModifyCount = 0;
		std::deque<FloatTensor2D> UndoList, RedoList;

		MyAudioData(const MyAudioData&) = delete;
		MyAudioData(MyAudioData&&) noexcept = default;
		MyAudioData& operator=(const MyAudioData&) = delete;
		MyAudioData& operator=(MyAudioData&&) noexcept = default;

		~MyAudioData();
		void SaveCache() const;
	};
}