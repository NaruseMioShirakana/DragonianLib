#pragma once
#include <Mui.h>
#include <numbers>
#include <Windows.h>
#include <User/Mui_Engine.h>

#include "Libraries/AvCodec/AvCodec.h"

namespace SimpleF0Labeler
{
	using FloatTensor1D = DragonianLib::Tensor<DragonianLib::Float32, 1, DragonianLib::Device::CPU>;
	using FloatTensor2D = DragonianLib::Tensor<DragonianLib::Float32, 2, DragonianLib::Device::CPU>;
	using Int16Tensor2D = DragonianLib::Tensor<DragonianLib::Int16, 2, DragonianLib::Device::CPU>;
	using ImageTensor = DragonianLib::Tensor<DragonianLib::Int32, 2, DragonianLib::Device::CPU>;

	class PitchLabel
	{
	public:
		static std::wstring PitchToLabel(float pitch_val)
		{
			auto pitch = int(round(pitch_val));
			if (pitch < 0 || pitch > 1200)
				return L"Null";
			const int realPitch = pitch / 10;
			pitch = pitch % 10;
			if (pitch == 0)
				return Labels[realPitch % 12] + std::to_wstring(realPitch / 12);
			return Labels[realPitch % 12] + std::to_wstring(realPitch / 12) + L',' + std::to_wstring(pitch * 10) + L'\'';
		}

		static float F0ToPitch(float f0_val)
		{
			return (12 * log(float(f0_val) / CenterC) / std::numbers::ln2_v<float>) + CenterCPitch;
		}

		static float PitchToF0(float pitch_val)
		{
			return (powf(2.f, (pitch_val - CenterCPitch) / 12.f)) * CenterC;
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
			FloatTensor2D _Spec,
			FloatTensor2D _Mel,
			std::wstring _F0Path,
			bool Modified
		) :
			SamplingRate(_SamplingRate),
			Audio(std::move(_Audio)),
			F0(std::move(_F0)),
			Spec(std::move(_Spec)),
			Mel(std::move(_Mel)),
			F0Path(std::move(_F0Path))
		{
			if (Modified)
				ModifyCount = 1;
		}

		DragonianLib::Int64 SamplingRate;
		FloatTensor2D Audio;
		FloatTensor2D F0;
		FloatTensor2D Spec;
		FloatTensor2D Mel;
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