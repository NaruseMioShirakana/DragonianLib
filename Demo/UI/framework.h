#pragma once
#include <sdkddkver.h>
#include <Windows.h>
#include <Mui.h>
#include <User/Mui_Engine.h>
#include "Libraries/AvCodec/AvCodec.h"

namespace Mui::Ctrl
{
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
			return (12 * log(float(f0_val) / CenterC) / log(2.f)) + CenterCPitch;
		}

		static float PitchToF0(float pitch_val)
		{
			return (powf(2.f, (pitch_val - CenterCPitch) / 12.f)) * CenterC;
		}

	private:
		constexpr static float CenterC = 261.626f;
		constexpr static float CenterCPitch = 4.f * 12.f;
		constexpr static const wchar_t* Labels[12] = { L"C", L"C#", L"D", L"D#", L"E", L"F", L"F#", L"G", L"G#", L"A", L"A#", L"B" };
	};
}