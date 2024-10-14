/**
 * FileName: AvCodec.h
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include <string>
#include "MyTemplateLibrary/Vector.h"
struct AVFrame;
struct SwrContext;
struct AVCodecContext;
struct AVFormatContext;
struct AVPacket;

namespace DragonianLib
{
    struct MidiEvent
    {
        double Time = 0;
        long MidiNote = 0;
        long Velocity = 0;
        MidiEvent(double t = 0.0, long m = 0, long v = 0) :Time(t), MidiNote(m), Velocity(v) {}
        bool operator<(const MidiEvent& b) const { return Time < b.Time; }
    };

    struct EstPedalEvents
    {
        double OnsetTime = 0.0;
        double OffsetTime = 0.0;
    };

    struct EstNoteEvents
    {
        double OnsetTime = 0.0;
        double OffsetTime = 0.0;
        long MidiNote = 0;
        long Velocity = 0;
        EstNoteEvents(double a, double b, long c, long d) :OnsetTime(a), OffsetTime(b), MidiNote(c), Velocity(d) {}
    };

    struct MidiTrack
    {
        DragonianLibSTL::Vector<EstNoteEvents> NoteEvents;
        DragonianLibSTL::Vector<EstPedalEvents> PedalEvents;
        MidiTrack() = default;
        MidiTrack(DragonianLibSTL::Vector<EstNoteEvents>&& ene, DragonianLibSTL::Vector<EstPedalEvents>&& epe) : NoteEvents(std::move(ene)), PedalEvents(std::move(epe)) {}
    };

	class AvCodec
	{
	public:
		DragonianLibSTL::Vector<unsigned char> Decode(
			const char* _AudioPath,
			int _OutSamplingRate,
			int _OutChannels = 1,
			bool _OutFloat = false,
			bool _OutPlanar = false
		);

		DragonianLibSTL::Vector<float> DecodeFloat(
			const char* _AudioPath,
			int _OutSamplingRate,
			int _OutChannels = 1,
			bool _OutPlanar = false
		);

		DragonianLibSTL::Vector<int16_t> DecodeSigned16(
			const char* _AudioPath,
			int _OutSamplingRate,
			int _OutChannels = 1,
			bool _OutPlanar = false
		);

		void Encode(
			const char* _OutPutPath,
			const DragonianLibSTL::Vector<unsigned char>& _PcmData,
			int _SrcSamplingRate,
			int _OutSamplingRate,
			int _SrcChannels,
			int _OutChannels = 1,
			bool _IsFloat = false,
			bool _IsPlanar = false
		);
		void Release();
		void Init();
		AvCodec();
		~AvCodec();
		AvCodec(const AvCodec&) = delete;
		AvCodec(AvCodec&&) = delete;
		AvCodec operator=(const AvCodec&) = delete;
		AvCodec operator=(AvCodec&&) = delete;
	private:
		AVFrame* inFrame = nullptr;
		SwrContext* swrContext = nullptr;
		AVCodecContext* avCodecContext = nullptr;
		AVFormatContext* avFormatContext = nullptr;
		AVPacket* packet = nullptr;
		uint8_t* outBuffer[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr };
	};

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
	struct RiffWaveHeader {
		char				RiffHeader[4] = { 'R','I','F','F' };
		uint32_t			RiffChunkSize;
		char				WaveHeader[4] = { 'W','A','V','E' };
		char				FmtHeader[4] = { 'f','m','t',' ' };
		uint32_t			WaveChunkSize;
		uint16_t			AudioFormat;
		uint16_t			ChannelCount;
		uint32_t			SamplingRate;
		uint32_t			ByteRate;
		uint16_t			SampleAlign;
		uint16_t			SampleBits;
		char				DataChunkHeader[4] = { 'd','a','t','a' };
		uint32_t			DataChunkSize;
	};
#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	void WritePCMData(
		const wchar_t* _OutPutPath,
		const DragonianLibSTL::Vector<unsigned char>& PCMDATA,
		int _SamplingRate,
		int _Channels = 1,
		bool _IsFloat = false,
		bool _IsPlanar = false
	);

	void WritePCMData(
		const wchar_t* _OutPutPath,
		const DragonianLibSTL::Vector<float>& PCMDATA,
		int _SamplingRate,
		int _Channels = 1,
		bool _IsPlanar = false
	);

	void WritePCMData(
		const wchar_t* _OutPutPath,
		const DragonianLibSTL::Vector<int16_t>& PCMDATA,
		int _SamplingRate,
		int _Channels = 1,
		bool _IsPlanar = false
	);

	void WritePCMData(
		const wchar_t* _OutPutPath,
		const unsigned char* PCMDATA,
		size_t _BufferSize,
		int _SamplingRate,
		int _Channels = 1,
		bool _IsFloat = false,
		bool _IsPlanar = false
	);

    void WriteMidiFile(const std::wstring& _Path, const MidiTrack& _Events, long _Begin, long _TPS, long tempo = 500000);

}

