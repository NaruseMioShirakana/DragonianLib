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

// Forward declarations to avoid including unnecessary headers
struct AVFrame;
struct SwrContext;
struct AVCodecContext;
struct AVFormatContext;
struct AVPacket;

namespace DragonianLib
{
	// MidiEvent is a struct that contains the time, MIDI note, and velocity of a MIDI event
	struct MidiEvent
	{
		double Time = 0; // Time of the event
		long MidiNote = 0; // MIDI note
		long Velocity = 0; // Velocity
		MidiEvent(double t = 0.0, long m = 0, long v = 0) :Time(t), MidiNote(m), Velocity(v) {}
		bool operator<(const MidiEvent& b) const { return Time < b.Time; } // Comparison operator for sorting
	};

	// EstPedalEvents is a struct that contains the onset time and offset time of a MIDI pedal event
	struct EstPedalEvents
	{
		double OnsetTime = 0.0; // Onset time of the pedal event
		double OffsetTime = 0.0; // Offset time of the pedal event
	};

	// EstNoteEvents is a struct that contains the onset time, offset time, MIDI note, and velocity of a MIDI note event
	struct EstNoteEvents
	{
		double OnsetTime = 0.0; // Onset time of the note event
		double OffsetTime = 0.0; // Offset time of the note event
		long MidiNote = 0; // MIDI note
		long Velocity = 0; // Velocity
		EstNoteEvents(double a, double b, long c, long d) :OnsetTime(a), OffsetTime(b), MidiNote(c), Velocity(d) {}
	};

	// MidiTrack is a struct that contains all the events of a MIDI track
	struct MidiTrack
	{
		DragonianLibSTL::Vector<EstNoteEvents> NoteEvents; // Note events
		DragonianLibSTL::Vector<EstPedalEvents> PedalEvents; // Pedal events
		MidiTrack() = default;
		MidiTrack(DragonianLibSTL::Vector<EstNoteEvents>&& ene, DragonianLibSTL::Vector<EstPedalEvents>&& epe) : NoteEvents(std::move(ene)), PedalEvents(std::move(epe)) {}
	};

	/**
	 * @class AvCodec
	 * @brief Implementation of audio codec
	 */
	class AvCodec
	{
	public:
		AvCodec();
		~AvCodec();
		AvCodec(const AvCodec&) = delete;
		AvCodec(AvCodec&&) = delete;
		AvCodec operator=(const AvCodec&) = delete;
		AvCodec operator=(AvCodec&&) = delete;

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutSamplingRate Sampling rate of the output audio
		 * @param OutChannels Channels of the output audio
		 * @param OutFloat Output audio is float
		 * @param OutPlanar Output audio is planar
		 * @return Raw PCM data
		 */
		DragonianLibSTL::Vector<unsigned char> Decode(
			const char* AudioPath,
			int OutSamplingRate,
			int OutChannels = 1,
			bool OutFloat = false,
			bool OutPlanar = false
		);

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutSamplingRate Sampling rate of the output audio
		 * @param OutChannels Channels of the output audio
		 * @param OutPlanar Output audio is planar
		 * @return Float PCM data
		 */
		DragonianLibSTL::Vector<float> DecodeFloat(
			const char* AudioPath,
			int OutSamplingRate,
			int OutChannels = 1,
			bool OutPlanar = false
		);

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutSamplingRate Sampling rate of the output audio
		 * @param OutChannels Channels of the output audio
		 * @param OutPlanar Output audio is planar
		 * @return Signed 16-bit PCM data
		 */
		DragonianLibSTL::Vector<int16_t> DecodeSigned16(
			const char* AudioPath,
			int OutSamplingRate,
			int OutChannels = 1,
			bool OutPlanar = false
		);

		/**
		 * @brief Encode an audio file
		 * @param OutPutPath Path to the output audio file
		 * @param PcmData Raw PCM data
		 * @param SrcSamplingRate Sampling rate of the input audio
		 * @param OutSamplingRate Sampling rate of the output audio
		 * @param SrcChannels Channels of the input audio
		 * @param OutChannels Channels of the output audio
		 * @param IsFloat Input audio is float
		 * @param IsPlanar Input audio is planar
		 */
		void Encode(
			const char* OutPutPath,
			const DragonianLibSTL::Vector<unsigned char>& PcmData,
			int SrcSamplingRate,
			int OutSamplingRate,
			int SrcChannels,
			int OutChannels = 1,
			bool IsFloat = false,
			bool IsPlanar = false
		);

		// Release the encoder/decoder
		void Release();

		// Initialize the encoder/decoder
		void Init();

	private:
		AVFrame* InFrame = nullptr; // Input frame
		SwrContext* SwrContext = nullptr; // Resampling context
		AVCodecContext* AvCodecContext = nullptr; // Codec context
		AVFormatContext* AvFormatContext = nullptr; // Format context
		AVPacket* Packet = nullptr; // Packet
		uint8_t* OutBuffer[8] = { nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr , nullptr }; // Output buffer
	};

#ifdef _MSC_VER
#pragma pack(push, 1)
#else
#pragma pack(1)
#endif
	// RiffWaveHeader is a struct that contains the header information of a RIFF WAV file
	struct RiffWaveHeader {
		char				RiffHeader[4] = { 'R','I','F','F' }; // RIFF header
		uint32_t			RiffChunkSize; // RIFF chunk size
		char				WaveHeader[4] = { 'W','A','V','E' }; // WAVE header
		char				FmtHeader[4] = { 'f','m','t',' ' }; // fmt header
		uint32_t			WaveChunkSize; // WAVE chunk size
		uint16_t			AudioFormat; // Audio format
		uint16_t			ChannelCount; // Channel count
		uint32_t			SamplingRate; // Sampling rate
		uint32_t			ByteRate; // Byte rate
		uint16_t			SampleAlign; // Sample alignment
		uint16_t			SampleBits; // Sample bits
		char				DataChunkHeader[4] = { 'd','a','t','a' }; // Data chunk header
		uint32_t			DataChunkSize; // Data chunk size
	};
#ifdef _MSC_VER
#pragma pack(pop)
#else
#pragma pack()
#endif

	/**
	 * @brief Write PCM data to a file
	 * @param OutPutPath Path to the output file
	 * @param PcmData PCM data
	 * @param SamplingRate Sampling rate
	 * @param Channels Channels
	 * @param IsFloat Data is float format
	 * @param IsPlanar Data is planar
	 */
	void WritePCMData(
		const wchar_t* OutPutPath,
		const DragonianLibSTL::Vector<unsigned char>& PcmData,
		int SamplingRate,
		int Channels = 1,
		bool IsFloat = false,
		bool IsPlanar = false
	);

	/**
	 * @brief Write float PCM data to a file
	 * @param OutPutPath Path to the output file
	 * @param PcmData PCM data
	 * @param SamplingRate Sampling rate
	 * @param Channels Channels
	 * @param IsPlanar Data is planar
	 */
	void WritePCMData(
		const wchar_t* OutPutPath,
		const DragonianLibSTL::Vector<float>& PcmData,
		int SamplingRate,
		int Channels = 1,
		bool IsPlanar = false
	);

	/**
	 * @brief Write signed 16-bit PCM data to a file
	 * @param OutPutPath Path to the output file
	 * @param PcmData PCM data
	 * @param SamplingRate Sampling rate
	 * @param Channels Channels
	 * @param IsPlanar Data is planar
	 */
	void WritePCMData(
		const wchar_t* OutPutPath,
		const DragonianLibSTL::Vector<int16_t>& PcmData,
		int SamplingRate,
		int Channels = 1,
		bool IsPlanar = false
	);

	/**
	 * @brief Write PCM data to a file
	 * @param OutPutPath Path to the output file
	 * @param PcmData PCM data
	 * @param BufferSize Buffer size
	 * @param SamplingRate Sampling rate
	 * @param Channels Channels
	 * @param IsFloat Data is float format
	 * @param IsPlanar Data is planar
	 */
	void WritePCMData(
		const wchar_t* OutPutPath,
		const unsigned char* PcmData,
		size_t BufferSize,
		int SamplingRate,
		int Channels = 1,
		bool IsFloat = false,
		bool IsPlanar = false
	);

	/**
	 * @brief Write a MIDI file
	 * @param Path Path to the output file
	 * @param Events MIDI track events
	 * @param Begin Begin time
	 * @param TPS Ticks per second
	 * @param Tempo Tempo
	 */
	void WriteMidiFile(
		const std::wstring& Path,
		const MidiTrack& Events,
		long Begin,
		long TPS,
		long Tempo = 500000
	);

}
