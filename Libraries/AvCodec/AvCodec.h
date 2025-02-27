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
#include "Libraries/MyTemplateLibrary/Vector.h"

// Forward declarations to avoid including unnecessary headers
struct AVFrame;
struct SwrContext;
struct AVCodecContext;
struct AVFormatContext;
struct AVPacket;

_D_Dragonian_Lib_Space_Begin

namespace AvCodec
{
	double CalculateRMS(const float* Begin, const float* End);
	double CalculateDB(double rms);
	double CalculateDB(const float* Begin, const float* End);

	struct SlicerSettings
	{
		int32_t SamplingRate = 48000;
		double Threshold = -60.;
		double MinLength = 3.;
		int32_t WindowLength = 2048;
		int32_t HopSize = 512;
	};

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
		AvCodec() = default;
		~AvCodec();
		AvCodec(const AvCodec&) = delete;
		AvCodec(AvCodec&&) = delete;
		AvCodec operator=(const AvCodec&) = delete;
		AvCodec operator=(AvCodec&&) = delete;

		enum PCMFormat {
			PCM_FORMAT_NONE = -1,
			PCM_FORMAT_UINT8,				///< unsigned 8 bits
			PCM_FORMAT_INT16,				///< signed 16 bits
			PCM_FORMAT_INT32,				///< signed 32 bits
			PCM_FORMAT_FLOAT32,				///< float
			PCM_FORMAT_FLOAT64,				///< double

			PCM_FORMAT_UINT8_PLANAR,        ///< unsigned 8 bits, planar
			PCM_FORMAT_INT16_PLANAR,        ///< signed 16 bits, planar
			PCM_FORMAT_INT32_PLANAR,        ///< signed 32 bits, planar
			PCM_FORMAT_FLOAT32_PLANAR,      ///< float, planar
			PCM_FORMAT_FLOAT64_PLANAR,      ///< double, planar
			PCM_FORMAT_INT64,				///< signed 64 bits
			PCM_FORMAT_INT64_PLANAR,        ///< signed 64 bits, planar

			PCM_FORMAT_NB					///< Number of sample formats. DO NOT USE if linking dynamically
		};

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutputSamplingRate Sampling rate of the output audio
		 * @param OutputFormat Format of the output audio
		 * @param OutputStero Whether the output audio is stereo
		 * @return Raw PCM data
		 */
		DragonianLibSTL::Vector<unsigned char> Decode(
			const std::wstring& AudioPath,
			int OutputSamplingRate,
			PCMFormat OutputFormat = PCM_FORMAT_FLOAT32,
			bool OutputStero = false
		);

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutputSamplingRate Sampling rate of the output audio
		 * @param OutputStero Whether the output audio is stereo
		 * @param OutputPlanar Whether the output audio is planar
		 * @return Float PCM data
		 */
		DragonianLibSTL::Vector<float> DecodeFloat(
			const std::wstring& AudioPath,
			int OutputSamplingRate,
			bool OutputStero = false,
			bool OutputPlanar = false
		);

		/**
		 * @brief Decode an audio file
		 * @param AudioPath Path to the audio file
		 * @param OutputSamplingRate Sampling rate of the output audio
		 * @param OutputStero Whether the output audio is stereo
		 * @param OutputPlanar Whether the output audio is planar
		 * @return Signed 16-bit PCM data
		 */
		DragonianLibSTL::Vector<int16_t> DecodeSigned16(
			const std::wstring& AudioPath,
			int OutputSamplingRate,
			bool OutputStero = false,
			bool OutputPlanar = false
		);

		/**
		 * @brief Encode an audio file
		 * @param OutputPath Path to the output audio file
		 * @param PCMData Raw PCM data
		 * @param SamplingRate Sampling rate of the input audio
		 * @param PCMType Type of the input audio
		 * @param EncoderFormatID ID of the encoder format
		 * @param IsStero Whether the input audio is stereo
		 */
		void Encode(
			const std::wstring& OutputPath,
			const DragonianLibSTL::ConstantRanges<Byte>& PCMData,
			int SamplingRate,
			PCMFormat PCMType = PCM_FORMAT_FLOAT32,
			int EncoderFormatID = 0,
			bool IsStero = false
		);

		// Release the encoder/decoder
		void Release();

		// Initialize the encoder/decoder
		void Init();

	private:
		AVFrame* InFrame = nullptr; // Input frame
		AVFrame* OutFrame = nullptr; // Input frame
		SwrContext* SwrContext = nullptr; // Resampling context
		AVCodecContext* AvCodecContext = nullptr; // Codec context
		AVFormatContext* AvFormatContext = nullptr; // Format context
		AVPacket* Packet = nullptr; // Packet
		bool InputMode = false;
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
	 * @param OutputPath Path to the output file
	 * @param PCMData PCM data
	 * @param SamplingRate Sampling rate
	 * @param DataFormat Data format
	 * @param IsStero Whether the data is stereo
	 */
	void WritePCMData(
		const std::wstring& OutputPath,
		const TemplateLibrary::ConstantRanges<Byte>& PCMData,
		int SamplingRate,
		AvCodec::PCMFormat DataFormat = AvCodec::PCM_FORMAT_FLOAT32,
		bool IsStero = false
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

	/**
	 * @brief Slice audio data into segments
	 * @param PcmData Input audio data
	 * @param SlicerSettings Slice settings
	 * @return
	 */
	DragonianLibSTL::Vector<size_t> SliceAudio(
		const DragonianLibSTL::Vector<float>& PcmData,
		const SlicerSettings& SlicerSettings
	);

	DragonianLibSTL::Vector<size_t> SliceAudio(
		const TemplateLibrary::ConstantRanges<float>& PcmData,
		const SlicerSettings& SlicerSettings
	);

	template <typename T>
	std::pair<DragonianLibSTL::Vector<T>, size_t> CrossCorrelation(
		float SamplingRate,
		const DragonianLibSTL::Vector<T>& Signal1,
		float Signal1Begin,
		float Signal1End,
		const DragonianLibSTL::Vector<T>& Signal2,
		float Signal2Begin,
		float SliceOverlap,
		const T& Factor
	)
	{
		const auto Signal1Size = Signal1.Size();
		const auto Signal2Size = Signal2.Size();

		const auto Signal2Overlap = static_cast<size_t>(SliceOverlap * SamplingRate);

		if (Signal2Overlap < 32)
			_D_Dragonian_Lib_Throw_Exception("The slice overlap is too small.");

		const auto Signal1BeginIndex = static_cast<size_t>(Signal1Begin * SamplingRate);
		const auto Signal1EndIndex = static_cast<size_t>(Signal1End * SamplingRate);
		const auto Signal1Sz = Signal1EndIndex - Signal1BeginIndex;

		if (Signal1EndIndex <= Signal1BeginIndex)
			_D_Dragonian_Lib_Throw_Exception("End time is less than start time.");

		const auto Signal2BeginIndex = static_cast<size_t>(Signal2Begin * SamplingRate);
		const auto Signal2EndIndex = Signal2BeginIndex + Signal2Overlap;

		if (Signal1EndIndex > Signal1Size || Signal2EndIndex > Signal2Size)
			_D_Dragonian_Lib_Throw_Exception("The slice overlap is too large.");

		if (Signal1Sz < Signal2Overlap)
			_D_Dragonian_Lib_Throw_Exception("The slice of the first signal is too short.");

		DragonianLibSTL::Vector<float> Result(Signal1Sz - Signal2Overlap + 1);

		{
			auto _Signal1Begin = Signal1.Data() + Signal1BeginIndex;
			const auto _Signal1End = Signal1.Data() + Signal1EndIndex;
			const auto _Signal2Begin = Signal2.Data() + Signal2BeginIndex;
			const auto _Signal2End = Signal2.Data() + Signal2EndIndex;
			for (auto& i : Result)
			{
				T Sum = 0;
				for (auto j = _Signal1Begin, k = _Signal2Begin; k != _Signal2End; ++j, ++k)
					Sum += (*j++) * (*k++);
				i = Sum;
				++_Signal1Begin;
			}
		}

		{
			auto _Signal1Begin = Signal1.Data() + Signal1BeginIndex;
			DragonianLibSTL::Vector<T> Signal1Pow2(Signal1Sz);
			for (auto& i : Signal1Pow2)
				i = (T)powf(float(*(_Signal1Begin++)), 2.f);
			auto Signal1Pow2Begin = Signal1Pow2.Data();
			for (auto& i : Result)
			{
				T Sum = Factor;
				auto j = Signal1Pow2Begin;
				for (auto k = 0; k < Signal2Overlap; ++j, ++k)
					Sum += (*j++);
				i /= (T)sqrtf(float(Sum));
				++Signal1Pow2Begin;
			}
		}

		return { std::move(Result), std::distance(Result.Begin(), std::max_element(Result.Begin(), Result.End())) };
	}
}

_D_Dragonian_Lib_Space_End
