/**
 * @file AvCodec.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief Implementation of audio codec
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Tensor.h"

_D_Dragonian_Lib_Space_Begin

namespace AvCodec
{
	double CalculateRMS(const float* Begin, const float* End);
	double CalculateDB(double rms);
	double CalculateDB(const float* Begin, const float* End);
	UInt64 GetAVTimeBase();

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
		AvCodec() = delete;
		~AvCodec() = delete;
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

		static bool PCMFormatIsPlannar(PCMFormat Format) noexcept;

		static UInt64 PCMFormatBytes(PCMFormat Format) noexcept;

		template <typename _ThisType>
		static PCMFormat Type2PCMFormat(bool IsPlannar = false)
		{
			if constexpr (std::is_same_v<_ThisType, UInt8>)
				return IsPlannar ? PCM_FORMAT_UINT8_PLANAR : PCM_FORMAT_UINT8;
			else if constexpr (std::is_same_v<_ThisType, Int16>)
				return IsPlannar ? PCM_FORMAT_INT16_PLANAR : PCM_FORMAT_INT16;
			else if constexpr (std::is_same_v<_ThisType, Int32>)
				return IsPlannar ? PCM_FORMAT_INT32_PLANAR : PCM_FORMAT_INT32;
			else if constexpr (std::is_same_v<_ThisType, Float32>)
				return IsPlannar ? PCM_FORMAT_FLOAT32_PLANAR : PCM_FORMAT_FLOAT32;
			else if constexpr (std::is_same_v<_ThisType, Float64>)
				return IsPlannar ? PCM_FORMAT_FLOAT64_PLANAR : PCM_FORMAT_FLOAT64;
			else if constexpr (std::is_same_v<_ThisType, Int64>)
				return IsPlannar ? PCM_FORMAT_INT64_PLANAR : PCM_FORMAT_INT64;
			else
				return PCM_FORMAT_NONE;
		}

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
		) = delete;

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
		) = delete;

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
		) = delete;

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
		) = delete;

		// Release the encoder/decoder
		void Release() = delete;

		// Initialize the encoder/decoder
		void Init() = delete;
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
	 * @class AudioFrame
	 * @brief Implementation of audio frame
	 */
	class AudioFrame
	{
	public:
		friend class AudioCodec;

		AudioFrame(
			long _SamplingRate, long _ChannelCount, AvCodec::PCMFormat _Format, long _SampleCount	
		);

		static AudioFrame CreateReference();

		/**
		 * @brief Get the audio frame raw pointer
		 * @return Audio frame raw pointer (AVFrame*)
		 */
		template <typename _ThisType>
		decltype(auto) Get(this _ThisType&& Self)
		{
			return std::forward<_ThisType>(Self)._MyFrame.get();
		}

		const uint8_t* const* GetDataPointerArray() const;

		int GetLinesize() const;

		AudioFrame& SetDataPointer(uint8_t** _Data, ULong _BufferCount);

		AudioFrame& CopyData(const uint8_t* const* _Data, ULong _BufferCount, ULong _SampleCount, ULong _PaddingCount = 0);

	private:
		AudioFrame() = default;
		std::shared_ptr<void> _MyFrame = nullptr;
	};

	/**
	 * @class AudioPacket
	 * @brief Implementation of audio packet
	 */
	class AudioPacket
	{
	public:
		friend class AudioCodec;
		friend class AudioIStream;
		friend class AudioOStream;

		/**
		 * @brief Get the audio packet raw pointer
		 * @return Audio packet raw pointer (AVPacket*)
		 */
		template <typename _ThisType>
		decltype(auto) Get(this _ThisType&& Self)
		{
			return std::forward<_ThisType>(Self)._MyPacket.get();
		}
	protected:
		AudioPacket() = default;
		std::shared_ptr<void> _MyPacket = nullptr; ///< Packet pointer

	public:
		static AudioPacket New();
		static AudioPacket CreateReference();
	};

	/**
	 * @class AudioResampler
	 * @brief Implementation of audio resampler, resample pcm data
	 */
	class AudioResampler
	{
	public:
		struct AudioResamplerSettings
		{
			UInt32 _InputSamplingRate = 0; AvCodec::PCMFormat _InputFormat = AvCodec::PCM_FORMAT_NONE; UInt32 _InputChannels = 0;
			UInt32 _OutputSamplingRate = 0; AvCodec::PCMFormat _OutputFormat = AvCodec::PCM_FORMAT_NONE; UInt32 _OutputChannels = 0;
		};

		AudioResampler() = default;

		AudioResampler(
			const AudioResamplerSettings& _Settings
		);

		/**
		 * @brief Reset the resampler
		 * @param _Settings Resampler settings
		 */
		void Reset(
			const AudioResamplerSettings& _Settings
		);

		/**
		 * @brief Resample audio data
		 * @param _OutputFrame Output audio frame
		 * @param _InputFrame Input audio frame
		 */
		void Resample(
			AudioFrame& _OutputFrame,
			const AudioFrame& _InputFrame
		) const;

		/**
		 * @brief Resample audio data
		 * @param _InputFrame Input audio frame
		 * @return Output audio frame
		 */
		AudioFrame Resample(
			const AudioFrame& _InputFrame
		) const;

		/**
		 * @brief Resample audio data
		 * @param _OutputData Output data, a [channel_count, sample_count] array
		 * @param _OutputSampleCount Output sample count
		 * @param _InputData Input data, a [channel_count, sample_count] array
		 * @param _InputSampleCount Input sample count
		 * @return Number of samples resampled
		 */
		Int32 Resample(
			void* const* _OutputData,
			size_t _OutputSampleCount,
			const void* const* _InputData,
			size_t _InputSampleCount
		) const;

		/**
		 * @brief Resample audio data
		 * @param _InputData Input data, a [channel_count, sample_count] array
		 * @param _InputSampleCount Input sample count
		 * @param _OutputSampleCount Output sample count, set by this function, can be nullptr if not needed
		 * @param _Alloc Memory allocation function
		 * @param _Free Memory free function
		 * @return Resampled audio data
		 */
		std::shared_ptr<void> Resample(
			const void* const* _InputData,
			size_t _InputSampleCount,
			size_t* _OutputSampleCount,
			void* (*_Alloc)(size_t) = nullptr,
			void (*_Free)(void*) = nullptr
		) const;

		/**
		 * @brief Get the output buffer with the input sample count
		 * @param _InputSampleCount Input sample count
		 * @param _Alloc Memory allocation function
		 * @param _Free Memory free function
		 * @param _OutputSampleCount Output sample count, set by this function, can be nullptr if not needed
		 * @return Output buffer
		 */
		std::shared_ptr<void> GetOutputBuffer(
			size_t _InputSampleCount,
			void* (*_Alloc)(size_t) = nullptr,
			void (*_Free)(void*) = nullptr,
			size_t* _OutputSampleCount = nullptr
		) const;

		/**
		 * @brief Get the output sample count with the input sample count
		 * @param _InputSampleCount Input sample count
		 * @return Output sample count
		 */
		size_t GetOutputSampleCount(
			size_t _InputSampleCount
		) const;

		bool Enabled() const noexcept
		{
			return _MySwrContext != nullptr;
		}

	private:
		std::shared_ptr<void> _MySwrContext = nullptr; // Resampling context
		AudioResamplerSettings _MySettings;
		mutable std::mutex _MyMutex;
	};

	/**
	 * @class AudioCodec
	 * @brief Implementation of audio codec
	 */
	class AudioCodec
	{
	public:
		enum CodecType
		{
			DECODER = 0,
			ENCODER = 1
		};

		struct AudioCodecSettings
		{
			CodecType _Type = DECODER; ///< Codec type

			UInt32 _Format = 0; ///< Codec format of ffmpeg, if the type is ENCODER, it is the encoder format, otherwise it is the decoder format
			void* _ParameterDict = nullptr; ///< Codec parameter of ffmpeg (AVDictionary)
			void* _Parameters = nullptr; ///< Codec parameter of ffmpeg (AVCodecParameters)
			
			UInt32 _InputSamplingRate = 0; ///< Input sampling rate
			UInt32 _InputChannels = 0; ///< Input channels

			UInt32 _OutputSamplingRate = 0; ///< Output sampling rate
			UInt32 _OutputChannels = 0; ///< Output channels

			AvCodec::PCMFormat _InputSampleFormat = AvCodec::PCM_FORMAT_NONE; ///< Input sample format, if the type is ENCODER, it must be set
			AvCodec::PCMFormat _OutputSampleFormat = AvCodec::PCM_FORMAT_NONE; ///< Output sample format, if the type is DECODER, it must be set
		};

		AudioCodec() = default;

		AudioCodec(
			const AudioCodecSettings& _Settings
		);

		void Reset(
			const AudioCodecSettings& _Settings
		);

		template <typename _ThisType>
		decltype(auto) GetResampler(this _ThisType&& Self)
		{
			return std::forward<_ThisType>(Self)._MyResampler;
		}

		bool Enabled() const noexcept
		{
			return _MyContext != nullptr;
		}

		Long GetFrameSize() const;

		UInt32 GetBitRate() const;
	private:
		std::shared_ptr<void> _MyContext = nullptr; // Decode context
		AudioResampler _MyResampler;
		bool _NeedResample = false;
		AudioCodecSettings _MySettings;
		mutable std::mutex _MyMutex;

	public:
		TemplateLibrary::Vector<AudioFrame> Decode(const AudioPacket& _Packet) const;
		TemplateLibrary::Vector<AudioPacket> Encode(const AudioFrame& _Frame) const;
	};

	/**
	 * @class AudioIStream
	 * @brief Implementation of audio input stream
	 */
	class AudioIStream
	{
	public:
		friend AudioIStream OpenInputStream(const std::wstring& _Path);

	protected:
		AudioIStream() = default;
		std::shared_ptr<void> _MyFormatContext = nullptr; // Format context
		long _MyStreamIndex = 0; // Stream index
		std::wstring _MyExtension; // File extension

	public:
		bool Enabled() const noexcept
		{
			return _MyFormatContext != nullptr;
		}
		bool IsEnd() const noexcept
		{
			return _MyCode < 0;
		}
		AudioCodec::AudioCodecSettings GetCodecSettings() const noexcept;
		void Reset(const std::wstring& _Path = L"");
		AudioIStream& operator>>(AudioPacket& _Packet);
		UInt64 GetDurations() const;

	private:
		int _MyCode = 0;

	public:
		/**
		 * @brief Decode all audio data
		 * @param _OutputSamplingRate Output sampling rate
		 * @param _OutputChannels Output channel count
		 * @param _OutputPlanar Whether the output data is planar
		 * @param _ParameterDict Codec parameter dictionary (AVDictionary)
		 * @return Decoded audio data
		 */
		template <typename _RetType = Float32>
		Tensor<_RetType, 2, Device::CPU> DecodeAll(
			UInt32 _OutputSamplingRate,
			UInt32 _OutputChannels = 1,
			bool _OutputPlanar = false,
			void* _ParameterDict = nullptr
		)
		{
			if (!Enabled())
				_D_Dragonian_Lib_Throw_Exception("Stream is not initialized!");
			if (_MyStreamIndex == -1)
				_D_Dragonian_Lib_Throw_Exception("Stream is not opened!");
			if (_OutputChannels == 0)
				_D_Dragonian_Lib_Throw_Exception("Output channel count is zero!");
			if (_OutputSamplingRate == 0)
				_D_Dragonian_Lib_Throw_Exception("Output sampling rate is zero!");
			const auto MOutputFormat = AvCodec::Type2PCMFormat<_RetType>(_OutputPlanar);
			if (MOutputFormat == AvCodec::PCM_FORMAT_NONE)
				_D_Dragonian_Lib_Throw_Exception("Invalid output format!");

			auto DecoderSetting = GetCodecSettings();
			DecoderSetting._ParameterDict = _ParameterDict;
			DecoderSetting._OutputSamplingRate = _OutputSamplingRate;
			DecoderSetting._OutputSampleFormat = MOutputFormat;
			DecoderSetting._OutputChannels = _OutputChannels;
			auto _MyCodec = AudioCodec(DecoderSetting);

			const auto nSample = size_t(GetDurations() * DecoderSetting._OutputSamplingRate / GetAVTimeBase());
			const auto nBuffer = nSample + (nSample >> 1);

			auto MyBuf = TemplateLibrary::Vector(_OutputChannels, TemplateLibrary::Vector<_RetType>());
			for (auto& Buf : MyBuf)
				Buf.Reserve(nBuffer);

			if (_OutputPlanar)
			{
				_D_Dragonian_Lib_Rethrow_Block(
					{
						AudioPacket Packet;
						while (!(*this >> Packet).IsEnd())
						{
							auto Frames = _MyCodec.Decode(Packet);
							for (auto& Frame : Frames)
							{
								for (UInt32 i = 0; i < _OutputChannels; ++i)
								{
									auto& Buf = MyBuf[i];
									auto Data = (const _RetType* const)Frame.GetDataPointerArray()[i];
									auto Size = Frame.GetLinesize() / sizeof(_RetType);
									Buf.Insert(Buf.End(), Data, Data + Size);
								}
							}
						}
					}
				);
			}


			_D_Dragonian_Lib_Rethrow_Block(
				{
					auto & OutputBuffer = MyBuf[0];
					AudioPacket Packet;
					while (!(*this >> Packet).IsEnd())
					{
						auto Frames = _MyCodec.Decode(Packet);
						for (auto& Frame : Frames)
						{
							auto Data = (const _RetType* const)Frame.GetDataPointerArray()[0];
							auto Size = Frame.GetLinesize() / sizeof(_RetType);
							OutputBuffer.Insert(OutputBuffer.End(), Data, Data + Size);
						}
					}
				}
			);

			const auto MBufSize = static_cast<SizeType>(MyBuf[0].Size());

			Dimensions<2> OutputDimensions{
				(_OutputPlanar ? static_cast<SizeType>(_OutputChannels) : static_cast<SizeType>(MBufSize / _OutputChannels)),
				(_OutputPlanar ? static_cast<SizeType>(MBufSize) : static_cast<SizeType>(_OutputChannels))
			};

			auto Ret = Tensor<_RetType, 2, Device::CPU>::Zeros(OutputDimensions).Evaluate();

			for (auto [i, Buf] : TemplateLibrary::Enumrate(MyBuf))
				memcpy(Ret.Data() + i * Buf.Size(), Buf.Data(), Buf.Size() * sizeof(_RetType));

			return Ret;
		}
	};

	/**
	 * @class AudioOStream
	 * @brief Implementation of audio output stream
	 */
	class AudioOStream
	{
	public:
		friend AudioOStream OpenOutputStream(
			UInt32 _OutputSamplingRate,
			const std::wstring& _Path,
			AvCodec::PCMFormat _OutputDataFormat,
			UInt32 _OutputChannelCount,
			Int32 _OutputCodecID
		);
		AudioCodec::AudioCodecSettings GetCodecSettings() const noexcept;

	protected:
		AudioOStream() = default;
		std::shared_ptr<void> _MyFormatContext = nullptr; // Format context
		long _MyStreamIndex = -1; // Stream index
		std::wstring _MyExtension; // File extension

	public:
		bool Enabled() const noexcept
		{
			return _MyFormatContext != nullptr;
		}

		void Close()
		{
			_MyFormatContext = nullptr;
			_MyStreamIndex = -1;
			_MyExtension.clear();
		}

		void Reset(
			UInt32 _OutputSamplingRate,
			const std::wstring& _Path = L"",
			AvCodec::PCMFormat _OutputDataFormat = AvCodec::PCM_FORMAT_FLOAT32,
			UInt32 _OutputChannelCount = 1,
			Int32 _OutputCodecID = -1
		);

		AudioOStream& SetBitRate(UInt32 _BitRate);

		AudioOStream& operator<<(const AudioPacket& _Packet);

		/**
		 * @brief Encode all PCM data
		 * @param _PCMData PCM data
		 * @param _InputSamplingRate Input sampling rate
		 * @param _InputChannelCount Input channel count
		 * @param _InputPlanar Whether the input data is planar
		 * @param _ParameterDict Codec Parameter dictionary (AVDictionary)
		 */
		template <typename _InputType>
		void EncodeAll(
			const TemplateLibrary::ConstantRanges<_InputType>& _PCMData,
			UInt32 _InputSamplingRate,
			UInt32 _InputChannelCount = 1,
			bool _InputPlanar = false,
			void* _ParameterDict = nullptr
		)
		{
			if (!Enabled())
				_D_Dragonian_Lib_Throw_Exception("Stream is not initialized!");
			if (_MyStreamIndex == -1)
				_D_Dragonian_Lib_Throw_Exception("Stream is not opened!");
			if (!_PCMData.Size())
				_D_Dragonian_Lib_Throw_Exception("PCM data is empty!");
			if (_InputChannelCount < 1)
				_D_Dragonian_Lib_Throw_Exception("Invalid channel count!");
			if (_InputSamplingRate < 640)
				_D_Dragonian_Lib_Throw_Exception("Invalid sampling rate!");
			if (_InputChannelCount > 8)
				_D_Dragonian_Lib_Throw_Exception("Too many channels!");
			const auto _InputDataFormat = AvCodec::Type2PCMFormat<_InputType>(_InputPlanar);
			if (_InputDataFormat == AvCodec::PCM_FORMAT_NONE)
				_D_Dragonian_Lib_Throw_Exception("Invalid sample format!");

			auto DecoderSetting = GetCodecSettings();
			DecoderSetting._ParameterDict = _ParameterDict;
			DecoderSetting._InputSamplingRate = _InputSamplingRate;
			DecoderSetting._InputSampleFormat = _InputDataFormat;
			DecoderSetting._InputChannels = _InputChannelCount;

			auto Codec = AudioCodec(DecoderSetting);
			SetBitRate(Codec.GetBitRate());

			const auto TotalSampleCount = _PCMData.Size() / _InputChannelCount;
			const _InputType* Buffer[8]{
				_PCMData.Data(), nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr
			};
			UInt64 AudioShape[]{ 1 ,TotalSampleCount, _InputChannelCount };
			if (_InputPlanar)
			{
				std::swap(AudioShape[0], AudioShape[2]);
				for (UInt32 i = 0; i < AudioShape[0]; ++i)
					Buffer[i] = _PCMData.Data() + i * AudioShape[1];
			}

			const auto FrameSize = static_cast<UInt64>(Codec.GetFrameSize() ? Codec.GetFrameSize() : 2048);
			AudioFrame MyFrame(
				static_cast<long>(_InputSamplingRate),
				static_cast<long>(_InputChannelCount),
				_InputDataFormat,
				static_cast<long>(FrameSize)
			);
			for (UInt64 Time = 0; Time < AudioShape[1]; Time += FrameSize)
			{
				const auto ChannelSampleCount = std::min(FrameSize, AudioShape[1] - Time);
				const auto ChannelPaddingCount = FrameSize - ChannelSampleCount;
				const auto SampleCount = ChannelSampleCount * AudioShape[2];
				const auto PaddingCount = ChannelPaddingCount * AudioShape[2];
				TemplateLibrary::Vector<AudioPacket> Packets;

				_D_Dragonian_Lib_Rethrow_Block(Packets = Codec.Encode(MyFrame.CopyData((const uint8_t* const*)Buffer, (ULong)AudioShape[0], (ULong)SampleCount, (ULong)PaddingCount)););

				for (auto& Packet : Packets)
					*this << Packet;

				for (UInt32 i = 0; i < AudioShape[0]; ++i)
					Buffer[i] += SampleCount;
			}
			Close();
		}
	};

	/**
	 * @brief Open an audio input stream
	 * @param _Path Path to the input file
	 * @return Audio input stream
	 */
	AudioIStream OpenInputStream(const std::wstring& _Path);

	/**
	 * @brief Open an audio output stream
	 * @param _OutputSamplingRate Output sampling rate
	 * @param _Path Path to the output file
	 * @param _OutputDataFormat Output data format
	 * @param _OutputChannelCount Output channel count
	 * @param _OutputCodecID Output codec ID (AVCodecID)
	 * @return Audio output stream
	 */
	AudioOStream OpenOutputStream(
		UInt32 _OutputSamplingRate,
		const std::wstring& _Path,
		AvCodec::PCMFormat _OutputDataFormat = AvCodec::PCM_FORMAT_FLOAT32,
		UInt32 _OutputChannelCount = 1,
		Int32 _OutputCodecID = -1
	);

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
	 * @return Slice positions
	 */
	DragonianLibSTL::Vector<size_t> SliceAudio(
		const DragonianLibSTL::Vector<float>& PcmData,
		const SlicerSettings& SlicerSettings
	);

	/**
	 * @brief Slice audio data into segments
	 * @param PcmData Input audio data
	 * @param SlicerSettings Slice settings
	 * @return Slice positions
	 */
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
