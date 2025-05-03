#include "Libraries/AvCodec/AvCodec.h"
#include "Libraries/Base.h"
#include "Libraries/Util/Logger.h"
#include "Libraries/Util/StringPreprocess.h"
#include "libremidi/writer.hpp"
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
}
#include <fstream>

_D_Dragonian_Lib_Space_Begin

namespace AvCodec
{
	static DLogger GetAvcodecLogger()
	{
		static DLogger AvCodecLogger = std::make_shared<Logger>(
			*_D_Dragonian_Lib_Namespace GetDefaultLogger(),
			L"AvCodec"
		);
		return AvCodecLogger;
	}

	static void* DefaultAlloc(size_t Size)
	{
		return malloc(Size);
	}

	static void DefaultFree(void* Ptr)
	{
		free(Ptr);
	}

	double CalculateRMS(const float* Begin, const float* End)
	{
		double sum = 0.0;
		const double count = static_cast<double>(End - Begin);
		while (Begin != End) {
			sum += static_cast<double>(*Begin) * static_cast<double>(*Begin);
			++Begin;
		}
		return sqrt(sum / count);
	}

	double CalculateDB(double RMS)
	{
		if (RMS <= 1e-10 || isnan(RMS))
			return -std::numeric_limits<double>::infinity();
		return 20.0 * log10(RMS);
	}

	double CalculateDB(const float* Begin, const float* End)
	{
		double sum = 0.0;
		const double count = static_cast<double>(End - Begin);
		while (Begin != End) {
			sum += static_cast<double>(*Begin) * static_cast<double>(*Begin);
			++Begin;
		}
		return CalculateDB(sqrt(sum / count));
	}

	UInt64 GetAVTimeBase()
	{
		return AV_TIME_BASE;
	}

	static AVSampleFormat PCMFormat2AVSampleFormat(const AvCodec::PCMFormat& Format) noexcept
	{
		switch (Format)
		{
			case AvCodec::PCM_FORMAT_UINT8: return AV_SAMPLE_FMT_U8;
			case AvCodec::PCM_FORMAT_INT16: return AV_SAMPLE_FMT_S16;
			case AvCodec::PCM_FORMAT_INT32: return AV_SAMPLE_FMT_S32;
			case AvCodec::PCM_FORMAT_FLOAT32: return AV_SAMPLE_FMT_FLT;
			case AvCodec::PCM_FORMAT_FLOAT64: return AV_SAMPLE_FMT_DBL;
			case AvCodec::PCM_FORMAT_UINT8_PLANAR: return AV_SAMPLE_FMT_U8P;
			case AvCodec::PCM_FORMAT_INT16_PLANAR: return AV_SAMPLE_FMT_S16P;
			case AvCodec::PCM_FORMAT_INT32_PLANAR: return AV_SAMPLE_FMT_S32P;
			case AvCodec::PCM_FORMAT_FLOAT32_PLANAR: return AV_SAMPLE_FMT_FLTP;
			case AvCodec::PCM_FORMAT_FLOAT64_PLANAR: return AV_SAMPLE_FMT_DBLP;
			case AvCodec::PCM_FORMAT_INT64: return AV_SAMPLE_FMT_S64;
			case AvCodec::PCM_FORMAT_INT64_PLANAR: return AV_SAMPLE_FMT_S64P;
			case AvCodec::PCM_FORMAT_NONE: return AV_SAMPLE_FMT_NONE;
			case AvCodec::PCM_FORMAT_NB: return AV_SAMPLE_FMT_NB;
		}
		return AV_SAMPLE_FMT_NONE;
	}

	static AvCodec::PCMFormat AVSampleFormat2PCMFormat(const AVSampleFormat& Format) noexcept
	{
		switch (Format)
		{
			case AV_SAMPLE_FMT_U8: return AvCodec::PCM_FORMAT_UINT8;
			case AV_SAMPLE_FMT_S16: return AvCodec::PCM_FORMAT_INT16;
			case AV_SAMPLE_FMT_S32: return AvCodec::PCM_FORMAT_INT32;
			case AV_SAMPLE_FMT_FLT: return AvCodec::PCM_FORMAT_FLOAT32;
			case AV_SAMPLE_FMT_DBL: return AvCodec::PCM_FORMAT_FLOAT64;
			case AV_SAMPLE_FMT_U8P: return AvCodec::PCM_FORMAT_UINT8_PLANAR;
			case AV_SAMPLE_FMT_S16P: return AvCodec::PCM_FORMAT_INT16_PLANAR;
			case AV_SAMPLE_FMT_S32P: return AvCodec::PCM_FORMAT_INT32_PLANAR;
			case AV_SAMPLE_FMT_FLTP: return AvCodec::PCM_FORMAT_FLOAT32_PLANAR;
			case AV_SAMPLE_FMT_DBLP: return AvCodec::PCM_FORMAT_FLOAT64_PLANAR;
			case AV_SAMPLE_FMT_S64: return AvCodec::PCM_FORMAT_INT64;
			case AV_SAMPLE_FMT_S64P: return AvCodec::PCM_FORMAT_INT64_PLANAR;
			case AV_SAMPLE_FMT_NONE: return AvCodec::PCM_FORMAT_NONE;
			case AV_SAMPLE_FMT_NB: return AvCodec::PCM_FORMAT_NB;
		}
		return AvCodec::PCM_FORMAT_NONE;
	}

	bool AvCodec::PCMFormatIsPlannar(PCMFormat Format) noexcept
	{
		if (Format == PCM_FORMAT_UINT8_PLANAR ||
			Format == PCM_FORMAT_INT16_PLANAR ||
			Format == PCM_FORMAT_INT32_PLANAR ||
			Format == PCM_FORMAT_FLOAT32_PLANAR ||
			Format == PCM_FORMAT_FLOAT64_PLANAR ||
			Format == PCM_FORMAT_INT64_PLANAR)
			return true;
		return false;
	}

	UInt64 AvCodec::PCMFormatBytes(PCMFormat Format) noexcept
	{
		return av_get_bytes_per_sample(PCMFormat2AVSampleFormat(Format));
	}

	/*DragonianLibSTL::Vector<unsigned char> AvCodec::Decode(
		const std::wstring& AudioPath,
		int OutputSamplingRate,
		PCMFormat OutputFormat,
		bool OutputStero
	)
	{
		Release();
		Init();

		char ErrorMessage[1024];

		int ErrorCode = avformat_open_input(&AvFormatContext, WideStringToUTF8(AudioPath).c_str(), nullptr, nullptr);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		ErrorCode = avformat_find_stream_info(AvFormatContext, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		int streamIndex = -1;
		for (unsigned i = 0; i < AvFormatContext->nb_streams; ++i) {
			const AVMediaType avMediaType = AvFormatContext->streams[i]->codecpar->codec_type;
			if (avMediaType == AVMEDIA_TYPE_AUDIO) {
				streamIndex = static_cast<int>(i);
			}
		}

		if (streamIndex == -1)
			_D_Dragonian_Lib_Throw_Exception("input file has no audio stream!");

		const AVCodecParameters* avCodecParameters = AvFormatContext->streams[streamIndex]->codecpar;
		const AVCodecID avCodecId = avCodecParameters->codec_id;
		const AVCodec* avCodec = avcodec_find_decoder(avCodecId);
		if (avCodec == nullptr)
			_D_Dragonian_Lib_Throw_Exception("unable to find a matching decoder!");
		if (AvCodecContext == nullptr)
			_D_Dragonian_Lib_Throw_Exception("Can't Get Decoder Info");

		ErrorCode = avcodec_parameters_to_context(AvCodecContext, avCodecParameters);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		ErrorCode = avcodec_open2(AvCodecContext, avCodec, nullptr);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		const int inSampleRate = AvCodecContext->sample_rate;
		const AVSampleFormat inFormat = AvCodecContext->sample_fmt;
		const int inChannelCount = AvCodecContext->ch_layout.nb_channels;

		const int outSampleRate = OutputSamplingRate;
		const AVSampleFormat outFormat = PCMFormat2AVSampleFormat(OutputFormat);
		const int outChannelCount = OutputStero ? 2 : 1;

		const auto nSample = size_t(AvFormatContext->duration * OutputSamplingRate / AV_TIME_BASE) * outChannelCount;
		const auto sampleBytes = av_get_bytes_per_sample(outFormat);
		const auto nBytes = nSample * sampleBytes;
		DragonianLibSTL::Vector<uint8_t> outData;
		outData.Reserve(nBytes);

		AVChannelLayout inChannelLayout, outChannelLayout;

		av_channel_layout_default(&inChannelLayout, inChannelCount);
		av_channel_layout_default(&outChannelLayout, outChannelCount);

		ErrorCode = swr_alloc_set_opts2(
			&SwrContext,
			&outChannelLayout,
			outFormat,
			outSampleRate,
			&inChannelLayout,
			inFormat,
			inSampleRate,
			0,
			nullptr
		);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		ErrorCode = swr_init(SwrContext);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		TemplateLibrary::Vector OutputBuffer(outChannelCount, TemplateLibrary::Vector<Byte>());
		for (auto& i : OutputBuffer)
			i.Reserve(nBytes);

		while (av_read_frame(AvFormatContext, Packet) >= 0)
		{
			if (Packet->stream_index == streamIndex)
			{
				ErrorCode = avcodec_send_packet(AvCodecContext, Packet);
				if (ErrorCode)
				{
					av_packet_unref(Packet);
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}

				while (!avcodec_receive_frame(AvCodecContext, InFrame))
				{
					auto dstNbSamples = av_rescale_rnd(
						InFrame->nb_samples,
						outSampleRate,
						inSampleRate,
						AV_ROUND_ZERO
					);

					if (av_sample_fmt_is_planar(outFormat))
						for (int i = 0; i < outChannelCount; ++i)
							OutBuffer[i] = (uint8_t*)av_malloc(dstNbSamples * sizeof(double));
					else
						OutBuffer[0] = (uint8_t*)av_malloc(dstNbSamples * sizeof(double) * outChannelCount);

					ErrorCode = swr_convert(
						SwrContext,
						OutBuffer,
						int(dstNbSamples),
						InFrame->data,
						InFrame->nb_samples
					);
					if (ErrorCode < 0)
					{
						av_frame_unref(InFrame);
						av_strerror(ErrorCode, ErrorMessage, 1024);
						_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
					}

					if (av_sample_fmt_is_planar(outFormat))
						for (int i = 0; i < outChannelCount; ++i)
							OutputBuffer[i].Insert(OutputBuffer[i].End(), OutBuffer[i], OutBuffer[i] + dstNbSamples * sampleBytes);
					else
						OutputBuffer[0].Insert(OutputBuffer[0].End(), OutBuffer[0], OutBuffer[0] + dstNbSamples * sampleBytes * outChannelCount);

					for (int i = 0; i < outChannelCount; ++i)
						if (OutBuffer[i])
						{
							av_free(OutBuffer[i]);
							OutBuffer[i] = nullptr;
						}
					av_frame_unref(InFrame);
				}
			}
			av_packet_unref(Packet);
		}

		for (auto& i : OutputBuffer)
		{
			i.Insert(i.End(), av_sample_fmt_is_planar(outFormat) ? nBytes / 2 - i.Size() : nBytes - i.Size(), 0);
			outData.Insert(outData.End(), i.Begin(), i.End());
		}

		Release();
		return outData;
	}

	DragonianLibSTL::Vector<float> AvCodec::DecodeFloat(
		const std::wstring& AudioPath,
		int OutputSamplingRate,
		bool OutputStero,
		bool OutputPlanar
	)
	{
		auto Ret = Decode(
			AudioPath,
			OutputSamplingRate,
			OutputPlanar ? PCM_FORMAT_FLOAT32_PLANAR : PCM_FORMAT_FLOAT32,
			OutputStero
		);
		auto Alloc = Ret.GetAllocator();
		auto Data = Ret.Release();
		auto Ptr = (float*)Data.first;
		auto Size = Data.second / sizeof(float);
		return { &Ptr, Size, Alloc };
	}

	DragonianLibSTL::Vector<int16_t> AvCodec::DecodeSigned16(
		const std::wstring& AudioPath,
		int OutputSamplingRate,
		bool OutputStero,
		bool OutputPlanar
	)
	{
		auto Ret = Decode(
			AudioPath,
			OutputSamplingRate,
			OutputPlanar ? PCM_FORMAT_INT16_PLANAR : PCM_FORMAT_INT16,
			OutputStero
		);
		auto Alloc = Ret.GetAllocator();
		auto Data = Ret.Release();
		auto Ptr = (int16_t*)Data.first;
		auto Size = Data.second / sizeof(int16_t);
		return { &Ptr, Size, Alloc };
	}

	void AvCodec::Encode(
		const std::wstring& OutputPath,
		const DragonianLibSTL::ConstantRanges<Byte>& PCMData,
		int SamplingRate,
		PCMFormat PCMType,
		int EncoderFormatID,
		bool IsStero
	)
	{
		Release();

		char ErrorMessage[1024];

		const auto Suffix = std::filesystem::path(OutputPath).extension().string();
		AVCodecID CodecId;
		if (Suffix == ".mp3")
			CodecId = AV_CODEC_ID_MP3;
		else if (Suffix == ".aac" || Suffix == ".m4a")
			CodecId = AV_CODEC_ID_AAC;
		else if (Suffix == ".flac")
			CodecId = AV_CODEC_ID_FLAC;
		else if (Suffix == ".wav")
			CodecId = AV_CODEC_ID_PCM_S16LE;
		else if (Suffix == ".ogg")
			CodecId = AV_CODEC_ID_VORBIS;
		else if (Suffix == ".opus")
			CodecId = AV_CODEC_ID_OPUS;
		else if (Suffix == ".wma")
			CodecId = AV_CODEC_ID_WMAV2;
		else if (Suffix == ".wav")
		{
			WritePCMData(OutputPath, PCMData, SamplingRate, PCMType, IsStero);
			return;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Unsupported output format!");

		auto Encoder = avcodec_find_encoder(CodecId);
		if (!Encoder)
			_D_Dragonian_Lib_Throw_Exception("Codec not found");

		auto ErrorCode = avformat_alloc_output_context2(&AvFormatContext, nullptr, nullptr, WideStringToUTF8(OutputPath).c_str());
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		AvCodecContext = avcodec_alloc_context3(Encoder);
		if (!AvCodecContext)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio codec context");
		AvCodecContext->codec_id = Encoder->id;
		AvCodecContext->codec_type = AVMEDIA_TYPE_AUDIO;
		AvCodecContext->sample_fmt = Encoder->sample_fmts[EncoderFormatID];
		AvCodecContext->sample_rate = SamplingRate;
		if (!IsStero)
			AvCodecContext->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			AvCodecContext->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
		AvCodecContext->time_base = { 1, SamplingRate };

		ErrorCode = avcodec_open2(AvCodecContext, Encoder, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		auto Stream = avformat_new_stream(AvFormatContext, nullptr);
		if (!Stream)
			_D_Dragonian_Lib_Throw_Exception("Could not create stream");

		Stream->time_base = { 1, SamplingRate };
		Stream->codecpar->codec_id = Encoder->id;
		Stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
		Stream->codecpar->format = AvCodecContext->sample_fmt;
		Stream->codecpar->bit_rate = AvCodecContext->bit_rate;
		Stream->codecpar->sample_rate = AvCodecContext->sample_rate;
		if (!IsStero)
			Stream->codecpar->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			Stream->codecpar->ch_layout = AV_CHANNEL_LAYOUT_STEREO;

		if (!(AvFormatContext->oformat->flags & AVFMT_NOFILE))
		{
			ErrorCode = avio_open(&AvFormatContext->pb, WideStringToUTF8(OutputPath).c_str(), AVIO_FLAG_WRITE);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
		}

		ErrorCode = avformat_write_header(AvFormatContext, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		OutFrame = av_frame_alloc();
		if (!OutFrame)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");

		OutFrame->nb_samples = AvCodecContext->frame_size ? AvCodecContext->frame_size : 2048;
		OutFrame->format = AvCodecContext->sample_fmt;
		if (!IsStero)
			OutFrame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			OutFrame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
		OutFrame->sample_rate = SamplingRate;

		ErrorCode = av_frame_get_buffer(OutFrame, 0);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		Packet = av_packet_alloc();
		if (!Packet)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate AVPacket");

		const AVSampleFormat InFormat = PCMFormat2AVSampleFormat(PCMType);
		if (InFormat != AvCodecContext->sample_fmt)
		{
			InFrame = av_frame_alloc();
			if (!InFrame)
				_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");

			InFrame->nb_samples = AvCodecContext->frame_size ? AvCodecContext->frame_size : 2048;
			InFrame->format = InFormat;
			if (!IsStero)
				InFrame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
			else
				InFrame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
			InFrame->sample_rate = SamplingRate;

			ErrorCode = av_frame_get_buffer(InFrame, 0);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			AVChannelLayout inChannelLayout, outChannelLayout;
			if (!IsStero)
			{
				inChannelLayout = AV_CHANNEL_LAYOUT_MONO;
				outChannelLayout = AV_CHANNEL_LAYOUT_MONO;
			}
			else
			{
				inChannelLayout = AV_CHANNEL_LAYOUT_STEREO;
				outChannelLayout = AV_CHANNEL_LAYOUT_STEREO;
			}
			SwrContext = swr_alloc();
			ErrorCode = swr_alloc_set_opts2(
				&SwrContext,
				&outChannelLayout,
				AvCodecContext->sample_fmt,
				SamplingRate,
				&inChannelLayout,
				InFormat,
				SamplingRate,
				0,
				nullptr
			);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			ErrorCode = swr_init(SwrContext);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
		}

		bool IsPlanar = av_sample_fmt_is_planar(InFormat);
		const auto ChannelCount = static_cast<UInt64>(OutFrame->ch_layout.nb_channels);
		const auto SSegmentCount = (UInt64)OutFrame->nb_samples * ChannelCount * av_get_bytes_per_sample(InFormat);

		const TemplateLibrary::Array PCMShape{
			IsPlanar ? ChannelCount : 1ull,
			IsPlanar ? PCMData.Size() / ChannelCount : PCMData.Size()
		};
		const auto SegmentCount = IsPlanar ? SSegmentCount / ChannelCount : SSegmentCount;
		const auto PCMPointer = PCMData.Data();
		UInt64 PCMOffset = 0;
		while (PCMOffset < PCMShape[1]) {
			const auto FrameSize = std::min(
				SegmentCount,
				PCMShape[1] - PCMOffset
			);

			auto FixFrameFn = [&](const AVFrame* Frame ,UInt64 i)
				{
					//Frame->data[i][0 : FrameSize]  PCMData[i][PCMOffset : PCMOffset + FrameSize]
					memcpy(Frame->data[i], PCMPointer + i * PCMShape[1] + PCMOffset, FrameSize);
					if (FrameSize < SegmentCount)
						memset(Frame->data[i] + FrameSize, 0, SegmentCount - FrameSize);
				};

			if (InFormat != AvCodecContext->sample_fmt)
			{
				if (IsPlanar)
					for (UInt64 i = 0; i < ChannelCount; ++i)
						FixFrameFn(InFrame, i);
				else
					FixFrameFn(InFrame, 0);

				ErrorCode = swr_convert(
					SwrContext,
					OutFrame->data,
					OutFrame->nb_samples,
					InFrame->data,
					InFrame->nb_samples
				);
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}
			}
			else
			{
				if (IsPlanar)
					for (UInt64 i = 0; i < ChannelCount; ++i)
						FixFrameFn(OutFrame, i);
				else
					FixFrameFn(OutFrame, 0);
			}

			ErrorCode = avcodec_send_frame(AvCodecContext, OutFrame);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			PCMOffset += FrameSize;

			while (true) {
				ErrorCode = avcodec_receive_packet(AvCodecContext, Packet);
				if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
					break;
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}

				ErrorCode = av_interleaved_write_frame(AvFormatContext, Packet);
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}
				av_packet_unref(Packet);
			}
		}

		avcodec_send_frame(AvCodecContext, nullptr);
		while (true) {
			ErrorCode = avcodec_receive_packet(AvCodecContext, Packet);
			if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
				break;
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			ErrorCode = av_interleaved_write_frame(AvFormatContext, Packet);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
			av_packet_unref(Packet);
		}
		av_write_trailer(AvFormatContext);

		Release();
	}

	void AvCodec::Release()
	{
		if (Packet)
			av_packet_free(&Packet);
		if (InFrame)
			av_frame_free(&InFrame);
		if (OutFrame)
			av_frame_free(&OutFrame);
		if (SwrContext)
		{
			swr_close(SwrContext);
			swr_free(&SwrContext);
		}
		if (AvCodecContext)
			avcodec_free_context(&AvCodecContext);
		if (AvFormatContext)
		{
			if (InputMode)
				avformat_close_input(&AvFormatContext);
			else
			{
				if (!(AvFormatContext->oformat->flags & AVFMT_NOFILE))
					avio_closep(&AvFormatContext->pb);
				avformat_free_context(AvFormatContext);
			}
		}
		for (auto& out_buffer : OutBuffer)
			if (out_buffer)
			{
				av_free(out_buffer);
				out_buffer = nullptr;
			}
		InFrame = nullptr;
		OutFrame = nullptr;
		SwrContext = nullptr;
		AvCodecContext = nullptr;
		AvFormatContext = nullptr;
		Packet = nullptr;
		InputMode = false;
	}

	void AvCodec::Init()
	{
		InFrame = av_frame_alloc();
		SwrContext = swr_alloc();
		AvCodecContext = avcodec_alloc_context3(nullptr);
		AvFormatContext = avformat_alloc_context();
		Packet = av_packet_alloc();

		InputMode = true;

		if (!AvFormatContext || !Packet || !InFrame)
		{
			Release();
			throw std::bad_alloc();
		}
	}

	AvCodec::~AvCodec()
	{
		Release();
	}*/

	/*void AvCodec::Encode(
		const std::wstring& OutputPath,
		const DragonianLibSTL::ConstantRanges<Byte>& PCMData,
		int SamplingRate,
		PCMFormat PCMType,
		int EncoderFormatID,
		bool IsStero
	)
	{
		AvCodecContext = avcodec_alloc_context3(Encoder);
		if (!AvCodecContext)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio codec context");
		AvCodecContext->codec_id = Encoder->id;
		AvCodecContext->codec_type = AVMEDIA_TYPE_AUDIO;
		AvCodecContext->sample_fmt = Encoder->sample_fmts[EncoderFormatID];
		AvCodecContext->sample_rate = SamplingRate;
		if (!IsStero)
			AvCodecContext->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			AvCodecContext->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
		AvCodecContext->time_base = { 1, SamplingRate };

		ErrorCode = avcodec_open2(AvCodecContext, Encoder, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		auto Stream = avformat_new_stream(AvFormatContext, nullptr);
		if (!Stream)
			_D_Dragonian_Lib_Throw_Exception("Could not create stream");

		Stream->time_base = { 1, SamplingRate };
		Stream->codecpar->codec_id = Encoder->id;
		Stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
		Stream->codecpar->format = AvCodecContext->sample_fmt;
		Stream->codecpar->bit_rate = AvCodecContext->bit_rate;
		Stream->codecpar->sample_rate = AvCodecContext->sample_rate;
		if (!IsStero)
			Stream->codecpar->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			Stream->codecpar->ch_layout = AV_CHANNEL_LAYOUT_STEREO;

		if (!(AvFormatContext->oformat->flags & AVFMT_NOFILE))
		{
			ErrorCode = avio_open(&AvFormatContext->pb, WideStringToUTF8(OutputPath).c_str(), AVIO_FLAG_WRITE);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
		}

		ErrorCode = avformat_write_header(AvFormatContext, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		OutFrame = av_frame_alloc();
		if (!OutFrame)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");

		OutFrame->nb_samples = AvCodecContext->frame_size ? AvCodecContext->frame_size : 2048;
		OutFrame->format = AvCodecContext->sample_fmt;
		if (!IsStero)
			OutFrame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
		else
			OutFrame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
		OutFrame->sample_rate = SamplingRate;

		ErrorCode = av_frame_get_buffer(OutFrame, 0);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 1024);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		Packet = av_packet_alloc();
		if (!Packet)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate AVPacket");

		const AVSampleFormat InFormat = PCMFormat2AVSampleFormat(PCMType);
		if (InFormat != AvCodecContext->sample_fmt)
		{
			InFrame = av_frame_alloc();
			if (!InFrame)
				_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");

			InFrame->nb_samples = AvCodecContext->frame_size ? AvCodecContext->frame_size : 2048;
			InFrame->format = InFormat;
			if (!IsStero)
				InFrame->ch_layout = AV_CHANNEL_LAYOUT_MONO;
			else
				InFrame->ch_layout = AV_CHANNEL_LAYOUT_STEREO;
			InFrame->sample_rate = SamplingRate;

			ErrorCode = av_frame_get_buffer(InFrame, 0);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			AVChannelLayout inChannelLayout, outChannelLayout;
			if (!IsStero)
			{
				inChannelLayout = AV_CHANNEL_LAYOUT_MONO;
				outChannelLayout = AV_CHANNEL_LAYOUT_MONO;
			}
			else
			{
				inChannelLayout = AV_CHANNEL_LAYOUT_STEREO;
				outChannelLayout = AV_CHANNEL_LAYOUT_STEREO;
			}
			SwrContext = swr_alloc();
			ErrorCode = swr_alloc_set_opts2(
				&SwrContext,
				&outChannelLayout,
				AvCodecContext->sample_fmt,
				SamplingRate,
				&inChannelLayout,
				InFormat,
				SamplingRate,
				0,
				nullptr
			);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			ErrorCode = swr_init(SwrContext);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
		}

		bool IsPlanar = av_sample_fmt_is_planar(InFormat);
		const auto ChannelCount = static_cast<UInt64>(OutFrame->ch_layout.nb_channels);
		const auto SSegmentCount = (UInt64)OutFrame->nb_samples * ChannelCount * av_get_bytes_per_sample(InFormat);

		const TemplateLibrary::Array PCMShape{
			IsPlanar ? ChannelCount : 1ull,
			IsPlanar ? PCMData.Size() / ChannelCount : PCMData.Size()
		};
		const auto SegmentCount = IsPlanar ? SSegmentCount / ChannelCount : SSegmentCount;
		const auto PCMPointer = PCMData.Data();
		UInt64 PCMOffset = 0;
		while (PCMOffset < PCMShape[1]) {
			const auto FrameSize = std::min(
				SegmentCount,
				PCMShape[1] - PCMOffset
			);

			auto FixFrameFn = [&](const AVFrame* Frame ,UInt64 i)
				{
					//Frame->data[i][0 : FrameSize]  PCMData[i][PCMOffset : PCMOffset + FrameSize]
					memcpy(Frame->data[i], PCMPointer + i * PCMShape[1] + PCMOffset, FrameSize);
					if (FrameSize < SegmentCount)
						memset(Frame->data[i] + FrameSize, 0, SegmentCount - FrameSize);
				};

			if (InFormat != AvCodecContext->sample_fmt)
			{
				if (IsPlanar)
					for (UInt64 i = 0; i < ChannelCount; ++i)
						FixFrameFn(InFrame, i);
				else
					FixFrameFn(InFrame, 0);

				ErrorCode = swr_convert(
					SwrContext,
					OutFrame->data,
					OutFrame->nb_samples,
					InFrame->data,
					InFrame->nb_samples
				);
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}
			}
			else
			{
				if (IsPlanar)
					for (UInt64 i = 0; i < ChannelCount; ++i)
						FixFrameFn(OutFrame, i);
				else
					FixFrameFn(OutFrame, 0);
			}

			ErrorCode = avcodec_send_frame(AvCodecContext, OutFrame);
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			PCMOffset += FrameSize;

			while (true) {
				ErrorCode = avcodec_receive_packet(AvCodecContext, Packet);
				if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
					break;
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}

				ErrorCode = av_interleaved_write_frame(AvFormatContext, Packet);
				if (ErrorCode < 0)
				{
					av_strerror(ErrorCode, ErrorMessage, 1024);
					_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
				}
				av_packet_unref(Packet);
			}
		}

		avcodec_send_frame(AvCodecContext, nullptr);
		while (true) {
			ErrorCode = avcodec_receive_packet(AvCodecContext, Packet);
			if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
				break;
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			ErrorCode = av_interleaved_write_frame(AvFormatContext, Packet);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
			av_packet_unref(Packet);
		}
		av_write_trailer(AvFormatContext);

		Release();
	}*/

	AudioFrame AudioFrame::CreateReference()
	{
		AudioFrame RetFrame;
		auto Frame = av_frame_alloc();
		if (!Frame)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");
		RetFrame._MyFrame = std::shared_ptr<void>(
			Frame,
			[](void* Frame)
			{
				if (Frame)
				{
					av_frame_unref((AVFrame*)Frame);
					av_frame_free((AVFrame**)&Frame);
				}
			}
		);
		return RetFrame;
	}

	AudioFrame::AudioFrame(
		long _SamplingRate,
		long _ChannelCount,
		AvCodec::PCMFormat _Format,
		long _SampleCount
	)
	{
		auto Frame = av_frame_alloc();
		if (!Frame)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio frame");

		Frame->nb_samples = _SampleCount;
		Frame->format = PCMFormat2AVSampleFormat(_Format);
		av_channel_layout_default(&Frame->ch_layout, _ChannelCount);
		Frame->sample_rate = _SamplingRate;
		int ErrorCode = av_frame_get_buffer(Frame, 0);
		if (ErrorCode < 0)
		{
			char ErrorMessage[128];
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		_MyFrame = std::shared_ptr<void>(
			Frame,
			[](void* Frame)
			{
				if (Frame)
					av_frame_free((AVFrame**)&Frame);
			}
		);
	}

	const uint8_t* const* AudioFrame::GetDataPointerArray() const
	{
		auto Frame = (AVFrame*)_MyFrame.get();
		if (!Frame || Frame->nb_samples == 0 || Frame->data[0] == nullptr)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized");
		return Frame->data;
	}

	int AudioFrame::GetLinesize(int i) const
	{
		auto Frame = (AVFrame*)_MyFrame.get();
		if (!Frame || Frame->nb_samples == 0 || Frame->linesize[0] == 0)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized");
		return Frame->linesize[i];
	}

	int AudioFrame::GetSampleCount() const
	{
		auto Frame = (AVFrame*)_MyFrame.get();
		if (!Frame)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized");
		return Frame->nb_samples;
	}

	AudioFrame& AudioFrame::SetDataPointer(uint8_t** _Data, ULong _BufferCount)
	{
		auto Frame = (AVFrame*)_MyFrame.get();
		if (!Frame || Frame->nb_samples == 0 || Frame->data[0] == nullptr)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized");

		_BufferCount = std::min(_BufferCount, static_cast<ULong>(Frame->ch_layout.nb_channels));
		if (!av_sample_fmt_is_planar(static_cast<AVSampleFormat>(Frame->format)) && _BufferCount > 1)
			_D_Dragonian_Lib_Throw_Exception("Buffer count should be 1 for packed format");

		for (ULong i = 0; i < _BufferCount; ++i)
			Frame->data[i] = _Data[i];

		return *this;
	}

	AudioFrame& AudioFrame::CopyData(const uint8_t* const* _Data, ULong _BufferCount, ULong _SampleCount, ULong _PaddingCount)
	{
		auto Frame = (AVFrame*)_MyFrame.get();
		if (!Frame || Frame->nb_samples == 0 || Frame->data[0] == nullptr)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized");

		const auto TotalSampleCount = _SampleCount + _PaddingCount;
		const bool IsPlanar = av_sample_fmt_is_planar(static_cast<AVSampleFormat>(Frame->format));
		const bool IsPack = !IsPlanar;
		const auto Bps = size_t(av_get_bytes_per_sample(static_cast<AVSampleFormat>(Frame->format)));

		_BufferCount = std::min(_BufferCount, static_cast<ULong>(Frame->ch_layout.nb_channels));
		if (IsPlanar && std::cmp_not_equal(TotalSampleCount, Frame->nb_samples))
			_D_Dragonian_Lib_Throw_Exception("Total sample count (including padding) should be equal to frame's sample count");
		if (IsPack && TotalSampleCount != Frame->nb_samples * static_cast<ULong>(Frame->ch_layout.nb_channels))
			_D_Dragonian_Lib_Throw_Exception("Total sample count (including padding) should be equal to frame's sample count");
		if (IsPack && _BufferCount != 1)
			_D_Dragonian_Lib_Throw_Exception("Buffer count should be 1 for packed format");

		const auto BufferSize = _SampleCount * Bps;
		const auto PaddingSize = _PaddingCount * Bps;

		if (BufferSize)
			for (ULong i = 0; i < _BufferCount; ++i)
				memcpy(Frame->data[i], _Data[i], BufferSize);
		if (PaddingSize)
			for (ULong i = 0; i < _BufferCount; ++i)
				memset(Frame->data[i] + BufferSize, 0, PaddingSize);

		return *this;
	}

	AudioPacket AudioPacket::New()
	{
		AudioPacket _Packet;
		auto Packet = av_packet_alloc();
		if (!Packet)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio packet");
		_Packet._MyPacket = std::shared_ptr<void>(
			Packet,
			[](void* Packet)
			{
				if (Packet)
					av_packet_free((AVPacket**)&Packet);
			}
		);
		return _Packet;
	}

	AudioPacket AudioPacket::CreateReference()
	{
		AudioPacket _Packet;
		auto Packet = av_packet_alloc();
		if (!Packet)
			_D_Dragonian_Lib_Throw_Exception("Could not allocate audio packet");
		_Packet._MyPacket = std::shared_ptr<void>(
			Packet,
			[](void* Packet)
			{
				if (Packet)
				{
					av_packet_unref((AVPacket*)Packet);
					av_packet_free((AVPacket**)&Packet);
				}
			}
		);
		return _Packet;
	}

	AudioResampler::AudioResampler(const AudioResamplerSettings& _Settings)
	{
		_D_Dragonian_Lib_Rethrow_Block(Reset(_Settings););
	}

	void AudioResampler::Reset(
		const AudioResamplerSettings& _Settings
	)
	{
		std::lock_guard lg(_MyMutex);

		char ErrorMessage[128];
		_MySettings = _Settings;

		const int inSampleRate = static_cast<int>(_MySettings._InputSamplingRate);
		const AVSampleFormat inFormat = PCMFormat2AVSampleFormat(_MySettings._InputFormat);
		const int inChannelCount = static_cast<int>(_MySettings._InputChannels);

		const int outSampleRate = static_cast<int>(_MySettings._OutputSamplingRate);
		const AVSampleFormat outFormat = PCMFormat2AVSampleFormat(_MySettings._OutputFormat);
		const int outChannelCount = static_cast<int>(_MySettings._OutputChannels);

		AVChannelLayout inChannelLayout, outChannelLayout;

		av_channel_layout_default(&inChannelLayout, inChannelCount);
		av_channel_layout_default(&outChannelLayout, outChannelCount);

		SwrContext* _Sampler = swr_alloc();

		int ErrorCode = swr_alloc_set_opts2(
			&_Sampler,
			&outChannelLayout,
			outFormat,
			outSampleRate,
			&inChannelLayout,
			inFormat,
			inSampleRate,
			0,
			nullptr
		);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		ErrorCode = swr_init(_Sampler);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		_MySwrContext = std::shared_ptr<void>(
			_Sampler,
			[](void* Context)
			{
				if (Context)
				{
					swr_close((SwrContext*)Context);
					swr_free((SwrContext**)&Context);
				}
			}
		);
	}

	void AudioResampler::Resample(
		AudioFrame& _OutputFrame,
		const AudioFrame& _InputFrame
	) const
	{
		std::lock_guard lg(_MyMutex);
		AVFrame* OutFrame = (AVFrame*)_OutputFrame.Get();
		AVFrame* InFrame = (AVFrame*)_InputFrame.Get();
		if (!Enabled())
			_D_Dragonian_Lib_Throw_Exception("Resampler is not initialized");
		int ErrorCode = swr_convert(
			static_cast<SwrContext*>(_MySwrContext.get()),
			OutFrame->data,
			OutFrame->nb_samples,
			InFrame->data,
			InFrame->nb_samples
		);
		if (ErrorCode < 0)
		{
			char ErrorMessage[128];
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}
	}

	AudioFrame AudioResampler::Resample(
		const AudioFrame& _InputFrame
	) const
	{
		AVFrame* InFrame = (AVFrame*)_InputFrame.Get();
		auto dstNbSamples = GetOutputSampleCount(InFrame->nb_samples);
		_D_Dragonian_Lib_Rethrow_Block(
			{
				AudioFrame _OutputFrame(
					_MySettings._OutputSamplingRate,
					_MySettings._OutputChannels,
					_MySettings._OutputFormat,
					(long)dstNbSamples
				);
				Resample(_OutputFrame, _InputFrame);
				return _OutputFrame;
			}
		);
	}

	Int32 AudioResampler::Resample(
		void* const* _OutputData,
		size_t _OutputSampleCount,
		const void* const* _InputData,
		size_t _InputSampleCount
	) const
	{
		std::lock_guard lg(_MyMutex);
		if (!Enabled())
			_D_Dragonian_Lib_Throw_Exception("Resampler is not initialized");
		int Ret = swr_convert(
			static_cast<SwrContext*>(_MySwrContext.get()),
			(UInt8* const*)_OutputData,
			static_cast<int>(_OutputSampleCount),
			(const UInt8* const*)_InputData,
			static_cast<int>(_InputSampleCount)
		);
		if (Ret < 0)
		{
			char ErrorMessage[128];
			av_strerror(Ret, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}
		return Ret;
	}

	std::shared_ptr<void> AudioResampler::Resample(
		const void* const* _InputData,
		size_t _InputSampleCount,
		size_t* _OutputSampleCount,
		void* (*_Alloc)(size_t),
		void (*_Free)(void*)
	) const
	{
		if (!_Free)
			_Free = DefaultFree;
		size_t OutputSampleCount;
		std::shared_ptr<void> Ret;
		_D_Dragonian_Lib_Rethrow_Block(Ret = GetOutputBuffer(_InputSampleCount, _Alloc, _Free, &OutputSampleCount););
		if (_OutputSampleCount) *_OutputSampleCount = OutputSampleCount;
		_D_Dragonian_Lib_Rethrow_Block(Resample((void**)Ret.get(), OutputSampleCount, _InputData, _InputSampleCount););
		return Ret;
	}

	std::shared_ptr<void> AudioResampler::GetOutputBuffer(
		size_t _InputSampleCount,
		void* (*_Alloc)(size_t),
		void (*_Free)(void*),
		size_t* _OutputSampleCount
	) const
	{
		if (bool(_Free) xor bool(_Alloc))
			_D_Dragonian_Lib_Throw_Exception("Both _Alloc and _Free must be set or unset");
		if (!_Alloc) _Alloc = DefaultAlloc;
		if (!_Free) _Free = DefaultFree;

		const auto dstNbSamples = GetOutputSampleCount(_InputSampleCount);
		const auto Bps = av_get_bytes_per_sample(PCMFormat2AVSampleFormat(_MySettings._OutputFormat));
		const auto OutputChannel = _MySettings._OutputChannels;
		if (_OutputSampleCount) *_OutputSampleCount = dstNbSamples;

		auto OutBuffer = _Alloc(sizeof(void*) * OutputChannel);
		if (!OutBuffer) _D_Dragonian_Lib_Throw_Exception("Memory allocation failed");
		for (size_t i = 0; i < OutputChannel; ++i)
		{
			((uint8_t**)OutBuffer)[i] = (uint8_t*)_Alloc(dstNbSamples * Bps);
			if (!((uint8_t**)OutBuffer)[i])
			{
				for (size_t j = 0; j < i; ++j)
					_Free(((uint8_t**)OutBuffer)[j]);
				_Free(OutBuffer);
				_D_Dragonian_Lib_Throw_Exception("Memory allocation failed");
			}
		}

		return {
			OutBuffer,
			[_Free, OutputChannel](void* _MyBuf)
			{
				auto Buf = (void**)_MyBuf;
				for (UInt32 i = 0; i < OutputChannel; ++i)
					_Free(Buf[i]);
				_Free(_MyBuf);
			}
		};
	}

	size_t AudioResampler::GetOutputSampleCount(
		size_t _InputSampleCount
	) const
	{
		return av_rescale_rnd(
			static_cast<int64_t>(_InputSampleCount),
			_MySettings._OutputSamplingRate,
			_MySettings._InputSamplingRate,
			AV_ROUND_ZERO
		);
	}

	AudioCodec::AudioCodec(
		const AudioCodecSettings& _Settings
	)
	{
		Reset(_Settings);
	}

	void AudioCodec::Reset(
		const AudioCodecSettings& _Settings
	)
	{
		std::lock_guard lg(_MyMutex);

		char ErrorMessage[128];
		int ErrorCode;
		_MySettings = _Settings;

		if (_MySettings._Type == DECODER)
		{
			if (_MySettings._OutputSampleFormat == AvCodec::PCM_FORMAT_NONE)
				_D_Dragonian_Lib_Throw_Exception("Input sample format is not set!");

			auto InCodecFormat = (AVCodecID)_MySettings._Format;
			const AVCodec* AVDecodec = avcodec_find_decoder(InCodecFormat);
			if (AVDecodec == nullptr)
				_D_Dragonian_Lib_Throw_Exception("Codec not found");

			AVCodecContext* DecoderContext = avcodec_alloc_context3(AVDecodec);
			if (!DecoderContext)
				_D_Dragonian_Lib_Throw_Exception("Could not create audio codec context");

			ErrorCode = avcodec_parameters_to_context(DecoderContext, static_cast<const AVCodecParameters*>(_MySettings._Parameters));
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 128);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			ErrorCode = avcodec_open2(DecoderContext, AVDecodec, static_cast<AVDictionary**>(_MySettings._ParameterDict));
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 128);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			_MyContext = std::shared_ptr<void>(
				DecoderContext,
				[](void* Context)
				{
					if (Context)
						avcodec_free_context((AVCodecContext**)&Context);
				}
			);
			_MySettings._InputSampleFormat = AVSampleFormat2PCMFormat(DecoderContext->sample_fmt);

			if (_MySettings._InputSampleFormat != _MySettings._OutputSampleFormat ||
				_MySettings._InputChannels != _MySettings._OutputChannels ||
				_MySettings._InputSamplingRate != _MySettings._OutputSamplingRate)
			{
				_MyResampler.Reset(
					{
						_MySettings._InputSamplingRate,
						_MySettings._InputSampleFormat,
						_MySettings._InputChannels,
						_MySettings._OutputSamplingRate,
						_MySettings._OutputSampleFormat,
						_MySettings._OutputChannels
					}
				);
				_NeedResample = true;
			}
		}
		else
		{
			if (_MySettings._InputSampleFormat == AvCodec::PCM_FORMAT_NONE)
				_D_Dragonian_Lib_Throw_Exception("Input sample format is not set!");

			auto OutCodecFormat = (AVCodecID)_MySettings._Format;
			auto AVEncodec = avcodec_find_encoder(OutCodecFormat);
			if (!AVEncodec)
				_D_Dragonian_Lib_Throw_Exception("Codec not found");
			{
				bool Found = false;
				for (auto Iter = AVEncodec->sample_fmts; *Iter != AV_SAMPLE_FMT_NONE; ++Iter)
					if (*Iter == PCMFormat2AVSampleFormat(_MySettings._OutputSampleFormat))
					{
						Found = true;
						break;
					}
				if (!Found)
					_MySettings._OutputSampleFormat = AVSampleFormat2PCMFormat(AVEncodec->sample_fmts[0]);
			}

			AVCodecContext* EncoderContext = avcodec_alloc_context3(AVEncodec);
			if (!EncoderContext)
				_D_Dragonian_Lib_Throw_Exception("Could not create audio codec context");

			EncoderContext->codec_id = AVEncodec->id;
			EncoderContext->codec_type = AVMEDIA_TYPE_AUDIO;
			EncoderContext->sample_fmt = PCMFormat2AVSampleFormat(_MySettings._OutputSampleFormat);
			EncoderContext->sample_rate = static_cast<int>(_MySettings._OutputSamplingRate);
			av_channel_layout_default(&EncoderContext->ch_layout, static_cast<int>(_MySettings._OutputChannels));
			EncoderContext->time_base = { 1, static_cast<int>(_MySettings._OutputSamplingRate) };

			ErrorCode = avcodec_open2(EncoderContext, AVEncodec, static_cast<AVDictionary**>(_MySettings._ParameterDict));
			if (ErrorCode)
			{
				av_strerror(ErrorCode, ErrorMessage, 128);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			_MyContext = std::shared_ptr<void>(
				EncoderContext,
				[](void* Context)
				{
					if (Context)
						avcodec_free_context((AVCodecContext**)&Context);
				}
			);

			if (_MySettings._InputSampleFormat != _MySettings._OutputSampleFormat ||
				_MySettings._InputChannels != _MySettings._OutputChannels ||
				_MySettings._InputSamplingRate != _MySettings._OutputSamplingRate)
			{
				_MyResampler.Reset(
					{
						_MySettings._InputSamplingRate,
						_MySettings._InputSampleFormat,
						_MySettings._InputChannels,
						_MySettings._OutputSamplingRate,
						_MySettings._OutputSampleFormat,
						_MySettings._OutputChannels
					}
				);
				_NeedResample = true;
			}
		}
	}

	Long AudioCodec::GetFrameSize() const
	{
		if (!_MyContext)
			_D_Dragonian_Lib_Throw_Exception("Codec is not initialized!");
		return static_cast<AVCodecContext*>(_MyContext.get())->frame_size;
	}

	UInt32 AudioCodec::GetBitRate() const
	{
		if (!_MyContext)
			_D_Dragonian_Lib_Throw_Exception("Codec is not initialized!");
		auto MyContext = static_cast<AVCodecContext*>(_MyContext.get());
		return static_cast<UInt32>(MyContext->bit_rate);
	}

	TemplateLibrary::Vector<AudioFrame> AudioCodec::Decode(const AudioPacket& _Packet) const
	{
		if (_MySettings._Type != DECODER)
			_D_Dragonian_Lib_Throw_Exception("Codec is not a decoder!");

		std::lock_guard lg(_MyMutex);
		if (!_Packet._MyPacket)
			_D_Dragonian_Lib_Throw_Exception("Packet is not initialized!");
		if (!_MyContext)
			_D_Dragonian_Lib_Throw_Exception("Codec is not initialized!");

		auto MyPacket = static_cast<AVPacket*>(_Packet._MyPacket.get());
		auto MyContext = static_cast<AVCodecContext*>(_MyContext.get());

		if (int ErrorCode = avcodec_send_packet(MyContext, MyPacket); ErrorCode)
		{
			char ErrorMessage[128];
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		TemplateLibrary::Vector<AudioFrame> OFrames;
		int Ret = 0;
		while (!Ret)
		{
			auto FrameBuffer = AudioFrame::CreateReference();
			Ret = avcodec_receive_frame(MyContext, static_cast<AVFrame*>(FrameBuffer.Get()));
			if (Ret == AVERROR(EAGAIN) || Ret == AVERROR_EOF)
				break;
			if (Ret)
			{
				char ErrorMessage[128];
				av_strerror(Ret, ErrorMessage, 128);
				GetAvcodecLogger()->LogWarn(UTF8ToWideString(ErrorMessage));
				continue;
			}
			if (_NeedResample)
				_D_Dragonian_Lib_Rethrow_Block(FrameBuffer = _MyResampler.Resample(FrameBuffer););
			OFrames.EmplaceBack(std::move(FrameBuffer));
		}
		return OFrames;
	}

	TemplateLibrary::Vector<AudioPacket> AudioCodec::Encode(const AudioFrame& _Frame) const
	{
		if (_MySettings._Type != ENCODER)
			_D_Dragonian_Lib_Throw_Exception("Codec is not an encoder!");

		std::lock_guard lg(_MyMutex);

		if (!_Frame._MyFrame)
			_D_Dragonian_Lib_Throw_Exception("Frame is not initialized!");
		if (!_MyContext)
			_D_Dragonian_Lib_Throw_Exception("Codec is not initialized!");

		char ErrorMessage[128];
		auto MyFrame = static_cast<AVFrame*>(_Frame._MyFrame.get());
		auto MyContext = static_cast<AVCodecContext*>(_MyContext.get());

		AudioFrame ResampledFrame = _Frame;
		if (_NeedResample)
		{
			_D_Dragonian_Lib_Rethrow_Block(ResampledFrame = _MyResampler.Resample(_Frame););
			MyFrame = static_cast<AVFrame*>(ResampledFrame.Get());
		}

		int ErrorCode = avcodec_send_frame(MyContext, MyFrame);
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		TemplateLibrary::Vector<AudioPacket> OPackets;
		while (true) {
			auto MPacket = AudioPacket::CreateReference();
			ErrorCode = avcodec_receive_packet(MyContext, static_cast<AVPacket*>(MPacket.Get()));
			if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
				break;
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 128);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			/*ErrorCode = av_interleaved_write_frame(AvFormatContext, Packet);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}*/
			OPackets.EmplaceBack(std::move(MPacket));
		}
		return OPackets;
	}

	TemplateLibrary::Vector<AudioPacket> AudioCodec::EncodeTail() const
	{
		auto MyContext = static_cast<AVCodecContext*>(_MyContext.get());
		TemplateLibrary::Vector<AudioPacket> OPackets;
		avcodec_send_frame(MyContext, nullptr);
		while (true) {
			auto MPacket = AudioPacket::CreateReference();
			auto ErrorCode = avcodec_receive_packet(MyContext, static_cast<AVPacket*>(MPacket.Get()));
			if (ErrorCode == AVERROR(EAGAIN) || ErrorCode == AVERROR_EOF)
				break;
			if (ErrorCode < 0)
			{
				char ErrorMessage[128];
				av_strerror(ErrorCode, ErrorMessage, 1024);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}

			OPackets.EmplaceBack(std::move(MPacket));
		}
		return OPackets;
	}

	AudioCodec::AudioCodecSettings AudioIStream::GetCodecSettings() const noexcept
	{
		const auto MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		const auto MyCodecPar = MyContext->streams[_MyStreamIndex]->codecpar;

		AudioCodec::AudioCodecSettings Ret;
		Ret._Type = AudioCodec::DECODER;
		Ret._Parameters = MyContext->streams[_MyStreamIndex]->codecpar;
		Ret._Format = static_cast<int>(MyCodecPar->codec_id);
		Ret._InputSamplingRate = MyCodecPar->sample_rate;
		Ret._InputChannels = MyCodecPar->ch_layout.nb_channels;
		Ret._InputSampleFormat = static_cast<AvCodec::PCMFormat>(MyCodecPar->format);

		return Ret;
	}

	void AudioIStream::Reset(const std::wstring& _Path)
	{
		if (_Path.empty())
		{
			_MyFormatContext = nullptr;
			_MyStreamIndex = -1;
			_MyExtension.clear();
		}

		AVFormatContext* _MyContext = nullptr;
		char ErrorMessage[128];

		int ErrorCode = avformat_open_input(&_MyContext, WideStringToUTF8(_Path).c_str(), nullptr, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		ErrorCode = avformat_find_stream_info(_MyContext, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		int _MyIndex = -1;
		for (unsigned i = 0; i < _MyContext->nb_streams; ++i)
			if (_MyContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
				_MyIndex = static_cast<int>(i);

		if (_MyIndex == -1)
			_D_Dragonian_Lib_Throw_Exception("Input file has no audio stream!");

		_MyFormatContext = std::shared_ptr<void>(
			_MyContext,
			[](void* _Context)
			{
				if (_Context)
					avformat_close_input((AVFormatContext**)&_Context);
			}
		);
		_MyStreamIndex = _MyIndex;
		_MyExtension = std::filesystem::path(_Path).extension().wstring().substr(1);
	}

	AudioIStream& AudioIStream::operator>>(AudioPacket& _Packet)
	{
		if (!_MyFormatContext)
			_D_Dragonian_Lib_Throw_Exception("Stream is not initialized!");
		if (_MyStreamIndex == -1)
			_D_Dragonian_Lib_Throw_Exception("Stream is not opened!");

		_Packet = AudioPacket::CreateReference();
		auto _MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		auto _MyPacket = static_cast<AVPacket*>(_Packet._MyPacket.get());

		while ((_MyCode = av_read_frame(_MyContext, _MyPacket)) >= 0)
		{
			if (_MyPacket->stream_index == _MyStreamIndex)
				break;
			av_packet_unref(_MyPacket);
		}
		return *this;
	}

	UInt64 AudioIStream::GetDurations() const
	{
		auto MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		return MyContext->streams[_MyStreamIndex]->duration;
	}

	std::pair<UInt64, UInt64> AudioIStream::GetTimeBase() const
	{
		auto MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		return { MyContext->streams[_MyStreamIndex]->time_base.num, MyContext->streams[_MyStreamIndex]->time_base.den };
	}

	AudioCodec::AudioCodecSettings AudioOStream::GetCodecSettings() const noexcept
	{
		const auto MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		const auto MyCodecPar = MyContext->streams[_MyStreamIndex]->codecpar;

		AudioCodec::AudioCodecSettings Ret;
		Ret._Type = AudioCodec::ENCODER;
		Ret._Parameters = MyContext->streams[_MyStreamIndex]->codecpar;
		Ret._Format = static_cast<int>(MyCodecPar->codec_id);
		Ret._OutputSamplingRate = MyCodecPar->sample_rate;
		Ret._OutputChannels = MyCodecPar->ch_layout.nb_channels;
		Ret._OutputSampleFormat = static_cast<AvCodec::PCMFormat>(MyCodecPar->format);

		return Ret;
	}

	void AudioOStream::Reset(
		UInt32 _OutputSamplingRate,
		const std::wstring& _Path,
		AvCodec::PCMFormat _OutputDataFormat,
		UInt32 _OutputChannelCount,
		Int32 _OutputCodecID
	)
	{
		if (_Path.empty())
		{
			_MyFormatContext = nullptr;
			_MyStreamIndex = -1;
			_MyExtension.clear();
		}

		_MyExtension = std::filesystem::path(_Path).extension().wstring().substr(1);
		AVCodecID CodecId;
		if (_OutputCodecID < 0)
		{
			if (_MyExtension == L"mp3")
				CodecId = AV_CODEC_ID_MP3;
			else if (_MyExtension == L"aac" || _MyExtension == L"m4a")
				CodecId = AV_CODEC_ID_AAC;
			else if (_MyExtension == L"flac")
				CodecId = AV_CODEC_ID_FLAC;
			else if (_MyExtension == L"wav")
				CodecId = AV_CODEC_ID_PCM_S16LE;
			else if (_MyExtension == L"ogg")
				CodecId = AV_CODEC_ID_VORBIS;
			else if (_MyExtension == L"opus")
				CodecId = AV_CODEC_ID_OPUS;
			else if (_MyExtension == L"wma")
				CodecId = AV_CODEC_ID_WMAV2;
			else if (_MyExtension == L"wav")
			{
				if (_OutputDataFormat == AvCodec::PCM_FORMAT_UINT8 || _OutputDataFormat == AvCodec::PCM_FORMAT_UINT8_PLANAR)
					CodecId = AV_CODEC_ID_PCM_U8;
				else if (_OutputDataFormat == AvCodec::PCM_FORMAT_INT16 || _OutputDataFormat == AvCodec::PCM_FORMAT_INT16_PLANAR)
					CodecId = AV_CODEC_ID_PCM_S16LE;
				else if (_OutputDataFormat == AvCodec::PCM_FORMAT_INT32 || _OutputDataFormat == AvCodec::PCM_FORMAT_INT32_PLANAR)
					CodecId = AV_CODEC_ID_PCM_S32LE;
				else if (_OutputDataFormat == AvCodec::PCM_FORMAT_FLOAT32 || _OutputDataFormat == AvCodec::PCM_FORMAT_FLOAT32_PLANAR)
					CodecId = AV_CODEC_ID_PCM_F32LE;
				else if (_OutputDataFormat == AvCodec::PCM_FORMAT_INT64 || _OutputDataFormat == AvCodec::PCM_FORMAT_INT64_PLANAR)
					CodecId = AV_CODEC_ID_PCM_S64LE;
				else if (_OutputDataFormat == AvCodec::PCM_FORMAT_FLOAT64 || _OutputDataFormat == AvCodec::PCM_FORMAT_FLOAT64_PLANAR)
					CodecId = AV_CODEC_ID_PCM_F64LE;
				else
					_D_Dragonian_Lib_Throw_Exception("Unsupported output format!");
			}
			else
				_D_Dragonian_Lib_Throw_Exception("Unsupported output format!");
		}
		else
			CodecId = static_cast<AVCodecID>(_OutputCodecID);

		bool Found = false;
		auto Codec = avcodec_find_encoder(CodecId);
		for (auto Iter = Codec->sample_fmts; *Iter != AV_SAMPLE_FMT_NONE; ++Iter)
			if (*Iter == PCMFormat2AVSampleFormat(_OutputDataFormat))
			{
				Found = true;
				break;
			}
		if (!Found)
		{
			_D_Dragonian_Lib_Namespace AvCodec::GetAvcodecLogger()->LogWarn(
				L"Codec does not support the specified sample format, using the first supported format instead"
			);
			_OutputDataFormat = AVSampleFormat2PCMFormat(Codec->sample_fmts[0]);
		}

		char ErrorMessage[128];
		AVFormatContext* _MyContext = nullptr;

		int ErrorCode = avformat_alloc_output_context2(&_MyContext, nullptr, nullptr, WideStringToUTF8(_Path).c_str());
		if (ErrorCode)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		auto Stream = avformat_new_stream(_MyContext, nullptr);
		if (!Stream)
			_D_Dragonian_Lib_Throw_Exception("Could not create stream");


		Stream->time_base = { 1, static_cast<Int32>(_OutputSamplingRate) };
		Stream->codecpar->codec_id = CodecId;
		Stream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
		Stream->codecpar->format = PCMFormat2AVSampleFormat(_OutputDataFormat);
		Stream->codecpar->sample_rate = static_cast<int>(_OutputSamplingRate);
		av_channel_layout_default(&Stream->codecpar->ch_layout, static_cast<int>(_OutputChannelCount));

		if (!(_MyContext->oformat->flags & AVFMT_NOFILE))
		{
			ErrorCode = avio_open(&_MyContext->pb, WideStringToUTF8(_Path).c_str(), AVIO_FLAG_WRITE);
			if (ErrorCode < 0)
			{
				av_strerror(ErrorCode, ErrorMessage, 128);
				_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
			}
		}

		ErrorCode = avformat_write_header(_MyContext, nullptr);
		if (ErrorCode < 0)
		{
			av_strerror(ErrorCode, ErrorMessage, 128);
			_D_Dragonian_Lib_Throw_Exception(ErrorMessage);
		}

		_MyFormatContext = std::shared_ptr<void>(
			_MyContext,
			[](void* Context)
			{
				if (Context)
				{
					av_write_trailer((AVFormatContext*)Context);
					if (!(((AVFormatContext*)Context)->oformat->flags & AVFMT_NOFILE))
						avio_closep(&((AVFormatContext*)Context)->pb);
					avformat_free_context((AVFormatContext*)(Context));
				}
			}
		);
		_MyStreamIndex = 0;
	}

	AudioOStream& AudioOStream::SetBitRate(UInt32 _BitRate)
	{
		if (!_MyFormatContext)
			_D_Dragonian_Lib_Throw_Exception("Stream is not initialized!");
		if (_MyStreamIndex == -1)
			_D_Dragonian_Lib_Throw_Exception("Stream is not opened!");
		auto _MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		auto _MyStream = _MyContext->streams[_MyStreamIndex];
		_MyStream->codecpar->bit_rate = _BitRate;
		return *this;
	}

	AudioOStream& AudioOStream::operator<<(const AudioPacket& _Packet)
	{
		if (!_MyFormatContext)
			_D_Dragonian_Lib_Throw_Exception("Stream is not initialized!");
		if (_MyStreamIndex == -1)
			_D_Dragonian_Lib_Throw_Exception("Stream is not opened!");
		if (!_Packet._MyPacket)
			_D_Dragonian_Lib_Throw_Exception("Packet is not initialized!");
		auto _MyContext = static_cast<AVFormatContext*>(_MyFormatContext.get());
		auto _MyPacket = static_cast<AVPacket*>(_Packet._MyPacket.get());
		_MyPacket->stream_index = _MyStreamIndex;
		av_interleaved_write_frame(_MyContext, _MyPacket);
		return *this;
	}

	AudioIStream OpenInputStream(const std::wstring& _Path)
	{
		AudioIStream _MyStream;
		_D_Dragonian_Lib_Rethrow_Block(_MyStream.Reset(_Path););
		return _MyStream;
	}

	AudioOStream OpenOutputStream(
		UInt32 _OutputSamplingRate,
		const std::wstring& _Path,
		AvCodec::PCMFormat _OutputDataFormat,
		UInt32 _OutputChannelCount,
		Int32 _OutputCodecID
	)
	{
		AudioOStream _MyStream;
		_D_Dragonian_Lib_Rethrow_Block(_MyStream.Reset(_OutputSamplingRate, _Path, _OutputDataFormat, _OutputChannelCount, _OutputCodecID););
		return _MyStream;
	}

	static TemplateLibrary::Vector<UInt8> Transpose2Packed(const TemplateLibrary::ConstantRanges<Byte>& PCMData, AVSampleFormat SampleFormat)
	{
		TemplateLibrary::Vector<UInt8> Ret(PCMData.Size());
		const auto Samples = PCMData.Size() / av_get_bytes_per_sample(SampleFormat) / 2;
		if (SampleFormat == AV_SAMPLE_FMT_U8P || SampleFormat == AV_SAMPLE_FMT_U8)
		{
			for (UInt64 i = 0; i < Samples; ++i)
			{
				Ret[i * 2] = PCMData[i];
				Ret[i * 2 + 1] = PCMData[i + Samples];
			}
		}
		else if (SampleFormat == AV_SAMPLE_FMT_S16P || SampleFormat == AV_SAMPLE_FMT_S16)
		{
			const auto RetDataPtr = (Int16*)Ret.Data();
			const auto PCMDataPtr = (const Int16*)PCMData.Data();
			for (UInt64 i = 0; i < Samples; ++i)
			{
				RetDataPtr[i * 2] = PCMDataPtr[i];
				RetDataPtr[i * 2 + 1] = PCMDataPtr[i + Samples];
			}
		}
		else if (SampleFormat == AV_SAMPLE_FMT_S32P || SampleFormat == AV_SAMPLE_FMT_S32)
		{
			const auto RetDataPtr = (Int32*)Ret.Data();
			const auto PCMDataPtr = (const Int32*)PCMData.Data();
			for (UInt64 i = 0; i < Samples; ++i)
			{
				RetDataPtr[i * 2] = PCMDataPtr[i];
				RetDataPtr[i * 2 + 1] = PCMDataPtr[i + Samples];
			}
		}
		else if (SampleFormat == AV_SAMPLE_FMT_FLTP || SampleFormat == AV_SAMPLE_FMT_FLT)
		{
			const auto RetDataPtr = (Float32*)Ret.Data();
			const auto PCMDataPtr = (const Float32*)PCMData.Data();
			for (UInt64 i = 0; i < Samples; ++i)
			{
				RetDataPtr[i * 2] = PCMDataPtr[i];
				RetDataPtr[i * 2 + 1] = PCMDataPtr[i + Samples];
			}
		}
		else if (SampleFormat == AV_SAMPLE_FMT_DBLP || SampleFormat == AV_SAMPLE_FMT_DBL)
		{
			const auto RetDataPtr = (Float64*)Ret.Data();
			const auto PCMDataPtr = (const Float64*)PCMData.Data();
			for (UInt64 i = 0; i < Samples; ++i)
			{
				RetDataPtr[i * 2] = PCMDataPtr[i];
				RetDataPtr[i * 2 + 1] = PCMDataPtr[i + Samples];
			}
		}
		else if (SampleFormat == AV_SAMPLE_FMT_S64P || SampleFormat == AV_SAMPLE_FMT_S64)
		{
			const auto RetDataPtr = (Int64*)Ret.Data();
			const auto PCMDataPtr = (const Int64*)PCMData.Data();
			for (UInt64 i = 0; i < Samples; ++i)
			{
				RetDataPtr[i * 2] = PCMDataPtr[i];
				RetDataPtr[i * 2 + 1] = PCMDataPtr[i + Samples];
			}
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Unsupported sample format!");

		return Ret;
	}

	void WritePCMData(
		const std::wstring& OutputPath,
		const TemplateLibrary::ConstantRanges<Byte>& PCMData,
		int SamplingRate,
		AvCodec::PCMFormat DataFormat,
		bool IsStero
	)
	{
		AVSampleFormat SampleFormat = PCMFormat2AVSampleFormat(DataFormat);
		const auto BytesPerSamples = av_get_bytes_per_sample(SampleFormat);
		bool IsFloat = SampleFormat == AV_SAMPLE_FMT_FLT || SampleFormat == AV_SAMPLE_FMT_DBL || SampleFormat == AV_SAMPLE_FMT_FLTP || SampleFormat == AV_SAMPLE_FMT_DBLP;
		int Channels = IsStero ? 2 : 1;
		bool IsPlannar = av_sample_fmt_is_planar(SampleFormat);
		const RiffWaveHeader Header{
			.RiffChunkSize = uint32_t(36 + PCMData.Size()),
			.WaveChunkSize = 16,
			.AudioFormat = uint16_t(IsFloat ? 0x0003 : 0x0001),
			.ChannelCount = uint16_t(Channels),
			.SamplingRate = uint32_t(SamplingRate),
			.ByteRate = uint32_t(SamplingRate * BytesPerSamples * Channels),
			.SampleAlign = IsPlannar ? uint16_t(BytesPerSamples) : uint16_t(BytesPerSamples * Channels),
			.SampleBits = uint16_t(BytesPerSamples * 8),
			.DataChunkSize = (uint32_t)PCMData.Size()
		};
		FileGuard File;
		File.Open(OutputPath, L"wb");
		if (!File.Enabled())
			_D_Dragonian_Lib_Throw_Exception("could not open file!");
		fwrite(&Header, 1, sizeof(RiffWaveHeader), File);

		if (av_sample_fmt_is_planar(SampleFormat))
		{
			auto Output = Transpose2Packed(PCMData, SampleFormat);
			fwrite(Output.Data(), 1, Output.Size(), File);
		}
		else
			fwrite(PCMData.Data(), 1, PCMData.Size(), File);
	}

	void WriteMidiFile(
		const std::wstring& Path,
		const MidiTrack& Events,
		long Begin,
		long TPS,
		long Tempo
	)
	{
		libremidi::writer Writer;
		Writer.add_track();
		std::vector<MidiEvent> MidiEvents;
		for (const auto& NoteEvent : Events.NoteEvents)
		{
			MidiEvents.emplace_back(NoteEvent.OnsetTime, NoteEvent.MidiNote, NoteEvent.Velocity);
			MidiEvents.emplace_back(NoteEvent.OffsetTime, NoteEvent.MidiNote, 0);
		}
		Writer.add_event(0, 0, libremidi::meta_events::tempo(Tempo));
		std::sort(MidiEvents.begin(), MidiEvents.end());  // NOLINT(modernize-use-ranges)
		long PreviousTicks = Begin;
		for (const auto& Event : MidiEvents) {
			const long CurrentTicks = long((Event.Time - Begin) * TPS);
			if (CurrentTicks >= 0)
			{
				long DiffTicks = CurrentTicks - PreviousTicks;
				DiffTicks = std::max(0l, DiffTicks);
				PreviousTicks = CurrentTicks;
				if (Event.Velocity)
					Writer.add_event(DiffTicks, 0, libremidi::channel_events::note_on(0, uint8_t(unsigned long(Event.MidiNote)), uint8_t(unsigned long(Event.Velocity))));
				else
					Writer.add_event(DiffTicks, 0, libremidi::channel_events::note_off(0, uint8_t(unsigned long(Event.MidiNote)), uint8_t(unsigned long(Event.Velocity))));
			}
		}
		Writer.add_event(0, 0, libremidi::meta_events::end_of_track());
		auto OutputFileStream = std::ofstream(Path, std::ios::out | std::ios::binary);
		if (!OutputFileStream.is_open())
			_D_Dragonian_Lib_Throw_Exception("Could not write file!");
		Writer.write(OutputFileStream);
	}

	DragonianLibSTL::Vector<size_t> SliceAudio(
		const DragonianLibSTL::Vector<float>& PcmData,
		const SlicerSettings& SlicerSettings
	)
	{
		const auto MinSamples = size_t(SlicerSettings.MinLength) * SlicerSettings.SamplingRate;
		if (PcmData.Size() < MinSamples)
			return { 0, PcmData.Size() };

		DragonianLibSTL::Vector<unsigned long long> SlicePos;
		SlicePos.EmplaceBack(0);
		auto TotalCount = static_cast<ptrdiff_t>(PcmData.Size() - SlicerSettings.WindowLength);

		ptrdiff_t LastPos = 0;
		bool LastIsVocalPart = CalculateDB(PcmData.Begin(), PcmData.Begin() + SlicerSettings.WindowLength) > SlicerSettings.Threshold;
		for (ptrdiff_t Pos = SlicerSettings.HopSize; Pos < TotalCount; Pos += SlicerSettings.HopSize)
		{
			const auto DB = CalculateDB(
				PcmData.Begin() + Pos,
				PcmData.Begin() + Pos + SlicerSettings.WindowLength
			);
			const auto IsVocalPart = DB > SlicerSettings.Threshold;

			if (Pos - LastPos < ptrdiff_t(MinSamples))
				continue;

			if ((IsVocalPart && !LastIsVocalPart) || (!IsVocalPart && LastIsVocalPart))
			{
				SlicePos.EmplaceBack(Pos + SlicerSettings.HopSize / 2);
				LastPos = Pos;
			}
			LastIsVocalPart = IsVocalPart;
		}

		SlicePos.EmplaceBack(PcmData.Size());
		return SlicePos;
	}

	DragonianLibSTL::Vector<size_t> SliceAudio(
		const TemplateLibrary::ConstantRanges<float>& PcmData,
		const SlicerSettings& SlicerSettings
	)
	{
		const auto MinSamples = size_t(SlicerSettings.MinLength) * SlicerSettings.SamplingRate;
		if (PcmData.Size() < MinSamples)
			return { 0, PcmData.Size() };

		DragonianLibSTL::Vector<unsigned long long> SlicePos;
		SlicePos.EmplaceBack(0);
		auto TotalCount = static_cast<ptrdiff_t>(PcmData.Size() - SlicerSettings.WindowLength);

		ptrdiff_t LastPos = 0;
		bool LastIsVocalPart = CalculateDB(PcmData.Begin(), PcmData.Begin() + SlicerSettings.WindowLength) > SlicerSettings.Threshold;
		for (ptrdiff_t Pos = SlicerSettings.HopSize; Pos < TotalCount; Pos += SlicerSettings.HopSize)
		{
			const auto DB = CalculateDB(
				PcmData.Begin() + Pos,
				PcmData.Begin() + Pos + SlicerSettings.WindowLength
			);
			const auto IsVocalPart = DB > SlicerSettings.Threshold;

			if (Pos - LastPos < ptrdiff_t(MinSamples))
				continue;

			if ((IsVocalPart && !LastIsVocalPart) || (!IsVocalPart && LastIsVocalPart))
			{
				SlicePos.EmplaceBack(Pos + SlicerSettings.HopSize / 2);
				LastPos = Pos;
			}
			LastIsVocalPart = IsVocalPart;
		}

		SlicePos.EmplaceBack(PcmData.Size());
		return SlicePos;
	}
}

_D_Dragonian_Lib_Space_End