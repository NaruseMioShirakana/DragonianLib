#include "../AvCodec.h"
#include "Libraries/Base.h"
#include "Libraries/MyTemplateLibrary/Array.h"
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

	AVSampleFormat PCMFormat2AVSampleFormat(const AvCodec::PCMFormat& Format) noexcept
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
		default: return AV_SAMPLE_FMT_NONE;
		}
	}

	DragonianLibSTL::Vector<unsigned char> AvCodec::Decode(
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
	}

	TemplateLibrary::Vector<UInt8> Transpose2Packed(const TemplateLibrary::ConstantRanges<Byte>& PCMData, AVSampleFormat SampleFormat)
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
		const RiffWaveHeader Header{
			.RiffChunkSize = uint32_t(36 + PCMData.Size()),
			.WaveChunkSize = 16,
			.AudioFormat = uint16_t(IsFloat ? 0x0003 : 0x0001),
			.ChannelCount = uint16_t(Channels),
			.SamplingRate = uint32_t(SamplingRate),
			.ByteRate = uint32_t(SamplingRate * BytesPerSamples * Channels),
			.SampleAlign = uint16_t(BytesPerSamples * Channels),
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
		std::sort(MidiEvents.begin(), MidiEvents.end());
		long PreviousTicks = Begin;
		for (const auto& Event : MidiEvents) {
			const long CurrentTicks = long((Event.Time - Begin) * TPS);
			if (CurrentTicks >= 0)
			{
				long DiffTicks = CurrentTicks - PreviousTicks;
				if (DiffTicks < 0)
					DiffTicks = 0;
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