#include "../AvCodec.h"
#include "Libraries/Base.h"
#include "libremidi/writer.hpp"
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
}
#include <fstream>

double DragonianLib::AvCodec::CalculateRMS(const float* Begin, const float* End)
{
	double sum = 0.0;
	const double count = static_cast<double>(End - Begin);
	while (Begin != End) {
		sum += static_cast<double>(*Begin) * static_cast<double>(*Begin);
		++Begin;
	}
	return sqrt(sqrt(sum / count));
}

double DragonianLib::AvCodec::CalculateDB(double RMS)
{
	if (RMS <= 1e-10 || isnan(RMS))
		return -std::numeric_limits<double>::infinity();
	return 20.0 * log10(RMS);
}

double DragonianLib::AvCodec::CalculateDB(const float* Begin, const float* End)
{
	double sum = 0.0;
	const double count = static_cast<double>(End - Begin);
	while (Begin != End) {
		sum += static_cast<double>(*Begin) * static_cast<double>(*Begin);
		++Begin;
	}
	return CalculateDB(sqrt(sum / count));
}

DragonianLibSTL::Vector<unsigned char> DragonianLib::AvCodec::AvCodec::Decode(
	const char* AudioPath,
	int OutSamplingRate,
	int OutChannels,
	bool OutFloat,
	bool OutPlanar
)
{
	char ErrorMessage[1024];

	int ErrorCode = avformat_open_input(&AvFormatContext, AudioPath, nullptr, nullptr);
	if (ErrorCode)
	{
		av_strerror(ErrorCode, ErrorMessage, 1024);
		throw std::exception(ErrorMessage);
	}

	ErrorCode = avformat_find_stream_info(AvFormatContext, nullptr);
	if (ErrorCode < 0)
	{
		av_strerror(ErrorCode, ErrorMessage, 1024);
		throw std::exception(ErrorMessage);
	}

	int streamIndex = -1;
	for (unsigned i = 0; i < AvFormatContext->nb_streams; ++i) {
		const AVMediaType avMediaType = AvFormatContext->streams[i]->codecpar->codec_type;
		if (avMediaType == AVMEDIA_TYPE_AUDIO) {
			streamIndex = static_cast<int>(i);
		}
	}

	if (streamIndex == -1)
		throw std::exception("input file has no audio stream!");

	const AVCodecParameters* avCodecParameters = AvFormatContext->streams[streamIndex]->codecpar;
	const AVCodecID avCodecId = avCodecParameters->codec_id;
	const AVCodec* avCodec = avcodec_find_decoder(avCodecId);
	if (avCodec == nullptr)
		throw std::exception("unable to find a matching decoder!");
	if (AvCodecContext == nullptr)
		throw std::exception("Can't Get Decoder Info");

	ErrorCode = avcodec_parameters_to_context(AvCodecContext, avCodecParameters);
	if (ErrorCode < 0)
	{
		av_strerror(ErrorCode, ErrorMessage, 1024);
		throw std::exception(ErrorMessage);
	}

	ErrorCode = avcodec_open2(AvCodecContext, avCodec, nullptr);
	if (ErrorCode)
	{
		av_strerror(ErrorCode, ErrorMessage, 1024);
		throw std::exception(ErrorMessage);
	}

	const int inSampleRate = AvCodecContext->sample_rate;
	const AVSampleFormat inFormat = AvCodecContext->sample_fmt;
	const int inChannelCount = AvCodecContext->ch_layout.nb_channels;

	const int outSampleRate = OutSamplingRate;
	const AVSampleFormat outFormat = OutPlanar ?
		(OutFloat ? AV_SAMPLE_FMT_FLTP : AV_SAMPLE_FMT_S16P) :
		(OutFloat ? AV_SAMPLE_FMT_FLT : AV_SAMPLE_FMT_S16);
	if (OutChannels > inChannelCount)
		OutChannels = inChannelCount;
	const int outChannelCount = OutChannels;

	const auto nSample = size_t(AvFormatContext->duration * OutSamplingRate / AV_TIME_BASE) * outChannelCount;
	const auto sampleBytes = (OutFloat ? sizeof(float) : sizeof(int16_t));
	const auto nBytes = nSample * sampleBytes;
	DragonianLibSTL::Vector<uint8_t> outData(nBytes);

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
		throw std::exception(ErrorMessage);
	}

	ErrorCode = swr_init(SwrContext);
	if (ErrorCode)
	{
		av_strerror(ErrorCode, ErrorMessage, 1024);
		throw std::exception(ErrorMessage);
	}

	auto OutPtr = outData.Data();
	while (av_read_frame(AvFormatContext, Packet) >= 0)
	{
		if (Packet->stream_index == streamIndex)
		{
			ErrorCode = avcodec_send_packet(AvCodecContext, Packet);
			if (ErrorCode)
			{
				av_packet_unref(Packet);
				av_strerror(ErrorCode, ErrorMessage, 1024);
				throw std::exception(ErrorMessage);
			}

			while (!avcodec_receive_frame(AvCodecContext, InFrame))
			{
				auto dstNbSamples = av_rescale_rnd(
					InFrame->nb_samples,
					outSampleRate,
					inSampleRate,
					AV_ROUND_ZERO
				);

				if (OutPlanar)
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
					throw std::exception(ErrorMessage);
				}

				if (OutPlanar)
				{
					for (int i = 0; i < outChannelCount; ++i)
					{
						memcpy(OutPtr, OutBuffer[0], sampleBytes * dstNbSamples);
						OutPtr += sampleBytes * dstNbSamples;
					}
				}
				else
				{
					memcpy(OutPtr, OutBuffer[0], sampleBytes * dstNbSamples * outChannelCount);
					OutPtr += sampleBytes * dstNbSamples * outChannelCount;
				}

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
	const auto OutPtrEnd = outData.End();
	while (OutPtr < OutPtrEnd)
		*(OutPtr++) = 0;

	return outData;
}

DragonianLibSTL::Vector<float> DragonianLib::AvCodec::AvCodec::DecodeFloat(
	const char* AudioPath,
	int OutSamplingRate,
	int OutChannels,
	bool OutPlanar
)
{
	auto Ret = Decode(
		AudioPath,
		OutSamplingRate,
		OutChannels,
		true,
		OutPlanar
	);
	auto Alloc = Ret.GetAllocator();
	auto Data = Ret.Release();
	auto Ptr = (float*)Data.first;
	auto Size = Data.second / sizeof(float);
	return { &Ptr, Size, Alloc };
}

DragonianLibSTL::Vector<int16_t> DragonianLib::AvCodec::AvCodec::DecodeSigned16(
	const char* AudioPath,
	int OutSamplingRate,
	int OutChannels,
	bool OutPlanar
)
{
	auto Ret = Decode(
		AudioPath,
		OutSamplingRate,
		OutChannels,
		false,
		OutPlanar
	);
	auto Alloc = Ret.GetAllocator();
	auto Data = Ret.Release();
	auto Ptr = (int16_t*)Data.first;
	auto Size = Data.second / sizeof(int16_t);
	return { &Ptr, Size, Alloc };
}

void DragonianLib::AvCodec::AvCodec::Encode(
	const char* OutPutPath,
	const DragonianLibSTL::Vector<unsigned char>& PcmData,
	int SrcSamplingRate,
	int OutSamplingRate,
	int SrcChannels,
	int OutChannels,
	bool IsFloat,
	bool IsPlanar
)
{
	_D_Dragonian_Lib_Not_Implemented_Error;
	OutBuffer[0] = nullptr;
	UNUSED(OutBuffer);
}

void DragonianLib::AvCodec::AvCodec::Release()
{
	if (Packet)
		av_packet_free(&Packet);
	if (InFrame)
		av_frame_free(&InFrame);
	if (SwrContext)
	{
		swr_close(SwrContext);
		swr_free(&SwrContext);
	}
	if (AvCodecContext)
		avcodec_free_context(&AvCodecContext);
	if (AvFormatContext)
		avformat_close_input(&AvFormatContext);
	InFrame = nullptr;
	for (auto out_buffer : OutBuffer)
		if (out_buffer)
			av_free(out_buffer);
	SwrContext = nullptr;
	AvCodecContext = nullptr;
	AvFormatContext = nullptr;
	Packet = nullptr;
}

void DragonianLib::AvCodec::AvCodec::Init()
{
	InFrame = av_frame_alloc();
	SwrContext = swr_alloc();
	AvCodecContext = avcodec_alloc_context3(nullptr);
	AvFormatContext = avformat_alloc_context();
	Packet = av_packet_alloc();

	if (!AvFormatContext || !Packet || !InFrame)
	{
		Release();
		throw std::bad_alloc();
	}
}

DragonianLib::AvCodec::AvCodec::AvCodec()
{
	Init();
}

DragonianLib::AvCodec::AvCodec::~AvCodec()
{
	Release();
}

void DragonianLib::AvCodec::WritePCMData(
	const wchar_t* OutPutPath,
	const DragonianLibSTL::Vector<unsigned char>& PcmData,
	int SamplingRate,
	int Channels,
	bool IsFloat,
	bool IsPlanar
)
{
	UNUSED(IsPlanar);
	const uint32_t sampleBytes = uint32_t(IsFloat ? sizeof(float) : sizeof(int16_t));
	const RiffWaveHeader Header{
		.RiffChunkSize = uint32_t(36 + PcmData.Size()),
		.WaveChunkSize = 16,
		.AudioFormat = uint16_t(IsFloat ? 0x0003 : 0x0001),
		.ChannelCount = uint16_t(Channels),
		.SamplingRate = uint32_t(SamplingRate),
		.ByteRate = uint32_t(SamplingRate * sampleBytes * Channels),
		.SampleAlign = uint16_t(sampleBytes * Channels),
		.SampleBits = uint16_t(sampleBytes * 8),
		.DataChunkSize = (uint32_t)PcmData.Size()
	};
	FileGuard File;
	File.Open(OutPutPath, L"wb");
	if (!File.Enabled())
		throw std::exception("could not open file!");
	fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
	fwrite(PcmData.Data(), 1, PcmData.Size(), File);
}

void DragonianLib::AvCodec::WritePCMData(
	const wchar_t* OutPutPath,
	const DragonianLibSTL::Vector<int16_t>& PcmData,
	int SamplingRate,
	int Channels,
	bool IsPlanar
)
{
	UNUSED(IsPlanar);
	constexpr uint32_t sampleBytes = uint32_t(sizeof(int16_t));
	const RiffWaveHeader Header{
		.RiffChunkSize = uint32_t(36 + PcmData.Size() * sizeof(int16_t)),
		.WaveChunkSize = 16,
		.AudioFormat = uint16_t(0x0001),
		.ChannelCount = uint16_t(Channels),
		.SamplingRate = uint32_t(SamplingRate),
		.ByteRate = uint32_t(SamplingRate * sampleBytes * Channels),
		.SampleAlign = uint16_t(sampleBytes * Channels),
		.SampleBits = uint16_t(sampleBytes * 8),
		.DataChunkSize = (uint32_t)(PcmData.Size() * sizeof(int16_t))
	};
	FileGuard File;
	File.Open(OutPutPath, L"wb");
	if (!File.Enabled())
		throw std::exception("could not open file!");
	fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
	fwrite(PcmData.Data(), 1, PcmData.Size() * sizeof(int16_t), File);
}

void DragonianLib::AvCodec::WritePCMData(
	const wchar_t* OutPutPath,
	const DragonianLibSTL::Vector<float>& PcmData,
	int SamplingRate,
	int Channels,
	bool IsPlanar
)
{
	UNUSED(IsPlanar);
	constexpr uint32_t sampleBytes = uint32_t(sizeof(float));
	const RiffWaveHeader Header{
		.RiffChunkSize = uint32_t(36 + PcmData.Size() * sizeof(float)),
		.WaveChunkSize = 16,
		.AudioFormat = uint16_t(0x0003),
		.ChannelCount = uint16_t(Channels),
		.SamplingRate = uint32_t(SamplingRate),
		.ByteRate = uint32_t(SamplingRate * sampleBytes * Channels),
		.SampleAlign = uint16_t(sampleBytes * Channels),
		.SampleBits = uint16_t(sampleBytes * 8),
		.DataChunkSize = (uint32_t)(PcmData.Size() * sizeof(float))
	};
	FileGuard File;
	File.Open(OutPutPath, L"wb");
	if (!File.Enabled())
		throw std::exception("could not open file!");
	fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
	fwrite(PcmData.Data(), 1, PcmData.Size() * sizeof(float), File);
}

void DragonianLib::AvCodec::WritePCMData(
	const wchar_t* OutPutPath,
	const unsigned char* PcmData,
	size_t BufferSize,
	int SamplingRate,
	int Channels,
	bool IsFloat,
	bool IsPlanar
)
{
	UNUSED(IsPlanar);
	const uint32_t sampleBytes = uint32_t(IsFloat ? sizeof(float) : sizeof(int16_t));
	const RiffWaveHeader Header{
		.RiffChunkSize = uint32_t(36 + BufferSize),
		.WaveChunkSize = 16,
		.AudioFormat = uint16_t(IsFloat ? 0x0003 : 0x0001),
		.ChannelCount = uint16_t(Channels),
		.SamplingRate = uint32_t(SamplingRate),
		.ByteRate = uint32_t(SamplingRate * sampleBytes * Channels),
		.SampleAlign = uint16_t(sampleBytes * Channels),
		.SampleBits = uint16_t(sampleBytes * 8),
		.DataChunkSize = (uint32_t)BufferSize
	};
	FileGuard File;
	File.Open(OutPutPath, L"wb");
	if (!File.Enabled())
		throw std::exception("could not open file!");
	fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
	fwrite(PcmData, 1, BufferSize, File);
}

void DragonianLib::AvCodec::WriteMidiFile(
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

DragonianLibSTL::Vector<size_t> DragonianLib::AvCodec::SliceAudio(
	const DragonianLibSTL::Vector<float>& PcmData,
	const SlicerSettings& SlicerSettings
)
{
	if (PcmData.Size() < size_t(SlicerSettings.MinLength) * SlicerSettings.SamplingRate)
		return { 0, PcmData.Size() };

	DragonianLibSTL::Vector<unsigned long long> SlicePos;
	bool VocalPart = CalculateDB(PcmData.Begin(), PcmData.Begin() + SlicerSettings.WindowLength) > SlicerSettings.Threshold;
	SlicePos.EmplaceBack(0);
	auto TotalCount = static_cast<ptrdiff_t>(PcmData.Size() - SlicerSettings.WindowLength);
	for (ptrdiff_t Pos = 0; Pos < TotalCount; Pos += SlicerSettings.HopSize)
	{
		const auto DB = CalculateDB(
			PcmData.Begin() + Pos,
			PcmData.Begin() + Pos + SlicerSettings.WindowLength
		);
		if (DB > SlicerSettings.Threshold)
		{
			if (!VocalPart)
			{
				SlicePos.EmplaceBack(Pos);
				VocalPart = true;
			}
		}
		else
		{
			if (VocalPart)
			{
				SlicePos.EmplaceBack(Pos);
				VocalPart = false;
			}
		}
	}

	SlicePos.EmplaceBack(PcmData.Size());
	return SlicePos;
}