#include "AvCodec.h"
#include "Base.h"
//#include "Util/StringPreprocess.h"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
}

DragonianLibSTL::Vector<unsigned char> DragonianLib::AvCodec::Decode(
    const char* _AudioPath,
    int _OutSamplingRate,
    int _OutChannels,
    bool _OutFloat,
    bool _OutPlanar
)
{
    char ErrorMessage[1024];

    int ErrorCode = avformat_open_input(&avFormatContext, _AudioPath, nullptr, nullptr);
    if (ErrorCode) 
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    ErrorCode = avformat_find_stream_info(avFormatContext, nullptr);
    if (ErrorCode < 0)
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    int streamIndex = -1;
    for (unsigned i = 0; i < avFormatContext->nb_streams; ++i) {
        const AVMediaType avMediaType = avFormatContext->streams[i]->codecpar->codec_type;
        if (avMediaType == AVMEDIA_TYPE_AUDIO) {
            streamIndex = static_cast<int>(i);
        }
    }

    if (streamIndex == -1)
        throw std::exception("input file has no audio stream!");

    const AVCodecParameters* avCodecParameters = avFormatContext->streams[streamIndex]->codecpar;
    const AVCodecID avCodecId = avCodecParameters->codec_id;
    const AVCodec* avCodec = avcodec_find_decoder(avCodecId);
    if (avCodec == nullptr)
        throw std::exception("unable to find a matching decoder!");
    if (avCodecContext == nullptr) 
        throw std::exception("Can't Get Decoder Info");
    
    ErrorCode = avcodec_parameters_to_context(avCodecContext, avCodecParameters);
    if(ErrorCode < 0)
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    ErrorCode = avcodec_open2(avCodecContext, avCodec, nullptr);
    if (ErrorCode)
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    const int inSampleRate = avCodecContext->sample_rate;
    const AVSampleFormat inFormat = avCodecContext->sample_fmt;
    const int inChannelCount = avCodecContext->ch_layout.nb_channels;

    const int outSampleRate = _OutSamplingRate;
    const AVSampleFormat outFormat = _OutPlanar ?
        (_OutFloat ? AV_SAMPLE_FMT_FLTP : AV_SAMPLE_FMT_S16P) :
        (_OutFloat ? AV_SAMPLE_FMT_FLT : AV_SAMPLE_FMT_S16);
    if (_OutChannels > inChannelCount)
        _OutChannels = inChannelCount;
    const int outChannelCount = _OutChannels;

    const auto nSample = size_t(avFormatContext->duration * _OutSamplingRate / AV_TIME_BASE) * outChannelCount;
    const auto sampleBytes = (_OutFloat ? sizeof(float) : sizeof(int16_t));
    const auto nBytes = nSample * sampleBytes;
    DragonianLibSTL::Vector<uint8_t> outData(nBytes);
    
	AVChannelLayout inChannelLayout, outChannelLayout;

    av_channel_layout_default(&inChannelLayout, inChannelCount);
    av_channel_layout_default(&outChannelLayout, outChannelCount);

    ErrorCode = swr_alloc_set_opts2(
        &swrContext,
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

    ErrorCode = swr_init(swrContext);
    if (ErrorCode)
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    auto OutPtr = outData.Data();
    while (av_read_frame(avFormatContext, packet) >= 0)
    {
        if (packet->stream_index == streamIndex)
        {
            ErrorCode = avcodec_send_packet(avCodecContext, packet);
            if (ErrorCode)
            {
                av_packet_unref(packet);
                av_strerror(ErrorCode, ErrorMessage, 1024);
                throw std::exception(ErrorMessage);
            }

            while (!avcodec_receive_frame(avCodecContext, inFrame))
            {
                auto dstNbSamples = av_rescale_rnd(
                    inFrame->nb_samples,
                    outSampleRate,
                    inSampleRate,
                    AV_ROUND_ZERO
                );

                if(_OutPlanar)
		            for (int i = 0; i < outChannelCount; ++i)
		            	outBuffer[i] = (uint8_t*)av_malloc(dstNbSamples * sizeof(double));
                else
                    outBuffer[0] = (uint8_t*)av_malloc(dstNbSamples * sizeof(double) * outChannelCount);

                ErrorCode = swr_convert(
                    swrContext,
                    outBuffer,
                    int(dstNbSamples),
                    inFrame->data,
                    inFrame->nb_samples
                );
                if (ErrorCode < 0)
                {
                    av_frame_unref(inFrame);
                    av_strerror(ErrorCode, ErrorMessage, 1024);
                    throw std::exception(ErrorMessage);
                }

                if (_OutPlanar)
                {
                    for (int i = 0; i < outChannelCount; ++i)
                    {
                        memcpy(OutPtr, outBuffer[0], sampleBytes * dstNbSamples);
                        OutPtr += sampleBytes * dstNbSamples;
                    }
                }
                else
                {
                    memcpy(OutPtr, outBuffer[0], sampleBytes * dstNbSamples * outChannelCount);
                    OutPtr += sampleBytes * dstNbSamples * outChannelCount;
                }

                for (int i = 0; i < outChannelCount; ++i)
                    if (outBuffer[i])
                    {
                        av_free(outBuffer[i]);
                        outBuffer[i] = nullptr;
                    }
                av_frame_unref(inFrame);
            }
        }
        av_packet_unref(packet);
    }
    const auto OutPtrEnd = outData.End();
    while (OutPtr < OutPtrEnd)
        *(OutPtr++) = 0;

    return outData;
}

DragonianLibSTL::Vector<float> DragonianLib::AvCodec::DecodeFloat(const char* _AudioPath, int _OutSamplingRate, int _OutChannels, bool _OutPlanar)
{
    auto Ret = Decode(
        _AudioPath,
        _OutSamplingRate,
        _OutChannels,
        true,
        _OutPlanar
    );
    auto Alloc = Ret.GetAllocator();
    auto Data = Ret.Release();
    auto Ptr = (float*)Data.first;
    auto Size = Data.second / sizeof(float);
    return { &Ptr, Size, Alloc };
}

DragonianLibSTL::Vector<int16_t> DragonianLib::AvCodec::DecodeSigned16(const char* _AudioPath, int _OutSamplingRate, int _OutChannels, bool _OutPlanar)
{
    auto Ret = Decode(
        _AudioPath,
        _OutSamplingRate,
        _OutChannels,
        false,
        _OutPlanar
    );
    auto Alloc = Ret.GetAllocator();
    auto Data = Ret.Release();
    auto Ptr = (int16_t*)Data.first;
    auto Size = Data.second / sizeof(int16_t);
    return { &Ptr, Size, Alloc };
}

void DragonianLib::AvCodec::Encode(
    const char* _OutPutPath,
    const DragonianLibSTL::Vector<unsigned char>& _PcmData,
    int _SrcSamplingRate,
    int _OutSamplingRate,
    int _SrcChannels,
    int _OutChannels,
    bool _IsFloat,
    bool _IsPlanar
)
{
    outBuffer[0] = nullptr;
    UNUSED(outBuffer);
}


void DragonianLib::AvCodec::Release()
{
    if (packet)
        av_packet_free(&packet);
    if (inFrame)
        av_frame_free(&inFrame);
    if (swrContext)
    {
        swr_close(swrContext);
	    swr_free(&swrContext);
    }
    if (avCodecContext)
    	avcodec_free_context(&avCodecContext);
    if (avFormatContext)
        avformat_close_input(&avFormatContext);
    inFrame = nullptr;
    for (auto out_buffer : outBuffer)
        if (out_buffer)
            av_free(out_buffer);
    swrContext = nullptr;
    avCodecContext = nullptr;
    avFormatContext = nullptr;
    packet = nullptr;
}

void DragonianLib::AvCodec::Init()
{
    inFrame = av_frame_alloc();
    swrContext = swr_alloc();
    avCodecContext = avcodec_alloc_context3(nullptr);
    avFormatContext = avformat_alloc_context();
    packet = av_packet_alloc();

    if (!avFormatContext || !packet || !inFrame)
    {
        Release();
    	throw std::bad_alloc();
    }
}

DragonianLib::AvCodec::AvCodec()
{
    Init();
}

DragonianLib::AvCodec::~AvCodec()
{
    Release();
}

void DragonianLib::WritePCMData(
    const wchar_t* _OutPutPath,
    const DragonianLibSTL::Vector<unsigned char>& PCMDATA,
    int _SamplingRate,
    int _Channels,
    bool _IsFloat,
    bool _IsPlanar
)
{
    UNUSED(_IsPlanar);
    const uint32_t sampleBytes = uint32_t(_IsFloat ? sizeof(float) : sizeof(int16_t));
    const RiffWaveHeader Header{
        .RiffChunkSize = uint32_t(36 + PCMDATA.Size()),
        .WaveChunkSize = 16,
        .AudioFormat = uint16_t(_IsFloat ? 0x0003 : 0x0001),
        .ChannelCount = uint16_t(_Channels),
        .SamplingRate = uint32_t(_SamplingRate),
        .ByteRate = uint32_t(_SamplingRate * sampleBytes * _Channels),
        .SampleAlign = uint16_t(sampleBytes * _Channels),
        .SampleBits = uint16_t(sampleBytes * 8),
        .DataChunkSize = (uint32_t)PCMDATA.Size()
    };
    FileGuard File;
    File.Open(_OutPutPath, L"wb");
    if (!File.Enabled())
        throw std::exception("could not open file!");
    fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
    fwrite(PCMDATA.Data(), 1, PCMDATA.Size(), File);
}

void DragonianLib::WritePCMData(
    const wchar_t* _OutPutPath,
    const DragonianLibSTL::Vector<int16_t>& PCMDATA,
    int _SamplingRate,
    int _Channels,
    bool _IsPlanar
)
{
    UNUSED(_IsPlanar);
    constexpr uint32_t sampleBytes = uint32_t(sizeof(int16_t));
    const RiffWaveHeader Header{
        .RiffChunkSize = uint32_t(36 + PCMDATA.Size() * sizeof(int16_t)),
        .WaveChunkSize = 16,
        .AudioFormat = uint16_t(0x0001),
        .ChannelCount = uint16_t(_Channels),
        .SamplingRate = uint32_t(_SamplingRate),
        .ByteRate = uint32_t(_SamplingRate * sampleBytes * _Channels),
        .SampleAlign = uint16_t(sampleBytes * _Channels),
        .SampleBits = uint16_t(sampleBytes * 8),
        .DataChunkSize = (uint32_t)(PCMDATA.Size() * sizeof(int16_t))
    };
    FileGuard File;
    File.Open(_OutPutPath, L"wb");
    if (!File.Enabled())
        throw std::exception("could not open file!");
    fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
    fwrite(PCMDATA.Data(), 1, PCMDATA.Size() * sizeof(int16_t), File);
}

void DragonianLib::WritePCMData(
    const wchar_t* _OutPutPath,
    const DragonianLibSTL::Vector<float>& PCMDATA,
    int _SamplingRate,
    int _Channels,
    bool _IsPlanar
)
{
    UNUSED(_IsPlanar);
    constexpr uint32_t sampleBytes = uint32_t(sizeof(float));
    const RiffWaveHeader Header{
        .RiffChunkSize = uint32_t(36 + PCMDATA.Size() * sizeof(float)),
        .WaveChunkSize = 16,
        .AudioFormat = uint16_t(0x0003),
        .ChannelCount = uint16_t(_Channels),
        .SamplingRate = uint32_t(_SamplingRate),
        .ByteRate = uint32_t(_SamplingRate * sampleBytes * _Channels),
        .SampleAlign = uint16_t(sampleBytes * _Channels),
        .SampleBits = uint16_t(sampleBytes * 8),
        .DataChunkSize = (uint32_t)(PCMDATA.Size() * sizeof(float))
    };
    FileGuard File;
    File.Open(_OutPutPath, L"wb");
    if (!File.Enabled())
        throw std::exception("could not open file!");
    fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
    fwrite(PCMDATA.Data(), 1, PCMDATA.Size() * sizeof(float), File);
}

void DragonianLib::WritePCMData(
    const wchar_t* _OutPutPath,
    const unsigned char* PCMDATA,
    size_t _BufferSize,
    int _SamplingRate,
    int _Channels,
    bool _IsFloat,
    bool _IsPlanar
)
{
    UNUSED(_IsPlanar);
    const uint32_t sampleBytes = uint32_t(_IsFloat ? sizeof(float) : sizeof(int16_t));
    const RiffWaveHeader Header{
        .RiffChunkSize = uint32_t(36 + _BufferSize),
        .WaveChunkSize = 16,
        .AudioFormat = uint16_t(_IsFloat ? 0x0003 : 0x0001),
        .ChannelCount = uint16_t(_Channels),
        .SamplingRate = uint32_t(_SamplingRate),
        .ByteRate = uint32_t(_SamplingRate * sampleBytes * _Channels),
        .SampleAlign = uint16_t(sampleBytes * _Channels),
        .SampleBits = uint16_t(sampleBytes * 8),
        .DataChunkSize = (uint32_t)_BufferSize
    };
    FileGuard File;
    File.Open(_OutPutPath, L"wb");
    if (!File.Enabled())
        throw std::exception("could not open file!");
    fwrite(&Header, 1, sizeof(RiffWaveHeader), File);
    fwrite(PCMDATA, 1, _BufferSize, File);
}
