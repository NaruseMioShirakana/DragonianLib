#include "AvCodec.h"

libsvcstd::Vector<double> libsvc::AvCodec::Arange(double start, double end, double step, double div)
{
    libsvcstd::Vector<double> output(size_t((end - start) / step));
    auto outputptr = output.Begin();
    const auto outputptrend = output.End();
    while (outputptr != outputptrend)
    {
        *(outputptr++) = start / div;
        start += step;
    }
    return output;
}

libsvcstd::Vector<int16_t> libsvc::AvCodec::Decode(const char* path, int samplingRate)
{
    libsvcstd::Vector<uint8_t> outData;
    char ErrorMessage[1024];

    int ErrorCode = avformat_open_input(&avFormatContext, path, nullptr, nullptr);
    if (ErrorCode) 
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }

    ErrorCode = avformat_find_stream_info(avFormatContext, nullptr);
    if (ErrorCode)
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
    
    avcodec_parameters_to_context(avCodecContext, avCodecParameters);
    ret = avcodec_open2(avCodecContext, avCodec, nullptr);
    if (ErrorCode)
    {
        av_strerror(ErrorCode, ErrorMessage, 1024);
        throw std::exception(ErrorMessage);
    }
    const AVSampleFormat inFormat = avCodecContext->sample_fmt;
    constexpr AVSampleFormat  outFormat = AV_SAMPLE_FMT_S16;
    const int inSampleRate = avCodecContext->sample_rate;
    const int outSampleRate = sr;

    const auto nSample = static_cast<size_t>(avFormatContext->duration * sr / AV_TIME_BASE);

    uint64_t in_ch_layout = avCodecContext->channel_layout;
    if (path.substr(path.rfind(L'.')) == L".wav")
    {
        const auto head = GetHeader(path);
        if (head.NumOfChan == 1)
            in_ch_layout = AV_CH_LAYOUT_MONO;
        else if (head.NumOfChan == 2)
            in_ch_layout = AV_CH_LAYOUT_STEREO;
        else
            throw std::exception("unsupported Channel Num");
    }
    constexpr uint64_t out_ch_layout = AV_CH_LAYOUT_MONO;
    swr_alloc_set_opts(swrContext, out_ch_layout, outFormat, outSampleRate,
        static_cast<int64_t>(in_ch_layout), inFormat, inSampleRate, 0, nullptr
    );
    swr_init(swrContext);
    const int outChannelCount = av_get_channel_layout_nb_channels(out_ch_layout);
    int currentIndex = 0;
    out_buffer = (uint8_t*)av_malloc(2ull * sr);
    while (av_read_frame(avFormatContext, packet) >= 0) {
        if (packet->stream_index == streamIndex) {
            avcodec_send_packet(avCodecContext, packet);
            ret = avcodec_receive_frame(avCodecContext, inFrame);
            if (ret == 0) {
                swr_convert(swrContext, &out_buffer, 2ull * sr,
                    (const uint8_t**)inFrame->data, inFrame->nb_samples);
                const int out_buffer_size = av_samples_get_buffer_size(nullptr, outChannelCount, (inFrame->nb_samples * sr / inSampleRate) - 1, outFormat, 1);
                outData.insert(outData.end(), out_buffer, out_buffer + out_buffer_size);
            }
            ++currentIndex;
            av_packet_unref(packet);
        }
    }
    //Wav outWav(static_cast<unsigned long>(sr), static_cast<unsigned long>(outData.size()), outData.data());
    auto outWav = reinterpret_cast<int16_t*>(outData.data());
    const auto RawWavLen = int64_t(outData.size()) / 2;
    if (nSample != static_cast<size_t>(RawWavLen))
    {
        const double interpOff = static_cast<double>(RawWavLen) / static_cast<double>(nSample);
        const auto x0 = arange(0.0, static_cast<double>(RawWavLen), 1.0, 1.0);
        std::vector<double> y0(RawWavLen);
        for (int64_t i = 0; i < RawWavLen; ++i)
            y0[i] = outWav[i] ? static_cast<double>(outWav[i]) : NAN;
        const auto yi = new double[nSample];
        auto xi = arange(0.0, static_cast<double>(RawWavLen), interpOff, 1.0);
        while (xi.size() < nSample)
            xi.push_back(*(xi.end() - 1) + interpOff);
        while (xi.size() > nSample)
            xi.pop_back();
        interp1(x0.data(), y0.data(), static_cast<int>(RawWavLen), xi.data(), static_cast<int>(nSample), yi);
        std::vector<short> DataChun(nSample);
        for (size_t i = 0; i < nSample; ++i)
            DataChun[i] = isnan(yi[i]) ? 0i16 : static_cast<short>(yi[i]);
        delete[] yi;
        return DataChun;
    }
    return { outWav , outWav + RawWavLen };
}

void libsvc::AvCodec::Release()
{
    if (packet)
        av_packet_free(&packet);
    if (inFrame)
        av_frame_free(&inFrame);
    if (outBuffer)
        av_free(outBuffer);
    if (swrContext)
        swr_free(&swrContext);
    if (avCodecContext)
        avcodec_free_context(&avCodecContext);
    if (avFormatContext)
        avformat_close_input(&avFormatContext);
    inFrame = nullptr;
    outBuffer = nullptr;
    swrContext = nullptr;
    avCodecContext = nullptr;
    avFormatContext = nullptr;
    packet = nullptr;
}

void libsvc::AvCodec::Init()
{
    inFrame = av_frame_alloc();
    swrContext = swr_alloc();
    avCodecContext = avcodec_alloc_context3(nullptr);
    avFormatContext = avformat_alloc_context();
    packet = av_packet_alloc();
    outBuffer = nullptr;

    if (!avFormatContext || !packet || !inFrame)
    {
        Release();
    	throw std::bad_alloc();
    }
}

libsvc::AvCodec::AvCodec()
{
    Init();
}

libsvc::AvCodec::~AvCodec()
{
    Release();
}