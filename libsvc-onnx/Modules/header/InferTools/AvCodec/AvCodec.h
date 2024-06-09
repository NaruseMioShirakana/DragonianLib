#pragma once
#include <vector>
#include <string>
#include "InferTools/inferTools.hpp"
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
}

class AvCodec
{
public:
	static std::vector<double> arange(double start, double end, double step = 1.0, double div = 1.0);
	std::vector<short> codec(const std::wstring& path, int sr);
	void release();
	void init();
	AvCodec();
	~AvCodec();
private:
    AVFrame* inFrame;
    uint8_t* out_buffer;
    SwrContext* swrContext;
    AVCodecContext* avCodecContext;
    AVFormatContext* avFormatContext;
    AVPacket* packet;
};
