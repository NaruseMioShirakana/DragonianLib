#pragma once
#include "Vector.h"
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
}

namespace libsvc
{
	class AvCodec
	{
	public:
		static libsvcstd::Vector<double> Arange(double start, double end, double step = 1.0, double div = 1.0);
		libsvcstd::Vector<int16_t> Decode(const char* path, int samplingRate);
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
		uint8_t* outBuffer = nullptr;
	};

}

