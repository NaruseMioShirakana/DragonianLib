#pragma once
#include "Vector.h"
namespace DlCodecStft
{
    class STFT
    {
    public:
        STFT() = default;
        ~STFT();
        STFT(int WindowSize, int HopSize, int FFTSize = 0);
        inline static double PI = 3.14159265358979323846;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
    private:
    	int WINDOW_SIZE = 2048;
    	int HOP_SIZE = WINDOW_SIZE / 4;
    	int FFT_SIZE = WINDOW_SIZE / 2 + 1;
    };

    class Mel
    {
    public:
        Mel() = delete;
        ~Mel() = default;
        Mel(int WindowSize, int HopSize, int SamplingRate, int MelSize = 0, double FreqMin = 20., double FreqMax = 11025.);
        std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<int16_t>& audioData) const;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> GetMel(const DragonianLibSTL::Vector<double>& audioData) const;
        std::pair<DragonianLibSTL::Vector<float>, int64_t> operator()(const DragonianLibSTL::Vector<double>& audioData) const;
    private:
        STFT stft;
        int MEL_SIZE = 128;
        int FFT_SIZE = 0;
        int sr = 22050;
        DragonianLibSTL::Vector<float> MelBasis;
    };
}
