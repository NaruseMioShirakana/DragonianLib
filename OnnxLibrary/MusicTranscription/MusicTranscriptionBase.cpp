#include "MusicTranscriptionBase.hpp"
#include "Base.h"
#include <fstream>

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
	    void MeanFliter(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _Signal, size_t WindowLength)
		{
			using DragonianLibSTL::Vector;
			const auto FrameCount = _Signal.Size();
			const auto ClassSize = _Signal[0].Size();
			Vector Result(FrameCount, Vector<float>(ClassSize));
			if (WindowLength > FrameCount || WindowLength < 2)
				return;

			auto WndSz = (float)(WindowLength % 2 ? WindowLength : WindowLength + 1);

			const size_t half = WindowLength / 2;

			for(size_t pitch = 0; pitch < ClassSize; ++pitch)
			{
				for (size_t i = 0; i < half; ++i)
					Result[i][pitch] = _Signal[i][pitch];

				for (size_t i = half; i < FrameCount - half; i++) {
					float sum = 0.0f;
					for (size_t j = i - half; j <= i + half; j++)
						sum += _Signal[j][pitch];
					Result[i][pitch] = (sum / WndSz);
				}

				for (size_t i = FrameCount - half; i < FrameCount; ++i)
					Result[i][pitch] = _Signal[i][pitch];
			}

			_Signal = std::move(Result);
			//return Result;
		}

		bool operator<(const EstNoteTp& a, const EstNoteTp& b)
		{
			return a.Begin < b.Begin;
		}

		EstNoteTp operator+(const EstNoteTp& a, const EstNoteTp& b)
		{
			return { a.Begin + b.Begin,a.End + b.End,a.OnsetShift + b.OnsetShift,a.OffsetShift + b.OffsetShift,a.NormalizedVelocity + b.NormalizedVelocity };
		}

		MusicTranscription::MusicTranscription(ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : Env_(_ThreadCount, _DeviceID, _Provider), _callback(std::move(_Callback))
		{
			
		}

		MidiTrack MusicTranscription::Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize) const
		{
			_D_Dragonian_Lib_Not_Implemented_Error;
		}

	}
}
