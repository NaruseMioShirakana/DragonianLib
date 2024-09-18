#include "MusicTranscriptionBase.hpp"
#include "Base.h"
#include "libremidi/writer.hpp"
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

			const size_t half = WindowLength / 2; // 窗口半径，向下取整

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

		void WriteMidiFile(const std::wstring& _Path, const MidiTrack& _Events, long _Begin, long _TPS, long tempo)
		{
			libremidi::writer _Writer;
			_Writer.add_track();
			std::vector<MidiEvent> _events;
			for (const auto& it : _Events.NoteEvents)
			{
				_events.emplace_back(it.OnsetTime, it.MidiNote, it.Velocity);
				_events.emplace_back(it.OffsetTime, it.MidiNote, 0);
			}
			_Writer.add_event(0, 0, libremidi::meta_events::tempo(tempo));
			std::sort(_events.begin(), _events.end());
			long previous_ticks = _Begin;
			for (const auto& it : _events) {
				const long this_ticks = long((it.Time - _Begin) * _TPS);
				if (this_ticks >= 0)
				{
					long diff_ticks = this_ticks - previous_ticks;
					if (diff_ticks < 0)
						diff_ticks = 0;
					previous_ticks = this_ticks;
					if (it.Velocity)
						_Writer.add_event(diff_ticks, 0, libremidi::channel_events::note_on(0, uint8_t(unsigned long(it.MidiNote)), uint8_t(unsigned long(it.Velocity))));
					else
						_Writer.add_event(diff_ticks, 0, libremidi::channel_events::note_off(0, uint8_t(unsigned long(it.MidiNote)), uint8_t(unsigned long(it.Velocity))));
				}
			}
			_Writer.add_event(0, 0, libremidi::meta_events::end_of_track());
			auto ofs = std::ofstream(_Path, std::ios::out | std::ios::binary);
			if (!ofs.is_open())
				DragonianLibThrow("Could not write file!");
			_Writer.write(ofs);
		}

		bool operator<(const EstNoteTp& a, const EstNoteTp& b)
		{
			return a.Begin < b.Begin;
		}

		bool operator<(const MidiEvent& a, const MidiEvent& b)
		{
			return a.Time < b.Time;
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
			DragonianLibNotImplementedError;
		}

	}
}