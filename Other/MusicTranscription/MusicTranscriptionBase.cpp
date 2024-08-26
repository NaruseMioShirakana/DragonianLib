#include "MusicTranscriptionBase.hpp"
#include "Base.h"
#include "libremidi/writer.hpp"
#include <fstream>

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		void WriteMidiFile(const std::wstring& _Path, const MidiTrack& _Events, long _Begin, long _TPS)
		{
			libremidi::writer _Writer;
			_Writer.add_track();
			std::vector<MidiEvent> _events;
			for (const auto& it : _Events.NoteEvents)
			{
				_events.emplace_back(it.OnsetTime, it.MidiNote, it.Velocity);
				_events.emplace_back(it.OffsetTime, it.MidiNote, 0);
			}
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