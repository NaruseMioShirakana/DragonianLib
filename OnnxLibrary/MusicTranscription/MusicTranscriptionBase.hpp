/**
 * FileName: MusicTranscriptionBase.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "MyTemplateLibrary/Vector.h"
#include "EnvManager.hpp"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		using ProgressCallback = std::function<void(size_t, size_t)>;
		using OrtTensors = std::vector<Ort::Value>;

		void MeanFliter(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _Signal, size_t WindowLength);

		struct MidiEvent
		{
			double Time = 0;
			long MidiNote = 0;
			long Velocity = 0;
			MidiEvent(double t = 0.0, long m = 0, long v = 0) :Time(t), MidiNote(m), Velocity(v) {}
		};

		struct EstPedalEvents
		{
			double OnsetTime = 0.0;
			double OffsetTime = 0.0;
		};

		struct EstNoteEvents
		{
			double OnsetTime = 0.0;
			double OffsetTime = 0.0;
			long MidiNote = 0;
			long Velocity = 0;
			EstNoteEvents(double a, double b, long c, long d) :OnsetTime(a), OffsetTime(b), MidiNote(c), Velocity(d) {}
		};

		struct MidiTrack
		{
			DragonianLibSTL::Vector<EstNoteEvents> NoteEvents;
			DragonianLibSTL::Vector<EstPedalEvents> PedalEvents;
			MidiTrack() = default;
			MidiTrack(DragonianLibSTL::Vector<EstNoteEvents>&& ene, DragonianLibSTL::Vector<EstPedalEvents>&& epe) : NoteEvents(std::move(ene)), PedalEvents(std::move(epe)) {}
		};

		struct Hparams
		{
			std::wstring ModelPath{}, OnSetPath{}, FramePath{};
			long SamplingRate = 16000;
			long ClassesCount = 88;
			long LowestPitch = 21;
			float SegmentTime = 10.0f;
			float HopTime = 1.0f;
			float FrameTime = 100.0;
			long VelocityScale = 128;
			double OnsetThreshold = 0.3;
			double OffsetThreshold = 0.3;
			double FrameThreshold = 0.1;
			double PedalOffsetThreshold = 0.2;
			long OnsetAligSize = 2;
			long OffsetAligSize = 4;
			double MinFrameSize = 4.f;
			long FliterSize = 6;
			long FliterCount = 3;
			bool UseByteDanceMethod = false;
		};

		struct EstNoteTp
		{
			long Begin = 0;
			long End = 0;
			double OnsetShift = 0.0;
			double OffsetShift = 0.0;
			double NormalizedVelocity = 0.0;
			EstNoteTp(long a, long b, double c, double d, double e) :Begin(a), End(b), OnsetShift(c), OffsetShift(d), NormalizedVelocity(e) {}
		};

		void WriteMidiFile(const std::wstring& _Path, const MidiTrack& _Events, long _Begin, long _TPS, long tempo = 500000);
		// operators used to sort
		bool operator<(const EstNoteTp& a, const EstNoteTp& b);
		bool operator<(const MidiEvent& a, const MidiEvent& b);
		EstNoteTp operator+(const EstNoteTp& a, const EstNoteTp& b);

		class MusicTranscription
		{
		public:
			MusicTranscription(ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
			virtual ~MusicTranscription() = default;
			MusicTranscription(MusicTranscription&& _Right) = delete;
			MusicTranscription(const MusicTranscription& _Left) = delete;
			MusicTranscription operator=(MusicTranscription&& _Right) = delete;
			MusicTranscription operator=(const MusicTranscription& _Left) = delete;
			virtual MidiTrack Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize = 1) const;
		protected:
			DragonianLibOrtEnv Env_;
			ProgressCallback _callback;
		};
	}
}