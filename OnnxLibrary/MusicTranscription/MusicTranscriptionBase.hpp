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
#include "Libraries/MyTemplateLibrary/Vector.h"
#include "Libraries/EnvManager.hpp"
#include "Libraries/AvCodec/AvCodec.h"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		using AvCodec::EstNoteEvents;
		using AvCodec::MidiTrack;
		using AvCodec::EstPedalEvents;
		using AvCodec::MidiEvent;

		using ProgressCallback = std::function<void(size_t, size_t)>;
		using OrtTensors = std::vector<Ort::Value>;

		void MeanFliter(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& _Signal, size_t WindowLength);

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

		// operators used to sort
		bool operator<(const EstNoteTp& a, const EstNoteTp& b);
		EstNoteTp operator+(const EstNoteTp& a, const EstNoteTp& b);

		class MusicTranscription
		{
		public:
			MusicTranscription(ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
			virtual ~MusicTranscription() = default;
			MusicTranscription(MusicTranscription&& _Right) = default;
			MusicTranscription(const MusicTranscription& _Left) = default;
			MusicTranscription& operator=(MusicTranscription&& _Right) = default;
			MusicTranscription& operator=(const MusicTranscription& _Left) = default;
			virtual AvCodec::MidiTrack Inference(const DragonianLibSTL::Vector<float>& _InputAudio, const Hparams& _Config, int64_t _BatchSize = 1) const;
		protected:
			std::shared_ptr<DragonianLibOrtEnv> Env_;
			ProgressCallback _callback;
		};
	}
}
