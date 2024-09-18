/**
 * FileName: MoePianoTranScription.hpp
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
#include "MusicTranscriptionBase.hpp"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		class MoePianoTranScription : public MusicTranscription
		{
		public:
			MoePianoTranScription(const Hparams& _Config, ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
			~MoePianoTranScription() override;
			MidiTrack Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize = 1) const override;

		private:
			void Destory();
			MoePianoTranScription(const MoePianoTranScription&) = delete;
			MoePianoTranScription(MoePianoTranScription&&) = delete;
			MoePianoTranScription& operator=(const MoePianoTranScription&) = delete;
			MoePianoTranScription& operator=(MoePianoTranScription&&) = delete;

			Ort::Session* OnSetModel = nullptr;
			Ort::Session* FrameModel = nullptr;
			Ort::Session* CQTModel = nullptr;
			long sample_rate = 16000;
			long classes_num = 88;
		};
	}
}