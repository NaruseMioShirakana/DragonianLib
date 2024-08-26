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