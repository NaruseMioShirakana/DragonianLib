#pragma once
#include "MusicTranscriptionBase.hpp"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		class ByteDancePianoTranScription : public MusicTranscription
		{
		public:
			struct NetOutPuts
			{
				DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> frame_output, reg_onset_output, reg_offset_output, velocity_output, reg_pedal_onset_output, reg_pedal_offset_output, pedal_frame_output, onset_output, onset_shift_output, offset_output, offset_shift_output, pedal_onset_output, pedal_offset_output;
				NetOutPuts(DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>* input) :
					frame_output(std::move(input[0])),
					reg_onset_output(std::move(input[1])),
					reg_offset_output(std::move(input[2])),
					velocity_output(std::move(input[3])),
					reg_pedal_onset_output(std::move(input[4])),
					reg_pedal_offset_output(std::move(input[5])),
					pedal_frame_output(std::move(input[6])) {}
			};
			ByteDancePianoTranScription(const Hparams& _Config, ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);
			~ByteDancePianoTranScription() override;
			MidiTrack Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize = 1) const override;
			//DragonianLibSTL::Vector<est_pedal_events> output_dict_to_detected_pedals(const NetOutPuts& output_dict) const;
		private:
			static MidiTrack frame_to_note_info(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& frame_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& offset_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& velocity_output, const Hparams& _Config);

			static MidiTrack toMidiEvents(NetOutPuts& output_dict, const Hparams& _Config);

			static std::tuple<DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>, DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>> get_binarized_output_from_regression(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>&, float, int);

			static DragonianLibSTL::Vector<EstNoteEvents> output_dict_to_detected_notes(const NetOutPuts& output_dict, const Hparams& _Config);

			void Destory();
			ByteDancePianoTranScription(const ByteDancePianoTranScription&) = delete;
			ByteDancePianoTranScription(ByteDancePianoTranScription&&) = delete;
			ByteDancePianoTranScription& operator=(const ByteDancePianoTranScription&) = delete;
			ByteDancePianoTranScription& operator=(ByteDancePianoTranScription&&) = delete;

			Ort::Session* PianoTranScriptionModel = nullptr;
			//size_t segment_samples = 160000;
			long sample_rate = 16000;
			long classes_num = 88;
			const DragonianLibSTL::Vector<const char*> inputNames = { "audio" };
			const DragonianLibSTL::Vector<const char*> outputNames = { "frame_output", "reg_onset_output", "reg_offset_output", "velocity_output", "reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output" };
		};


	}
}

