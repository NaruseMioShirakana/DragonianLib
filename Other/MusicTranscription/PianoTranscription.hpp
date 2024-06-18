#pragma once
#include "Vector.h"
#include "EnvManager.hpp"
namespace libmts
{
	struct MidiEvent
	{
		double time = 0;
		long midi_note = 0;
		long velocity = 0;
		MidiEvent(double t = 0.0, long m = 0, long v = 0) :time(t), midi_note(m), velocity(v) {}
	};

	struct est_pedal_events
	{
		double onset_time = 0.0;
		double offset_time = 0.0;
	};

	struct est_note_events
	{
		double onset_time = 0.0;
		double offset_time = 0.0;
		long midi_note = 0;
		long velocity = 0;
		est_note_events(double a, double b, long c, long d) :onset_time(a), offset_time(b), midi_note(c), velocity(d) {}
	};

	struct midi_events
	{
		DragonianLibSTL::Vector<est_note_events> note;
		DragonianLibSTL::Vector<est_pedal_events> pedal;
		midi_events() = default;
		midi_events(DragonianLibSTL::Vector<est_note_events>&& ene, DragonianLibSTL::Vector<est_pedal_events>&& epe) : note(std::move(ene)), pedal(std::move(epe)) {}
	};

	class PianoTranScription
	{
	public:
		using ProgressCallback = std::function<void(size_t, size_t)>;
		using OrtTensors = std::vector<Ort::Value>;
		struct Hparams
		{
			std::wstring path;
			long sample_rate = 16000;
			long classes_num = 88;
			long begin_note = 21;
			float segment_seconds = 10.0f;
			float hop_seconds = 1.0f;
			float frames_per_second = 100.0;
			long velocity_scale = 128;
			double onset_threshold = 0.3;
			double offset_threshold = 0.3;
			double frame_threshold = 0.1;
			double pedal_offset_threshold = 0.2;
			long onset_ali = 2;
			long offset_ali = 4;
			bool use_official_method = false;
		};

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

		struct est_note_tp
		{
			long bgn = 0;
			long fin = 0;
			double onset_shift = 0.0;
			double offset_shift = 0.0;
			double normalized_velocity = 0.0;
			est_note_tp(long a, long b, double c, double d, double e) :bgn(a), fin(b), onset_shift(c), offset_shift(d), normalized_velocity(e) {}
		};

		PianoTranScription(const Hparams& _Config, const ProgressCallback& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider);

		~PianoTranScription();

		midi_events Infer(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize = 1) const;

		static midi_events frame_to_note_info(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& frame_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& offset_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& velocity_output, const Hparams& _Config);

		static midi_events toMidiEvents(NetOutPuts& output_dict, const Hparams& _Config);

		static std::tuple<DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>, DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>> get_binarized_output_from_regression(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>&, float, int);

		static DragonianLibSTL::Vector<est_note_events> output_dict_to_detected_notes(const NetOutPuts& output_dict, const Hparams& _Config);

		//DragonianLibSTL::Vector<est_pedal_events> output_dict_to_detected_pedals(const NetOutPuts& output_dict) const;
	private:
		void Destory();
		PianoTranScription(const PianoTranScription&) = delete;
		PianoTranScription(PianoTranScription&&) = delete;
		PianoTranScription& operator=(const PianoTranScription&) = delete;
		PianoTranScription& operator=(PianoTranScription&&) = delete;

		Ort::Session* PianoTranScriptionModel = nullptr;
		size_t segment_samples = 160000;
		long sample_rate = 16000;
		long classes_num = 88;

		ProgressCallback _callback;
		DragonianLib::DragonianLibOrtEnv Env_;
		DragonianLibSTL::Vector<const char*> inputNames = { "audio" };
		DragonianLibSTL::Vector<const char*> outputNames = { "frame_output", "reg_onset_output", "reg_offset_output", "velocity_output", "reg_pedal_onset_output", "reg_pedal_offset_output", "pedal_frame_output" };
	};

	// operators used to sort
	bool operator<(const PianoTranScription::est_note_tp& a, const PianoTranScription::est_note_tp& b);
	bool operator<(const MidiEvent& a, const MidiEvent& b);
	PianoTranScription::est_note_tp operator+(const PianoTranScription::est_note_tp& a, const PianoTranScription::est_note_tp& b);

	void WriteMidiFile(const std::wstring& _Path, const midi_events& _Events, long _Begin, long _TPS);
}
