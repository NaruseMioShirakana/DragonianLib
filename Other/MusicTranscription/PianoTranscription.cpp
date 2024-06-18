#include "PianoTranscription.hpp"
#include <algorithm>
#include "Base.h"
#include "libremidi/writer.hpp"
#include <fstream>

namespace libmts
{
	PianoTranScription::PianoTranScription(const Hparams& _Config, const ProgressCallback& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : Env_(_ThreadCount, _DeviceID, _Provider)
	{
		sample_rate = _Config.sample_rate;
		classes_num = _Config.classes_num;

		try
		{
			PianoTranScriptionModel = new Ort::Session(*Env_.GetEnv(), _Config.path.c_str(), *Env_.GetSessionOptions());
		}
		catch (Ort::Exception& e)
		{
			Destory();
			DragonianLibThrow(e.what());
		}
		_callback = _Callback;
	}

	void PianoTranScription::Destory()
	{
		delete PianoTranScriptionModel;
		PianoTranScriptionModel = nullptr;
	}

	PianoTranScription::~PianoTranScription()
	{
		Destory();
	}

	// Infer Function
	midi_events PianoTranScription::Infer(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize) const
	{
		const size_t audio_len = _Audio.Size();
		const size_t pad_len = size_t(ceil(double(audio_len) / double(segment_samples))) * segment_samples;
		if (audio_len < pad_len)
			_Audio.Insert(_Audio.end(), pad_len - audio_len, 0.f);
		size_t progress = 0;
		const size_t progressMax = (pad_len / segment_samples) * 2 - 1;
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>> output[7];

		DragonianLibSTL::Vector<float> segments;
		segments.Reserve(segment_samples * progressMax);
		for (size_t i = 0; i + segment_samples <= pad_len; i += segment_samples / 2)
			segments.Insert(segments.end(), _Audio.Data() + i, _Audio.Data() + i + segment_samples);

		for (size_t i = 0; i < segments.Size(); i += segment_samples * _BatchSize)
		{
			_callback(progress, progressMax);
			OrtTensors inputTensors, outTensors;
			if (progress + _BatchSize > progressMax)
				_BatchSize = int64_t(progressMax - progress);
			const int64_t inputShape[] = { _BatchSize, int64_t(segment_samples) };
			inputTensors.emplace_back(Ort::Value::CreateTensor(*Env_.GetMemoryInfo(),
				segments.Data() + i,
				segment_samples * _BatchSize,
				inputShape,
				2));
			try
			{
				outTensors = PianoTranScriptionModel->Run(Ort::RunOptions{ nullptr },
				   inputNames.Data(),
				   inputTensors.data(),
				   inputTensors.size(),
				   outputNames.Data(),
				   outputNames.Size()
			   );
			}
			catch (Ort::Exception& e)
			{
				DragonianLibThrow(e.what());
			}
			progress += _BatchSize;
			for (size_t ita = 0; ita < 7; ++ita)
			{
				const auto Tdata = outTensors[ita].GetTensorData<float>();
				const auto TShape = outTensors[ita].GetTensorTypeAndShapeInfo().GetShape();
				for (int64_t it = 0; it < _BatchSize; ++it)
				{
					DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> tmp;
					tmp.Reserve(TShape[1]);
					for (int64_t iter = 0; iter < TShape[1]; ++iter)
						tmp.EmplaceBack(Tdata + (it * TShape[1] + iter) * TShape[2], Tdata + (it * TShape[1] + iter + 1) * TShape[2]);
					output[ita].EmplaceBack(tmp);
				}
			}
		}
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> routput[7];
		if (progressMax > 1)
		{
			for (size_t i = 0; i < 7; ++i)
			{
				const auto b_size = int64_t((output[i][0].Size() - 1) / 4);
				const auto e_size = int64_t((output[i][0].Size() - 1) * 3 / 4);
				routput[i].Reserve((progressMax + 2) * 2 * b_size);

				size_t it = 0;
				routput[i].Insert(routput[i].end(), output[i][it].begin(), output[i][it].begin() + e_size);
				++it;
				for (; it < output[i].Size() - 1; ++it)
					routput[i].Insert(routput[i].end(), output[i][it].begin() + b_size, output[i][it].begin() + e_size);
				it = output[i].Size() - 1;
				routput[i].Insert(routput[i].end(), output[i][it].begin() + b_size, output[i][it].begin() + e_size + b_size);
			}
		}

		NetOutPuts netOutputs(routput);
		midi_events midiEvents;
		if (!_Config.use_official_method)
			midiEvents = frame_to_note_info(netOutputs.frame_output, netOutputs.reg_offset_output, netOutputs.velocity_output, _Config);
		else
			midiEvents = toMidiEvents(netOutputs, _Config);

		_callback(progress, progressMax);
		return midiEvents;
	}

	// detect note info with output dict (MyMethod, and It works better in my software)
	midi_events PianoTranScription::frame_to_note_info(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& frame_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& reg_offset_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& velocity_output, const Hparams& _Config)
	{
		const auto Temp = get_binarized_output_from_regression(reg_offset_output, float(_Config.offset_threshold), _Config.offset_ali);
		const auto offset = std::get<0>(Temp);
		DragonianLibSTL::Vector<est_note_events> outputs;
		double onset = 0.0;
		const long class_size = long(frame_output[0].Size()), duration_size = long(frame_output.Size());
		for (long pitch = 0; pitch < class_size; ++pitch)
		{
			bool begin = false;
			for (long duration = 0; duration < duration_size; ++duration)
			{
				if (!begin && frame_output[duration][pitch] >= float(_Config.frame_threshold))
				{
					begin = true;
					onset = double(duration);
					continue;
				}
				if (begin)
				{
					if ((frame_output[duration][pitch] < float(_Config.frame_threshold)) ||
						(double(duration) - onset > 600.0) ||
						(duration == duration_size - 1) ||
						(offset[duration][pitch] == 1.0f))
					{
						begin = false;
						outputs.EmplaceBack(onset / double(_Config.frames_per_second), double(duration) / double(_Config.frames_per_second), pitch + _Config.begin_note, long(velocity_output[long(onset)][pitch] * float(_Config.velocity_scale) + 1));
					}
				}
			}
		}
		return { std::move(outputs),{} };
	}

	// detect note info with onset offset & frame (Orginal Method)
	DragonianLibSTL::Vector<PianoTranScription::est_note_tp> note_detection_with_onset_offset_regress(
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& frame_output,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& onset_output,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& onset_shift_output,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& offset_output,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& offset_shift_output,
		const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& velocity_output,
		float frame_threshold,
		long nk)
	{
		const long frames_num = long(frame_output.Size());

		DragonianLibSTL::Vector<PianoTranScription::est_note_tp> output_tuples;

		bool begin = false, bframe_disappear = false, boffset_occur = false;
		long bgn = 9999, fin = 0, frame_disappear = 9999, offset_occur = 9999;

		for (long i = 0; i < frames_num; ++i)
		{
			if (onset_output[i][nk] == 1.0f)
			{
				if (begin)
				{
					fin = std::max(i - 1, 0l);
					output_tuples.EmplaceBack(bgn, fin, double(onset_shift_output[bgn][nk]), 0.0, double(velocity_output[bgn][nk]));
					bframe_disappear = false;
					boffset_occur = false;
				}
				bgn = i;
				begin = true;
			}
			if (begin && i > bgn)
			{
				if (frame_output[i][nk] <= frame_threshold && !bframe_disappear)
				{
					frame_disappear = i;
					bframe_disappear = true;
				}
				if ((offset_output[i][nk] == 1.0f) && !boffset_occur)
				{
					offset_occur = i;
					boffset_occur = true;
				}
				if (bframe_disappear)
				{
					if (boffset_occur && offset_occur - bgn > frame_disappear - offset_occur)
						fin = offset_occur;
					else
						fin = frame_disappear;
					output_tuples.EmplaceBack(bgn, fin, double(onset_shift_output[bgn][nk]), double(offset_shift_output[fin][nk]), double(velocity_output[bgn][nk]));
					bframe_disappear = false;
					boffset_occur = false;
					begin = false;
				}
				if (begin && (i - bgn >= 600 || i == frames_num - 1))
				{
					fin = i;
					output_tuples.EmplaceBack(bgn, fin, double(onset_shift_output[bgn][nk]), double(offset_shift_output[fin][nk]), double(velocity_output[bgn][nk]));
					bframe_disappear = false;
					boffset_occur = false;
					begin = false;
				}
			}
		}
		std::sort(output_tuples.begin(), output_tuples.end());
		return output_tuples;
	}

	// detect note info with output dict (Orginal Method)
	DragonianLibSTL::Vector<est_note_events> PianoTranScription::output_dict_to_detected_notes(const NetOutPuts& output_dict, const Hparams& _Config)
	{
		const long class_num = long(output_dict.frame_output[0].Size());
		DragonianLibSTL::Vector<est_note_tp> est_tuples;
		DragonianLibSTL::Vector<long> est_midi_notes;
		for (long piano_note = 0; piano_note < class_num; ++piano_note)
		{
			auto est_tuples_per_note = note_detection_with_onset_offset_regress(
				output_dict.frame_output,
				output_dict.onset_output,
				output_dict.onset_shift_output,
				output_dict.offset_output,
				output_dict.offset_shift_output,
				output_dict.velocity_output,
				float(_Config.frame_threshold),
				piano_note
			);
			if (est_tuples_per_note.Empty())
				continue;
			est_tuples.Insert(est_tuples.end(), est_tuples_per_note.begin(), est_tuples_per_note.end());
			for (size_t ii = 0; ii < est_tuples_per_note.Size(); ++ii)
				est_midi_notes.EmplaceBack(piano_note + _Config.begin_note);
		}
		DragonianLibSTL::Vector<est_note_events> est_on_off_note_vels;
		est_on_off_note_vels.Reserve(est_tuples.Size());
		for (size_t i = 0; i < est_tuples.Size(); ++i)
		{
			est_on_off_note_vels.EmplaceBack((double(est_tuples[i].fin) + est_tuples[i].onset_shift) / double(_Config.frames_per_second),
				(double(est_tuples[i].bgn) + est_tuples[i].offset_shift) / double(_Config.frames_per_second),
				est_midi_notes[i],
				long(est_tuples[i].normalized_velocity * _Config.velocity_scale));
		}
		return est_on_off_note_vels;
	}

	//DragonianLibSTL::Vector<PianoTranScription::est_pedal_events> PianoTranScription::output_dict_to_detected_pedals(const NetOutPuts& output_dict) const
	//{
	//	return {};
	//}

	// NetOutputs to MidiEvents (Orginal Method)
	midi_events PianoTranScription::toMidiEvents(NetOutPuts& output_dict, const Hparams& _Config)
	{
		DragonianLibSTL::Vector<est_pedal_events> _pedal;
		auto Temp = get_binarized_output_from_regression(output_dict.reg_onset_output, float(_Config.onset_threshold), _Config.onset_ali);
		output_dict.onset_output = std::move(std::get<0>(Temp));
		output_dict.onset_shift_output = std::move(std::get<1>(Temp));
		Temp = get_binarized_output_from_regression(output_dict.reg_offset_output, float(_Config.offset_threshold), _Config.offset_ali);
		output_dict.offset_output = std::move(std::get<0>(Temp));
		output_dict.offset_shift_output = std::move(std::get<1>(Temp));
		DragonianLibSTL::Vector<est_note_events> _note = output_dict_to_detected_notes(output_dict, _Config);
		return { std::move(_note),std::move(_pedal) };
	}

	// If the class normal distribution is satisfied, return true, else false
	bool is_monotonic_neighbour(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& x, long n, long neighbour, long k)
	{
		bool monotonic = true;
		for (long i = 0; i < neighbour; ++i)
		{
			if (n - i < 0)
				continue;
			if (x[n - i][k] < x[n - i - 1][k] || x[n + i][k] < x[n + i + 1][k])
				monotonic = false;
		}
		return monotonic;
	}

	// Look for Midi events on the timeline
	std::tuple<DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>, DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>> PianoTranScription::get_binarized_output_from_regression(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& reg_output, float threshold, int neighbour)
	{
		const long frames_num = long(reg_output.Size());
		const long class_num = long(reg_output[0].Size());
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> binary_output(frames_num, DragonianLibSTL::Vector<float>(class_num, 0.0));
		DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>> shift_output(frames_num, DragonianLibSTL::Vector<float>(class_num, 0.0));
		for (long k = 0; k < class_num; ++k)
		{
			for (long n = 0; n < frames_num; ++n)
			{
				if (reg_output[n][k] > threshold && is_monotonic_neighbour(reg_output, n, neighbour, k))
				{
					binary_output[n][k] = 1.0f;
					if (reg_output[n - 1][k] > reg_output[n + 1][k])
						shift_output[n][k] = (reg_output[n + 1][k] - reg_output[n - 1][k]) / (reg_output[n][k] - reg_output[n + 1][k]) / 2.0f;
					else
						shift_output[n][k] = (reg_output[n + 1][k] - reg_output[n - 1][k]) / (reg_output[n][k] - reg_output[n - 1][k]) / 2.0f;
				}
			}
		}
		return{ binary_output ,shift_output };
	}

	// operators used to sort
	bool operator<(const PianoTranScription::est_note_tp& a, const PianoTranScription::est_note_tp& b)
	{
		return a.bgn < b.bgn;
	}
	bool operator<(const MidiEvent& a, const MidiEvent& b)
	{
		return a.time < b.time;
	}
	PianoTranScription::est_note_tp operator+(const PianoTranScription::est_note_tp& a, const PianoTranScription::est_note_tp& b)
	{
		return { a.bgn + b.bgn,a.fin + b.fin,a.onset_shift + b.onset_shift,a.offset_shift + b.offset_shift,a.normalized_velocity + b.normalized_velocity };
	}

	void WriteMidiFile(const std::wstring& _Path, const midi_events& _Events, long _Begin, long _TPS)
	{
		libremidi::writer _Writer;
		_Writer.add_track();
		std::vector<MidiEvent> _events;
		for (const auto& it : _Events.note)
		{
			_events.emplace_back(it.onset_time, it.midi_note, it.velocity);
			_events.emplace_back(it.offset_time, it.midi_note, 0);
		}
		std::sort(_events.begin(), _events.end());
		long previous_ticks = _Begin;
		for (const auto& it : _events) {
			const long this_ticks = long((it.time - _Begin) * _TPS);
			if (this_ticks >= 0)
			{
				long diff_ticks = this_ticks - previous_ticks;
				if (diff_ticks < 0)
					diff_ticks = 0;
				previous_ticks = this_ticks;
				if (it.velocity)
					_Writer.add_event(diff_ticks, 0, libremidi::channel_events::note_on(0, uint8_t(unsigned long(it.midi_note)), uint8_t(unsigned long(it.velocity))));
				else
					_Writer.add_event(diff_ticks, 0, libremidi::channel_events::note_off(0, uint8_t(unsigned long(it.midi_note)), uint8_t(unsigned long(it.velocity))));
			}
		}
		_Writer.add_event(0, 0, libremidi::meta_events::end_of_track());
		auto ofs = std::ofstream(_Path, std::ios::out | std::ios::binary);
		if (!ofs.is_open())
			DragonianLibThrow("Could not write file!");
		_Writer.write(ofs);
	}
}
