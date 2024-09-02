#include "ByteDancePianoTranScription.hpp"
#include <algorithm>
#include "Base.h"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{
		ByteDancePianoTranScription::ByteDancePianoTranScription(const Hparams& _Config, ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : MusicTranscription(std::move(_Callback), _ThreadCount, _DeviceID, _Provider)
		{
			sample_rate = _Config.SamplingRate;
			classes_num = _Config.ClassesCount;

			try
			{
				PianoTranScriptionModel = new Ort::Session(*Env_.GetEnv(), _Config.ModelPath.c_str(), *Env_.GetSessionOptions());
			}
			catch (Ort::Exception& e)
			{
				Destory();
				DragonianLibThrow(e.what());
			}
		}

		void ByteDancePianoTranScription::Destory()
		{
			delete PianoTranScriptionModel;
			PianoTranScriptionModel = nullptr;
		}

		ByteDancePianoTranScription::~ByteDancePianoTranScription()
		{
			Destory();
		}

		// Infer Function
		MidiTrack ByteDancePianoTranScription::Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize) const
		{
			const auto segment_samples = size_t(_Config.SegmentTime * float(_Config.SamplingRate));
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
			MidiTrack midiEvents;
			if (!_Config.UseByteDanceMethod)
				midiEvents = frame_to_note_info(netOutputs.frame_output, netOutputs.reg_offset_output, netOutputs.velocity_output, _Config);
			else
				midiEvents = toMidiEvents(netOutputs, _Config);

			_callback(progress, progressMax);
			return midiEvents;
		}

		// detect note info with output dict (MyMethod, and It works better in my software)
		MidiTrack ByteDancePianoTranScription::frame_to_note_info(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& frame_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& reg_offset_output, const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& velocity_output, const Hparams& _Config)
		{
			const auto Temp = get_binarized_output_from_regression(reg_offset_output, float(_Config.OffsetThreshold), _Config.OffsetAligSize);
			const auto offset = std::get<0>(Temp);
			DragonianLibSTL::Vector<EstNoteEvents> outputs;
			double onset = 0.0;
			const long class_size = long(frame_output[0].Size()), duration_size = long(frame_output.Size());
			for (long pitch = 0; pitch < class_size; ++pitch)
			{
				bool begin = false;
				for (long duration = 0; duration < duration_size; ++duration)
				{
					if (!begin && frame_output[duration][pitch] >= float(_Config.FrameThreshold))
					{
						begin = true;
						onset = double(duration);
						continue;
					}
					if (begin)
					{
						if ((frame_output[duration][pitch] < float(_Config.FrameThreshold)) ||
							(double(duration) - onset > 600.0) ||
							(duration == duration_size - 1) ||
							(offset[duration][pitch] == 1.0f))
						{
							begin = false;
							outputs.EmplaceBack(onset / double(_Config.FrameTime), double(duration) / double(_Config.FrameTime), pitch + _Config.LowestPitch, long(velocity_output[long(onset)][pitch] * float(_Config.VelocityScale) + 1));
						}
					}
				}
			}
			return { std::move(outputs),{} };
		}

		// detect note info with onset offset & frame (Orginal Method)
		DragonianLibSTL::Vector<EstNoteTp> note_detection_with_onset_offset_regress(
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

			DragonianLibSTL::Vector<EstNoteTp> output_tuples;

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
		DragonianLibSTL::Vector<EstNoteEvents> ByteDancePianoTranScription::output_dict_to_detected_notes(const NetOutPuts& output_dict, const Hparams& _Config)
		{
			const long class_num = long(output_dict.frame_output[0].Size());
			DragonianLibSTL::Vector<EstNoteTp> est_tuples;
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
					float(_Config.FrameThreshold),
					piano_note
				);
				if (est_tuples_per_note.Empty())
					continue;
				est_tuples.Insert(est_tuples.end(), est_tuples_per_note.begin(), est_tuples_per_note.end());
				for (size_t ii = 0; ii < est_tuples_per_note.Size(); ++ii)
					est_midi_notes.EmplaceBack(piano_note + _Config.LowestPitch);
			}
			DragonianLibSTL::Vector<EstNoteEvents> est_on_off_note_vels;
			est_on_off_note_vels.Reserve(est_tuples.Size());
			for (size_t i = 0; i < est_tuples.Size(); ++i)
			{
				est_on_off_note_vels.EmplaceBack((double(est_tuples[i].End) + est_tuples[i].OnsetShift) / double(_Config.FrameTime),
					(double(est_tuples[i].Begin) + est_tuples[i].OffsetShift) / double(_Config.FrameTime),
					est_midi_notes[i],
					long(est_tuples[i].NormalizedVelocity * _Config.VelocityScale));
			}
			return est_on_off_note_vels;
		}

		//DragonianLibSTL::Vector<PianoTranScription::est_pedal_events> PianoTranScription::output_dict_to_detected_pedals(const NetOutPuts& output_dict) const
		//{
		//	return {};
		//}

		// NetOutputs to MidiEvents (Orginal Method)
		MidiTrack ByteDancePianoTranScription::toMidiEvents(NetOutPuts& output_dict, const Hparams& _Config)
		{
			DragonianLibSTL::Vector<EstPedalEvents> _pedal;
			auto Temp = get_binarized_output_from_regression(output_dict.reg_onset_output, float(_Config.OnsetThreshold), _Config.OnsetAligSize);
			output_dict.onset_output = std::move(std::get<0>(Temp));
			output_dict.onset_shift_output = std::move(std::get<1>(Temp));
			Temp = get_binarized_output_from_regression(output_dict.reg_offset_output, float(_Config.OffsetThreshold), _Config.OffsetAligSize);
			output_dict.offset_output = std::move(std::get<0>(Temp));
			output_dict.offset_shift_output = std::move(std::get<1>(Temp));
			DragonianLibSTL::Vector<EstNoteEvents> _note = output_dict_to_detected_notes(output_dict, _Config);
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
		std::tuple<DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>, DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>> ByteDancePianoTranScription::get_binarized_output_from_regression(const DragonianLibSTL::Vector<DragonianLibSTL::Vector<float>>& reg_output, float threshold, int neighbour)
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

	}
}