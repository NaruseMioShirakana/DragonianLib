#include "MoePianoTranScription.hpp"
#include "Base.h"

namespace DragonianLib
{
	namespace LibMusicTranscription
	{

		MoePianoTranScription::MoePianoTranScription(const Hparams& _Config, ProgressCallback&& _Callback, unsigned _ThreadCount, unsigned _DeviceID, unsigned _Provider) : MusicTranscription(std::move(_Callback), _ThreadCount, _DeviceID, _Provider)
		{
			sample_rate = _Config.SamplingRate;
			classes_num = _Config.ClassesCount;

			try
			{
				OnSetModel = new Ort::Session(*Env_.GetEnv(), (_Config.OnSetPath).c_str(), *Env_.GetSessionOptions());
				FrameModel = new Ort::Session(*Env_.GetEnv(), (_Config.FramePath).c_str(), *Env_.GetSessionOptions());
				if(!_Config.ModelPath.empty())
					CQTModel = new Ort::Session(*Env_.GetEnv(), (_Config.ModelPath).c_str(), *Env_.GetSessionOptions());
			}
			catch (Ort::Exception& e)
			{
				Destory();
				DragonianLibThrow(e.what());
			}
		}

		void MoePianoTranScription::Destory()
		{
			delete OnSetModel;
			OnSetModel = nullptr;
			delete FrameModel;
			FrameModel = nullptr;
			delete CQTModel;
			CQTModel = nullptr;
		}

		MoePianoTranScription::~MoePianoTranScription()
		{
			Destory();
		}

		MidiTrack MoePianoTranScription::Inference(DragonianLibSTL::Vector<float> _Audio, const Hparams& _Config, int64_t _BatchSize) const
		{
			_BatchSize = 1;
			const auto SegmentSamples = size_t(_Config.SegmentTime * float(_Config.SamplingRate));
			const size_t AudioLength = _Audio.Size();
			const size_t PaddingLength = size_t(ceil(double(AudioLength) / double(SegmentSamples))) * SegmentSamples;
			if (AudioLength < PaddingLength)
				_Audio.Insert(_Audio.end(), PaddingLength - AudioLength, 0.f);
			size_t Progress = 0;
			const size_t ProgressMax = (PaddingLength / SegmentSamples) * 2 - 1;

			DragonianLibSTL::Vector<float> Segments;
			Segments.Reserve(SegmentSamples * ProgressMax);
			for (size_t i = 0; i + SegmentSamples <= PaddingLength; i += SegmentSamples / 2)
				Segments.Insert(Segments.end(), _Audio.Data() + i, _Audio.Data() + i + SegmentSamples);

			for (size_t i = 0; i < Segments.Size(); i += SegmentSamples * _BatchSize)
			{
				OrtTensors inputTensors, outTensors;
				if (Progress + _BatchSize > ProgressMax)
					_BatchSize = int64_t(ProgressMax - Progress);
				//const int64_t inputShape[] = { _BatchSize, int64_t(SegmentSamples) };

				_callback(++Progress, ProgressMax);
			}
		}

	}
}