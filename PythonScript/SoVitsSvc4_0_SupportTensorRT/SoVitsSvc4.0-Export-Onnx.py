import argparse
import json
import onnx
import onnxsim
import torch

import onnxexport.utils as utils
from onnxexport.model import SynthesizerTrn

parser = argparse.ArgumentParser(description='SoVitsSvc OnnxExport')

def OnnxExport(ModelPath=None, SpeakerMix = True):
    Device = torch.device("cpu")
    Hparams = utils.get_hparams_from_file(f"{ModelPath}/config.json")
    SoVitsSvcModel = SynthesizerTrn(
        Hparams.data.filter_length // 2 + 1,
        Hparams.train.segment_size // Hparams.data.hop_length,
        **Hparams.model)
    _ = utils.load_checkpoint(f"{ModelPath}/model.pth", SoVitsSvcModel, None)
    _ = SoVitsSvcModel.eval().to(Device)
    for i in SoVitsSvcModel.parameters():
        i.requires_grad = False
    
    NumFrames = 200

    TestUnits = torch.rand(1, NumFrames, SoVitsSvcModel.gin_channels)
    TestPitch = torch.rand(1, NumFrames)
    TestVolume = torch.rand(1, NumFrames)
    TestMel2Units = torch.LongTensor(torch.arange(0, NumFrames)).unsqueeze(0)
    TestUnVoice = torch.ones(1, NumFrames, dtype=torch.float32)
    TestNoise = torch.randn(1, 192, NumFrames)
    TestSpeaker = torch.LongTensor([[0]])

    if len(Hparams.spk) < 2:
        SpeakerMix = False
    
    if SpeakerMix:
        SpeakerMixData = []
        SpeakerCount = len(Hparams.spk)
        for i in range(SpeakerCount):
            SpeakerMixData.append(1.0/float(SpeakerCount))
        TestSpeaker = torch.tensor(SpeakerMixData)
        SoVitsSvcModel.export_chara_mix(Hparams.spk)
        TestSpeaker = TestSpeaker.unsqueeze(0)
        TestSpeaker = TestSpeaker.repeat(NumFrames, 1)
        TestSpeaker = TestSpeaker.unsqueeze(0)
    
    SoVitsSvcModel.eval()

    if SpeakerMix:
        DynamicAxis = {
            "Units": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "F0": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "Mel2Units": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "UnVoice": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "Noise": {
                0 : "BatchSize",
                2 : "FrameCount"
            },
            "Speaker":{
                0 : "BatchSize",
                1 : "FrameCount"
            }
        }
    else:
        DynamicAxis = {
            "Units": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "F0": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "Mel2Units": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "UnVoice": {
                0 : "BatchSize",
                1 : "FrameCount"
            },
            "Noise": {
                0 : "BatchSize",
                2 : "FrameCount"
            },
            "Speaker":{
                0 : "BatchSize"
            }
        }
    
    InputNames = ["Units", "F0", "Mel2Units", "UnVoice", "Noise", "Speaker"]
    OutputNames = ["Audio"]

    if SoVitsSvcModel.vol_embedding:
        InputNames.append("Volume")
        VolDaxis = {
            "Volume" : {
                0 : "BatchSize",
                1 : "FrameCount"
            }
        }
        DynamicAxis.update(VolDaxis)
        TestInputs = (
            TestUnits.to(Device),
            TestPitch.to(Device),
            TestMel2Units.to(Device),
            TestUnVoice.to(Device),
            TestNoise.to(Device),
            TestSpeaker.to(Device),
            TestVolume.to(Device)
        )
    else:
        TestInputs = (
            TestUnits.to(Device),
            TestPitch.to(Device),
            TestMel2Units.to(Device),
            TestUnVoice.to(Device),
            TestNoise.to(Device),
            TestSpeaker.to(Device)
        )

    SoVitsSvcModel.dec.OnnxExport()

    torch.onnx.export(
        SoVitsSvcModel,
        TestInputs,
        f"{ModelPath}/Model.onnx",
        dynamic_axes=DynamicAxis,
        do_constant_folding=True,
        opset_version=18,
        verbose=False,
        input_names=InputNames,
        output_names=OutputNames
    )

    try:
        import numpy as np
        from onnxconverter_common import auto_mixed_precision
        def validate_fn(predict, target):
            return np.mean((np.array(predict) - np.array(target)) ** 2) < 0.0001
        OnnxModel = onnx.load(f"{ModelPath}/Model.onnx")
        OnnxInputs = {key : value.cpu().numpy() for key, value in zip(InputNames, TestInputs)}
        auto_mixed_precision.auto_convert_mixed_precision(OnnxModel, OnnxInputs, validate_fn=validate_fn, keep_io_types=True)
    except Exception as e:
        print("Failed to convert to mixed precision, saving float32 model only, error:", e)

    ModelF32, _ = onnxsim.simplify(f"{ModelPath}/Model.onnx")
    onnx.save(ModelF32, f"{ModelPath}/ModelF32.onnx")

    VecLayer = "layer-12" if SoVitsSvcModel.gin_channels == 768 else "layer-9"
    SpeakerList = []
    for key in Hparams.spk.keys():
        SpeakerList.append(key)

    MoeVSConf = {
        "Folder" : f"{ModelPath}",
        "Name" : f"{ModelPath}",
        "Type" : "SoVits",
        "Rate" : Hparams.data.sampling_rate,
        "Hop" : Hparams.data.hop_length,
        "Hubert": f"vec-{SoVitsSvcModel.gin_channels}-{VecLayer}",
        "SoVits4": True,
        "SoVits3": False,
        "CharaMix": SpeakerMix,
        "Volume": SoVitsSvcModel.vol_embedding,
        "HiddenSize": SoVitsSvcModel.gin_channels,
        "Characters": SpeakerList,
        "Cluster": ""
    }

    with open(f"{ModelPath}/OnnxConfig.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)


if __name__ == '__main__':
    parser.add_argument('-n', '--model_name', type=str, default="OnnxSoVits", help='模型文件夹名')
    args = parser.parse_args()
    path = args.model_name
    OnnxExport(path)
