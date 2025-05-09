import argparse
import json
import onnx
import onnxsim
import torch

import ncnnonnxexport.utils as utils
from ncnnonnxexport.model import SynthesizerTrn

parser = argparse.ArgumentParser(description='SoVitsSvc OnnxExport')

def OnnxExport(path=None):
    device = torch.device("cpu")
    hps = utils.get_hparams_from_file(f"{path}/config.json")
    SVCVITS = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = utils.load_checkpoint(f"{path}/model.pth", SVCVITS, None)
    _ = SVCVITS.eval().to(device)
    for i in SVCVITS.parameters():
        i.requires_grad = False
    
    num_frames =        200

    test_hidden_unit =  torch.rand(SVCVITS.gin_channels, num_frames)
    test_pitch =        torch.rand(num_frames)
    test_vol =          torch.rand(num_frames)
    test_uv =           torch.ones(num_frames, dtype=torch.float32)
    test_noise =        torch.randn(num_frames, 192)
    test_sid =          torch.LongTensor([0])
    export_mix =        True
    if SVCVITS.n_speaker <= 1:
        export_mix = False
    
    if export_mix:
        spk_mix = []
        n_spk = len(hps.spk)
        for i in range(n_spk):
            spk_mix.append(1.0/float(n_spk))
        test_sid = torch.tensor(spk_mix)
        SVCVITS.export_chara_mix(hps.spk)
        test_sid = test_sid.unsqueeze(0)
        test_sid = test_sid.repeat(num_frames, 1)

    if SVCVITS.n_speaker == 1:
        parameter = SVCVITS.emb_g.weight.clone()
        parameter = torch.repeat_interleave(parameter, 8, 0)
        SVCVITS.emb_g = torch.nn.Embedding(8, SVCVITS.gin_channels)
        SVCVITS.emb_g.weight = torch.nn.Parameter(parameter, False)
    
    SVCVITS.eval()

    if export_mix:
        daxes = {
            "c": [0],
            "f0": [0],
            "uv": [0],
            "noise": [1],
            "sid":[0]
        }
    else:
        daxes = {
            "c": [0],
            "f0": [0],
            "uv": [0],
            "noise": [1]
        }
    
    input_names = ["c", "f0", "uv", "noise", "sid"]
    output_names = ["audio", ]

    if SVCVITS.vol_embedding:
        input_names.append("vol")
        vol_dadict = {"vol" : [1]}
        daxes.update(vol_dadict)
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device),
            test_vol.to(device)
        )
    else:
        test_inputs = (
            test_hidden_unit.to(device),
            test_pitch.to(device),
            test_uv.to(device),
            test_noise.to(device),
            test_sid.to(device)
        )

    SVCVITS.dec.OnnxExport()

    torch.onnx.export(
        SVCVITS,
        test_inputs,
        f"{path}/Model.onnx",
        dynamic_axes=daxes,
        do_constant_folding=False,
        opset_version=18,
        verbose=False,
        input_names=input_names,
        output_names=output_names
    )

    model, _ = onnxsim.simplify(f"{path}/Model.onnx")
    onnx.save(model, f"{path}/Model.onnx")

    vec_lay = "layer-12" if SVCVITS.gin_channels == 768 else "layer-9"
    spklist = []
    for key in hps.spk.keys():
        spklist.append(key)

    MoeVSConf = {
        "Folder" : f"{path}",
        "Name" : f"{path}",
        "Type" : "SoVits",
        "Rate" : hps.data.sampling_rate,
        "Hop" : hps.data.hop_length,
        "Hubert": f"vec-{SVCVITS.gin_channels}-{vec_lay}",
        "SoVits4": True,
        "SoVits3": False,
        "CharaMix": export_mix,
        "Volume": SVCVITS.vol_embedding,
        "HiddenSize": SVCVITS.gin_channels,
        "Characters": spklist,
        "Cluster": ""
    }

    with open(f"{path}/OnnxConfig.json", 'w') as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent = 4)


if __name__ == '__main__':
    parser.add_argument('-n', '--model_name', type=str, default="SoVits", help='模型文件夹名')
    args = parser.parse_args()
    path = args.model_name
    OnnxExport(path)
