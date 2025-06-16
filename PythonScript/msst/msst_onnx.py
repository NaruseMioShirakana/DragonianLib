import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from ml_collections import ConfigDict
import pnnx

def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    if model_type == 'mel_band_roformer':
        from modules.bs_roformer import MelBandRoformer
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        from modules.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config.model)
        )

    return model, config

def export(model_type, config_path, model_path=None, sim = False):
    output_path = os.path.splitext(config_path)[0] + ".onnx" if model_path else None
    model, config = get_model_from_config(model_type, config_path)
    if model_path and os.path.exists(model_path):
        try:
            if model_type in ['htdemucs', 'apollo']:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                if 'state' in state_dict:
                    state_dict = state_dict['state']
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            else:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Model file {model_path} not found. Use empty model.")
    model.eval()
    model.requires_grad_(False)
    inputs = torch.randn(1, 2, 512, 1025, 2)
    inputs.requires_grad_(False)
    #model = torch.jit.trace(model, inputs, check_trace=False)
    if output_path:
        pnnx.export(
            model,
            model_path+".pt",
            inputs,
            check_trace=False
        )


export(
    model_type='mel_band_roformer',
    config_path="D:\\VSGIT\\MSST-WebUI-main\\configs_backup\\vocal_models\\melband_roformer_instvox_duality_v2.ckpt.yaml",
    model_path="D:\\VSGIT\\MSST-WebUI-main\\configs_backup\\vocal_models\\melband_roformer_instvox_duality_v2.ckpt",
    sim=True
)