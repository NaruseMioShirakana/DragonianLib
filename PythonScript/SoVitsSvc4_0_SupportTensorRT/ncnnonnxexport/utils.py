
import numpy as np
import torch
import json

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)
a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
b = f0_mel_min * a - 1.

def f0_to_coarse(f0):
  f0_mel = 1127 * (1 + f0 / 700).log()
  f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
  return torch.clip(torch.round(f0_mel), min=1., max=float(f0_bin - 1)).long()

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
              new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model, optimizer, learning_rate, iteration

def get_hparams_from_file(config_path, infer_mode = False):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)
  hparams =HParams(**config) if not infer_mode else InferHParams(**config)
  return hparams

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def get(self,index):
    return self.__dict__.get(index)

class InferHParams(HParams):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = InferHParams(**v)
      self[k] = v

  def __getattr__(self,index):
    return self.get(index)
