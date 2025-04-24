import torch
import torch.nn.functional as F
import scipy

dimm = 8
upp = 3

f0 = torch.ones(1, 500, 1) * torch.arange(1, dimm + 1)
rad = f0 #/ 48000
print(rad[..., -1:])
rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
rad_acc = F.pad(rad_acc, (0, 0, 1, -1))
rad += rad_acc
rad = rad.reshape(f0.shape[0], -1, 1)
rad = torch.multiply(rad, torch.arange(1, dimm + 1, device=f0.device).reshape(1, 1, -1))
rand_ini = torch.rand(1, 1, dimm, device=f0.device)
rand_ini[..., 0] = 0
rad += rand_ini
sines = torch.sin(2 * 3.1415926535 * rad)