
from lib.lib_v5.nets_new import CascadedNet
import torch
from torch import nn
import torch.nn.functional as F
import os

class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        from lib.lib_v5.nets_61968KB import BaseASPPNet, layers
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)

        self.stg2_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)

        self.stg3_bridge = layers.Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)

        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)

        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1

        self.offset = 128

    def infer(self, x, split_bin, value):
        mix = x.detach()
        x = x.clone()

        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        aux1 = torch.cat(
            [
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:]),
            ],
            dim=2,
        )

        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        mask[:, :, : split_bin] = torch.pow(
            mask[:, :, : split_bin],
            1 + value / 3,
        )
        mask[:, :, split_bin :] = torch.pow(
            mask[:, :, split_bin :],
            1 + value,
        )

        return mask * mix

    def forward(self, x_mag, split_bin, value):
        h = self.infer(x_mag, split_bin, value)

        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]

        return h

class CascadedNet(nn.Module):
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()
        from lib.lib_v5.nets_new import BaseNet, layers_new
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            layers_new.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )

        self.stg1_high_band_net = BaseNet(
            2, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            layers_new.Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def infer(self, x):
        x = x[:, :, : self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode="replicate",
        )

        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode="replicate",
            )
            return mask, aux
        else:
            return mask

    def forward(self, x):
        mask = self.infer(x)
        pred_mag = x * mask

        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
            assert pred_mag.size()[3] > 0

        return pred_mag

for path in os.listdir("tools/uvr5/uvr5_weights/"):
    if path.endswith(".pth"):
        if "DeEcho" in path:
            nout = 64 if "DeReverb" in path else 48
            model = CascadedNet(672 * 2, nout=nout)
            static_dict = torch.load("tools/uvr5/uvr5_weights/" + path, map_location="cpu")
            model.load_state_dict(static_dict, strict=True)

            Magnitude = torch.randn(1, 2, 673, 512)
            torch.onnx.export(
                model,
                (Magnitude),
                "tools/uvr5/uvr5_weights/onnx_dereverb_" + path.replace(".pth", ".onnx"),
                input_names=["Magnitude"],
                output_names=["Output"],
                opset_version=18,
                do_constant_folding=True,
            )
        else:
            model = CascadedASPPNet(672 * 2)
            static_dict = torch.load( "tools/uvr5/uvr5_weights/" + path, map_location="cpu")
            model.load_state_dict(static_dict, strict=True)

            Magnitude = torch.randn(1, 2, 673, 512)
            SplitBin = torch.LongTensor([85])
            Value = torch.FloatTensor([0.1])
            torch.onnx.export(
                model,
                (Magnitude, SplitBin, Value),
                "tools/uvr5/uvr5_weights/onnx_dereverb_" + path.replace(".pth", ".onnx"),
                input_names=["Magnitude", "SplitBin", "Value"],
                output_names=["Output"],
                opset_version=18,
                do_constant_folding=True,
            )


