import warnings
warnings.filterwarnings("ignore")
import copy
import math
import os
import pdb

import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from module.commons import init_weights, get_padding
from module.mrte_model import MRTE
from module.quantize import ResidualVectorQuantizer
# from text import symbols
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2
from torch.cuda.amp import autocast
import contextlib,random


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version = "v2",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.version = version

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, text, ge, speed=1,test=None):
        y_mask = torch.ones_like(y[:1,:1,:])

        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.ones_like(text).to(y.dtype).unsqueeze(0)

        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)
        
        y = self.encoder2(y * y_mask, y_mask)
        if(speed!=1):
            y = F.interpolate(y, size=int(y.shape[-1] / speed)+1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
            
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, g=None):
        if(g!=None):
            g = g.detach()
        x_mask = torch.ones_like(x)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask

class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)
        # self.wns = nn.ModuleList([weight_norm(num_features=ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)

    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0)).unsqueeze(-1)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        d = (
            torch.sum(x**2, 1, keepdim=True)
            + torch.sum(self.embedding.weight**2, 1)
            - 2 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies


class Quantizer(torch.nn.Module):
    def __init__(self, embed_dim=512, n_code_groups=4, n_codes=160):
        super(Quantizer, self).__init__()
        assert embed_dim % n_code_groups == 0
        self.quantizer_modules = nn.ModuleList(
            [
                Quantizer_module(n_codes, embed_dim // n_code_groups)
                for _ in range(n_code_groups)
            ]
        )
        self.n_code_groups = n_code_groups
        self.embed_dim = embed_dim

    def forward(self, xin):
        # B, C, T
        B, C, T = xin.shape
        xin = xin.transpose(1, 2)
        x = xin.reshape(-1, self.embed_dim)
        x = torch.split(x, self.embed_dim // self.n_code_groups, dim=-1)
        min_indicies = []
        z_q = []
        for _x, m in zip(x, self.quantizer_modules):
            _z_q, _min_indicies = m(_x)
            z_q.append(_z_q)
            min_indicies.append(_min_indicies)  # B * T,
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean(
            (z_q - xin.detach()) ** 2
        )
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        codes = torch.stack(min_indicies, -1).reshape(B, T, self.n_code_groups)
        return z_q, loss, codes.transpose(1, 2)

    def embed(self, x):
        # idx: N, 4, T
        x = x.transpose(1, 2)
        x = torch.split(x, 1, 2)
        ret = []
        for q, embed in zip(x, self.quantizer_modules):
            q = embed.embedding(q.squeeze(-1))
            ret.append(q)
        ret = torch.cat(ret, -1)
        return ret.transpose(1, 2)  # N, C, T


class CodePredictor(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        n_q=8,
        dims=1024,
        ssl_dim=768,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.vq_proj = nn.Conv1d(ssl_dim, hidden_channels, 1)
        self.ref_enc = modules.MelStyleEncoder(
            ssl_dim, style_vector_dim=hidden_channels
        )

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        self.out_proj = nn.Conv1d(hidden_channels, (n_q - 1) * dims, 1)
        self.n_q = n_q
        self.dims = dims

    def forward(self, x, x_mask, refer, codes, infer=False):
        x = x.detach()
        x = self.vq_proj(x * x_mask) * x_mask
        g = self.ref_enc(refer, x_mask)
        x = x + g
        x = self.encoder(x * x_mask, x_mask)
        x = self.out_proj(x * x_mask) * x_mask
        logits = x.reshape(x.shape[0], self.n_q - 1, self.dims, x.shape[-1]).transpose(
            2, 3
        )
        target = codes[1:].transpose(0, 1)
        if not infer:
            logits = logits.reshape(-1, self.dims)
            target = target.reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, target)
            return loss
        else:
            _, top10_preds = torch.topk(logits, 10, dim=-1)
            correct_top10 = torch.any(top10_preds == target.unsqueeze(-1), dim=-1)
            top3_acc = 100 * torch.mean(correct_top10.float()).detach().cpu().item()

            print("Top-10 Accuracy:", top3_acc, "%")

            pred_codes = torch.argmax(logits, dim=-1)
            acc = 100 * torch.mean((pred_codes == target).float()).detach().cpu().item()
            print("Top-1 Accuracy:", acc, "%")

            return pred_codes.transpose(0, 1)


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v2",
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        # self.enc_q = PosteriorEncoder(
        #     spec_channels,
        #     inter_channels,
        #     hidden_channels,
        #     5,
        #     1,
        #     16,
        #     gin_channels=gin_channels,
        # )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        # self.version=os.environ.get("version","v1")
        if self.version == "v1":
            self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        self.ssl_dim = ssl_dim
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        if freeze_quantizer:
            self.ssl_proj.requires_grad_(False)
            self.quantizer.requires_grad_(False)
            # self.enc_p.text_embedding.requires_grad_(False)
            # self.enc_p.encoder_text.requires_grad_(False)
            # self.enc_p.mrte.requires_grad_(False)

    def forward(self, codes, text, refer, noise_scale=0.5):
        refer_mask = torch.ones_like(refer[:1,:1,:])
        if (self.version == "v1"):
            ge = self.ref_enc(refer * refer_mask, refer_mask)
        else:
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            dquantized = torch.cat([quantized, quantized]).permute(1, 2, 0)
            quantized = dquantized.contiguous().view(1, self.ssl_dim, -1)

        _, m_p, logs_p, y_mask = self.enc_p(
            quantized, text, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=ge, reverse=True)
        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        _, codes, _, _ = self.quantizer(ssl)
        return codes.transpose(0, 1)


def set_no_grad(net_g):
    for name, param in net_g.named_parameters():
        param.requires_grad=False


from x_transformers.x_transformers import RotaryEmbedding


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask = None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out
    

class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


class TextEmbedding(nn.Module):
    def __init__(self, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):
        batch = text.shape[0]
        if drop_text:
            text = torch.zeros_like(text)
        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)
        return text
    

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x
    

class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)
    

class Attention(nn.Module):
    def __init__(
        self,
        processor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim = None,  # if not None -> joint attention
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(self,x,c = None,mask = None,rope=None,c_rope=None,) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)
        

from x_transformers.x_transformers import apply_rotary_pos_emb


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(self,attn: Attention,x,mask = None,rope=None,) -> torch.FloatTensor:
        batch_size = x.shape[0]

        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_mask = None
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)
        return x



class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x



class DiT(nn.Module):
    def __init__(
        self,*,dim,depth=8,heads=8,dim_head=64,dropout=0.1,ff_mult=4,mel_dim=100,text_dim=None,conv_layers=0,long_skip_connection=False,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.d_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(self, x0, cond0, time, dt_base_bootstrap, text0, drop_audio_cond=False, drop_text=False):
        x=x0.transpose(2,1)
        cond=cond0.transpose(2,1)
        text=text0.transpose(2,1)
        mask = None

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        dt = self.d_embed(dt_base_bootstrap)
        t+=dt
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)###need to change
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
    

class CFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        dit : DiT
    ):
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator : DiT = dit
        self.in_channels = in_channels
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, mu, prompt, t, d, cfg_rate):
        ##d = 1 / sample_steps
        v_pred = self.estimator(x, prompt, t, d, mu, drop_audio_cond=False,drop_text=False).transpose(2, 1)
        neg = self.estimator(x, prompt, t, d, mu, drop_audio_cond=True, drop_text=True).transpose(2, 1)
        v_pred=v_pred+(v_pred-neg)*cfg_rate
        x = x + d * v_pred
        #t = t + d
        return x

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, temperature=1.0, inference_cfg_rate=0):
        """Forward diffusion"""
        B, T = mu.size(0), mu.size(1)
        x = torch.randn([B, self.in_channels, T], device=mu.device,dtype=mu.dtype) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x,dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu=mu.transpose(2,1)
        t = 0
        d = 1 / self.sample_steps
        for j in range(self.sample_steps):
            t_tensor = torch.ones(x.shape[0], device=x.device,dtype=mu.dtype) * t
            d_tensor = torch.ones(x.shape[0], device=x.device,dtype=mu.dtype) * d
            v_pred = self.estimator(x, prompt_x, x_lens, t_tensor,d_tensor, mu, drop_audio_cond=False,drop_text=False).transpose(2, 1)
            if inference_cfg_rate>1e-5:
                neg = self.estimator(x, prompt_x, x_lens, t_tensor, d_tensor, mu, drop_audio_cond=True, drop_text=True).transpose(2, 1)
                v_pred=v_pred+(v_pred-neg)*inference_cfg_rate
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0
        return x


class SynthesizerTrnV3(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 semantic_frame_rate=None,
                 freeze_quantizer=None,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.model_dim=512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(inter_channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout)
        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ['25hz', "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == '25hz':
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(
            dimension=ssl_dim,
            n_q=1,
            bins=1024
        )
        self.freeze_quantizer=freeze_quantizer
        inter_channels2=512
        self.bridge=nn.Sequential(
            nn.Conv1d(inter_channels, inter_channels2, 1, stride=1),
            nn.LeakyReLU()
        )
        self.wns1=Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8,gin_channels=gin_channels)
        self.linear_mel=nn.Conv1d(inter_channels2,100,1,stride=1)
        self.cfm = CFM(100, DiT(**dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2, conv_layers=4)),)
        if self.freeze_quantizer==True:
            set_no_grad(self.ssl_proj)
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)

    @torch.no_grad()
    def decode_encp(self, codes, text, refer, ge=None):
        if(ge==None):
            ge = self.ref_enc(refer[:,:704], None)
        quantized = self.quantizer.decode(codes.unsqueeze(0))
        if self.semantic_frame_rate == '25hz':
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")
        x, _, _, _ = self.enc_p(quantized, text, ge)
        fea=self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")
        fea, _ = self.wns1(fea, ge)
        return fea, ge

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        _, codes, _, _ = self.quantizer(ssl)
        return codes.transpose(0, 1)


class SynthesizerTrnV3b(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 semantic_frame_rate=None,
                 freeze_quantizer=None,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.model_dim=512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(inter_channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout)
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)###Rollback
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                             upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)


        ssl_dim = 768
        assert semantic_frame_rate in ['25hz', "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == '25hz':
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(
            dimension=ssl_dim,
            n_q=1,
            bins=1024
        )
        self.freeze_quantizer=freeze_quantizer

        inter_channels2=512
        self.bridge=nn.Sequential(
            nn.Conv1d(inter_channels, inter_channels2, 1, stride=1),
            nn.LeakyReLU()
        )
        self.wns1=Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8,gin_channels=gin_channels)
        self.linear_mel=nn.Conv1d(inter_channels2,100,1,stride=1)
        self.cfm = CFM(100,DiT(**dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2, conv_layers=4)),)#text_dim is condition feature dim


    def forward(self, ssl, y, mel,ssl_lengths,y_lengths, text, text_lengths,mel_lengths):#ssl_lengths no need now
        with autocast(enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:,:704] * y_mask, y_mask)
            # ge = self.ref_enc(y * y_mask, y_mask)#change back, new spec setting is whole 24k
            # ge=None
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(
                    ssl, layers=[0]
                )
                quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")##BCT
                x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        fea=self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")##BCT
        fea, y_mask_ = self.wns1(fea, mel_lengths, ge)
        learned_mel = self.linear_mel(fea)
        B=ssl.shape[0]
        prompt_len_max = mel_lengths*2/3
        prompt_len = (torch.rand([B], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)#
        minn=min(mel.shape[-1],fea.shape[-1])
        mel=mel[:,:,:minn]
        fea=fea[:,:,:minn]
        cfm_loss= self.cfm(mel, mel_lengths, prompt_len, fea)#fea==cond,y_lengths==target_mel_lengths#ge not need
        return commit_loss,cfm_loss,F.mse_loss(learned_mel, mel),o, ids_slice, y_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), quantized

    @torch.no_grad()
    def decode_encp(self, codes,text, refer,ge=None):
        # print(2333333,refer.shape)
        # ge=None
        if(ge==None):
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:,:704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2)*2)]).to(codes.device)
        y_lengths1 = torch.LongTensor([int(codes.size(2)*2.5*1.5)]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == '25hz':
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")##BCT
        x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea=self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")##BCT
        ####more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, ge)
        return fea,ge

    def extract_latent(self, x):
        ssl =  self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0,1)

