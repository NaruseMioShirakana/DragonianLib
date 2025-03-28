import torch
from torch import nn
from torch.nn import functional as F

import onnxexport.attentions as attentions
import onnxexport.commons as commons
import onnxexport.modules as modules
from onnxexport.utils import f0_to_coarse

def normalize_f0(f0, uv, random_scale=True):
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum
    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    return f0_norm

class ResidualCouplingBlock(nn.Module):
    def __init__(
            self, channels, hidden_channels, kernel_size, dilation_rate,
            n_layers, n_flows=4, gin_channels=0, share_parameter=False
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
        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                    gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, g=None, reverse=False, self_attn_mask = None):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, g=g, reverse=reverse)
        return x

class TransformerCouplingBlock(nn.Module):
    def __init__(
            self, channels, hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout, n_flows=4,gin_channels=0,share_parameter=False
        ): 
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = nn.ModuleList()
        self.wn = attentions.FFT(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout,
            isflow = True, gin_channels = self.gin_channels
        ) if share_parameter else None
        for i in range(n_flows):
            self.flows.append(
                modules.TransformerCouplingLayer(
                    channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout,
                    filter_channels, mean_only=True, wn_sharing_parameter=self.wn,
                    gin_channels = self.gin_channels)
                )
            self.flows.append(modules.Flip())

    def forward(self, x, g=None, reverse=False, self_attn_mask = None):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, g=g, self_attn_mask=self_attn_mask, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, g=g, self_attn_mask=self_attn_mask, reverse=reverse)
        return x


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
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, g=None):
        x = self.pre(x)
        x = self.enc(x, g=g)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs


class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, f0=None, z=None):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + z * torch.exp(logs))
        return z, m, logs


class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x)
        x = self.decoder(x)
        x = self.proj(x)
        return x


class SynthesizerTrn(nn.Module):
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
                 gin_channels,
                 ssl_dim,
                 n_speakers,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name = "nsf-hifigan",
                 use_depthwise_conv = False,
                 use_automatic_f0_prediction = True,
                 flow_share_parameter = False,
                 n_flow_layer = 4,
                 n_layers_trans_flow = 3,
                 use_transformer_flow = False,
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
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.n_speaker = n_speakers
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv":use_depthwise_conv
        }
        
        modules.set_Conv1dModel(self.use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)
        else:
            print("[?] Unkown vocoder: use default(nsf-hifigan)")
            from vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(inter_channels, hidden_channels, filter_channels, n_heads, n_layers_trans_flow, 5, p_dropout, n_flow_layer, gin_channels=gin_channels, share_parameter=flow_share_parameter)
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter=flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.predict_f0 = False
        self.export_mix = False

    def export_chara_mix(self, speakers_mix):
        speaker_map = torch.zeros((len(speakers_mix), 1, 1, self.gin_channels))
        i = 0
        for key in speakers_mix.keys():
            spkidx = speakers_mix[key]
            speaker_map[i] = self.emb_g(torch.LongTensor([[spkidx]]))
            i = i + 1
        speaker_map = speaker_map.unsqueeze(0)
        self.register_buffer("speaker_map", speaker_map)
        self.export_mix = True

    def forward(self, c, f0, mel2ph, uv, noise=None, g=None, vol = None):
        decoder_inp = F.pad(c, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, c.shape[-1]])
        c = torch.gather(decoder_inp, 1, mel2ph_).transpose(1, 2)  # [B, T, H]
        
        if self.export_mix:   # [B, N, S]  *  [1, S, 1, 1, H]
            g = g.permute(2, 0, 1)  # [S, B, N]
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2], 1))  # [1, S, B, N, 1]
            g = g * self.speaker_map  # [1, S, B, N, H]
            g = torch.sum(g, dim=1) # [1, B, N, H]
            g = g.squeeze(0).transpose(1, 2) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)

        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0

        x = self.pre(c) + self.emb_uv(uv.long()).transpose(1, 2) + vol

        if self.use_automatic_f0_prediction and self.predict_f0:
            lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
            norm_lf0 = normalize_f0(lf0, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        self_attn_mask = commons.subsequent_mask(x.size(2)).to(device=x.device, dtype=x.dtype)
        z_p, _, _ = self.enc_p(x, f0=f0_to_coarse(f0), z=noise)
        z = self.flow(z_p, g=g, self_attn_mask=self_attn_mask, reverse=True)
        o = self.dec(z, g=g, f0=f0)
        return o

