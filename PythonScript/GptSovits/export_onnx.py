import os
import json
import onnx
import torch
import onnxsim

from torch.nn import Module
from feature_extractor import cnhubert
from onnxruntime import InferenceSession
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForMaskedLM
import AR.models.t2s_model_onnx as t2s

from module.models_onnx import SynthesizerTrn, SynthesizerTrnV3


class LinearSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.return_complex = False

        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=self.return_complex,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        return spec


class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        from librosa.filters import mel as librosa_mel_fn
        mel_basis = torch.from_numpy(librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=self.f_min, fmax=self.f_max))
        self.register_buffer(
            "mel_basis",
            mel_basis,
            persistent=False
        )

    def forward(
        self, x
    ):
        linear = self.spectrogram(x)
        spec = torch.matmul(self.mel_basis, linear)
        return torch.log(torch.clamp(spec, min=1e-5))


root_path = os.path.dirname(os.path.abspath(__file__))
onnx_path = os.path.join(root_path, "onnx")
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

class BertWrapper(Module):
    def __init__(self):
        bert_path = os.environ.get(
            "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        )
        super(BertWrapper, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

    def forward(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        res = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        return torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
    def export_onnx(self):
        vocab_dict = { k: v for k, v in self.tokenizer.get_vocab().items() }
        vocab_path = os.path.join(onnx_path, "Vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab_dict, f, indent=4)
        dummy_input = torch.randint(0, 100, (1, 20)).long()
        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(onnx_path, "Bert.onnx"),
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "Bert.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "Bert.onnx"))
        print("Exported BERT to ONNX format.")


class CnHubertWrapper(Module):
    def __init__(self):
        super(CnHubertWrapper, self).__init__()
        cnhubert_base_path = os.environ.get(
            "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        )
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.model = cnhubert.get_model().model

    def forward(self, signal):
        return self.model(signal)["last_hidden_state"]
    
    def export_onnx(self):
        dummy_input = torch.randn(1, 16000 * 10)
        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(onnx_path, "CnHubert.onnx"),
            input_names=["signal"],
            output_names=["output"],
            dynamic_axes={"signal": {0: "batch_size", 1: "sequence_length"}},
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "CnHubert.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "CnHubert.onnx"))
        print("Exported CN-Hubert to ONNX format.")


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, path, top_k=20, cache_size=2000):
        super().__init__()
        dict_s1 = torch.load(path, map_location="cpu")
        config = dict_s1["config"]
        self.model = t2s.Text2SemanticDecoder(config=config)
        self.load_state_dict(dict_s1["weight"])
        self.cache_size = cache_size
        self.top_k = top_k

def export_ar(path, top_k=20, cache_size=2000):
    model_l = Text2SemanticLightningModule(path, top_k=top_k, cache_size=cache_size)
    model = model_l.model

    x = torch.randint(0, 100, (1, 20)).long()
    x_len = torch.tensor([20]).long()
    y = torch.randint(0, 100, (1, 20)).long()
    y_len = torch.tensor([20]).long()
    bert_feature = torch.randn(1, 20, 1024)
    top_p = torch.tensor([0.8])
    repetition_penalty = torch.tensor([1.35])
    temperature = torch.tensor([0.6])

    prompt_processor = t2s.PromptProcessor(cache_len=cache_size, model=model, top_k=top_k)
    decode_next_token = t2s.DecodeNextToken(cache_len=cache_size, model=model, top_k=top_k)

    torch.onnx.export(
        prompt_processor,
        (x, x_len, y, y_len, bert_feature, top_p, repetition_penalty, temperature),
        os.path.join(onnx_path, "PromptProcessor.onnx"),
        input_names=["x", "x_len", "y", "y_len", "bert_feature", "top_p", "repetition_penalty", "temperature"],
        output_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "samples"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "sequence_length"},
            "y": {0: "batch_size", 1: "sequence_length"},
            "bert_feature": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=18,
    )

    sim, _ = onnxsim.simplify(os.path.join(onnx_path, "PromptProcessor.onnx"))
    onnx.save_model(sim, os.path.join(onnx_path, "PromptProcessor.onnx"))

    y, k_cache, v_cache, xy_pos, y_idx, samples = prompt_processor(
        x, x_len, y, y_len, bert_feature, top_p, repetition_penalty, temperature
    )

    torch.onnx.export(
        decode_next_token,
        (y, k_cache, v_cache, xy_pos, y_idx, top_p, repetition_penalty, temperature),
        os.path.join(onnx_path, "DecodeNextToken.onnx"),
        input_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "top_p", "repetition_penalty", "temperature"],
        output_names=["y", "k_cache", "v_cache", "xy_pos", "y_idx", "samples"],
        dynamic_axes={
            "y": {0: "batch_size", 1: "sequence_length"},
            "k_cache": {1: "batch_size", 2: "sequence_length"},
            "v_cache": {1: "batch_size", 2: "sequence_length"},
        },
        opset_version=18
    )

    sim, _ = onnxsim.simplify(os.path.join(onnx_path, "DecodeNextToken.onnx"))
    onnx.save_model(sim, os.path.join(onnx_path, "DecodeNextToken.onnx"))


from io import BytesIO
def load_sovits_new(sovits_path):
    f=open(sovits_path,"rb")
    meta=f.read(2)
    if meta!="PK":
        data = b'PK' + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path,map_location="cpu", weights_only=False)


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class Extractor(Module):
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.extract_latent(x.transpose(1, 2))


from peft import LoraConfig, get_peft_model


head2version={
    b'00':["v1","v1",False],
    b'01':["v2","v2",False],
    b'02':["v2","v3",False],
    b'03':["v2","v3",True],
}
hash_pretrained_dict={
    "dc3c97e17592963677a4a1681f30c653":["v2","v2",False],#s2G488k.pth#sovits_v1_pretrained
    "43797be674a37c1c83ee81081941ed0f":["v2","v3",False],#s2Gv3.pth#sovits_v3_pretrained
    "6642b37f3dbb1f76882b69937c95a5f3":["v2","v2",False],#s2G2333K.pth#sovits_v2_pretrained
}
import hashlib
def get_hash_from_file(sovits_path):
    with open(sovits_path,"rb")as f:data=f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()
def get_sovits_version_from_path_fast(sovits_path):
    ###1-if it is pretrained sovits models, by hash
    hash=get_hash_from_file(sovits_path)
    if hash in hash_pretrained_dict:
        return hash_pretrained_dict[hash]
    ###2-new weights or old weights, by head
    with open(sovits_path,"rb")as f:version=f.read(2)
    if version!=b"PK":
        return head2version[version]
    ###3-old weights, by file size
    if_lora_v3=False
    size=os.path.getsize(sovits_path)
    '''
            v1weights:about 82942KB
                half thr:82978KB
            v2weights:about 83014KB
            v3weights:about 750MB
    '''
    if size < 82978 * 1024:
        model_version = version = "v1"
    elif size < 700 * 1024 * 1024:
        model_version = version = "v2"
    else:
        version = "v2"
        model_version = "v3"
    return version,model_version,if_lora_v3

def load_sovits_new(sovits_path):
    f=open(sovits_path,"rb")
    meta=f.read(2)
    if meta!="PK":
        data = b'PK' + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path,map_location="cpu", weights_only=False)


def norm_spec(x):
    spec_min = -12
    spec_max = 2
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


from BigVGAN import bigvgan


class BigVGAN(Module):
    def __init__(self, path):
        super(BigVGAN, self).__init__()
        self.model = bigvgan.BigVGAN.from_pretrained(path, use_cuda_kernel=False)

    def forward(self, x):
        return self.model(x)
    
    def export(self):
        dummy_input = torch.randn(1, 100, 20)
        self(dummy_input)
        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(onnx_path, "BigVGAN.onnx"),
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={"x": {0: "batch_size", 2: "sequence_length"}},
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "BigVGAN.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "BigVGAN.onnx"))
        print("Exported BigVGAN to ONNX format.")


class GSV(Module):
    def __init__(self, path):
        super(GSV, self).__init__()
        version, model_version, if_lora_v3=get_sovits_version_from_path_fast(path)
        dict_s2 = load_sovits_new(path)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if 'enc_p.text_embedding.weight'not in dict_s2['weight']:
            hps.model.version = "v2"#v3model,v2sybomls
        elif dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        #version=hps.model.version
        self.version = model_version
        # print("sovits版本:",hps.model.version)
        if model_version!="v3":
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model
            )
            model_version=version
        else:
            vq_model = SynthesizerTrnV3(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model
            )
        if if_lora_v3==False:
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            vq_model.load_state_dict(load_sovits_new("GPT_SoVITS/pretrained_models/s2Gv3.pth")["weight"])
            lora_rank=dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )
            vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
            print("loading sovits_v3_lora%s"%(lora_rank))
            vq_model.load_state_dict(dict_s2["weight"], strict=False)
            vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()
        self.vq_model = vq_model
        self.hps = hps
        self.ext = Extractor(self.vq_model)
        self.mel_fn = LogMelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=100,
            center=False,
            f_min=0,
            f_max=None,
        )
        self.spec_fn = LinearSpectrogram(
            n_fft=self.hps.data.filter_length,
            win_length=self.hps.data.win_length,
            hop_length=self.hps.data.hop_length,
            center=False,
        )

    def forward(self, text_seq, pred_semantic, ref_audio, ref_phoneme=None, prompt=None):
        refer = self.spec_fn(
            ref_audio,
        )
        if self.version == "v3":
            fea_ref, ge = self.vq_model.decode_encp(prompt, ref_phoneme, refer)
            ref_audio=torch.nn.functional.interpolate(
                ref_audio.unsqueeze(0), scale_factor=24000./self.hps.data.sampling_rate, mode="linear", align_corners=False
            ).squeeze(0)
            mel2 = self.mel_fn(
                ref_audio,
            )
            mel2 = norm_spec(mel2)
            fea_todo, ge = self.vq_model.decode_encp(pred_semantic, text_seq, refer, ge)
            return fea_ref, fea_todo, mel2
        else:
            return self.vq_model(pred_semantic.unsqueeze(0), text_seq, refer)[0, 0]
    
    def export(self):
        test_seq = torch.randint(0, 100, (1, 20)).long()
        pred_semantic = torch.randint(0, 100, (1, 20)).long()
        ref_audio = torch.randn(1, 16000 * 10)
        ref_seq = torch.randint(0, 100, (1, 20)).long()
        ref_prompt = torch.randint(0, 100, (1, 20)).long()
        if self.version != "v3":
            torch.onnx.export(
                self,
                (test_seq, pred_semantic, ref_audio),
                os.path.join(onnx_path, "GptSoVits.onnx"),
                input_names=["text_seq", "pred_semantic", "ref_audio"],
                output_names=["output"],
                dynamic_axes={
                    "text_seq": {0: "batch_size", 1: "sequence_length"},
                    "pred_semantic": {0: "batch_size", 1: "sequence_length"},
                    "ref_audio": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=18,
            )
        else:
            raise RuntimeError("v3 model export is not supported yet, because BigVGAN has performance issue.")
            torch.onnx.export(
                self,
                (test_seq, pred_semantic, ref_audio, ref_seq, ref_prompt),
                os.path.join(onnx_path, "GptSoVits.onnx"),
                input_names=["text_seq", "pred_semantic", "ref_audio", "ref_seq", "prompt"],
                output_names=["fea_ref", "fea_todo", "mel2"],
                dynamic_axes={
                    "text_seq": {0: "batch_size", 1: "sequence_length"},
                    "pred_semantic": {0: "batch_size", 1: "sequence_length"},
                    "ref_audio": {0: "batch_size", 1: "sequence_length"},
                    "ref_seq": {0: "batch_size", 1: "sequence_length"},
                    "prompt": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=18,
            )

            fea_ref, fea_todo, mel2 = self(
                test_seq, pred_semantic, ref_audio, ref_seq, ref_prompt
            )

            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            if (T_min > 468):
                mel2 = mel2[:, :, -468:]
                fea_ref = fea_ref[:, :, -468:]
                T_min = 468
            chunk_len = 934 - T_min
            idx = 0
            fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2)

            t = torch.FloatTensor([0.0])
            d = torch.FloatTensor([0.125])
            temperature = torch.FloatTensor([0.8])
            cfg_rate = torch.FloatTensor([0.0])
            
            B, T = fea.size(0), fea.size(2)
            x = torch.randn([B, self.vq_model.cfm.in_channels, T], device=fea.device,dtype=fea.dtype) * temperature
            prompt = torch.nn.functional.pad(mel2, (0, fea.shape[2] - mel2.shape[2]), mode="constant", value=0.0)
            self.vq_model.cfm.forward(x, fea, prompt, t, d, cfg_rate)

            torch.onnx.export(
                self.vq_model.cfm,
                (x, fea, prompt, t, d, cfg_rate),
                os.path.join(onnx_path, "GptSoVits_cfm.onnx"),
                input_names=["x", "fea", "mel2", "t", "d", "cfg_rate"],
                output_names=["output"],
                dynamic_axes={
                    "x": {0: "batch_size", 2: "sequence_length"},
                    "fea": {0: "batch_size", 2: "sequence_length"},
                    "mel2": {0: "batch_size", 2: "sequence_length"},
                    "t": {0: "batch_size"},
                    "d": {0: "batch_size"},
                    "cfg_rate": {0: "batch_size"},
                },
                opset_version=18,
            )

        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "GptSoVits.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "GptSoVits.onnx"))

        ref_units = torch.randn(1, 20, 768)
        torch.onnx.export(
            self.ext,
            ref_units,
            os.path.join(onnx_path, "Extractor.onnx"),
            input_names=["ref_units"],
            output_names=["output"],
            dynamic_axes={
                "ref_units": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=18,
        )
        sim, _ = onnxsim.simplify(os.path.join(onnx_path, "Extractor.onnx"))
        onnx.save_model(sim, os.path.join(onnx_path, "Extractor.onnx"))
        print("Exported GptSoVits to ONNX format.")

        
if __name__ == "__main__":
    #CnHubertWrapper().export_onnx()

    #BertWrapper().export_onnx()

    '''
    # Conv block and tConv block in UpSample1d, DownSample1d of BigVGAN activations has a dynamic shape, onnx export should have a static shape, if you want to export it, please modify the code in BigVGAN/alias_free_activation/torch/resample.py and BigVGAN/alias_free_activation/torch/act.py, eg

    class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, C=None): <----------------
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        --------------->  self.register_buffer("filter", filter.expand(C, -1, -1), persistent=False)
        --------------->  self.filter = self.filter.contiguous()
        --------------->  self.C = C

    def forward(self, x):

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter, stride=self.stride, groups=self.C     <------------------------------
        )
        x = x[..., self.pad_left : -self.pad_right]

        return x.contiguous()

    you need to set the shape of filter in __init__ function, if the shape of filter is not fixed, the onnx export will fail.

    BigVGAN(
        "Path"
    ).export()

    BigVGAN also has a great performance issue, propose to use pytorch. 

    '''

    GSV(
        "D:\\VSGIT\\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\t\\SoVITS_weights\\小特.pth"
        #"D:\\VSGIT\\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\pretrained_models\\s2Gv3.pth"
    ).export()

    export_ar(
        "D:\\VSGIT\\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\t\\GPT_weights\\小特.ckpt",
        #"D:\\VSGIT\\GPT-SoVITS-main\\GPT_SoVITS\\GPT-SoVITS-v3lora-20250228\\GPT_SoVITS\\pretrained_models\\s1v3.ckpt",
        top_k=10,
        cache_size=1500,
    )
    
