from fairseq.checkpoint_utils import load_model_ensemble_and_task
import torch



class ContentVec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        models, saved_cfg, task = load_model_ensemble_and_task(
          ["model/checkpoint_best_legacy_500.pt"],
          suffix="",
        )
        model = models[0]
        model.eval()

    def forward(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
          "source": feats.to(wav.device),
          "padding_mask": padding_mask.to(wav.device),
          "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = self.model.final_proj(logits[0])
        return feats.transpose(1, 2)
    
model = ContentVec()

model(torch.randn(16000))
