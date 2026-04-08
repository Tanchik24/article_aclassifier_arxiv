from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, PreTrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg = 4.0, gamma_pos = 1.0, clip = 0.05, eps = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()

        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1.0 - prob

        if self.clip is not None and self.clip > 0:
            shifted_prob_neg = torch.clamp(prob_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(torch.clamp(prob_pos, min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(torch.clamp(shifted_prob_neg, min=self.eps))
        loss = loss_pos + loss_neg

        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = torch.pow(1.0 - (prob_pos * targets + shifted_prob_neg * (1.0 - targets)), gamma)

        loss *= focal_weight
        return -loss.mean()


class SciBertMultiLabelConfig(PreTrainedConfig):
    model_type = "scibert_multilabel"

    def __init__(self, base_model_name="allenai/scibert_scivocab_cased", num_labels=148, dropout_prob=0.3, hidden_dim=768, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim


@dataclass
class MultiLabelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class SciBertForMultiLabelClassification(PreTrainedModel):
    config_class = SciBertMultiLabelConfig

    def __init__(self, config: SciBertMultiLabelConfig):
        super().__init__(config)

        encoder_cfg = AutoConfig.from_pretrained(config.base_model_name)
        self.encoder = AutoModel.from_config(encoder_cfg)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config.dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_dim, config.num_labels)
        )

        self.loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )

        emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(emb))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return MultiLabelOutput(
            loss=loss,
            logits=logits
        )
