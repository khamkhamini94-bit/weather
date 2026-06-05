import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import (
    BACKBONE,
    PRETRAINED,
    LOCAL_PRETRAINED,
    STOCHASTIC_DROP_PROB,
    NUM_CLASSES,
    DROPOUT_BEFORE_NECK,
)


class GeMPool(nn.Module):
    """Generalized Mean Pooling — between GAP (p=1) and GMP (p→∞)."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        H, W = x.shape[-2:]
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (H, W)
        ).pow(1.0 / self.p)


class WeatherClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=PRETRAINED,
            num_classes=0,
            drop_path_rate=STOCHASTIC_DROP_PROB,
        )

        # Load offline pretrained weights when online download is unavailable
        if not PRETRAINED and LOCAL_PRETRAINED:
            from pathlib import Path
            ckpt_path = Path(LOCAL_PRETRAINED)
            if ckpt_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(ckpt_path))
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {ckpt_path}")
            else:
                print(f"WARNING: LOCAL_PRETRAINED={LOCAL_PRETRAINED} not found, using random init")

        # Determine backbone feature dimension from spatial features
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone.forward_features(dummy)
            in_dim = feat.shape[1]

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem = GeMPool(p=3)

        # Dual-path pooled features concatenated: GAP + GeM → classifier
        self.neck = nn.Sequential(
            nn.Dropout(DROPOUT_BEFORE_NECK),
            nn.Linear(in_dim * 2, NUM_CLASSES),
        )

        self._init_neck()

    def _init_neck(self):
        for m in self.neck.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        gap = self.gap(feat)
        gem = self.gem(feat)
        pooled = torch.cat([gap.flatten(1), gem.flatten(1)], dim=1)
        return self.neck(pooled)


def build_model(num_classes=None):
    if num_classes is not None:
        # Override the NUM_CLASSES from config
        import config
        config.NUM_CLASSES = num_classes
    return WeatherClassifier()
