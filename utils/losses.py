import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLabelSmoothing(nn.Module):
    """Focal Loss combined with Label Smoothing.

    Focal Loss down-weights easy examples so the model focuses on hard ones.
    Label Smoothing prevents overconfidence on ambiguous class boundaries.
    """

    def __init__(self, num_classes, gamma=2.0, alpha=None, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha  # tensor of per-class weights, or None
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = (
                1.0 - self.smoothing
            ) * smooth_targets + self.smoothing / self.num_classes

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # p_t: weighted probability of the (smoothed) target class
        p_t = (probs * smooth_targets).sum(dim=1)
        focal_weight = (1.0 - p_t).pow(self.gamma)

        loss = -(smooth_targets * log_probs).sum(dim=1) * focal_weight

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t

        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Standard CE with label smoothing. Use when Focal Loss is not needed."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        nll = F.cross_entropy(
            logits, targets, label_smoothing=self.smoothing
        )
        return nll
