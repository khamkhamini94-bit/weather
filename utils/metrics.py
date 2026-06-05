import copy
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters()}
        self.backup = {}

    def update(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """Apply EMA weights to model (call before eval)."""
        self.backup = {
            name: p.data.clone() for name, p in self.model.named_parameters()
        }
        for name, p in self.model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights."""
        for name, p in self.model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = {}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_f1(preds, targets, average="macro"):
    """Compute macro F1 score."""
    return f1_score(targets, preds, average=average, zero_division=0)


def compute_all_metrics(preds, targets, class_names=None):
    """Return dict with accuracy, macro/weighted F1, per-class F1, confusion matrix."""
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {name: float(f1) for name, f1 in zip(class_names or [], per_class_f1)},
        "confusion_matrix": cm.tolist(),
    }
    return metrics
