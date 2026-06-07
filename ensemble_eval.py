"""5-fold ensemble evaluation: average softmax probabilities across folds.

Usage:
    python ensemble_eval.py                          # evaluate on data/val
    python ensemble_eval.py --on test                # evaluate on data/test if exists
    python ensemble_eval.py --weights output/weights  # custom weights dir
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, NUM_WORKERS, DATA_DIR
from data.dataset import TransformSubset, WeatherDataset, val_transforms
from models.convnext import build_model
from utils.metrics import compute_all_metrics


@torch.no_grad()
def ensemble_predict(models, loader, device):
    """Average softmax probabilities from all models."""
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Ensemble", unit="batch"):
        images = images.to(device)
        probs = []
        for model in models:
            logits = model(images)
            probs.append(F.softmax(logits, dim=1))
        avg_probs = torch.stack(probs).mean(dim=0)
        preds = avg_probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--weights", type=str, default="output/weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_dir = Path(args.weights)
    weight_paths = sorted(weights_dir.glob("fold_*/best.pth"))
    if not weight_paths:
        print(f"No fold_*/best.pth found in {weights_dir}")
        return

    # Detect num_classes from first checkpoint
    ckpt = torch.load(weight_paths[0], map_location="cpu", weights_only=True)
    neck_weight = [v for k, v in ckpt["model"].items()
                   if k.startswith("neck.") and k.endswith(".weight")][-1]
    num_classes = neck_weight.shape[0]

    # Load all fold models
    models = []
    for wp in weight_paths:
        model = build_model(num_classes=num_classes).to(device)
        ckpt = torch.load(wp, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models.append(model)
        print(f"Loaded {wp}")

    # Load dataset
    data_root = DATA_DIR / args.on
    ds = WeatherDataset(root=data_root)
    subset = TransformSubset(ds, list(range(len(ds))), val_transforms)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Ensembling {len(models)} models on {args.on} ({len(ds)} images)")

    preds, targets = ensemble_predict(models, loader, device)
    metrics = compute_all_metrics(preds, targets, ds.class_names)

    print(f"\n{'='*50}")
    print(f"Ensemble ({len(models)} folds) — {args.on}")
    print(f"{'='*50}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['weighted_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls_name, f1_val in metrics["per_class_f1"].items():
        bar = "█" * int(f1_val * 30) + "░" * (30 - int(f1_val * 30))
        print(f"    {cls_name:>12s}: {f1_val:.4f} {bar}")

    # Also show per-fold for comparison
    print(f"\n{'='*50}")
    print("Per-fold comparison:")
    print(f"{'='*50}")
    for i, wp in enumerate(weight_paths):
        model = models[i]
        all_preds, all_targets = [], []
        for images, targets in tqdm(loader, desc=f"Fold {i}", unit="batch", leave=False):
            logits = model(images.to(device))
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
        preds_fold = np.concatenate(all_preds)
        targets_fold = np.concatenate(all_targets)
        from utils.metrics import compute_f1
        print(f"  fold_{i}: Macro F1 = {compute_f1(preds_fold, targets_fold):.4f}")


if __name__ == "__main__":
    main()
