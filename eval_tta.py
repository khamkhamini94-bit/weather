"""TTA evaluation: original + horizontal flip → average.

Usage:
    python eval_tta.py --weights output/weights/fold_0/best.pth
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import BATCH_SIZE, DATA_DIR, NUM_CLASSES, IMAGE_SIZE
from data.dataset import val_transforms
from models.convnext import build_model


class RawDataset(Dataset):
    """Returns resized uint8 images — transforms applied in TTA loop."""

    def __init__(self, root):
        self.samples = []
        root = Path(root)
        classes = sorted(d for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {c.name: i for i, c in enumerate(classes)}
        for cls_dir in classes:
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((str(img_path), self.class_to_idx[cls_dir.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return img, label, path


@torch.no_grad()
def eval_tta(model, dataset, device):
    """TTA: original + horizontal flip, take softmax mean."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    for images_np, targets, _ in tqdm(loader, desc="TTA (orig+hflip)", unit="batch"):
        B = len(targets)

        # Original
        batch_orig = []
        for i in range(B):
            aug = val_transforms(image=images_np[i].numpy())
            batch_orig.append(aug["image"])
        batch_orig = torch.stack(batch_orig).to(device)

        # Horizontally flipped
        batch_flip = []
        for i in range(B):
            flipped = cv2.flip(images_np[i].numpy(), 1)
            aug = val_transforms(image=flipped)
            batch_flip.append(aug["image"])
        batch_flip = torch.stack(batch_flip).to(device)

        probs_all = []
        for batch_t in [batch_orig, batch_flip]:
            logits = model(batch_t)
            probs_all.append(F.softmax(logits, dim=1))

        avg_probs = torch.stack(probs_all).mean(dim=0)
        preds = avg_probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--on", type=str, default="val", choices=["val", "train"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.weights} (F1: {ckpt.get('f1', 'N/A')})")

    root = DATA_DIR / args.on
    ds = RawDataset(root)
    print(f"Evaluating on {args.on}: {len(ds)} images")

    # ── TTA ──
    preds_tta, targets = eval_tta(model, ds, device)

    # ── No-TTA baseline ──
    from data.dataset import ImageFolderDataset
    ds_notta = ImageFolderDataset(root, transform=val_transforms)
    loader = DataLoader(ds_notta, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds_nt = []
    all_targets_nt = []
    for images, labels in tqdm(loader, desc="No TTA ", unit="batch"):
        logits = model(images.to(device))
        all_preds_nt.append(logits.argmax(dim=1).cpu().numpy())
        all_targets_nt.append(labels.numpy())
    preds_nt = np.concatenate(all_preds_nt)
    targets_nt = np.concatenate(all_targets_nt)

    from utils.metrics import compute_all_metrics, compute_f1
    from data.dataset import WeatherDataset

    class_names = WeatherDataset(root).class_names

    f1_nt = compute_f1(preds_nt, targets_nt)
    f1_tta = compute_f1(preds_tta, targets)
    gain = f1_tta - f1_nt

    print(f"\n{'='*50}")
    print(f"No TTA   F1: {f1_nt:.4f}")
    print(f"TTA      F1: {f1_tta:.4f}  (orig + hflip)")
    print(f"Gain:        +{gain:.4f}")
    print(f"{'='*50}")

    metrics_tta = compute_all_metrics(preds_tta, targets, class_names)
    print(f"\nPer-class F1 (TTA):")
    for cls_name, f1_val in metrics_tta["per_class_f1"].items():
        bar = "█" * int(f1_val * 30) + "░" * (30 - int(f1_val * 30))
        print(f"  {cls_name:>12s}: {f1_val:.4f} {bar}")


if __name__ == "__main__":
    main()
