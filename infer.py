"""Weather Recognition — TTA Inference.

Usage:
    python infer.py --weights output/weights/fold_0/best.pth --input test_images/ --output preds.csv
    python infer.py --weights output/weights/fold_0/ema.pth --input test_images/ --tta 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import BATCH_SIZE, NUM_WORKERS, TTA_CROPS, IMAGE_SIZE
from data.dataset import val_transforms, tta_transforms
from models.convnext import build_model


class InferenceDataset(Dataset):
    """Reads images from a flat directory — no labels required."""

    def __init__(self, root, transform=None):
        import cv2

        self.samples = []
        self.transform = transform
        for img_path in sorted(Path(root).iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                self.samples.append(str(img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2

        path = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, path


@torch.no_grad()
def predict_single(model, loader, device):
    model.eval()
    all_paths = []
    all_probs = []

    for images, paths in tqdm(loader, desc="Infer"):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_paths.extend(paths)
        all_probs.append(probs.cpu().numpy())

    return all_paths, np.concatenate(all_probs)


@torch.no_grad()
def predict_tta(model, loader, device, n_crops):
    """TTA: average predictions over N random crops and flips."""
    model.eval()
    all_paths = []
    all_probs = []

    for images, paths in tqdm(loader, desc="Infer TTA"):
        images = images.to(device)
        batch_probs = []
        for _ in range(n_crops):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            batch_probs.append(probs)
        avg_probs = torch.stack(batch_probs).mean(dim=0)
        all_paths.extend(paths)
        all_probs.append(avg_probs.cpu().numpy())

    return all_paths, np.concatenate(all_probs)


def detect_num_classes(weights_path):
    """Read num_classes from checkpoint weight shape without loading full model."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    # The classifier is the last Linear layer in neck — find it dynamically
    neck_weights = [v for k, v in ckpt["model"].items() if k.startswith("neck.") and k.endswith(".weight")]
    return neck_weights[-1].shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="preds.csv")
    parser.add_argument("--tta", type=int, default=TTA_CROPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = detect_num_classes(args.weights)
    print(f"Detected {num_classes} classes from checkpoint")

    model = build_model(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded weights from {args.weights} (F1: {checkpoint.get('f1', 'N/A')})")

    if args.tta > 1:
        ds = InferenceDataset(args.input, transform=tta_transforms)
        n_tta = args.tta
    else:
        ds = InferenceDataset(args.input, transform=val_transforms)
        n_tta = 1

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    if n_tta > 1:
        paths, probs = predict_tta(model, loader, device, n_tta)
    else:
        paths, probs = predict_single(model, loader, device)

    preds = probs.argmax(axis=1)

    df = pd.DataFrame({"path": paths, "prediction": preds})
    for i in range(probs.shape[1]):
        df[f"prob_{i}"] = probs[:, i]
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
