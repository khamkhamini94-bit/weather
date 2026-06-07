import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    LLRD_DECAY,
    LABEL_SMOOTHING,
    FOCAL_GAMMA,
    FOCAL_ALPHA,
    T_0,
    T_MULT,
    EARLY_STOP_PATIENCE,
    MIN_EPOCHS,
    MIN_DELTA,
    K_FOLDS,
    FOLD_TO_RUN,
    USE_AMP,
    EMA_DECAY,
    USE_WEIGHTED_SAMPLER,
    MODEL_DIR,
    LOG_DIR,
    OUTPUT_DIR,
)
import config  # module reference for live FOLD_TO_RUN reads
from data.dataset import (
    TransformSubset,
    WeatherDataset,
    load_data,
    train_transforms,
    val_transforms,
)
from models.convnext import build_model, get_param_groups
from utils.losses import FocalLossWithLabelSmoothing
from utils.metrics import AverageMeter, ModelEMA, compute_all_metrics, compute_f1


@torch.no_grad()
def validate(model, loader, criterion, device, desc="Val"):
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        losses.update(loss.item(), images.size(0))

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1 = compute_f1(all_preds, all_targets)
    return losses.avg, f1


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    losses = AverageMeter()

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)

        if USE_AMP:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def run_fold(fold_idx, train_idx, val_idx, dataset, holdout_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}\nFold {fold_idx + 1}\n{'='*50}")

    train_ds = TransformSubset(dataset, train_idx, train_transforms)
    val_ds = TransformSubset(dataset, val_idx, val_transforms)

    sampler_args = {}
    if USE_WEIGHTED_SAMPLER:
        class_counts = np.bincount(train_ds.labels)
        weights = 1.0 / class_counts[train_ds.labels]
        sampler_args = {
            "sampler": WeightedRandomSampler(torch.tensor(weights), len(weights)),
            "shuffle": False,
        }
    else:
        sampler_args = {"shuffle": True}

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        **sampler_args,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    num_classes = len(dataset.class_names)
    model = build_model(num_classes=num_classes).to(device)
    ema = ModelEMA(model, decay=EMA_DECAY)

    criterion = FocalLossWithLabelSmoothing(
        num_classes=model.neck[-1].out_features,
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA,
        smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        get_param_groups(model, base_lr=LR, decay=LLRD_DECAY),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT
    )
    scaler = GradScaler() if USE_AMP else None

    # Early stopping uses holdout when available; fold-val is for monitoring only
    early_stop_loader = holdout_loader if holdout_loader is not None else val_loader
    early_stop_name = "holdout" if holdout_loader is not None else "val"

    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [],
               "holdout_loss": [], "holdout_f1": []}

    fold_dir = MODEL_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        ema.update()

        # Always evaluate on fold-val for monitoring
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Evaluate on holdout for early stopping (if provided)
        if holdout_loader is not None:
            holdout_loss, holdout_f1 = validate(model, holdout_loader, criterion, device, desc="Holdout")
            history["holdout_loss"].append(holdout_loss)
            history["holdout_f1"].append(holdout_f1)
            es_loss, es_f1 = holdout_loss, holdout_f1
        else:
            es_loss, es_f1 = val_loss, val_f1

        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        parts = [
            f"Epoch {epoch:3d}/{EPOCHS}",
            f"lr {lr:.2e}",
            f"train_loss {train_loss:.4f}",
            f"val_loss {val_loss:.4f}",
            f"val_f1 {val_f1:.4f}",
        ]
        if holdout_loader is not None:
            parts.append(f"holdout_loss {holdout_loss:.4f}")
            parts.append(f"holdout_f1 {holdout_f1:.4f}")
        parts.append(f"time {elapsed:.0f}s")
        print(" | ".join(parts))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Always save latest
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "f1": es_f1},
            fold_dir / "last.pth",
        )

        if es_f1 - best_f1 > MIN_DELTA:
            best_f1 = es_f1
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "f1": es_f1},
                fold_dir / "best.pth",
            )
        else:
            patience_counter += 1

        if epoch >= MIN_EPOCHS and patience_counter >= EARLY_STOP_PATIENCE:
            best_record = max(history[f"{early_stop_name}_f1"])
            print(
                f"Early stopping at epoch {epoch} — "
                f"no improvement > {MIN_DELTA:.4f} for {EARLY_STOP_PATIENCE} epochs, "
                f"best {early_stop_name} F1: {best_record:.4f}"
            )
            break

    # Save EMA weights
    ema.apply_shadow()
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "f1": val_f1, "ema": True},
        fold_dir / "ema.pth",
    )
    ema.restore()

    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Fold {fold_idx + 1} best F1: {best_f1:.4f}")
    return best_f1


def run_train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_ds, holdout_ds = load_data()

    print(
        f"Train samples: {len(train_ds)}, "
        f"Holdout: {len(holdout_ds) if holdout_ds else 0}, "
        f"Classes: {len(train_ds.class_names)}"
    )

    # Build holdout loader for early stopping (from data/val)
    holdout_loader = None
    if holdout_ds is not None:
        holdout_subset = TransformSubset(
            holdout_ds, list(range(len(holdout_ds))), val_transforms
        )
        holdout_loader = DataLoader(
            holdout_subset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    # Collect targets for stratified split
    all_targets = [train_ds[i][1] for i in range(len(train_ds))]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    folds_to_run = range(K_FOLDS) if config.FOLD_TO_RUN == -1 else [config.FOLD_TO_RUN]

    f1_scores = []
    for fold in folds_to_run:
        for fold_i, (tr_idx, val_idx) in enumerate(
            skf.split(range(len(train_ds)), all_targets)
        ):
            if fold_i != fold:
                continue
            f1 = run_fold(fold, tr_idx, val_idx, train_ds, holdout_loader)
            f1_scores.append(f1)

    # ── Final test on external test set ─────────────────────
    test_ds = _load_test_dataset()
    if test_ds is not None:
        print("\n" + "=" * 50)
        print("Final evaluation on test set")
        print("=" * 50)
        evaluate_on_test(test_ds)
    elif holdout_ds is not None:
        print("\n" + "=" * 50)
        print("Final evaluation on holdout set (no test set found)")
        print("=" * 50)
        evaluate_on_test(holdout_ds)

    if len(f1_scores) > 1:
        print(
            f"\nKFold F1: mean={np.mean(f1_scores):.4f}, "
            f"std={np.std(f1_scores):.4f}"
        )
        print(f"Per-fold: {f1_scores}")


def _load_test_dataset():
    """Load an external test dataset if available."""
    from config import TEST_DIR
    from pathlib import Path

    test_root = Path(TEST_DIR) if isinstance(TEST_DIR, str) else TEST_DIR
    if test_root.exists() and any(test_root.iterdir()):
        print(f"Loading test dataset from {test_root}")
        return WeatherDataset(root=test_root)
    return None


@torch.no_grad()
def evaluate_on_test(test_ds):
    """Evaluate the best fold model on a test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_weights = sorted(MODEL_DIR.glob("fold_*/best.pth"))
    if not best_weights:
        print("  No trained models found, skipping evaluation")
        return

    test_subset = TransformSubset(test_ds, list(range(len(test_ds))), val_transforms)
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    num_classes = len(test_ds.class_names)
    class_names = test_ds.class_names

    for weight_path in best_weights:
        model = build_model(num_classes=num_classes).to(device)
        ckpt = torch.load(weight_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        all_preds, all_targets = [], []
        for images, targets in tqdm(test_loader, desc="Test", leave=False):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = compute_all_metrics(all_preds, all_targets, class_names)
        fold_name = weight_path.parent.name

        print(f"\n  [{fold_name}]")
        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    Macro F1:     {metrics['macro_f1']:.4f}")
        print(f"    Weighted F1:  {metrics['weighted_f1']:.4f}")
        print(f"    Per-class F1:")
        for cls_name, f1_val in metrics["per_class_f1"].items():
            bar = "█" * int(f1_val * 30) + "░" * (30 - int(f1_val * 30))
            print(f"      {cls_name:>12s}: {f1_val:.4f} {bar}")
