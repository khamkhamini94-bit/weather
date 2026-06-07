import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
import cv2
import torch

from config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_DIR,
    COLOR_JITTER_PROB, COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION, COLOR_JITTER_HUE,
    AFFINE_PROB, HFLIP_PROB, RRC_SCALE,
    RANDOM_ERASING_PROB, RANDOM_ERASING_SCALE,
    USE_WEIGHTED_SAMPLER,
    TTA_CROPS, TTA_SCALE,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class WeatherDataset(Dataset):
    """Flat dataset: loads all images from class subdirectories, no transforms.

    Used as the base dataset for k-fold splitting (labels accessible without
    loading images). Also serves as the module-level dataset class so trainer.py
    imports work.
    """

    def __init__(self, root):
        self.samples = []  # (path, label)
        self.labels = []   # parallel label list
        self.transform = None  # applied externally via TransformSubset
        root = Path(root)
        classes = sorted(d for d in root.iterdir() if d.is_dir())
        self.class_to_idx = {c.name: i for i, c in enumerate(classes)}
        self.class_names = list(self.class_to_idx.keys())
        for cls_dir in classes:
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    label = self.class_to_idx[cls_dir.name]
                    self.samples.append((str(img_path), label))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label

    def get_labels(self, indices=None):
        if indices is None:
            return self.labels
        return [self.labels[i] for i in indices]


class ImageFolderDataset(WeatherDataset):
    """WeatherDataset alias with optional built-in transform.
    Kept for backward compatibility with build_loaders / infer paths.
    """
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform


class TransformSubset(Dataset):
    """Subset with override transform — used by k-fold training."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.samples = [dataset.samples[i] for i in self.indices]
        self.labels = [dataset.labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label

    def get_labels(self, indices=None):
        if indices is None:
            return self.labels
        return [self.labels[i] for i in indices]


train_transforms = A.Compose([
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=RRC_SCALE),
    A.HorizontalFlip(p=HFLIP_PROB),
    A.ColorJitter(
        brightness=COLOR_JITTER_BRIGHTNESS,
        contrast=COLOR_JITTER_CONTRAST,
        saturation=COLOR_JITTER_SATURATION,
        hue=COLOR_JITTER_HUE,
        p=COLOR_JITTER_PROB,
    ),
    A.Affine(translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=AFFINE_PROB),
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=RANDOM_ERASING_SCALE,
        hole_width_range=RANDOM_ERASING_SCALE,
        fill=0,
        p=RANDOM_ERASING_PROB,
    ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

tta_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])  # identical to val_transforms; hflip applied manually in TTA loop


def _make_sampler(labels):
    """WeightedRandomSampler for class-balanced sampling."""
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts[labels].float()
    return WeightedRandomSampler(weights, len(weights))


def build_loaders(data_dir, batch_size=None):
    if batch_size is None:
        batch_size = BATCH_SIZE

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_ds = ImageFolderDataset(train_dir, transform=train_transforms)
    val_ds = ImageFolderDataset(val_dir, transform=val_transforms)

    sampler_args = {}
    if USE_WEIGHTED_SAMPLER:
        sampler_args = {
            "sampler": _make_sampler(train_ds.labels),
            "shuffle": False,
        }
    else:
        sampler_args = {"shuffle": True}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=NUM_WORKERS, pin_memory=True,
        **sampler_args,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader, train_ds.class_to_idx


def load_data():
    """Load train + holdout datasets for k-fold training path."""
    train_ds = WeatherDataset(DATA_DIR / "train")
    holdout_dir = DATA_DIR / "val"
    holdout_ds = WeatherDataset(holdout_dir) if holdout_dir.exists() else None
    return train_ds, holdout_ds
