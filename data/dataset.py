import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2

from config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    COLOR_JITTER_PROB, COLOR_JITTER_BRIGHTNESS, COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION, COLOR_JITTER_HUE,
    AFFINE_PROB, HFLIP_PROB, RRC_SCALE, TTA_CROPS, TTA_SCALE,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
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
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label


train_transforms = A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=RRC_SCALE),
    A.HorizontalFlip(p=HFLIP_PROB),
    A.ColorJitter(
        brightness=COLOR_JITTER_BRIGHTNESS,
        contrast=COLOR_JITTER_CONTRAST,
        saturation=COLOR_JITTER_SATURATION,
        hue=COLOR_JITTER_HUE,
        p=COLOR_JITTER_PROB,
    ),
    A.Affine(translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=AFFINE_PROB),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

tta_transforms = A.Compose([
    A.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=TTA_SCALE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def build_loaders(data_dir, batch_size=None):
    if batch_size is None:
        batch_size = BATCH_SIZE

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    train_ds = ImageFolderDataset(train_dir, transform=train_transforms)
    val_ds = ImageFolderDataset(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader, train_ds.class_to_idx
