"""Split raw class folders into data/train/ and data/val/ with 80/20 stratified split."""
import shutil
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data"
TRAIN = DATA / "train"
VAL = DATA / "val"
SEED = 42

# Source class folders → target class names
CLASS_MAP = {
    "Cloudy": "cloudy",
    "Rain": "rainy",
    "Shine": "sunny",
    "snow": "snowy",
}

random.seed(SEED)

for src_name, tgt_name in CLASS_MAP.items():
    src_dir = DATA / src_name
    if not src_dir.is_dir():
        print(f"SKIP: {src_dir} not found")
        continue

    images = list(src_dir.glob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    labels = [tgt_name] * len(images)

    train_imgs, val_imgs = train_test_split(
        images, test_size=0.2, random_state=SEED, stratify=labels,
    )

    (TRAIN / tgt_name).mkdir(parents=True, exist_ok=True)
    (VAL / tgt_name).mkdir(parents=True, exist_ok=True)

    for p in train_imgs:
        shutil.copy2(p, TRAIN / tgt_name / p.name)
    for p in val_imgs:
        shutil.copy2(p, VAL / tgt_name / p.name)

    print(f"{src_name} → {tgt_name}: train={len(train_imgs)}, val={len(val_imgs)}")

print("\nDone. Original files preserved. Remove them manually when ready.")
