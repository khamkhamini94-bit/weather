import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "weights"
LOG_DIR = OUTPUT_DIR / "logs"

# ── Model ───────────────────────────────────────────
BACKBONE = "convnext_tiny"
PRETRAINED = False  # set to False when using offline weights
LOCAL_PRETRAINED = "pretrained/model.safetensors"  # local path to downloaded weights
STOCHASTIC_DROP_PROB = 0.15
NUM_CLASSES = 4
DROPOUT_BEFORE_NECK = 0.3

# ── Input ───────────────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4

# ── Augmentation ────────────────────────────────────
# Weather simulation augmentations REMOVED — label-destructive for 4-class (sun/cloud/rain/snow)
COLOR_JITTER_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.1
COLOR_JITTER_HUE = 0.05
AFFINE_PROB = 0.5
HFLIP_PROB = 0.5
RRC_SCALE = (0.7, 1.0)
RANDOM_ERASING_PROB = 0.3
RANDOM_ERASING_SCALE = (0.02, 0.15)  # hole size: 2%~15% of image area

# ── Data Loading ────────────────────────────────────
USE_WEIGHTED_SAMPLER = True

# ── Training ────────────────────────────────────────
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 0.05
LLRD_DECAY = 0.85  # per-stage LR multiplier for layer-wise LR decay
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None  # set to class weights tensor to enable
T_0 = 30  # CosineAnnealingWarmRestarts cycle
T_MULT = 1

# ── K-Fold ──────────────────────────────────────────
K_FOLDS = 5
FOLD_TO_RUN = 0  # -1 runs all folds sequentially

# ── Early Stopping ──────────────────────────────────
EARLY_STOP_PATIENCE = 6
MIN_EPOCHS = 8        # don't stop before this epoch
MIN_DELTA = 0.001     # F1 improvement must exceed this threshold to count

# ── Mixed Precision ─────────────────────────────────
USE_AMP = True

# ── EMA ─────────────────────────────────────────────
EMA_DECAY = 0.999

# ── TTA ─────────────────────────────────────────────
TTA_CROPS = 1  # disabled for speed; 4-class doesn't need TTA
TTA_SCALE = (0.85, 1.0)
