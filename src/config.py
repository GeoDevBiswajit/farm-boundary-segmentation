from pathlib import Path

IMAGE_SIZE=512
NUM_CHANNELS=4
SCALE_FACTOR = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 8
OUT_CLASS = 2
LR = 0.0001
FILTERS = [64, 128, 256, 512]
BCE_WEIGHTS = 0.3
DICE_WEIGHT= 0.7
EPOCHS = 1

#### Paths
ROOT= Path('../data')
train_image_dir = ROOT / "train" / "images"
train_mask_dir = ROOT / "train" / "masks"
val_image_dir = ROOT / "val" / "images"
val_mask_dir = ROOT / "val" / "masks"
test_image_dir = ROOT / "test" / "images"
test_mask_dir = ROOT / "test" / "masks"

train_image_paths = sorted(
    [str(p) for p in train_image_dir.glob("*.tif")]
)
train_mask_paths = sorted(
    [str(p) for p in train_mask_dir.glob("*.tif")])

test_image_paths = sorted(
    [str(p) for p in test_image_dir.glob("*.tif")]
)
test_mask_paths = sorted(
    [str(p) for p in test_mask_dir.glob("*.tif")])

val_image_paths = sorted(
    [str(p) for p in val_image_dir.glob("*.tif")]
)
val_mask_paths = sorted(
    [str(p) for p in val_mask_dir.glob("*.tif")])

OUTPUT = Path('../output')
CHECKPOINT_DIR = OUTPUT / "model"
LOG_DIR = OUTPUT / "log"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)