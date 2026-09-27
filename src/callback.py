from pathlib import Path
from keras.callbacks import (
    ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau,
    TensorBoard)

def create_callback(checkpoint_dir, log_dir, monitor="val_dice_coefficient", mode="max"):
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(checkpoint_dir/"best_model.keras", monitor=monitor, save_best_only=True, mode=mode),
        CSVLogger(filename=log_dir/"train.csv"),
        EarlyStopping(monitor=monitor, patience=10, mode=mode),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-6, mode=mode),
        TensorBoard(log_dir=log_dir)
    ]
    return callbacks
