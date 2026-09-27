from pathlib import Path
from datasets import build_dataset
from config import *
from losses import BCEDiceLoss
from metrices import DiceMetric
from model import build_unet
from keras.optimizers import Adam
from callback import create_callback

def train_model():
    train_dataset = build_dataset(image_paths=train_image_paths, mask_paths=train_mask_paths, training=True)
    test_dataset = build_dataset(test_image_paths, test_mask_paths)
    val_dataset = build_dataset(val_image_paths, val_mask_paths)
    model = build_unet(image_size=IMAGE_SIZE, channels=NUM_CHANNELS, filters=FILTERS, num_target_class=OUT_CLASS)
    bce_dice_loss  = BCEDiceLoss(bce_weight=BCE_WEIGHTS, dice_weight=DICE_WEIGHT)
    dice_metric = DiceMetric()
    callbacks = create_callback(checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR)

    model.compile(optimizer=Adam(learning_rate=LR), loss=bce_dice_loss , metrics=[dice_metric])
    model.fit(train_dataset, validation_data=val_dataset, 
              callbacks=callbacks,
              epochs=EPOCHS)
    return model

if __name__ == "__main__":
    train_model()
    
