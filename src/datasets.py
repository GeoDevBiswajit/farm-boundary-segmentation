import geopandas as gpd
from pathlib import Path
import rasterio
import numpy as np
import tensorflow as tf
from config import *

import warnings
warnings.filterwarnings('ignore')

#### Paths
ROOT= Path('data')
train_image_dir = ROOT / "train" / "images"
train_mask_dir = ROOT / "train" / "masks"

train_image_paths = sorted(
    [str(p) for p in train_image_dir.glob("*.tif")]
)

train_mask_paths = sorted(
    [str(p) for p in train_mask_dir.glob("*.tif")]
)
###$ VErify that image and mask files are align
# for img, msk in zip(train_image_paths[:5], train_mask_paths[:5]):
#     print(img.name, msk.name)

def get_file_paths(image_dir:Path, mask_dir:Path, extention="*.tif"):
    """
    Returns Sorted path of of image and mask
    """
    image_paths = sorted(image_dir.glob(extention))
    mask_paths = sorted(mask_dir.glob(extention))
    if len(image_paths) != len(mask_paths):
        raise ValueError('Number of Images and Masks are different')
    for img, mask in zip(image_paths, mask_paths):
        if img.stem != mask.stem:
            raise ValueError ( f"Filename mismatch:\n{img.name}\n{mask.name}")
    return image_paths, mask_paths

def _read_image_mask(image_path:str|bytes, mask_path:str|bytes):
    """
        Input:
        image_path (bytes or str)
        mask_path  (bytes or str)

    Output:
        image : numpy.ndarray
        mask  : numpy.ndarray
    """
    # TensorFlow passes bytes, so decode if needed
    if isinstance(image_path, bytes):
        image_path = image_path.decode()

    if isinstance(mask_path, bytes):
        mask_path = mask_path.decode()

    with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
        # band, height, weight in this order
        image = image_src.read()
        image = np.moveaxis(image, 0, -1)
        mask = mask_src.read()
        mask = np.moveaxis(mask, 0, -1)

        image = image.astype(np.float32)

        mask = mask.astype(np.uint8)
    return image, mask

def read_image_mask(image_path, mask_path):
    image, mask = tf.numpy_function(
        func=_read_image_mask, inp=[image_path, mask_path], Tout=[tf.float32, tf.uint8]
    )
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    mask.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return image, mask

def normalize_data(image:tf.Tensor, mask:tf.Tensor, scale_factor=SCALE_FACTOR):
    image =  tf.cast((image/scale_factor), dtype=tf.float32)
    
    return image, mask

def build_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, training=False, augment_fn=None, drop_reminder=True, buffer_size=BUFFER_SIZE):
    dataset = (tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
               .map(read_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
               .map(normalize_data, num_parallel_calls=tf.data.AUTOTUNE))

    if training:

        dataset = dataset.shuffle(buffer_size)

        if augment_fn is not None:

            dataset = dataset.map(
                augment_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=training
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

