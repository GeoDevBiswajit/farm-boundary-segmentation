# Farm Boundary Segmentation

A geospatial deep-learning pipeline for **automated farm boundary delineation from high-resolution satellite imagery** using **U-Net semantic segmentation**.

The project covers the complete model-development workflow, from geospatial raster preparation and image tiling to model training, custom loss functions, evaluation metrics, checkpointing, and TensorBoard experiment tracking.

---

## Overview

Farm boundary delineation is an important step in building field-level agricultural datasets. Manually digitizing field boundaries is time-consuming and difficult to scale.

This project uses semantic segmentation to classify pixels into:

* **Farm / field**
* **Background**

The resulting segmentation mask can then be used as the foundation for extracting individual agricultural field boundaries.

### Workflow

```text
Source Satellite Imagery
          │
          ▼
   Raster Preparation
          │
          ▼
   Mask Rasterization
          │
          ▼
     Image Tiling
          │
          ▼
 Train / Validation / Test Split
          │
          ▼
    tf.data Pipeline
          │
          ▼
       U-Net
          │
          ▼
   Pixel-wise Prediction
          │
          ▼
 Segmentation Evaluation
          │
          ▼
 Model Checkpoint + Logs
```

---

## Key Features

* Geospatial raster and vector data preparation
* Rasterization of field boundary masks
* Image and mask tiling
* Train/validation/test dataset preparation
* TensorFlow `tf.data` input pipeline
* U-Net semantic segmentation architecture
* Dice loss
* Weighted Binary Cross-Entropy + Dice loss
* Streaming Dice metric
* Model checkpointing
* CSV training history
* TensorBoard logging
* Config-driven training parameters

---

## Project Structure

```text
field_boundary_delineation/
│
├── data/
│   ├── raster/
│   │   ├── images/
│   │   ├── mask/
│   │   └── tiles/
│   │       ├── images/
│   │       └── masks/
│   │
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── test/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── vector/
│
├── output/
│   ├── model/
│   └── log/
│       ├── train.csv
│       └── train/
│
├── src/
│   ├── config.py
│   ├── data_preparation.ipynb
│   ├── datasets.py
│   ├── model.py
│   ├── losses.py
│   ├── metrices.py
│   ├── callback.py
│   └── train.py
│
├── requirments.txt
├── README.md
├── LICENSE
└── model_architecture.png
```

---

## Pipeline Components

### 1. Geospatial Data Preparation

`data_preparation.ipynb` handles the initial preparation of the training data.

The workflow includes:

* Reading geospatial raster and vector data
* Preparing source imagery
* Rasterizing farm boundary polygons into segmentation masks
* Generating image/mask tiles
* Exploring the prepared dataset
* Creating training, validation, and test datasets

The important aspect is that the **image and corresponding mask remain spatially aligned** throughout the preparation process.

---

### 2. Dataset Pipeline

`datasets.py` implements the TensorFlow input pipeline.

The dataset loader:

```text
GeoTIFF
   ↓
Image / Mask Reading
   ↓
Preprocessing
   ↓
Tensor Conversion
   ↓
tf.data.Dataset
   ↓
Batching / Training
```

Using `tf.data` allows the training pipeline to efficiently load and batch large numbers of image patches.

---

### 3. U-Net Model

`model.py` contains the U-Net segmentation architecture.

Conceptually:

```text
                 Input Image
                      │
                      ▼
              Encoder / Downsampling
                      │
                      ▼
                   Bottleneck
                      │
                      ▼
              Decoder / Upsampling
                      │
               Skip Connections
                      │
                      ▼
             Pixel-wise Prediction
```

Skip connections allow spatial information from the encoder to be passed to the decoder, which is particularly important for delineating detailed field boundaries.

The model architecture is visualized in:

```text
model_architecture.png
```

---

### 4. Custom Loss Functions

`losses.py` contains segmentation-specific loss functions.

The project includes:

* **Dice Loss**
* **Weighted Binary Cross-Entropy + Dice Loss**

Dice-based objectives are useful for segmentation problems where foreground pixels may represent a relatively small portion of the complete image.

A combined loss can be expressed conceptually as:

```text
Total Loss
    =
Weighted BCE
    +
Dice Loss
```

This combines pixel-level classification with overlap-based segmentation optimization.

---

### 5. Evaluation Metric

`metrices.py` contains the streaming Dice metric used during model training/evaluation.

Dice coefficient measures the overlap between the predicted segmentation and the reference mask:

```text
Dice = 2 × |Prediction ∩ Ground Truth|
       ---------------------------------
       |Prediction| + |Ground Truth|
```

A higher Dice value indicates greater spatial overlap between the predicted and reference field regions.

---

### 6. Training & Experiment Tracking

`train.py` acts as the main training entry point.

`callback.py` handles training-related callbacks such as:

* Model checkpointing
* CSV logging
* TensorBoard logging
* Training controls

Training outputs are stored under:

```text
output/
├── model/
└── log/
    ├── train.csv
    └── train/
```

This provides a reproducible record of training progress and model checkpoints.

---

## Configuration

Training parameters and paths are centralized in:

```text
src/config.py
```

This keeps dataset locations, model parameters, and training settings separate from the core implementation.

Typical configuration parameters include:

```text
Dataset paths
Image dimensions
Batch size
Learning rate
Number of epochs
Model output path
Logging paths
```

This makes experiments easier to reproduce and modify without changing the training logic.

---

## Why This Architecture?

The project separates the major stages of a deep-learning workflow:

```text
Data Preparation
       ↓
Dataset Loading
       ↓
Model Architecture
       ↓
Loss Functions
       ↓
Metrics
       ↓
Callbacks
       ↓
Training
```

This modular structure makes it easier to:

* modify the model independently
* experiment with different loss functions
* add or modify evaluation metrics
* change dataset preparation
* reproduce experiments
* maintain the training pipeline
* extend the project for future inference workflows

---

## Technology Stack

### Deep Learning

* Python
* TensorFlow / Keras
* U-Net
* Semantic Segmentation

### Geospatial

* Rasterio
* GeoPandas
* GDAL
* GeoTIFF
* GeoPackage

### Data Processing

* NumPy
* Pandas
* TensorFlow `tf.data`

### Experiment Tracking

* TensorBoard
* CSV training logs
* Keras model checkpoints

---

## Dataset

The project uses high-resolution satellite imagery and corresponding farm-boundary reference data.

The repository structure supports:

```text
Image
   +
Ground Truth Boundary
        ↓
Rasterized Mask
        ↓
Image / Mask Tile
        ↓
Training Dataset
```

Large satellite datasets and other data that cannot be redistributed are not included in the repository.

---

## Future Extensions

The current repository focuses primarily on the **training pipeline**.

Possible extensions include:

* Automated inference pipeline
* Large-scene tiled inference
* Overlapping tile prediction
* Prediction reconstruction
* Polygonization of segmentation masks
* Farm polygon geometry cleaning
* Quantitative test-set evaluation
* Model comparison
* Data augmentation
* Experiment tracking
* Batch inference over multiple AOIs
* Deployment as a geospatial inference service

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Biswajit Das**

Remote Sensing Engineer | Geospatial AI | Satellite Image Analysis

**Focus:** Remote Sensing · Geospatial Deep Learning · Agricultural Intelligence · Spatial Data Science
