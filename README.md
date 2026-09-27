# Farm Boundary Segmentation

A modular, production-oriented deep learning pipeline for extracting farm boundaries from high-resolution satellite imagery using semantic segmentation and geospatial post-processing.

## Overview

Accurate farm boundary extraction is an important component of satellite-based agricultural intelligence. Manually digitizing field boundaries over large geographic areas is time-consuming and difficult to scale.

This project develops an end-to-end workflow for automatically extracting agricultural field boundaries from high-resolution satellite imagery.

The pipeline is designed with a **modular architecture**, separating data preparation, model inference, post-processing, and geospatial vectorization. This makes individual components easier to test, replace, and integrate into a larger production system.

### Key capabilities

* High-resolution satellite image preprocessing
* Image tiling for deep-learning inference
* Semantic segmentation of agricultural fields
* Model inference on large satellite scenes
* Overlapping tile prediction and reconstruction
* Probability-based mask generation
* Morphological and spatial post-processing
* Connected-component / polygon extraction
* Polygon cleaning and filtering
* Raster-to-vector conversion
* Geospatial output generation
* Modular configuration for reproducible processing

---

## Pipeline Architecture

```text
                    Satellite Imagery
                           │
                           ▼
                  ┌──────────────────┐
                  │ Data Preparation │
                  │                  │
                  │ • Read imagery   │
                  │ • CRS handling   │
                  │ • Normalization  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Image Tiling     │
                  │                  │
                  │ • Fixed windows  │
                  │ • Overlap        │
                  │ • Patch creation │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Deep Learning    │
                  │ Segmentation     │
                  │                  │
                  │ • Model loading  │
                  │ • Batch infer.   │
                  │ • Probability   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Prediction       │
                  │ Reconstruction   │
                  │                  │
                  │ • Merge patches  │
                  │ • Thresholding   │
                  │ • Mask creation  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Post Processing  │
                  │                  │
                  │ • Noise removal  │
                  │ • Morphology     │
                  │ • Object filtering│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Vectorization   │
                  │                  │
                  │ • Polygonize    │
                  │ • Geometry fix  │
                  │ • Area filtering│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Farm Boundaries  │
                  │                  │
                  │ GeoJSON / GPKG   │
                  │ Shapefile / etc. │
                  └──────────────────┘
```

---

## Why a Modular Architecture?

A production geospatial ML workflow should not depend on a single notebook containing the entire processing chain.

Instead, the workflow is divided into independent components with clear responsibilities.

For example:

```text
Input
  │
  ├── preprocessing
  │
  ├── tiling
  │
  ├── inference
  │
  ├── reconstruction
  │
  ├── post-processing
  │
  └── vectorization
       │
       ▼
     Output
```

This approach makes it possible to:

* replace the segmentation model without rewriting the preprocessing pipeline
* change the tiling strategy independently
* tune post-processing parameters without retraining the model
* process different satellite datasets
* test individual modules independently
* run inference over many AOIs
* integrate the pipeline into an automated production workflow

---

## Project Structure

```text
farm-boundary-segmentation/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/
│   └── README.md
│
├── src/
│   ├── preprocessing/
│   │   ├── reader.py
│   │   ├── normalization.py
│   │   └── tiling.py
│   │
│   ├── inference/
│   │   ├── model.py
│   │   ├── predictor.py
│   │   └── batch_inference.py
│   │
│   ├── postprocessing/
│   │   ├── mask_processing.py
│   │   ├── morphology.py
│   │   └── filtering.py
│   │
│   ├── vectorization/
│   │   ├── polygonize.py
│   │   └── geometry.py
│   │
│   └── utils/
│       ├── raster.py
│       ├── logging.py
│       └── config.py
│
├── scripts/
│   ├── run_inference.py
│   └── generate_vectors.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_inference.py
│   └── test_vectorization.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

> The exact directory names can be adapted to the implementation in this repository. The important principle is that **model inference, geospatial processing, and post-processing remain separated modules**.

---

## Methodology

### 1. Image Preparation

Satellite imagery is first prepared for model inference.

Typical operations include:

* raster reading
* spatial reference handling
* band selection
* data type conversion
* normalization
* nodata handling
* spatial cropping

The preprocessing stage produces model-ready image arrays while preserving the spatial metadata required for subsequent geospatial processing.

---

### 2. Tiling

Large satellite scenes cannot always be passed directly to a deep-learning model because of GPU memory constraints.

The imagery is therefore divided into smaller patches.

```text
Large Satellite Scene
┌───────────────────────────────┐
│ ┌─────┐ ┌─────┐ ┌─────┐      │
│ │tile │ │tile │ │tile │      │
│ └─────┘ └─────┘ └─────┘      │
│ ┌─────┐ ┌─────┐ ┌─────┐      │
│ │tile │ │tile │ │tile │      │
│ └─────┘ └─────┘ └─────┘      │
└───────────────────────────────┘
```

Overlapping tiles can be used to reduce boundary artifacts between neighboring patches.

Each tile retains its spatial position so that predictions can later be reconstructed in the original geographic coordinate system.

---

### 3. Semantic Segmentation

The prepared image patches are passed through a semantic segmentation model.

The model predicts the probability of each pixel belonging to the target agricultural-field class.

Conceptually:

```text
Image Patch
     │
     ▼
Segmentation Model
     │
     ▼
Pixel-wise Probability
     │
     ▼
Binary Segmentation Mask
```

The model architecture is intentionally isolated from the rest of the pipeline so that different segmentation architectures can be evaluated without changing the surrounding geospatial workflow.

---

### 4. Prediction Reconstruction

Predictions from individual tiles are reconstructed into a continuous scene.

For overlapping tiles, predictions can be combined before thresholding to reduce discontinuities at tile boundaries.

The reconstructed output maintains the spatial relationship with the original satellite image.

---

### 5. Post-processing

Raw segmentation predictions may contain:

* small isolated objects
* holes
* fragmented boundaries
* narrow artifacts
* unwanted regions

Spatial post-processing is therefore applied before vectorization.

Depending on the use case, this stage may include:

* thresholding
* morphological operations
* connected-component analysis
* minimum-area filtering
* hole removal
* geometry simplification

Keeping this stage independent from model inference makes it possible to tune spatial rules without retraining the model.

---

### 6. Raster-to-Vector Conversion

The final segmentation mask is converted into vector geometries.

```text
Segmentation Mask
       │
       ▼
    Polygonize
       │
       ▼
Geometry Validation
       │
       ▼
Spatial Filtering
       │
       ▼
Farm Boundary Polygons
```

The resulting polygons can be exported as common geospatial formats such as:

* GeoJSON
* GeoPackage
* Shapefile

The output retains the coordinate reference system of the source data.

---

## Production-Oriented Design

The project follows several principles commonly required when moving a research workflow toward production.

### Separation of concerns

Each component performs a specific task.

```text
Data I/O
   ↓
Preprocessing
   ↓
Inference
   ↓
Prediction reconstruction
   ↓
Post-processing
   ↓
Vectorization
   ↓
Export
```

This prevents model-specific logic from becoming tightly coupled with geospatial processing.

### Configuration-driven execution

Processing parameters can be maintained separately from the implementation.

Example:

```yaml
input:
  image: data/input/image.tif

model:
  checkpoint: models/model.pth

inference:
  tile_size: 512
  overlap: 64
  batch_size: 8

postprocessing:
  threshold: 0.5
  min_area: 100
```

This allows different AOIs, models, and inference configurations to be processed without modifying the source code.

### Reproducibility

The pipeline separates:

* configuration
* model weights
* source code
* input data
* generated outputs

Large satellite datasets and model checkpoints should not be committed directly to Git. Instead, they can be referenced through external storage or model repositories.

### Scalability

The same processing logic can be applied to:

```text
Single Farm
     ↓
Multiple Farms
     ↓
AOI
     ↓
Satellite Scene
     ↓
Large Geographic Region
```

Batch processing can therefore be implemented without changing the core segmentation logic.

---

## Geospatial Considerations

Unlike a conventional computer-vision segmentation project, this workflow treats the imagery as geospatial data.

Important considerations include:

* CRS preservation
* affine transforms
* pixel-to-coordinate conversion
* raster dimensions
* spatial resolution
* nodata handling
* geometry validity
* polygon area
* spatial filtering

This allows the final segmentation results to be used directly in downstream GIS and agricultural applications.

---

## Example Workflow

```python
from pipeline import FarmBoundaryPipeline

pipeline = FarmBoundaryPipeline(
    config="configs/config.yaml"
)

result = pipeline.run(
    input_raster="data/input/satellite.tif",
    output="outputs/farm_boundaries.gpkg"
)
```

A typical execution performs:

```text
Satellite GeoTIFF
      ↓
Preprocessing
      ↓
Generate Tiles
      ↓
Model Inference
      ↓
Merge Predictions
      ↓
Post-processing
      ↓
Polygonization
      ↓
Farm Boundary GeoPackage
```

---

## Output

The primary output is a vector dataset containing extracted farm/field boundaries.

Example attributes may include:

| Field      | Description                   |
| ---------- | ----------------------------- |
| `id`       | Unique polygon identifier     |
| `area_m2`  | Polygon area in square meters |
| `geometry` | Farm boundary geometry        |

Additional confidence or model-derived attributes can be added depending on the downstream application.

---

## Applications

The extracted farm boundaries can serve as a foundation for:

* agricultural field inventory
* crop monitoring
* satellite-based crop analytics
* soil property estimation
* irrigation advisory
* crop health monitoring
* agricultural insurance
* farm-level credit assessment
* precision agriculture
* automated GIS database creation

---

## Technology Stack

**Programming**

* Python

**Deep Learning**

* PyTorch / TensorFlow
* Semantic segmentation
* GPU-based inference

**Geospatial**

* Rasterio
* GDAL
* GeoPandas
* Shapely
* NumPy

**Data Formats**

* GeoTIFF
* GeoJSON
* GeoPackage
* Shapefile

**Development**

* Git
* YAML configuration
* Modular Python package structure

---

## Model & Data

This repository is intended primarily to demonstrate the **engineering and geospatial processing workflow**.

Large proprietary satellite datasets, production imagery, and trained model weights are not included in the repository.

For reproducibility, users can provide their own compatible imagery and model checkpoint.

---

## Limitations

Performance depends on factors such as:

* spatial resolution of imagery
* image quality
* cloud and atmospheric conditions
* field size and shape
* landscape complexity
* training-data distribution
* model architecture
* segmentation threshold
* post-processing parameters

The extracted polygons should therefore be validated against appropriate reference data before being used in operational applications.

---

## Future Improvements

Potential extensions include:

* automated dataset preparation
* multi-class field segmentation
* uncertainty estimation
* confidence-based polygon filtering
* distributed batch inference
* GPU/CPU resource optimization
* automated model evaluation
* experiment tracking
* containerized deployment
* REST API integration
* cloud-based inference
* STAC-based satellite data ingestion

---

## Author

**Biswajit Das**

Remote Sensing Engineer | Geospatial AI | Satellite Image Analysis

Interests include:

* Remote Sensing
* Geospatial AI
* Deep Learning
* Agricultural Intelligence
* Satellite Image Processing
* Spatial Data Science

---

## License

Add an appropriate license based on the ownership and distribution rights of the code and datasets.
