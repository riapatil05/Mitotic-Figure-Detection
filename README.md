# Deep Learning-Based Mitotic Figure Detection for Computational Histopathology

A computer vision pipeline for mitotic figure detection in histopathology images using YOLOv8, with tiled whole-slide image (WSI) inference, spatial coordinate reconstruction, and duplicate suppression.

## Overview

Mitotic figure detection is an important task in computational pathology, supporting the assessment of cellular proliferation in tumour tissue. This project implements an object detection pipeline to localize mitotic figures in histopathology images and extends patch-level predictions to whole-slide image analysis.

The pipeline includes dataset validation, YOLOv8n training, augmentation experiments, inference benchmarking, and utilities for processing large whole-slide images through overlapping tiles.

## Key Features

- **Deep learning-based detection:** YOLOv8n-based localization of mitotic figures.
- **Training comparison:** Baseline and augmented model experiments evaluated under consistent validation settings.
- **Whole-slide inference:** Overlapping tile generation and spatial reconstruction of detections in slide-level coordinates.
- **Duplicate suppression:** Non-maximum suppression to consolidate detections across overlapping tiles.
- **Tissue-aware filtering:** Optional filtering of tiles based on tissue content.
- **Testing:** Automated tests for coordinate reconstruction and inference utilities.

## Results

### Model Evaluation

The baseline and augmented models were evaluated on the validation set using a confidence threshold of `0.001`, IoU threshold of `0.7`, and image size of `640 × 640`.

| Metric | Baseline YOLOv8n | Augmented YOLOv8n |
|---|---:|---:|
| Precision | 79.07% | 88.13% |
| Recall | 83.52% | 89.75% |
| mAP@50 | 87.97% | 91.95% |
| mAP@50–95 | 49.58% | 57.39% |

The augmentation experiment improved validation mAP@50–95 by **7.81 percentage points**.

### Inference Benchmark

| Metric | Result |
|---|---:|
| Hardware | NVIDIA Tesla T4 GPU |
| Model checkpoint size | Approximately 6.5 MB |
| Mean inference latency | 17.8 ms/image |
| Median inference latency | 14.93 ms/image |
| 95th-percentile latency | 29.98 ms/image |
| Throughput | 56.19 images/second |

Benchmark results were obtained using preloaded validation images, GPU warm-up, and synchronized GPU timing. Performance may vary with hardware, image dimensions, preprocessing, and inference settings.

### Whole-Slide Image Demonstration

The WSI pipeline was demonstrated on an unannotated `.svs` slide using tiled inference and global duplicate suppression.

| Setting | Result |
|---|---:|
| Total tiles | 30 |
| Tiles accepted with tissue filtering | 18 |
| Raw detections without tissue filtering | 6 |
| Detections after global NMS without tissue filtering | 5 |
| Raw detections with tissue filtering | 2 |
| Detections after global NMS with tissue filtering | 2 |
| Processing time | Approximately 2.2 seconds |

This is a pipeline demonstration, not a measure of WSI detection accuracy. The slide did not have ground-truth annotations, and performance on whole-slide images may differ from patch-level validation results.

## Project Structure

```text
mitotic-figure-detection/
├── configs/
│   ├── augmentation_experiment_v1.json
│   ├── baseline_training_config.json
│   └── mitosis_data.yaml
├── results/
│   ├── figures/
│   ├── metrics/
│   └── wsi/
├── src/
│   ├── wsi_inference.py
│   └── wsi_pipeline.py
├── tests/
│   └── test_wsi_inference.py
├── 01_mitosis_detection.ipynb
├── 02_wsi_inference.ipynb
├── project_manifest.json
├── requirements.txt
├── .gitignore
└── README.md
```

Notebook filenames reflect the current repository copy; verify that each notebook contains the intended workflow.

## Installation

Python 3.13 was used in the development environment. Dependency compatibility may vary across environments.

Clone the repository:

```bash
git clone https://github.com/riapatil05/mitotic-figure-detection.git
cd mitotic-figure-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-compatible PyTorch installation is recommended for GPU inference and training. Install the PyTorch build appropriate for your system from the official PyTorch installation guide.

## Dataset

The project uses the [Mitosis GJS dataset on Hugging Face](https://huggingface.co/datasets/LibreYOLO/mitosis-gjs3g).

- **Total images:** 304
- **Training images:** 213
- **Validation images:** 61
- **Test images:** 30
- **Ground-truth instances:** 436
- **Image dimensions:** 640 × 640 pixels
- **Dataset license:** CC BY 4.0

The dataset is not included in this repository. Obtain it from the original source and configure the paths in `configs/mitosis_data.yaml` before running training or evaluation.

The test split was evaluated during project development; it should not be interpreted as an untouched final holdout.

## Running the Project

### 1. Training and Model Evaluation

Open `01_mitosis_detection.ipynb` in Jupyter Notebook or Google Colab. Review the dataset configuration and paths before running the notebook.

The notebook documents the model training and evaluation workflow. Training requires access to the dataset and a compatible PyTorch/Ultralytics environment.

### 2. Whole-Slide Image Inference

Review `02_wsi_inference.ipynb` for the WSI workflow, and inspect the source files in `src/`.

The pipeline supports tiled processing of large slide images, reconstruction of detections in slide-level coordinates, and suppression of duplicate detections arising from overlapping tiles. Whole-slide image processing may require additional slide-reading dependencies and compatible image formats.

**Note:** Confirm that `02_wsi_inference.ipynb` is the intended WSI notebook before publishing or relying on its contents.

### 3. Run Tests

From the repository root:

```bash
pytest -q
```

The development test run completed with **4 passed tests** and one warning.

## Technical Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Pandas
- Matplotlib
- OpenCV
- pytest
- Jupyter Notebook / Google Colab

## Limitations

- The dataset is relatively small, so generalization to different laboratories, scanners, staining protocols, and tissue types requires further evaluation.
- Validation metrics reflect the specified dataset and evaluation settings; they do not guarantee equivalent performance on external data.
- The WSI demonstration used an unannotated slide, so WSI-level precision and recall were not established.
- Magnification metadata for the patch dataset was not documented.
- Model performance and inference speed depend on hardware, preprocessing, image size, and inference thresholds.

## Future Work

- Evaluate on independent, externally sourced histopathology datasets.
- Validate whole-slide predictions against annotated slides.
- Investigate domain adaptation across scanners, staining protocols, and tissue types.
- Extend evaluation with slide-level and region-level metrics.
- Explore improved calibration and post-processing for overlapping detections.

## License

The source code in this repository is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

The dataset is separately licensed under CC BY 4.0 and remains subject to its original terms.
