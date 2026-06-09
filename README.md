# WBC-MTL-SEGCLS 🔬

**Multi-task White Blood Cell segmentation and classification using a ResNet34-based U-Net architecture**
Simultaneous pixel-level segmentation and cell-type classification in a single forward pass.

---

# Overview

<p align="center">
  <img src="assets/gradcam_examples.png" width="500">
</p>
<p align="center">
  <em>Grad-CAM heatmaps highlighting regions influencing classification decisions.</em>
</p>

WBC-MTL-SEGCLS is a multi-task deep learning framework for automated White Blood Cell analysis from microscopy images.

The model simultaneously performs:

- **Semantic Segmentation** → nucleus / cytoplasm / background
- **Image Classification** → WBC type identification

Unlike single-task approaches, the model shares a **ResNet34 encoder** between tasks, improving feature reuse and representation learning.

---

# Key Idea

A single network learns:

- pixel-wise structure (segmentation)
- global morphology (classification)

This is achieved via a shared encoder + dual-head architecture.

---

# Dataset

The model uses the public dataset:
https://github.com/zxaoyou/segmentation_WBC

### Dataset composition:

- Dataset 1
- Dataset 2

Each sample includes:

- microscopy image (`.bmp`)
- segmentation mask (`.png`)
- class label (CSV files)

---

# Classes

## Segmentation Classes (Mask Encoding)

> ⚠️ These are grayscale mask values (not semantic labels)

| Pixel value | Class |
|-------------|-------|
| 0 | Background |
| 128 | Nucleus |
| 255 | Cytoplasm |

## Classification Classes

- Lymphocyte
- Monocyte
- Neutrophil
- Eosinophil

---

# Data Preprocessing

The preprocessing pipeline includes:

- Merging Dataset 1 and Dataset 2 CSV files
- Normalizing image IDs (zero-padded indexing)
- Removing missing image/mask pairs
- Filtering classes with < 20 samples
- Label encoding using `LabelEncoder`
- Stratified train/test split (80/20 based on class labels)

---

# Data Augmentation

Two augmentation pipelines are used:

## Standard augmentation

- Resize (256×256)
- Horizontal Flip
- Random 90° rotation
- ShiftScaleRotate
- Brightness/Contrast

## Minority-class augmentation

Applied to least frequent classes:

- Higher geometric distortion
- Gaussian blur
- Stronger augmentation probability

---

# Class Imbalance Handling

To handle dataset imbalance:

- WeightedRandomSampler (inverse class frequency sampling)
- Stronger augmentation for rare classes

> Note: No explicit dataset duplication or manual oversampling is used.

---

# Model Architecture

## Backbone

- ResNet-34 (ImageNet pretrained)

## Architecture Design

```
Input (256 × 256 × 3)
│
├── ResNet34 Encoder (pretrained on ImageNet)
│
│   enc0 = conv1 → bn1 → relu
│   enc1 = layer1
│   enc2 = layer2
│   enc3 = layer3
│   enc4 = layer4
│
│
├── Bottleneck
│   Conv(512 → 1024) + ReLU
│   Conv(1024 → 1024) + ReLU
│
│
├── Segmentation Decoder (U-Net style with skip connections)
│
│   up4: ConvTranspose(1024 → 512)
│   concat(up4, enc3)
│   dec4: Conv(768 → 512) + ReLU
│
│   up3: ConvTranspose(512 → 256)
│   concat(up3, enc2)
│   dec3: Conv(384 → 256) + ReLU
│
│   up2: ConvTranspose(256 → 128)
│   concat(up2, enc1)
│   dec2: Conv(192 → 128) + ReLU
│
│   up1: ConvTranspose(128 → 64)
│   concat(up1, enc0)
│   dec1: Conv(128 → 64) + ReLU
│
│
├── Segmentation Head
│   Conv(64 → 3)  → pixel-wise prediction
│   (Background / Nucleus / Cytoplasm)
│
│
└── Classification Head (from bottleneck)
    Global Average Pooling (1024 → 1×1)
    Flatten
    Fully Connected Layer: 1024 → 4
    (Lymphocyte / Monocyte / Neutrophil / Eosinophil)
```

---

# Loss Function

Multi-task optimization:

```
L_total = L_seg + 0.7 × L_cls
```

Where:

- `L_seg`: CrossEntropyLoss (segmentation)
- `L_cls`: CrossEntropyLoss (classification)

---

# Training Details

- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 8
- Epochs: 10
- Loss: weighted multi-task loss
- Device: CPU / CUDA (auto-detect)

---

# Evaluation Metrics

## Classification

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

## Segmentation

- Dice score (foreground classes only, excluding background)

---

# Final Results

## Classification Performance

- Accuracy:  **0.8481**
- Precision: **0.8816**
- Recall:    **0.8481**
- F1-score:  **0.8554**

## Segmentation Performance

- Dice Score: **0.9171**

---

# Classification Report

```text
              precision    recall  f1-score   support

  Lymphocyte       1.00      0.88      0.94        41
    Monocyte       0.87      0.72      0.79        18
  Neutrophil       0.60      0.92      0.73        13
  Eosinophil       0.75      0.86      0.80         7

    accuracy                           0.85        79
   macro avg       0.80      0.85      0.81        79
weighted avg       0.88      0.85      0.86        79
```

---

# Segmentation Encoding

Mask pixel values:

- `0`   → Background
- `128` → Nucleus
- `255` → Cytoplasm

---

# Visualization

The framework supports:

## Segmentation visualization

- Input image
- Ground truth mask
- Predicted mask

## Classification visualization

- Correct predictions
- Incorrect predictions

## Grad-CAM explainability

Highlights regions influencing classification decisions.

---

# Grad-CAM

Grad-CAM is applied on:

- Bottleneck layer (`model.center`)

> Important: Grad-CAM is computed on the bottleneck convolutional block, not encoder/decoder layers.

It produces heatmaps showing which regions of the cell most influence classification decisions.

---

# Checkpoints

Each checkpoint includes:

- Model weights
- Optimizer state
- Epoch number
- Model configuration (`n_classes_seg`, `n_classes_cls`)
- Class mappings (LabelEncoder classes)
- Evaluation metrics

Saved file:

```text
wbc_multitask_checkpoint.ckpt
```

---

# Inference Pipeline

At inference time:

1. Load model checkpoint
2. Forward pass returns:
   - segmentation map
   - classification logits
3. Apply argmax for predictions
4. Optional Grad-CAM visualization

---

# Requirements

```text
Python >= 3.9
torch >= 2.0
torchvision
albumentations
scikit-learn
opencv-python
matplotlib
numpy
pandas
```

Install:

```bash
pip install -r requirements.txt
```

---

# Visualization Results

### Segmentation Samples

<p align="center">
  <img src="assets/segmentation_samples.png" width="600">
</p>

_Side-by-side comparison of original image, ground-truth mask, and predicted mask._

---

### Classification Predictions

<p align="center">
  <img src="assets/prediction_results.png" width="600">
</p>

_Correctly classified samples (green) and incorrectly classified samples (red)._

---

### Confusion Matrix

<p align="center">
  <img src="assets/confusion_matrix.png" width="400">
</p>

_Per-class classification performance across all four WBC categories._

---

### Grad-CAM Explainability

<p align="center">
  <img src="assets/gradcam_examples.png" width="500">
</p>

_Grad-CAM heatmaps overlaid on test images showing which morphological regions drive the model's classification decisions._

---

# Notes

- Dataset must be downloaded manually from the official repository
- Images must contain both `.bmp` image and `.png` mask
- Classes with fewer than 20 samples are removed automatically
- Model supports CPU and GPU execution
- Grad-CAM uses forward/backward hooks on the bottleneck layer (`model.center`)

---

# License

MIT
