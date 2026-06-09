# WBC Dataset

This folder contains the raw data used for the WBC multi-task learning project.

## Structure

The dataset consists of two subsets:

- Dataset 1/
- Dataset 2/

Each dataset includes:
- .bmp images (microscopic blood cell images)
- .png masks (segmentation labels)

## Labels

Class labels are provided in the following CSV files:

- Class Labels of Dataset 1.csv
- Class Labels of Dataset 2.csv

Each row contains:
- image ID
- class label

## Preprocessing

During training:
- Image IDs are normalized to 3-digit format
- Masks are converted into 3 classes:
  - 0: background
  - 1: class 1 region
  - 2: class 2 region

## Notes

- Only images with both .bmp and .png files are used
- Rare classes (less than 20 samples) are filtered out during training
