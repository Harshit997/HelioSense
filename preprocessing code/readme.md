# ISRO SoLEXS Data Preprocessing Pipeline

This repository contains the complete preprocessing pipeline used to prepare **ISRO SoLEXS Level-1 spectral observations** and **NASA GOES XRS flux data** for deep learning-based solar flare forecasting.

The pipeline performs data cleaning, temporal synchronization, flare-aware interpolation, normalization, and exports a high-quality processed dataset suitable for model training.

---

# Repository Structure

```
.
├── combined.ipynb              # Merge ISRO and GOES datasets
├── preprocess.ipynb            # Test flare interpolation on sample files
├── final_preprocessing.ipynb   # Run preprocessing on the complete dataset
└── README.md
```

---

# Pipeline Overview

```
Raw ISRO SoLEXS Files
          │
          ▼
ISRO Data Cleaning
          │
          ▼
Raw GOES XRS Files
          │
          ▼
GOES Data Cleaning
          │
          ▼
Timestamp Synchronization
          │
          ▼
Merged Daily Dataset
          │
          ▼
Flare Region Detection
          │
          ▼
Flare-aware Interpolation
          │
          ▼
Dataset Cleaning
          │
          ▼
Background Flux Correction
          │
          ▼
Final Processed Dataset
```

---

# Prerequisites

Install the required Python packages before running the notebooks.

```bash
pip install pandas numpy scipy matplotlib pyarrow fastparquet netCDF4 tqdm
```

Depending on your environment, you may also need:

```bash
pip install jupyter notebook
```

---

# Input Data

The preprocessing requires two datasets.

## 1. ISRO SoLEXS Level-1 Data

Contains:

- Spectral channels
- Light curve counts
- Observation timestamps

---

## 2. NASA GOES XRS Data

Contains:

- XRS Short Flux
- XRS Long Flux
- Event timestamps

Both datasets should be downloaded beforehand and stored locally.

---

# Notebook Description

## 1. combined.ipynb

This notebook performs the initial preprocessing and merges both datasets.

### Operations

- Read ISRO daily files
- Read GOES XRS files
- Convert timestamps
- Remove invalid observations
- Remove NaN values
- Normalize spectral information
- Synchronize timestamps
- Merge both datasets
- Export merged daily Parquet files

Output:

```
merged_parquet/
    day1.parquet
    day2.parquet
    ...
```

---

## 2. preprocess.ipynb

This notebook is mainly used for validating the interpolation strategy on sample merged files.

Operations include:

- Load merged dataset
- Detect flare intervals
- Identify

    EVENT_START
    EVENT_PEAK
    EVENT_END

- Perform interpolation
- Visualize reconstructed flare curves

This notebook is useful for verifying preprocessing before running the complete pipeline.

---

## 3. final_preprocessing.ipynb

This notebook applies the complete preprocessing pipeline to every merged file.

Operations include:

### Flare Detection

Detects flare intervals using the flare status labels.

```
EVENT_START
      │
      ▼
EVENT_PEAK
      │
      ▼
EVENT_END
```

---

### Region 1 Interpolation

Interpolates missing XRS values between

```
EVENT_START
        │
        ▼
EVENT_PEAK
```

---

### Region 2 Interpolation

Interpolates missing XRS values between

```
EVENT_PEAK
        │
        ▼
EVENT_END
```

---

### Dataset Cleaning

- Remove remaining missing values
- Remove helper columns
- Reset indices
- Validate data consistency

---

### Background Flux Correction

Computes the minimum background XRS flux and replaces unrealistic values below the background threshold.

---

### Export

The cleaned dataset is exported as processed daily Parquet files.

Example output:

```
processed_data/
    2023-01-01.parquet
    2023-01-02.parquet
    ...
```

---

# Running the Pipeline

Run the notebooks in the following order.

## Step 1

```
combined.ipynb
```

Creates merged daily Parquet files.

---

## Step 2

(Optional)

```
preprocess.ipynb
```

Verify interpolation on sample files.

---

## Step 3

```
final_preprocessing.ipynb
```

Generate the final processed dataset.

---

# Output Dataset

The final processed dataset contains

- Observation timestamps
- Spectral channels
- Normalized light-curve counts
- GOES XRS flux
- Interpolated flare regions
- Clean numerical features

All files are stored in Parquet format for efficient storage and fast loading during model training.

---

# Features

- Automated preprocessing pipeline
- ISRO SoLEXS Level-1 support
- NASA GOES XRS integration
- Timestamp synchronization
- Physics-aware flare interpolation
- Missing value reconstruction
- Background flux correction
- Daily Parquet generation
- Training-ready dataset generation

---

# Workflow

```
ISRO SoLEXS Data
          │
          ▼
Cleaning & Normalization
          │
          ▼
GOES XRS Data
          │
          ▼
Cleaning
          │
          ▼
Timestamp Matching
          │
          ▼
Merged Dataset
          │
          ▼
Flare Detection
          │
          ▼
Interpolation
          │
          ▼
Cleaning
          │
          ▼
Background Flux Correction
          │
          ▼
Processed Dataset
```

---

# Notes

- Ensure the folder paths inside the notebooks are updated according to your local system before execution.
- Run the notebooks sequentially to avoid missing intermediate outputs.
- The final processed dataset generated by this pipeline is the input used for the model training and evaluation stages of the solar flare forecasting framework.

---

# Citation

If you use this preprocessing pipeline in your research or project, please cite the corresponding repository or associated publication.

---
