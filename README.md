# Uncertainty-aware Blur Prior (UBP)

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Run](#run)

# Introduction
This repository contains the implementation of **Uncertainty-aware Blur Prior (UBP)** for various brain and CLIP encoders.
```
UBP/                           # Root directory
├── README.md
├── base                       # Core implementation files
│   ├── data.py                # Data loading
│   ├── eeg_backbone.py        # EEG encoder backbone implementation
│   ├── inpating_data.py       # Inpainting data module for preprocessing
│   └── utils.py               # Utility functions
├── configs
│   ├── baseline.yaml          # Configuration for baseline experiments
│   └── ubp.yaml               # Configuration for UBP experiments
├── data                       # Directory for datasets
│   └── things-eeg
│       ├── Image_feature      # Pre-extracted image features
│       ├── Image_set          # Original image dataset
│       ├── Image_set_Resize   # Resized image dataset
│       ├── Preprocessed_data_250Hz_whiten # Preprocessed EEG data (whitened)
│       └── Raw_data
├── exp                        # Directory for experiment results
├── main.py                    # Main script for running experiments
├── preprocess
│   ├── process_eeg_whiten.py  # Script to preprocess and whiten EEG data
│   └── process_resize.py      # Script to resize image dataset
├── requirements.txt           # List of required Python packages
└── scripts
    ├── bash_preprocess.sh     # Bash script for preprocessing data
    └── exp.sh                 # Bash script for running experiments
```
## Environment Setup
- Python 3.8.19
- Cuda 12.0
- PyTorch 2.4.1
- Required libraries are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Data Preparation
1. Download the Things-EEG dataset from the [OSF repository](xxxxxxxx) and put them in the `data` dir.

2. Resize the downloaded images using the provided script:

```
python preprocess/process_resize.py
```

3. Convert the data to .pt format using the preprocessing script for all subjects:

```
/bin/bash scripts/bash_preprocess.sh
```

Finally, we have the directory tree:
```
|-- data
    |-- things-eeg
        |-- Image_set
        |-- Image_set_Resize
        |-- Raw_data
        |-- Preprocessed_data_250Hz_whiten
```
## Run
To run the experiments using the provided configurations, execute:
```
/bin/bash scripts/exp.sh
```

## Acknowledgement
The code is inspired by prior works on EEG retrieve tasks.

For anonymity, we have omitted the external links.