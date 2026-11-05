# Maneuver Detection

This repository contains code for detecting maneuvers from Loft Dynamics Helicopter Simulator flight data.
The package is named `maneuver_detection`.

> [!important]
> This project contains some `TODO`s for you to complete as part of your implementation.
> Make sure to address these or the project will not work as expected.


## Installation

Before you begin, ensure that you have downloaded the `data` folder from `Moodle` and placed it in the root directory of this project.

### Python

It is highly recommended to use `uv`.
Instructions for installing `uv` can be found [here](https://docs.astral.sh/uv/getting-started/installation/).

To install this package and its dependencies, run the following command:

```bash
uv sync
```

This will generate a virtual environment and install all required packages.
The virtual environment can be activated with:

Windows
```bash
. \.venv\Scripts\Activate
```

Linux/MacOS
```bash
source .venv/bin/activate
```

#### Pip (not recommended)
If you prefer to use `pip`, you can install the required packages as editable with:

```bash
python -m pip install -e .
```

## Usage
To run scripts for maneuver detection, use the following command:

```bash
python -m maneuver_detection.<script_name>
```

e.g., to run the training script:

```bash
python -m maneuver_detection.train
```

You could also use `uv run` to ensure the virtual environment is activated:

```bash
uv run python -m maneuver_detection.<script_name>
```

### Importing the Package

The package can be imported in your Python scripts as follows:

```python
import maneuver_detection
```

or

```python
from maneuver_detection import <module_name>
```

## Description

The project has the following folder structure:

```
maneuver_detection/
├── configs/                # Configuration files in YAML format
│   └── default.yml         # Default configuration file
├── data/                   # Folder containing flight data (to be downloaded separately)
├── models/                 # Folder to save trained models
├── src/maneuver_detection  # Main package folder
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings
│   ├── evaluate.py         # Evaluation script
│   ├── evaluation_metrics.py # Evaluation metrics functions
│   ├── helpers.py          # Helper functions
│   ├── preprocess.py       # Data preprocessing script
│   ├── test.py             # Testing script
│   └── train.py            # Training script
├── README.md               # This README file
└── pyproject.toml          # Project metadata and dependencies
```

## Configuration

The configuration settings for the project are defined in `config.py`.
You can modify these settings using `configs/default.yml` to change parameters such as data paths, thresholds, and more.
Take a look at the `Config` class in `config.py` for details on available configuration options.


## Workflow

The typical workflow for training and testing the maneuver detection model involves the following steps:
1. **Data Preprocessing**: Run `preprocess.py` to preprocess the flight data.
   (this takes a while, so only run it when your preprocessing has changed)

   ```bash
   python -m maneuver_detection.preprocess
   ```

2. **Model Training**: Run `train.py` to train the model using the preprocessed data.

   ```bash
   python -m maneuver_detection.train
   ```

3. **Model Testing**: Run `test.py` to evaluate the trained model on test data.

   ```bash
   python -m maneuver_detection.test
   ```


