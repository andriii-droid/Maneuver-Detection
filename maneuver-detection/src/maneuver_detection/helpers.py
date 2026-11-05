"""Helper functions for data extractions and model persistence."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def extract_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from the dataframe."""
    x = df.drop(columns=["Maneuver_Label"]).to_numpy()
    y = df["Maneuver_Label"].to_numpy()
    return x, y


def save_classifiers(classifiers: list[BaseEstimator], path: Path) -> None:
    """Save the list of classifiers to a file."""
    print(f"Saving classifiers to {path}")
    with open(path, "wb") as file:
        pickle.dump(classifiers, file)


def load_classifiers(path: Path) -> list[BaseEstimator]:
    """Load the list of classifiers from a file."""
    print(f"Loading classifiers from {path}")
    with open(path, "rb") as file:
        classifiers = pickle.load(file)
    return classifiers
