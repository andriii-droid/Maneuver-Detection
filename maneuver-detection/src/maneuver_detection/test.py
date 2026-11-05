"""Script to test trained classifiers on the test dataset and evaluate their performance."""

import matplotlib.pyplot as plt
import pandas as pd

from maneuver_detection.config import get_default_config
from maneuver_detection.helpers import extract_features_and_labels, load_classifiers

config = get_default_config()
classifiers = load_classifiers(config.classifiers_path)

df_test = pd.read_parquet(config.test_data_path)
x_test, y_test = extract_features_and_labels(df_test)

for classifier in classifiers:
    classifier_name = type(classifier).__name__
    print(f"Testing {classifier_name} ...")

    # TODO: Evaluate your classifier here

plt.show()
