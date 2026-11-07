"""Train and evaluate classifiers for maneuver detection."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from maneuver_detection.config import get_default_config
import maneuver_detection.evaluate as ev
from maneuver_detection.helpers import extract_features_and_labels, save_classifiers

config = get_default_config()
classifiers = [
    # TODO: Add your classifiers here
    LinearDiscriminantAnalysis(),
]

df_train = pd.read_parquet(config.train_data_path)
x_train, y_train = extract_features_and_labels(df_train)
print(df_train.columns[df_train.isna().any()].tolist())

for classifier in classifiers:
    classifier_name = type(classifier).__name__

    print(f"Training {classifier_name} ...")
    # TODO: Fit your classifier here
    classifier.fit(x_train, y_train)

    print(f"Evaluating {classifier_name} ...")
    # TODO: Evaluate your classifier here
    metrics = ev.evaluate(classifier_name, y_train, classifier.predict(x_train), get_default_config())

save_classifiers(classifiers, config.classifiers_path)
plt.show()
