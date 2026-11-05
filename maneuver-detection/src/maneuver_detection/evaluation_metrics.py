"""Module for evaluating maneuver detection performance metrics."""

from functools import cached_property

import numpy as np
import pandas as pd

from maneuver_detection.config import Config


class EvaluationMetrics:
    """Class to compute evaluation metrics for maneuver detection."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, config: Config) -> None:
        """Initialize.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            config: Project configuration.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self._config = config

        n_labels = len(self._config.all_labels)
        self._true_maneuver_counts = pd.DataFrame([[0] * n_labels], columns=self._config.all_labels)
        self._detected_maneuver_counts = self._true_maneuver_counts.copy()
        self._detect_maneuvers()

    @cached_property
    def accuracy(self) -> float:
        """Compute overall accuracy."""
        return (self.y_pred == self.y_true).mean()

    @cached_property
    def precision(self) -> float:
        """Compute precision across all maneuvers."""
        precisions = []
        support = []
        for label in self._config.all_labels:
            y_pred_label = self.y_pred == label
            y_true_label = self.y_true == label

            count_detected = np.count_nonzero(y_pred_label & y_true_label)
            count_predicted = np.count_nonzero(y_pred_label)
            count_true = np.count_nonzero(y_true_label)

            precision = 0.0
            if count_predicted > 0:
                precision = count_detected / count_predicted

            precisions.append(precision)
            support.append(count_true)

        return float(np.average(precisions, weights=support))

    @cached_property
    def maneuver_recall(self) -> float:
        """Compute maneuver recall across all maneuvers."""
        return self._maneuver_recall(self._config.all_labels)

    @cached_property
    def maneuver_recall_only_foreground(self) -> float:
        """Compute maneuver recall excluding 'no maneuver'."""
        return self._maneuver_recall(self._config.foreground_labels)

    @cached_property
    def per_maneuver_recalls(self) -> pd.DataFrame:
        """Compute per-maneuver recalls."""
        return self._detected_maneuver_counts / self._true_maneuver_counts

    def _maneuver_recall(self, labels: list[int]) -> float:
        detected_maneuvers = self._detected_maneuver_counts.loc[:, labels]
        true_maneuvers = self._true_maneuver_counts.loc[:, labels]

        total_detected_maneuvers = detected_maneuvers.sum(axis="columns").iloc[0]
        total_true_maneuvers = true_maneuvers.sum(axis="columns").iloc[0]

        return total_detected_maneuvers / total_true_maneuvers

    def _detect_maneuvers(self) -> None:
        maneuver_change_indices = np.flatnonzero(self.y_true[:-1] != self.y_true[1:])
        maneuver_start_indices = np.insert(maneuver_change_indices + 1, 0, 0)
        maneuver_end_indices = np.append(maneuver_start_indices[1:], len(self.y_true))

        for start_index, end_index in zip(maneuver_start_indices, maneuver_end_indices):
            true_label = self.y_true[start_index]
            predicted_labels = self.y_pred[start_index:end_index]

            self._true_maneuver_counts[true_label] += 1

            maneuver_accuracy = (predicted_labels == true_label).mean()
            if maneuver_accuracy >= self._config.maneuver_detection_threshold:
                self._detected_maneuver_counts[true_label] += 1
