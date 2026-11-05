"""Evaluate classifier predictions and plot evaluation results."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from maneuver_detection.config import Config
from maneuver_detection.evaluation_metrics import EvaluationMetrics


def evaluate(
    classifier_name: str, y_true: np.ndarray, y_pred: np.ndarray, config: Config
) -> EvaluationMetrics:
    """Evaluate classifier predictions and plot evaluation results.

    Args:
        classifier_name: Name of the classifier.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        config: Project configuration.
    """
    metrics = EvaluationMetrics(y_true, y_pred, config)
    plot_summary(classifier_name, metrics, config)
    plot_per_maneuver_recalls(classifier_name, metrics, config)
    return metrics


def plot_summary(classifier_name: str, metrics: EvaluationMetrics, config: Config) -> None:
    """Plot evaluation summary including confusion matrix and key metrics.

    Args:
       classifier_name: Name of the classifier.
       metrics: Computed evaluation metrics.
       config: Project configuration.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Evaluation Summary: {classifier_name}")

    ConfusionMatrixDisplay.from_predictions(
        y_true=metrics.y_true,
        y_pred=metrics.y_pred,
        ax=axs[0],
        colorbar=False,
        display_labels=config.label_names,
    )
    axs[0].set_title("Confusion Matrix")
    axs[0].xaxis.set_tick_params(rotation=90)

    metric_names = [
        "Maneuver Recall",
        "Maneuver Recall\n(excl. no maneuver)",
        "Precision",
        "Accuracy",
    ]
    metric_values = [
        metrics.maneuver_recall,
        metrics.maneuver_recall_only_foreground,
        metrics.precision,
        metrics.accuracy,
    ]
    metric_colors = ["blue", "green", "red", "black"]
    axs[1].bar(metric_names, metric_values, color=metric_colors)
    axs[1].set_ylim(0, 1)
    axs[1].grid(axis="y", linestyle="--")
    axs[1].set_title("Evaluation Metrics")
    for i, v in enumerate(metric_values):
        axs[1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")


def plot_per_maneuver_recalls(
    classifier_name: str, metrics: EvaluationMetrics, config: Config
) -> None:
    """Plot per-maneuver recall metrics.

    Args:
        classifier_name: Name of the classifier.
        metrics: Computed evaluation metrics.
        config: Project configuration.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(f"Per-Maneuver Recalls: {classifier_name}")
    metric_values = metrics.per_maneuver_recalls.iloc[0, :]
    ax.bar(config.label_names, metric_values)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--")
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
