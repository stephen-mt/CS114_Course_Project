"""Shared model-evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(
    model_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float | str]:
    """Return the metrics used throughout the project as one tidy row."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_class_0": precision[0],
        "recall_class_0": recall[0],
        "f1_class_0": f1[0],
        "precision_class_1": precision[1],
        "recall_class_1": recall[1],
        "f1_class_1": f1[1],
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
    }


def metrics_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Sort experiment rows by holdout macro F1."""
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
