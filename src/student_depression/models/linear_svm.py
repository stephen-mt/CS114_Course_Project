"""Educational dual-form linear SVM implementation."""

from __future__ import annotations

import numpy as np


class LinearSVMDual:
    """Linear SVM optimized with a simplified SMO procedure.

    This implementation prioritizes readability over performance and is intended
    for small educational experiments. Use scikit-learn's SVC for the full data.
    """

    def __init__(
        self,
        C: float = 1.0,
        tolerance: float = 1e-4,
        max_iter: int = 1_000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.alpha_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSVMDual:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("X must be 2D and contain one row per target value")
        if not np.isin(y_array, [0, 1]).all():
            raise ValueError("y must contain binary labels encoded as 0 and 1")

        labels = np.where(y_array == 0, -1.0, 1.0)
        n_samples = len(X_array)
        kernel = X_array @ X_array.T
        alpha = np.zeros(n_samples, dtype=float)
        bias = 0.0
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_iter):
            previous_alpha = alpha.copy()
            for i in range(n_samples):
                error_i = (alpha * labels) @ kernel[:, i] + bias - labels[i]
                violates_kkt = (labels[i] * error_i < -self.tolerance and alpha[i] < self.C) or (
                    labels[i] * error_i > self.tolerance and alpha[i] > 0
                )
                if not violates_kkt:
                    continue

                j = int(rng.integers(0, n_samples - 1))
                if j >= i:
                    j += 1
                error_j = (alpha * labels) @ kernel[:, j] + bias - labels[j]
                old_i, old_j = alpha[i], alpha[j]

                if labels[i] != labels[j]:
                    lower = max(0.0, alpha[j] - alpha[i])
                    upper = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    lower = max(0.0, alpha[i] + alpha[j] - self.C)
                    upper = min(self.C, alpha[i] + alpha[j])
                if lower == upper:
                    continue

                eta = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= labels[j] * (error_i - error_j) / eta
                alpha[j] = np.clip(alpha[j], lower, upper)
                alpha[i] += labels[i] * labels[j] * (old_j - alpha[j])

                bias_i = (
                    bias
                    - error_i
                    - labels[i] * (alpha[i] - old_i) * kernel[i, i]
                    - labels[j] * (alpha[j] - old_j) * kernel[i, j]
                )
                bias_j = (
                    bias
                    - error_j
                    - labels[i] * (alpha[i] - old_i) * kernel[i, j]
                    - labels[j] * (alpha[j] - old_j) * kernel[j, j]
                )
                if 0 < alpha[i] < self.C:
                    bias = bias_i
                elif 0 < alpha[j] < self.C:
                    bias = bias_j
                else:
                    bias = (bias_i + bias_j) / 2

            if np.linalg.norm(alpha - previous_alpha) < self.tolerance:
                break

        self.alpha_ = alpha
        self.intercept_ = float(bias)
        self.coef_ = (alpha * labels) @ X_array
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before decision_function")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)


# Backward-compatible name used by the original coursework notebooks.
LinearSVM_Dual = LinearSVMDual
