"""NumPy implementation of binary logistic regression."""

from __future__ import annotations

import numpy as np


class ScratchLogisticRegression:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1_000,
        l2: float = 0.1,
        threshold: float = 0.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.threshold = threshold
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, X: np.ndarray, y: np.ndarray) -> ScratchLogisticRegression:
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("X must be 2D and contain one row per target value")
        if not np.isin(y_array, [0, 1]).all():
            raise ValueError("y must contain binary labels encoded as 0 and 1")

        n_samples, n_features = X_array.shape
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0

        for _ in range(self.max_iter):
            scores = X_array @ self.coef_ + self.intercept_
            errors = self._sigmoid(scores) - y_array
            coef_gradient = (X_array.T @ errors) / n_samples
            coef_gradient += self.l2 * self.coef_
            intercept_gradient = float(np.mean(errors))

            self.coef_ -= self.learning_rate * coef_gradient
            self.intercept_ -= self.learning_rate * intercept_gradient

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        X_array = np.asarray(X, dtype=float)
        return self._sigmoid(X_array @ self.coef_ + self.intercept_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)


# Backward-compatible name used by the original coursework notebooks.
LogisticRegression = ScratchLogisticRegression
