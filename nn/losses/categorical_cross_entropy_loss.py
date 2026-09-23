"""Categorical Cross-Entropy loss implementation."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical Cross-Entropy loss for multi-class classification.

    L = -mean(sum(y * log(a), axis=1))
    """

    def __init__(self):
        """Initialize CCE loss."""
        self._predictions = None
        self._targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute categorical cross-entropy loss.

        Args:
            predictions: Predicted probabilities from softmax, shape (m, C).
            targets: One-hot encoded ground truth, shape (m, C).

        Returns:
            Scalar loss value (Python float).
        """
        self._predictions = predictions
        self._targets = targets
        eps = 1e-15
        a = np.clip(predictions, eps, 1.0 - eps)
        y = targets
        loss = -np.mean(np.sum(y * np.log(a), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute gradient w.r.t. predictions.

        Returns:
            Gradient of shape (m, C).
        """
        a = self._predictions
        y = self._targets
        m = a.shape[0]
        eps = 1e-15
        a_clipped = np.clip(a, eps, 1.0 - eps)
        grad = -(y / a_clipped) / m
        return grad
