"""Binary Cross-Entropy loss implementation."""

import numpy as np


class CrossEntropyLoss:
    """Binary Cross-Entropy loss for binary classification.

    L = -mean(y * log(a) + (1 - y) * log(1 - a))
    """

    def __init__(self):
        """Initialize BCE loss."""
        self._predictions = None
        self._targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        Args:
            predictions: Predicted probabilities, shape (m, 1) or (m,).
            targets: Ground truth labels (0 or 1), shape (m, 1) or (m,).

        Returns:
            Scalar loss value (Python float).
        """
        self._predictions = predictions
        self._targets = targets
        eps = 1e-15
        a = np.clip(predictions, eps, 1.0 - eps)
        y = targets
        loss = -np.mean(y * np.log(a) + (1.0 - y) * np.log(1.0 - a))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute gradient w.r.t. predictions.

        Returns:
            Gradient of shape (m, 1) or (m,).
        """
        a = self._predictions
        y = self._targets
        m = a.shape[0]
        eps = 1e-15
        a_clipped = np.clip(a, eps, 1.0 - eps)
        grad = -(y / a_clipped - (1.0 - y) / (1.0 - a_clipped)) / m
        return grad