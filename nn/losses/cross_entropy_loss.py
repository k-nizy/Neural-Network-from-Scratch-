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
        """Compute binary cross-entropy loss, averaged over all elements.

        Args:
            predictions: Predicted probabilities, shape (m, 1) or (m,).
            targets: Ground truth labels (0 or 1), same shape as predictions.

        Returns:
            Scalar loss value (Python float), the mean of the per-element
            BCE terms.
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

        Dividing by predictions.size (not shape[0]) keeps this the exact
        gradient of forward() for any input shape, including (m, C); for the
        documented (m, 1) / (m,) shapes size equals the batch size m.

        Returns:
            Gradient of the same shape as predictions.
        """
        a = self._predictions
        y = self._targets
        n = a.size
        eps = 1e-15
        a_clipped = np.clip(a, eps, 1.0 - eps)
        grad = -(y / a_clipped - (1.0 - y) / (1.0 - a_clipped)) / n
        return grad
