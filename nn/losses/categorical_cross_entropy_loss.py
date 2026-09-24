"""Categorical cross-entropy loss."""

import numpy as np


class CategoricalCrossEntropyLoss:
    """Categorical cross-entropy for one-hot targets, averaged per example.

    L = -mean(sum(y * log(a), axis=1))

    Not a Module, for the same reason as the binary loss: its backward()
    takes no grad_output because it is the top of the graph.
    """

    def __init__(self):
        """Set up the stored predictions and targets."""
        self._predictions = None
        self._targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the mean categorical cross-entropy.

        Predictions are clipped to [1e-15, 1 - 1e-15] before log.

        Args:
            predictions: Softmax probabilities, shape (m, C).
            targets: One-hot labels, shape (m, C).

        Returns:
            The loss as a Python float.
        """
        self._predictions = predictions
        self._targets = targets
        eps = 1e-15
        a = np.clip(predictions, eps, 1.0 - eps)
        y = targets
        loss = -np.mean(np.sum(y * np.log(a), axis=1))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Gradient -(y / a) / m; zero for non-target classes.

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
