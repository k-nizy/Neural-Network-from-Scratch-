"""Binary cross-entropy loss."""

import numpy as np


class CrossEntropyLoss:
    """Binary cross-entropy, averaged over all elements.

    L = -mean(y * log(a) + (1 - y) * log(1 - a))

    Deliberately not a Module: a loss has no forward/backward position
    in the same sense, and its backward() takes no grad_output because
    it sits at the top of the graph.
    """

    def __init__(self):
        """Set up the stored predictions and targets."""
        self._predictions = None
        self._targets = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute the mean binary cross-entropy.

        Predictions are clipped to [1e-15, 1 - 1e-15] before log so the
        loss stays finite.

        Args:
            predictions: Predicted probabilities, shape (m, 1) or (m,).
            targets: Labels (0 or 1), same shape as predictions.

        Returns:
            The loss as a Python float.
        """
        self._predictions = predictions
        self._targets = targets
        eps = 1e-15
        a = np.clip(predictions, eps, 1.0 - eps)
        y = targets
        loss = -np.mean(y * np.log(a) + (1.0 - y) * np.log(1.0 - a))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Gradient of the loss w.r.t. the predictions.

        Divides by predictions.size, so this is the exact gradient of
        forward() for both (m, 1) and (m,) inputs.

        Returns:
            Gradient, same shape as the predictions.
        """
        a = self._predictions
        y = self._targets
        n = a.size
        eps = 1e-15
        a_clipped = np.clip(a, eps, 1.0 - eps)
        grad = -(y / a_clipped - (1.0 - y) / (1.0 - a_clipped)) / n
        return grad
