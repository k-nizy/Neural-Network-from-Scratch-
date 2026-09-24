"""ReLU activation."""

import numpy as np

from nn.module import Module


class ReLU(Module):
    """ReLU activation: f(x) = max(0, x)."""

    def __init__(self):
        """Set up the stored mask."""
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute max(0, x) elementwise.

        Args:
            x: Input array of any shape.

        Returns:
            Same shape, negatives zeroed.
        """
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Pass gradients where x > 0, block them where x <= 0.

        Args:
            grad_output: Upstream gradient, same shape as the input.

        Returns:
            Same shape; 0 where the input was <= 0 (so x = 0 gets 0).
        """
        return grad_output * self._mask
