"""ReLU activation implementation."""

import numpy as np

from nn.module import Module


class ReLU(Module):
    """ReLU activation function: f(x) = max(0, x)."""

    def __init__(self):
        """Initialize ReLU activation."""
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ReLU forward pass.

        Args:
            x: Input array of any shape.

        Returns:
            Output array of same shape with negative values zeroed.
        """
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute ReLU backward pass.

        Args:
            grad_output: Upstream gradient of same shape as forward input.

        Returns:
            Gradient w.r.t. input, same shape.
        """
        return grad_output * self._mask