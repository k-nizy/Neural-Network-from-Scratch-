"""Sigmoid activation implementation."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))."""

    def __init__(self):
        """Initialize Sigmoid activation."""
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid forward pass.

        Args:
            x: Input array of any shape.

        Returns:
            Output array of same shape with values in (0, 1).
        """
        x_clipped = np.clip(x, -500, 500)
        self._output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute sigmoid backward pass.

        Uses the stored output ``a`` from the last forward call: the local
        derivative is ``a * (1 - a)``, so the input gradient is
        ``grad_output * a * (1 - a)``.

        Args:
            grad_output: Upstream gradient of same shape as forward input.

        Returns:
            Gradient w.r.t. the input, same shape as grad_output.
        """
        a = self._output
        return grad_output * a * (1.0 - a)
