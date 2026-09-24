"""Sigmoid activation."""

import numpy as np

from nn.module import Module


class Sigmoid(Module):
    """Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))."""

    def __init__(self):
        """Set up the stored output."""
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid of x.

        Inputs are clipped to +/-500 first so exp() never overflows.

        Args:
            x: Input array of any shape.

        Returns:
            Same shape, values in (0, 1).
        """
        x_clipped = np.clip(x, -500, 500)
        self._output = 1.0 / (1.0 + np.exp(-x_clipped))
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Multiply the upstream gradient by a * (1 - a).

        Uses the stored output a from the last forward call.

        Args:
            grad_output: Upstream gradient, same shape as the input.

        Returns:
            Same shape as grad_output.
        """
        a = self._output
        return grad_output * a * (1.0 - a)
