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

        Multiplies the upstream gradient by the mask stored during
        ``forward``: gradients pass through where the input was > 0 and
        are blocked (set to 0) where it was <= 0, so x = 0 gets zero
        gradient by convention.

        Args:
            grad_output: Upstream gradient of same shape as forward input.

        Returns:
            Gradient w.r.t. the input, same shape, 0 where input <= 0.
        """
        return grad_output * self._mask
