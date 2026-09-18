"""Softmax activation implementation."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation function.

    Computes stable row-wise softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    Uses max-shift technique for numerical stability.
    """

    def __init__(self):
        """Initialize Softmax activation."""
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute stable row-wise softmax forward pass.

        Args:
            x: Input array of shape (m, C) where m is batch size, C is classes.

        Returns:
            Output array of shape (m, C) with row-wise probabilities summing to 1.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self._output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute softmax backward pass using Jacobian-vector product.

        Args:
            grad_output: Upstream gradient of shape (m, C).

        Returns:
            Gradient w.r.t. input, shape (m, C).
        """
        m, C = grad_output.shape
        grad_input = np.zeros_like(grad_output)
        for i in range(m):
            a = self._output[i]
            g = grad_output[i]
            jac = np.diag(a) - np.outer(a, a)
            grad_input[i] = jac @ g
        return grad_input