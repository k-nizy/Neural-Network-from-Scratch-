"""Softmax activation implementation."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Softmax activation function.

    Computes stable row-wise softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    over the last axis. Uses the max-shift technique for numerical stability
    and a vectorized Jacobian-vector product for the backward pass.
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
        """Compute softmax backward pass using the Jacobian-vector product.

        Each row's Jacobian is ``diag(a) - a a^T``; applying it to the
        upstream gradient ``g`` and simplifying gives the vectorized
        closed form ``a * (g - sum(g * a))`` used here, so one matrix
        expression handles the whole batch with no per-example loop.

        Args:
            grad_output: Upstream gradient of shape (m, C), not mutated.

        Returns:
            Gradient w.r.t. the logits, shape (m, C), where row i is
            ``a_i * (g_i - <g_i, a_i>)``.
        """
        a = self._output
        dot = np.sum(grad_output * a, axis=1, keepdims=True)
        return a * (grad_output - dot)
