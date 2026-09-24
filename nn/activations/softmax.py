"""Softmax activation."""

import numpy as np

from nn.module import Module


class Softmax(Module):
    """Row-wise softmax over the last axis.

    Shifts each row by its max before exponentiating so large logits
    stay finite, and computes the backward pass with the vectorized
    Jacobian-vector product.
    """

    def __init__(self):
        """Set up the stored output."""
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute row-wise softmax probabilities.

        Args:
            x: Logits of shape (m, C), m = batch size.

        Returns:
            Probabilities of shape (m, C); each row sums to 1.
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self._output = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Apply each row's softmax Jacobian to the upstream gradient.

        Row i of the Jacobian is diag(a) - a a^T; applying it to g and
        simplifying gives a * (g - sum(g * a)), done for all rows at once.

        Args:
            grad_output: Upstream gradient of shape (m, C), not mutated.

        Returns:
            Gradient w.r.t. the logits, shape (m, C).
        """
        a = self._output
        dot = np.sum(grad_output * a, axis=1, keepdims=True)
        return a * (grad_output - dot)
