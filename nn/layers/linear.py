"""Fully connected linear layer."""

import numpy as np

from nn.module import Module


class Linear(Module):
    """Affine layer: Y = X @ W + b.

    Attributes:
        W: Weight matrix, shape (in_features, out_features).
        b: Bias vector, shape (out_features,), starts at zeros.
        dW: Gradient of the loss w.r.t. W, same shape as W.
        db: Gradient of the loss w.r.t. b, same shape as b.
    """

    def __init__(self, in_features: int, out_features: int):
        """Set up weights (Xavier uniform) and a zero bias.

        Args:
            in_features: Number of input features (n).
            out_features: Number of output neurons (C).
        """
        # Xavier uniform: keep activation variance stable across layers.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute Y = X @ W + b.

        Args:
            x: Input of shape (m, n), m = batch size.

        Returns:
            Output of shape (m, C). With C = 1 the shape is still (m, 1).
        """
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute dW, db and the input gradient from the upstream gradient.

        Batch gradients are summed (dW = X^T G, db = G.sum(axis=0));
        the input gradient is G W^T.

        Args:
            grad_output: Upstream gradient of shape (m, C).

        Returns:
            Gradient w.r.t. the input, shape (m, n).
        """
        x = self._x
        np.copyto(self.dW, x.T @ grad_output)
        np.copyto(self.db, grad_output.sum(axis=0))
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        """Return [(W, dW), (b, db)] -- the actual stored arrays."""
        return [(self.W, self.dW), (self.b, self.db)]
