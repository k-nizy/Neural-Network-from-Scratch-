"""Linear layer implementation."""

import numpy as np

from nn.module import Module


class Linear(Module):
    """Linear (affine) layer: Y = X @ W + b.

    Attributes:
        W: Weight matrix of shape (in_features, out_features).
        b: Bias vector of shape (out_features,).
        dW: Gradient of loss w.r.t. W, shape (in_features, out_features).
        db: Gradient of loss w.r.t. b, shape (out_features,).
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize Linear layer with Xavier uniform weights and zero bias.

        Args:
            in_features: Number of input features (n).
            out_features: Number of output features (C).
        """
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features, dtype=float)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass: Y = X @ W + b.

        Args:
            x: Input array of shape (m, n) where m is batch size.

        Returns:
            Output array of shape (m, C).
        """
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward pass gradients.

        Args:
            grad_output: Upstream gradient of shape (m, C).

        Returns:
            Gradient w.r.t. input X, shape (m, n).
        """
        x = self._x
        np.copyto(self.dW, x.T @ grad_output)
        np.copyto(self.db, grad_output.sum(axis=0))
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self):
        """Return learnable parameters and their gradients.

        Returns:
            List of (param, grad) pairs: [(W, dW), (b, db)].
        """
        return [(self.W, self.dW), (self.b, self.db)]