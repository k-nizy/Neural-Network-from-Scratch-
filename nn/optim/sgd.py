"""SGD optimizer implementation."""


class SGD:
    """Stochastic Gradient Descent optimizer.

    Updates parameters in place: param -= lr * grad
    """

    def __init__(self, parameters, lr: float):
        """Initialize SGD optimizer.

        Args:
            parameters: List of (param, grad) pairs from model.parameters().
            lr: Learning rate.
        """
        self._parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Perform one optimization step."""
        for param, grad in self._parameters:
            param -= self.lr * grad

    def zero_grad(self):
        """Zero out all gradients in place."""
        for _, grad in self._parameters:
            grad.fill(0.0)