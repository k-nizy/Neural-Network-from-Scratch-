"""SGD optimizer."""


class SGD:
    """Stochastic gradient descent: param -= lr * grad, in place.

    Not a Module: it only reads (param, grad) pairs and mutates the
    parameter arrays directly.
    """

    def __init__(self, parameters, lr: float):
        """Store the parameter list and learning rate.

        Args:
            parameters: List of (param, grad) pairs from model.parameters().
            lr: Learning rate.
        """
        self._parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Update every tracked parameter in place as param -= lr * grad.

        Arrays are never rebound, so the optimizer keeps mutating the
        same arrays the model's parameters() returned.
        """
        for param, grad in self._parameters:
            param -= self.lr * grad

    def zero_grad(self):
        """Clear every tracked gradient in place with fill(0.0)."""
        for _, grad in self._parameters:
            grad.fill(0.0)
