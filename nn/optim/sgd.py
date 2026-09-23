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
        """Perform one optimization step.

        For every tracked (param, grad) pair, updates the parameter in
        place as ``param -= lr * grad``. No array is rebound, so the
        optimizer keeps mutating the exact same arrays the model's
        ``parameters()`` returned.
        """
        for param, grad in self._parameters:
            param -= self.lr * grad

    def zero_grad(self):
        """Zero out all tracked gradients in place.

        Fills each stored gradient array with 0.0 (same array object, no
        rebinding), so newly computed gradients can be accumulated into
        or overwrite them on the next backward pass.
        """
        for _, grad in self._parameters:
            grad.fill(0.0)
