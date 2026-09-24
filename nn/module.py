"""Module base class shared by all layers and activations."""

class Module:
    """Base class for layers and activations.

    Defines the forward/backward contract. Modules with no learnable
    weights just use the defaults below.
    """

    def forward(self, x):
        """Compute the output from input x.

        Args:
            x: input array.

        Returns:
            The module's output.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """Propagate the upstream gradient to the input.

        Args:
            grad_output: gradient of the loss w.r.t. this module's output.

        Returns:
            Gradient of the loss w.r.t. this module's input.
        """
        raise NotImplementedError

    def parameters(self):
        """Return learnable parameters as [(param, grad), ...].

        Empty by default for modules without weights.
        """
        return []

    def zero_grad(self):
        """Reset stored gradients.

        Not used in this project: the optimizer clears gradients
        directly (Chapter 9), so modules don't need to override this.
        """
        pass
