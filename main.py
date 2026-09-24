"""Train a single linear neuron on the AND gate (Chapter 10)."""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

# Shared state so accuracy() works even without an explicit train() call.
_X = None
_Y = None
_linear = None
_sigmoid = None


def toy_data():
    """Return the 4-sample AND-gate dataset.

    AND is linearly separable, so one Linear + Sigmoid can fit it
    (XOR cannot).

    Returns:
        X: Inputs of shape (4, 2).
        y: Labels of shape (4, 1), y = x1 AND x2 for each row.
    """
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=float)
    y = np.array([
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ], dtype=float)
    return X, y


def train(epochs=4000, lr=1.0, seed=0):
    """Train Linear -> Sigmoid with binary cross-entropy on the AND data.

    Full-batch gradient descent: each epoch runs forward, loss, backward,
    one optimizer step, then clears the gradients.

    Args:
        epochs: Number of gradient-descent steps.
        lr: Learning rate for SGD.
        seed: Seed for NumPy's global RNG, so the run is reproducible.

    Returns:
        List of Python floats, the loss at every epoch.
    """
    global _X, _Y, _linear, _sigmoid

    np.random.seed(seed)
    _X, _Y = toy_data()
    _linear = Linear(2, 1)
    _sigmoid = Sigmoid()
    loss_fn = CrossEntropyLoss()
    opt = SGD(_linear.parameters(), lr)

    history = []
    for epoch in range(epochs):
        a = _sigmoid.forward(_linear.forward(_X))
        loss = loss_fn.forward(a, _Y)
        history.append(loss)

        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}  loss {loss:.6f}")

        grad = loss_fn.backward()
        grad = _sigmoid.backward(grad)
        _linear.backward(grad)

        opt.step()
        opt.zero_grad()

    return history


def accuracy(loss_history=None):
    """Accuracy of the trained model on the AND data, thresholded at 0.5.

    Args:
        loss_history: Optional loss history from train(); accepted for
            interface compatibility, not needed to score predictions.

    Returns:
        Accuracy as a Python float in [0.0, 1.0].
    """
    del loss_history
    if _linear is None or _X is None:
        train()

    a = _sigmoid.forward(_linear.forward(_X))
    preds = (a >= 0.5).astype(float)
    return float(np.mean(preds == _Y))


if __name__ == "__main__":
    losses = train(epochs=4000, lr=1.0, seed=0)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Accuracy: {accuracy(losses):.4f}")
