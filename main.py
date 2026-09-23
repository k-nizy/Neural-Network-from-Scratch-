"""Main training script: a single linear neuron trained on the AND dataset.

Wires together the pieces built in Chapters 1-9 -- a Linear layer, a Sigmoid
activation, a binary cross-entropy loss, and an SGD optimizer -- into one
full-batch training loop (Chapter 10), then reports the final loss and
classification accuracy when run as a script.
"""

import numpy as np

from nn.activations import Sigmoid
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.optim import SGD

# State shared between train() and accuracy(): the trained model and the
# data it was trained on. accuracy() can be called without a prior explicit
# train() call, in which case it trains with the default settings first.
_X = None
_Y = None
_linear = None
_sigmoid = None


def toy_data():
    """Return the 4-sample AND-gate toy dataset.

    The AND gate is the standard single-neuron sanity check: it is linearly
    separable, so a single Linear + Sigmoid can fit it exactly (XOR cannot).

    Returns:
        X: Input features of shape (4, 2); each row holds two input bits.
        y: Binary labels of shape (4, 1); y = x1 AND x2 for each row.
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
    """Train Linear -> Sigmoid with binary cross-entropy on the AND dataset.

    Full-batch gradient descent: every epoch computes the loss and gradients
    over all 4 samples, then the optimizer applies one update and clears the
    gradients for the next epoch.

    Args:
        epochs: Number of full-batch gradient-descent steps.
        lr: Learning rate passed to the SGD optimizer.
        seed: Seed for NumPy's global RNG, making the weight initialization
            reproducible.

    Returns:
        List of Python floats: the loss recorded at every epoch, in order.
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
    """Compute classification accuracy of the trained model on the AND data.

    Predictions are thresholded at 0.5 and compared against the labels.

    Args:
        loss_history: Optional loss history as returned by train(); accepted
            for interface compatibility and not needed to score predictions.

    Returns:
        Accuracy as a Python float between 0.0 and 1.0.
    """
    del loss_history  # not needed to score predictions
    if _linear is None or _X is None:
        train()

    a = _sigmoid.forward(_linear.forward(_X))
    preds = (a >= 0.5).astype(float)
    return float(np.mean(preds == _Y))


if __name__ == "__main__":
    losses = train(epochs=4000, lr=1.0, seed=0)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Accuracy: {accuracy(losses):.4f}")
