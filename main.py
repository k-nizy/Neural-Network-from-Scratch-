"""Main training script for the neural network from scratch."""

import numpy as np

from nn.layers import Linear
from nn.activations import Sigmoid
from nn.losses import CrossEntropyLoss
from nn.optim import SGD


_model = None
_loss_fn = None
_X_train = None
_y_train = None


def toy_data():
    """Generate a simple linearly separable toy dataset for binary classification.

    Returns:
        X: Input features of shape (4, 2).
        y: Binary labels of shape (4, 1).
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
        [1.0],
        [1.0],
    ], dtype=float)
    return X, y


def train(epochs=4000, lr=1.0, seed=0):
    """Train a single-layer neural network on the toy dataset.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed for reproducibility.

    Returns:
        List of loss values (Python floats) for each epoch.
    """
    global _model, _loss_fn, _X_train, _y_train

    np.random.seed(seed)
    _X_train, _y_train = toy_data()

    linear = Linear(2, 1)
    sigmoid = Sigmoid()
    _loss_fn = CrossEntropyLoss()
    _model = (linear, sigmoid)

    opt = SGD(linear.parameters(), lr)

    history = []
    for _ in range(epochs):
        z = linear.forward(_X_train)
        a = sigmoid.forward(z)
        loss = _loss_fn.forward(a, _y_train)
        history.append(loss)

        grad = _loss_fn.backward()
        grad = sigmoid.backward(grad)
        grad = linear.backward(grad)

        opt.step()
        opt.zero_grad()

    return history


def accuracy():
    """Compute accuracy on the training data.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    global _model, _X_train, _y_train

    if _model is None:
        train()

    linear, sigmoid = _model
    z = linear.forward(_X_train)
    a = sigmoid.forward(z)
    preds = (a >= 0.5).astype(float)
    acc = float(np.mean(preds == _y_train))
    return acc


if __name__ == "__main__":
    history = train(epochs=4000, lr=1.0, seed=0)
    acc = accuracy()
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Accuracy: {acc:.4f}")