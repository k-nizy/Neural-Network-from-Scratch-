# Formative 1, Part 1 — Neural Network from Scratch (NumPy only)

A small neural-network library built with NumPy only — no deep-learning
framework — plus `main.py`, which trains a single linear neuron on the AND
gate with binary cross-entropy and SGD.

## Structure

```
nn/
  module.py                                    Module: forward/backward contract
  layers/linear.py                             Linear: Y = X @ W + b, Xavier init
  activations/relu.py                          ReLU + mask backward
  activations/sigmoid.py                       Sigmoid, clipped stable forward
  activations/softmax.py                       row-wise softmax, vectorized JVP backward
  losses/cross_entropy_loss.py                 binary cross-entropy (not a Module)
  losses/categorical_cross_entropy_loss.py     categorical cross-entropy (not a Module)
  optim/sgd.py                                 in-place SGD step + zero_grad
main.py                                        toy_data, train, accuracy (AND gate)
```

## How to run

```bash
pytest                 # public stage checks (provided tests/)
ruff check nn/ main.py # lint + docstring checks
python main.py         # trains AND: loss -> ~0.004, accuracy 1.0000
```

## Key decisions

- **Shapes:** inputs `(m, n)`, weights `(n, C)`, bias `(C,)`, outputs `(m, C)`;
  a single output neuron is `C = 1`, so outputs stay `(m, 1)`.
- **Init:** Xavier/Glorot uniform, `W ~ U(±sqrt(6 / (in + out)))` — shape-aware,
  bias starts at zeros.
- **Gradients:** `Linear.backward` **sums** over the batch (`dW = XᵀG`,
  `db = G.sum(axis=0)`, `dX = GWᵀ`); losses **average** (`/ m`), so
  `Sigmoid + BCE` and `Softmax + CCE` both reduce to the classic
  `(a − y) / m` shortcut. BCE divides by element count so `backward` is the
  exact gradient of `forward` for `(m,)` and `(m, 1)` shapes alike.
- **Stability:** softmax shifts logits by the row max (rows of ±1e9 stay
  finite); sigmoid clips input to ±500; both losses clip predictions to
  `[1e-15, 1 − 1e-15]` before `log`.
- **Vectorization:** no data-level loops anywhere in `nn/`; `Softmax.backward`
  uses the closed-form JVP `a ∘ (g − (a·g) 1)` in one broadcasted expression.
- **In-place SGD:** `step()` does `param -= lr * grad` on the exact arrays
  `parameters()` returned; `zero_grad()` fills the stored gradient arrays.

## Verification

All 62 public checks pass (`pytest`), every stage reports `PASS` with
`LINT clean`, `ruff check nn/ main.py` exits 0, and training converges to
accuracy 1.0 across seeds 0/1/2/7/42. Gradients were additionally validated
by central-difference checks (Linear→Sigmoid→BCE and Softmax→CCE agree with
finite differences to ~1e-10; the vectorized softmax JVP matches the explicit
Jacobian to ~1e-16).

## Use of AI tooling

An AI coding assistant was used for debugging, lint cleanup, and explaining
course material — never as the author of the implementation. All design
decisions above were reviewed and understood by the author.

## References

- Course guide: `guide.pdf` (assignment spec, chapters 0.5–10).
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep
  feedforward neural networks," AISTATS 2010 (Xavier initialization).
- NumPy documentation, https://numpy.org/doc/.
- Ruff documentation, https://docs.astral.sh/ruff/.
