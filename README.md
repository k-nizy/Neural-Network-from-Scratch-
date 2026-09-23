# Formative 1, Part 1 — A Neural Network from Scratch in NumPy

A minimal autograd-style neural network library built with NumPy only — no
deep-learning framework anywhere in the project — plus a training script that
fits a single linear neuron to the AND gate.

## Structure

```
nn/
  module.py                       Module: the forward/backward contract (Ch 0.5)
  layers/linear.py                Linear: Y = X @ W + b, Xavier init (Ch 1-3)
  activations/relu.py             ReLU + mask backward (Ch 4)
  activations/sigmoid.py          Sigmoid + clipped stable forward (Ch 5)
  activations/softmax.py          row-wise softmax, max-shift, JVP backward (Ch 2, 7)
  losses/cross_entropy_loss.py    binary cross-entropy, not a Module (Ch 6)
  losses/categorical_cross_entropy_loss.py   categorical cross-entropy (Ch 8)
  optim/sgd.py                    in-place SGD step + zero_grad (Ch 9)
main.py                           training loop on the AND dataset (Ch 10)
```

## Conventions that everything else depends on

- **Shapes.** Input `X` is `(m, n)`; weights `W` are `(n, C)` (inputs by
  outputs); `b` is `(C,)` and broadcasts across the batch; outputs are
  `(m, C)`. A single output neuron is just `C = 1`, so `Linear(2, 1)` returns
  `(m, 1)` arrays — never a bare float.
- **Who stores what.** Layers/activations cache forward values (`_x`, `_mask`,
  `_output`) during `forward` so `backward` can reuse them. Losses store
  `(predictions, targets)` instead — they see the graph from the top.
- **Who is not a Module.** Losses and the optimizer are plain classes on
  purpose: they don't have a forward/backward position in the same sense.
  A loss's `backward()` takes no `grad_output` because it is the top of the
  graph; the SGD optimizer only reads `(param, grad)` pairs and mutates them
  in place.
- **Sum vs mean.** `Linear.backward` **sums** gradients over the batch
  (`dW = Xᵀ @ G`, `db = G.sum(axis=0)`) because each example contributes its
  own gradient; losses **average** (`/ m`) so the loss magnitude doesn't grow
  with batch size. Getting these two backwards trains silently wrong.
- **Binary vs categorical.** BCE divides by the number of *elements*
  (`predictions.size`), keeping `backward` the exact gradient of `forward`
  for both `(m,)` and `(m, 1)` inputs; CCE divides by batch size `m` over
  `(m, C)` one-hot targets, which makes Softmax + CCE reduce to the classic
  `(a - y) / m` shortcut.

## Design decisions worth defending

- **Xavier/Glorot uniform init:** `W ~ Uniform(±sqrt(6 / (in + out)))`, the
  shape-aware bound that keeps activation variance roughly constant across
  layers. A zeros or `randn * 0.01` init would fail the shape-aware check.
- **Numerical stability:** Softmax shifts logits by the row max before
  exponentiating, so `exp` never overflows even for logits like ±1e9.
  Sigmoid clips its input to ±500 before `exp`. Both losses clip predictions
  to `[1e-15, 1 - 1e-15]` before `log`/division, so a confident wrong
  prediction produces a huge-but-finite loss instead of `nan`.
- **Softmax backward** builds the Jacobian `diag(a) - a aᵀ` per row and
  applies it to the upstream gradient — an explicit per-example loop, which
  the assignment allows for this stage. The equivalent vectorized JVP form is
  `a * (g - (a·g).sum())`, verified to agree with the loop to machine
  precision.
- **SGD in-place contract:** `step()` does `param -= lr * grad` (rebinding a
  new array would desynchronize the optimizer from the layer's weights) and
  `zero_grad()` fills stored gradient arrays with 0. The optimizer holds
  references to the *same* arrays `Linear.parameters()` returned.

## Verification

```bash
pytest               # 62 public checks, stages 1-10
ruff check nn/ main.py
python main.py       # trains AND: loss -> ~0.004, accuracy 1.0000
```

Gradients were additionally validated by central-difference checks against
the analytic gradients (Linear chains through Sigmoid + BCE, and Softmax +
CCE), all agreeing to ~1e-10. Convergence was confirmed across seeds
0, 1, 2, 7, 42 — all reach accuracy 1.0 with finite, decreasing loss.

## What the AND dataset is doing here

The network is a *linear* classifier (one Linear, one activation, nothing in
between), so it can only learn linearly separable targets. XOR is the
textbook non-separable example and is impossible for this architecture
regardless of training; AND is linearly separable, which makes it a genuine
end-to-end sanity check of the pipeline rather than a test of the wrong
thing.

## Use of AI tooling

Development used an AI coding assistant for debugging, lint cleanup, and
explaining course material; all submitted implementation decisions were
reviewed and understood by the author. See the course policy on AI use.

## References

- Course guide: `guide.pdf` (assignment spec, chapters 0.5-10).
- X. Glorot and Y. Bengio, "Understanding the difficulty of training deep
  feedforward neural networks," AISTATS 2010 — the Xavier initialization
  bound used in `nn/layers/linear.py`.
- NumPy documentation, https://numpy.org/doc/ — array API semantics
  (broadcasting, `keepdims`, in-place ufuncs).
- Ruff (linter) documentation, https://docs.astral.sh/ruff/ — used for the
  documentation/style checks configured in `pyproject.toml`.
