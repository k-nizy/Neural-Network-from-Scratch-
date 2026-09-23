# Formative 1, Part 1 — Building a Neural Network from Scratch in NumPy

Read `guide.pdf` first. It walks you, chapter by chapter, from one neuron to a
full training loop. This folder is your starting point.

## 1. Environment

You need `conda` (Miniconda or Anaconda). From this folder, run once:

```bash
conda env create -f environment.yml
conda activate iml-formative1
```

Re-run `conda activate iml-formative1` in every new terminal. Check it worked:

```bash
python -c "import numpy, pytest; print('numpy', numpy.__version__)"
pytest --version
ruff --version
```

Python 3.11, NumPy 2.x, pytest 8.x, ruff. No deep-learning framework is
installed or permitted.

## 2. What's provided vs. what you build

**Provided — do not edit:**

```
guide.pdf            the assignment
environment.yml      the fixed dependency set
pyproject.toml       ruff + pytest configuration
conftest.py          test fixtures and the stage report
tests/               the public checks (a subset of what is graded)
```

**You build — everything under `nn/`, plus `main.py`:**

```
nn/
  __init__.py                       (empty)
  module.py                         Chapter 0.5
  layers/__init__.py  layers/linear.py
  activations/__init__.py  relu.py  sigmoid.py  softmax.py
  losses/__init__.py  cross_entropy_loss.py  categorical_cross_entropy_loss.py
  optim/__init__.py  sgd.py
main.py                             Chapter 10
README.md                           your own notes (you may overwrite this file)
```

Create these yourself, following the guide. Every `__init__.py` starts empty;
the guide tells you the one import line to add to each as you go.

## 3. Running the checks

Run everything from this folder (the submission root), with the environment
active. Each chapter ends with a "Validate before moving on" box in the guide —
run exactly what it says. In general:

```bash
pytest                       # all public checks (runs tests/ only)
pytest tests/layers/test_stage1_vector.py     # one stage
ruff check nn/               # documentation + style, required from Chapter 1 on
```

Each run prints a grouped **Stage** report and writes `stage<N>_report.json`.
`ruff check nn/` must exit 0 before you move past Chapter 1.

## 4. Important

The public `tests/` are a **subset**. Passing them is necessary, not
sufficient — grading runs a larger private suite plus a short technical
defense. The guide and the rubric list every property that is checked; read
both. Do not try to special-case the tests.

## 5. Submitting

Zip the submission root — `nn/`, `main.py`, your `README.md`, and the provided
files — exactly as laid out above. Do not rename files or move `tests/`.
