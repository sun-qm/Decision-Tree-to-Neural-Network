# Evaluation Protocol

This project now evaluates two different questions.

## 1. Exact Parser Correctness

This follows the Editable XAI parser logic:

- each decision node creates two first-layer branch neurons,
- each root-to-leaf trace is represented by ReLU conjunction, `ReLU(a + b - 1)`,
- traces with the same leaf class are aggregated with clipped ReLU, `cReLU(sum(paths))`.

The exact sparse implementation uses hard branch indicators so threshold equality is unambiguous:

```text
true branch:  x_j >= threshold
false branch: x_j <  threshold
```

The metric is exact agreement between the input decision tree and the generated neural network:

- prediction agreement,
- one-hot output-vector agreement,
- maximum output error,
- mismatch examples.

## 2. Paper-Style Training Evaluation

The evaluation script references the DJINN-style evaluation in `2.pdf`:

- fixed five-fold 80/20 train/test splits,
- classification accuracy, macro precision, and macro recall,
- training loss before and after optimization,
- comparison with random dense and random sparse same-architecture baselines.

It also follows Editable XAI for the algorithmic part:

- parsed tree-to-network initialization,
- zero padding with additional neurons initialized with `weights = 0` and `bias = 0`,
- extra pass-through depth when requested,
- retraining with gradient descent to use the padded capacity.

Run:

```bash
python3 -m examples.paper_style_evaluation
```

The key fields are:

- `equivalence_to_input_tree`: whether the parsed network exactly reproduces the tree.
- `tree_or_exact_parsed_nn`: the tree and exact parsed NN have identical dataset performance.
- `zero_padded_parsed_before_training`: soft differentiable parsed model before training.
- `zero_padded_parsed_after_training`: performance after zero padding and training.
- `random_dense_after_training` and `random_sparse_after_training`: same architecture but without tree-informed placement.

Finite dataset training can improve accuracy against ground-truth labels, but after training the network is no longer guaranteed to remain exactly equivalent to the original tree. This is expected: zero padding and training are used to increase model capacity beyond the user-authored rule tree.
