# Zero-Padded CoExplain and Sparse-Fixed DJINN Results

This experiment adds two fairness baselines:

- `coexplain_soft_zero_padded`: Editable-XAI-style dense zero padding. Tree
  connections are initialized from the soft parser, while extra dense zero
  connections are trainable and can become nonzero during training.
- `djinn_sparse_fixed`: DJINN-like architecture where every initially zero
  weight is masked after each optimizer step, so zero weights remain zero.

The experiment uses the same datasets, splits, learning rates, epochs, feature
scaling, alpha grid, and validation-loss alpha selection as the previous
performance feasibility run.

Raw outputs are in `results/performance_zero_padded_fairness/`.

## Primary Results

Regression reports test MSE, lower is better.

| Dataset | Setting | Source | CoExplain | CoExplain + Zero Padding | DJINN | DJINN Sparse Fixed | Path Expansion | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Diabetes | single tree | 4553.51 | 4203.36 | 5151.73 | **3852.24** | 5181.25 | 4218.95 | 5152.04 |
| Diabetes | ensemble 10 | 3536.45 | 3467.36 | 4353.90 | **3416.29** | 5246.88 | 3485.49 | 5006.45 |
| California | single tree | 0.5570 | 0.4515 | 0.3696 | **0.3423** | 0.6444 | 0.4081 | 0.3549 |
| California | ensemble 10 | 0.4623 | 0.3740 | 0.3453 | **0.3252** | 0.3803 | 0.3508 | 0.3306 |

Classification reports test accuracy, higher is better.

| Dataset | Setting | Source | CoExplain | CoExplain + Zero Padding | DJINN | DJINN Sparse Fixed | Path Expansion | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wine | single tree | 0.9056 | 0.9111 | 0.9500 | **0.9889** | 0.8278 | 0.9111 | 0.8556 |
| Wine | ensemble 10 | 0.9944 | **0.9944** | 0.9778 | **0.9944** | 0.9611 | **0.9944** | **0.9944** |
| Breast cancer | single tree | 0.9439 | 0.9632 | 0.9649 | 0.9684 | 0.9491 | 0.9667 | **0.9719** |
| Breast cancer | ensemble 10 | 0.9544 | 0.9737 | 0.9614 | **0.9842** | 0.9684 | 0.9737 | 0.9789 |

## Interpretation

1. Zero padding helps when extra capacity is useful, but it is not automatically
   better. It improves California and Wine single-tree performance, but hurts
   Diabetes and the classification ensembles.
2. The zero-padded CoExplain model is much larger than the fixed parser. For
   example, California ensemble uses about 469k parameters versus about 2.9k
   for fixed CoExplain/path expansion.
3. DJINN's unconstrained zero weights matter. `djinn_sparse_fixed` is
   consistently worse than original DJINN, especially on Diabetes and Wine.
   This supports the claim that DJINN's final performance benefits from allowing
   initially zero connections to become nonzero during training.
4. Fixed CoExplain and path expansion remain the most faithful compact models.
   Zero padding often improves predictive flexibility at the cost of lower tree
   fidelity and much larger model size.

## Fairness Takeaway

The earlier direct comparison between fixed `coexplain_soft` and original
`djinn` was not a pure initialization comparison. This run separates the two
questions:

- If CoExplain is also given trainable zero-padded capacity, it can close part
  of the prediction gap on California and Wine.
- If DJINN is forced to keep initially zero weights at zero, its performance
  drops substantially.

So the fair conclusion is:

`djinn` is a strong predictive model partly because it can densify after
initialization, while `coexplain_soft` and `path_expansion` are better compact
and editable tree-structured models. `coexplain_soft_zero_padded` is a useful
enhancement baseline, but its current padding size is large and should be tuned
or regularized before being treated as the main method.
