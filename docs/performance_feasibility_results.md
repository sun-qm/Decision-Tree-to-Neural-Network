# Performance Feasibility Baselines

This experiment evaluates the soft Editable-XAI tree-to-network parser against
source trees/forests, same-architecture random initialization, a DJINN-like
tree-informed architecture, an EntropyNet-like path expansion network, and a
parameter-matched MLP.

Sources used for the implementation choices:

- Requirement document: `/Users/amywang/Downloads/Performance Feasibility Study.pdf`
- Entropy Net reference: `/Users/amywang/Desktop/CP2107/Entropy_nets_from_decision_trees_to_neural_networks.pdf`
- DJINN reference implementation: <https://github.com/LLNL/DJINN>

## Setup

- Datasets: diabetes, California Housing, wine, breast cancer.
- Settings: one decision tree and a 10-tree random forest.
- Splits: five fixed 80/20 train/test splits, seeds 0-4; 20% of train is used
  for validation.
- Preprocessing: features are scaled to `[0, 1]`; regression targets are
  standardized for training and transformed back for reported metrics.
- Alpha grid for soft tree predicates: `0.5, 1, 2, 5, 10, 20, 50, 100`.
- Alpha selection: for `coexplain_soft`, `same_arch_random`, and
  `path_expansion`, the reported row is selected per split by minimum best
  validation loss.
- California Housing is capped at 2500 samples in the default feasibility
  configuration. Use `--no-dataset-cap` to rerun on the full sklearn dataset.

Raw outputs are in `results/performance_feasibility_full/`.

## Baselines

- `source_tree`: the fitted sklearn decision tree or random forest.
- `coexplain_soft`: Editable-XAI soft parser with sigmoid threshold predicates
  and Lukasiewicz/ReLU path AND.
- `same_arch_random`: the same soft parser architecture with random
  initialization.
- `djinn`: a PyTorch implementation of the LLNL DJINN tree-to-architecture
  mapping: growing hidden widths, pass-through feature connections, and Xavier
  child-split connections.
- `path_expansion`: EntropyNet-like fixed path expansion: soft branch
  probabilities are multiplied along each root-to-leaf path and leaf outputs are
  aggregated.
- `mlp`: a two-hidden-layer ReLU MLP with approximately matched parameter count.

## Primary Results

Regression reports MSE, lower is better.

| Dataset | Setting | Source | CoExplain Soft | Random Same Arch | DJINN-like | Path Expansion | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diabetes | single tree | 4553.51 | 4203.36 | 5400.85 | **3852.24** | 4218.95 | 5152.04 |
| Diabetes | ensemble 10 | 3536.45 | 3467.36 | 5400.85 | **3416.29** | 3485.49 | 5006.45 |
| California | single tree | 0.5570 | 0.4515 | 1.3785 | **0.3423** | 0.4081 | 0.3549 |
| California | ensemble 10 | 0.4623 | 0.3767 | 1.3785 | **0.3252** | 0.3508 | 0.3306 |

Classification reports accuracy, higher is better.

| Dataset | Setting | Source | CoExplain Soft | Random Same Arch | DJINN-like | Path Expansion | MLP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wine | single tree | 0.9056 | 0.9111 | 0.5056 | **0.9889** | 0.9111 | 0.8556 |
| Wine | ensemble 10 | 0.9944 | 0.9944 | 0.9722 | 0.9944 | 0.9944 | 0.9944 |
| Breast cancer | single tree | 0.9439 | 0.9632 | 0.4930 | 0.9684 | 0.9667 | **0.9719** |
| Breast cancer | ensemble 10 | 0.9544 | 0.9737 | 0.9807 | **0.9842** | 0.9737 | 0.9789 |

## Key Findings

1. Tree-informed soft initialization is consistently useful. `coexplain_soft`
   beats `same_arch_random` on every regression task and every single-tree
   classification task, and starts with much higher source-model fidelity.
2. `path_expansion` is the most faithful compact variant. It is usually close
   to `coexplain_soft` in final performance and often has slightly better
   initial/final fidelity to the source tree or forest.
3. `djinn` is a strong predictive baseline, especially after training, but it is
   much larger. Its trainable parameter count is often about 4-10x the soft
   parser/path-expansion count, and its final fidelity to the source tree is
   usually lower.
4. MLP can be competitive on California and breast cancer, but it has no tree
   semantics and weak initialization quality.
5. Validation usually selects large alpha values for regression and single-tree
   tasks, suggesting that near-hard predicates preserve useful tree behavior.
   Breast-cancer ensemble often selects softer alpha values around 10, which
   suggests that smoothness can help after ensembling.

## Fidelity Notes

For classification, fidelity is agreement with the source tree/forest label.
For regression, fidelity is MSE to the source tree/forest prediction, so lower
is better.

- Diabetes ensemble fidelity: `path_expansion` 23.50, `coexplain_soft` 26.29,
  `djinn` 780.20, `mlp` 2443.85.
- California ensemble fidelity: `coexplain_soft` 0.1248, `path_expansion`
  0.1380, `mlp` 0.1688, `djinn` 0.2244.
- Wine ensemble fidelity: `coexplain_soft` and `path_expansion` both 1.0000.
- Breast-cancer ensemble fidelity: `coexplain_soft` 0.9772,
  `path_expansion` 0.9702, `same_arch_random` 0.9667, `djinn` 0.9632.

## Recommendation

For an AAAI-style direction, use `path_expansion` or a topology-aware variant as
the main proposed compact baseline: it keeps the tree semantics, is small, and
trains competitively. Keep `coexplain_soft` as the direct Editable-XAI baseline,
`same_arch_random` as the initialization ablation, and `djinn`/MLP as predictive
baselines showing the trade-off between performance, size, and editability.
