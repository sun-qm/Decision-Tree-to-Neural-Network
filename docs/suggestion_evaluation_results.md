# Experimental Suggestion Evaluation

Run command:

```bash
.venv/bin/python -m examples.compare_suggestions
```

This evaluation does not modify the original parser. It adds experimental
models in `dt_to_nn/experimental_variants.py` and compares them against the
current adjacent-layer parser.

## Structure

For the shallow demo tree, the original parser and path-matrix parser have the
same size because there are almost no pass-through chains to remove.

| Tree | Parser | Neurons | Edges | Layers |
|---|---|---:|---:|---:|
| Demo tree | Original adjacent-layer | 12 | 18 | 3 |
| Demo tree | Path matrix | 12 | 18 | 3 |
| Deep unbalanced tree | Original adjacent-layer | 28 | 37 | 5 |
| Deep unbalanced tree | Path matrix | 15 | 27 | 3 |

Conclusion: suggestions 1 and 3 are most useful when the tree is deep or
unbalanced. They remove pass-through chains and force a fixed three-layer
representation.

## Predictive Comparison

Mean over three fixed 80/20 splits on the synthetic structured benchmark.

| Suggestion | Variant | Accuracy | Fidelity to original tree | Recommendation |
|---|---|---:|---:|---|
| 0 | Original adjacent-layer baseline | 0.891 +/- 0.010 | 0.589 | Keep as baseline |
| 1+3 | Path-matrix / skip | 0.856 +/- 0.020 | 0.628 | Adopt as experimental parser |
| 4 | Product AND | 0.815 +/- 0.023 | 0.694 | Keep optional |
| 4 | Min AND | 0.774 +/- 0.013 | 0.691 | Do not adopt |
| 4 | Soft-min AND | 0.737 +/- 0.020 | 0.746 | Keep for experiments only |
| 5 | Soft routing | 0.846 +/- 0.019 | 0.681 | Keep optional |
| 6 | Temperature annealing | 0.852 +/- 0.025 | 0.580 | Do not adopt yet |
| 10 | L2 structure regularization, lambda=1e-4 | 0.883 +/- 0.016 | 0.552 | Do not adopt as-is |
| 10 | L2 structure regularization, lambda=1e-2 | 0.852 +/- 0.017 | 0.546 | Do not adopt as-is |

## Additional Experiments

| Suggestion | Experiment | Result | Recommendation |
|---|---|---|---|
| 7 | Oblique predicate vs axis-aligned predicate | 0.993 vs 0.786 accuracy | Adopt in a future predicate API |
| 9 | Random forest to path-matrix ensemble | 0.906 NN ensemble vs 0.911 sklearn RF | Adopt as optional module |

## Decision

Adopt for future implementation:

- Suggestions 1 and 3 together: path-matrix parser with skip-style direct condition-to-path connections.
- Suggestion 8: compression and pruning are natural on the path matrix representation.
- Suggestion 9: ensemble conversion as an optional module.
- Suggestion 7: arbitrary predicates, especially oblique predicates, but only after designing a clean predicate API.

Keep as optional experiments:

- Suggestion 4 product AND.
- Suggestion 5 soft routing.

Do not adopt yet:

- Suggestion 4 min AND.
- Suggestion 6 annealing as currently implemented.
- Suggestion 10 simple L2-to-initial regularization. It did not preserve rule fidelity in this benchmark; stronger hard constraints are likely needed.

Main parser status:

- Keep the original exact parser unchanged for correctness and paper fidelity.
- Add path-matrix as an experimental alternative first, then promote it only after testing on larger and real datasets.
