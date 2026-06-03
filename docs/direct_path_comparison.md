# Direct Path Parser Comparison

Run command:

```bash
.venv/bin/python -m examples.compare_direct_path
```

The direct-path parser creates one path neuron for each root-to-leaf trace:

```text
path = ReLU(sum(required_branch_conditions) - (path_length - 1))
```

Then each path neuron connects directly to its class output.

## Exactness

Both parsers exactly matched the input decision tree on random and threshold-probe samples.

| Parser | Prediction agreement | Output-vector agreement | Max output error |
|---|---:|---:|---:|
| Original adjacent-layer | 1.000 | 1.000 | 0.000 |
| Direct path | 1.000 | 1.000 | 0.000 |

## Structure

For a shallow demo tree, both representations have the same size.

| Tree | Parser | Neurons | Edges | Layers |
|---|---|---:|---:|---:|
| Demo tree | Original adjacent-layer | 12 | 18 | 3 |
| Demo tree | Direct path | 12 | 18 | 3 |

For a deep unbalanced tree, direct path removes pass-through chains and partial path neurons.

| Tree | Parser | Neurons | Edges | Layers |
|---|---|---:|---:|---:|
| Deep unbalanced | Original adjacent-layer | 28 | 37 | 5 |
| Deep unbalanced | Direct path | 15 | 27 | 3 |

## Training Comparison

On the synthetic benchmark:

| Model | Accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|
| Tree before training | 0.605 | 0.599 | 0.592 |
| Original adjacent-layer after training | 0.940 | 0.942 | 0.937 |
| Direct path after training | 0.900 | 0.906 | 0.894 |

## Recommendation

Use direct path when:

- the goal is exact DT-to-NN equivalence,
- the tree is deep or unbalanced,
- deployment size and fixed-depth structure matter,
- partial path neurons are not needed for inspection.

Use the original recursive adjacent-layer parser when:

- strict paper-topology fidelity matters,
- you want to inspect intermediate partial paths,
- training performance on the target dataset is better with the deeper structure.

Current decision: keep the original parser as the default, and keep direct path as a compact alternative.
