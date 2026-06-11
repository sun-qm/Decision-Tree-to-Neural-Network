# Editable XAI Paper-Exact Parser Evaluation

## Implementation

`convert_tree_to_paper_exact_network()` follows Editable XAI Eq. (1)-(5):

```text
true condition  = sigmoid(+x_j - tau)
false condition = sigmoid(-x_j + tau)
path AND        = ReLU(left + right - 1)
pass through    = ReLU(source)
class OR        = cReLU(sum(class paths))
```

The first-layer weights are exactly `+1` or `-1`; no temperature multiplier is
used. Hidden layers have a minimum width of twice the number of decision nodes,
as specified in Section 4.2.4.

This implementation is separate from `convert_tree_to_network()`, which uses
hard branch indicators to preserve exact decision-tree behavior.

## Reproducible evaluation

Run:

```bash
.venv/bin/python -m examples.evaluate_paper_exact
```

Configuration:

```text
tree: dt_to_nn.demo.build_demo_tree()
random samples: 10,000
random range: [-2, 2] for each of 3 features
seed: 42
threshold probes: tau-epsilon, tau, tau+epsilon for every decision node
epsilon: 1e-9
```

Results:

| Parser and sample group | Label agreement | Exact one-hot agreement |
|---|---:|---:|
| Paper sigmoid, random | 94.36% | 0.00% |
| Paper sigmoid, threshold probes | 88.89% | 0.00% |
| Paper sigmoid, combined | 94.355% | 0.00% |
| Hard indicator, combined | 100.00% | 100.00% |

For the paper sigmoid parser on combined samples:

```text
mean absolute output error: 0.205684
maximum output error:       0.995959
```

## Interpretation

The new parser is exact with respect to the paper's stated neural-network
equations, but the resulting smooth network is not mathematically identical to
the discrete decision tree on every input. At a threshold, both complementary
sigmoid neurons output `0.5`. Away from thresholds they approach, but do not
equal, binary indicators.

Consequently:

- Formula/parameter fidelity to the paper is exact.
- Topological correspondence with the tree is preserved.
- Prediction agreement is empirical and depends on the tree and input
  distribution.
- Exact one-hot output equivalence should not be expected with a finite smooth
  sigmoid.
