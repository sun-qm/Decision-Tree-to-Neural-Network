# Current Evaluation Results

Run command:

```bash
.venv/bin/python -m examples.paper_style_evaluation
```

Backend:

- scikit-learn 1.6.1 for splits and metrics
- PyTorch 2.8.0 for cross-entropy backpropagation
- NumPy 2.0.2

## Parser Correctness

The exact sparse parser is still fully equivalent to the input decision tree:

| Metric | Value |
|---|---:|
| Samples | 1009 |
| Prediction agreement | 1.000 |
| Output-vector agreement | 1.000 |
| Max output error | 0.000 |
| Mismatches | 0 |

## Five Fixed 80/20 Splits

| Model | Accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|
| Input tree / exact parsed NN | 0.606 +/- 0.012 | 0.602 +/- 0.013 | 0.598 +/- 0.014 |
| sklearn decision tree trained on data | 0.838 +/- 0.033 | 0.849 +/- 0.018 | 0.837 +/- 0.027 |
| Zero-padded parsed NN before training | 0.607 +/- 0.014 | 0.603 +/- 0.014 | 0.599 +/- 0.015 |
| Threshold-only enhancement | 0.745 +/- 0.031 | 0.746 +/- 0.032 | 0.741 +/- 0.029 |
| Zero-padded parsed NN after training | 0.946 +/- 0.021 | 0.947 +/- 0.018 | 0.945 +/- 0.022 |
| Random dense same architecture | 0.542 +/- 0.000 | 0.271 +/- 0.000 | 0.500 +/- 0.000 |
| Random sparse same architecture | 0.601 +/- 0.118 | 0.390 +/- 0.239 | 0.570 +/- 0.140 |

## Interpretation

The exact parsed network is a correctness check: it reproduces the user tree exactly.

The PyTorch training runs answer a different question: whether the parsed network is a useful initialization for improving predictive performance. In this synthetic structured benchmark, threshold-only training improves the user-authored rule tree, while zero padding plus full topology training improves much more. After training, the model is no longer guaranteed to be exactly equivalent to the original tree, which matches Editable XAI's Enhance stage.
