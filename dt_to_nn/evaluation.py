"""Evaluation helpers for tree/network equivalence."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from random import Random
from typing import Any, Iterable, Sequence

from dt_to_nn.network import NeuralDecisionNetwork
from dt_to_nn.tree import TreeNode, iter_internal_nodes, predict_one, required_n_features


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    prediction_matches: int
    output_vector_matches: int
    max_output_error: float
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prediction_agreement(self) -> float:
        return self.prediction_matches / self.total if self.total else 1.0

    @property
    def output_vector_agreement(self) -> float:
        return self.output_vector_matches / self.total if self.total else 1.0

    @property
    def is_fully_consistent(self) -> bool:
        return self.prediction_matches == self.total and self.output_vector_matches == self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "prediction_matches": self.prediction_matches,
            "prediction_agreement": self.prediction_agreement,
            "output_vector_matches": self.output_vector_matches,
            "output_vector_agreement": self.output_vector_agreement,
            "max_output_error": self.max_output_error,
            "is_fully_consistent": self.is_fully_consistent,
            "mismatches": self.mismatches,
        }


def evaluate_equivalence(
    tree: TreeNode,
    network: NeuralDecisionNetwork,
    samples: Iterable[Sequence[float]],
    *,
    keep_mismatches: int = 10,
    tolerance: float = 0.0,
) -> EvaluationResult:
    """Compare tree labels and neural-network outputs on finite samples."""

    total = 0
    prediction_matches = 0
    output_vector_matches = 0
    max_output_error = 0.0
    mismatches: list[dict[str, Any]] = []

    for x in samples:
        total += 1
        tree_label = predict_one(tree, x)
        nn_label = network.predict(x)
        outputs = network.outputs(x)
        expected = {label: 1.0 if label == tree_label else 0.0 for label in network.classes}
        errors = {
            label: abs(outputs.get(label, 0.0) - expected[label])
            for label in network.classes
        }
        sample_max_error = max(errors.values(), default=0.0)
        max_output_error = max(max_output_error, sample_max_error)

        prediction_ok = tree_label == nn_label
        output_ok = sample_max_error <= tolerance
        prediction_matches += int(prediction_ok)
        output_vector_matches += int(output_ok)

        if (not prediction_ok or not output_ok) and len(mismatches) < keep_mismatches:
            mismatches.append(
                {
                    "x": list(x),
                    "tree_label": tree_label,
                    "network_label": nn_label,
                    "outputs": outputs,
                    "expected": expected,
                    "max_error": sample_max_error,
                }
            )

    return EvaluationResult(
        total=total,
        prediction_matches=prediction_matches,
        output_vector_matches=output_vector_matches,
        max_output_error=max_output_error,
        mismatches=mismatches,
    )


def make_grid(feature_values: dict[int, Sequence[float]]) -> list[list[float]]:
    """Create samples from a Cartesian product of per-feature values."""

    if not feature_values:
        return [[]]
    ordered_features = sorted(feature_values)
    n_features = ordered_features[-1] + 1
    samples: list[list[float]] = []
    for values in product(*(feature_values[index] for index in ordered_features)):
        sample = [0.0] * n_features
        for feature_index, value in zip(ordered_features, values):
            sample[feature_index] = float(value)
        samples.append(sample)
    return samples


def random_samples(
    n_features: int,
    count: int,
    *,
    low: float = -1.0,
    high: float = 1.0,
    seed: int = 0,
) -> list[list[float]]:
    """Generate deterministic random samples."""

    rng = Random(seed)
    return [
        [rng.uniform(low, high) for _ in range(n_features)]
        for _ in range(count)
    ]


def threshold_probe_samples(
    tree: TreeNode,
    *,
    base: Sequence[float] | None = None,
    epsilon: float = 1e-9,
) -> list[list[float]]:
    """Build samples that hit every threshold from both sides and at equality."""

    n_features = max(required_n_features(tree), len(base or []))
    template = list(base) if base is not None else [0.0] * n_features
    if len(template) < n_features:
        template.extend([0.0] * (n_features - len(template)))

    samples: list[list[float]] = []
    for node in iter_internal_nodes(tree):
        for value in (node.threshold - epsilon, node.threshold, node.threshold + epsilon):
            sample = list(template)
            sample[node.feature_index] = value
            samples.append(sample)
    return samples
