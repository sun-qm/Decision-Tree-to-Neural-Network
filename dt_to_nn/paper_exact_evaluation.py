"""Agreement evaluation for the exact Editable XAI formula implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.demo import build_demo_tree
from dt_to_nn.evaluation import (
    evaluate_equivalence,
    random_samples,
    threshold_probe_samples,
)
from dt_to_nn.paper_exact_converter import (
    PaperExactParsedNetwork,
    convert_tree_to_paper_exact_network,
)
from dt_to_nn.tree import TreeNode, predict_one, required_n_features


@dataclass(frozen=True)
class PaperAgreementResult:
    """Prediction and output agreement between a DT and smooth parsed NN."""

    total: int
    prediction_matches: int
    output_vector_matches: int
    mean_absolute_output_error: float
    max_output_error: float
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prediction_agreement(self) -> float:
        return self.prediction_matches / self.total if self.total else 1.0

    @property
    def output_vector_agreement(self) -> float:
        return self.output_vector_matches / self.total if self.total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "prediction_matches": self.prediction_matches,
            "prediction_agreement": self.prediction_agreement,
            "output_vector_matches": self.output_vector_matches,
            "output_vector_agreement": self.output_vector_agreement,
            "mean_absolute_output_error": self.mean_absolute_output_error,
            "max_output_error": self.max_output_error,
            "mismatches": self.mismatches,
        }


def evaluate_paper_exact_agreement(
    tree: TreeNode,
    network: PaperExactParsedNetwork,
    samples: Iterable[Sequence[float]],
    *,
    tolerance: float = 1e-12,
    keep_mismatches: int = 10,
) -> PaperAgreementResult:
    """Compare DT labels and one-hot outputs with the paper's smooth network."""

    sample_list = [list(sample) for sample in samples]
    if not sample_list:
        return PaperAgreementResult(0, 0, 0, 0.0, 0.0)

    x = np.asarray(sample_list, dtype=float)
    outputs = network.outputs(x)
    predictions = network.predict(x)
    class_index = {label: index for index, label in enumerate(network.classes)}

    prediction_matches = 0
    output_vector_matches = 0
    absolute_error_sum = 0.0
    max_output_error = 0.0
    mismatches: list[dict[str, Any]] = []

    for row, sample in enumerate(sample_list):
        tree_label = predict_one(tree, sample)
        nn_label = predictions[row]
        expected = np.zeros(len(network.classes), dtype=float)
        expected[class_index[tree_label]] = 1.0
        errors = np.abs(outputs[row] - expected)
        sample_max_error = float(np.max(errors, initial=0.0))
        prediction_ok = bool(tree_label == nn_label)
        output_ok = bool(sample_max_error <= tolerance)

        prediction_matches += int(prediction_ok)
        output_vector_matches += int(output_ok)
        absolute_error_sum += float(np.sum(errors))
        max_output_error = max(max_output_error, sample_max_error)

        if not prediction_ok and len(mismatches) < keep_mismatches:
            mismatches.append(
                {
                    "x": sample,
                    "tree_label": tree_label,
                    "network_label": nn_label.item()
                    if hasattr(nn_label, "item")
                    else nn_label,
                    "outputs": {
                        label: float(outputs[row, index])
                        for index, label in enumerate(network.classes)
                    },
                    "expected": {
                        label: float(expected[index])
                        for index, label in enumerate(network.classes)
                    },
                    "max_error": sample_max_error,
                }
            )

    denominator = len(sample_list) * max(1, len(network.classes))
    return PaperAgreementResult(
        total=len(sample_list),
        prediction_matches=prediction_matches,
        output_vector_matches=output_vector_matches,
        mean_absolute_output_error=absolute_error_sum / denominator,
        max_output_error=max_output_error,
        mismatches=mismatches,
    )


def run_paper_exact_evaluation(
    *,
    tree: TreeNode | None = None,
    random_count: int = 10_000,
    low: float = -2.0,
    high: float = 2.0,
    seed: int = 42,
    threshold_epsilon: float = 1e-9,
) -> dict[str, Any]:
    """Evaluate paper-formula and hard-indicator parsers on the same probes."""

    evaluated_tree = build_demo_tree() if tree is None else tree
    n_features = required_n_features(evaluated_tree)
    paper_network = convert_tree_to_paper_exact_network(
        evaluated_tree,
        n_features=n_features,
    )

    random = random_samples(
        n_features,
        random_count,
        low=low,
        high=high,
        seed=seed,
    )
    thresholds = threshold_probe_samples(
        evaluated_tree,
        epsilon=threshold_epsilon,
    )
    combined = random + thresholds

    hard_network = convert_tree_to_network(evaluated_tree)
    return {
        "interpretation": (
            "The parser matches the paper's smooth equations exactly; "
            "smooth sigmoid does not guarantee exact DT predictions or one-hot outputs."
        ),
        "network": paper_network.summary(),
        "paper_sigmoid_random": evaluate_paper_exact_agreement(
            evaluated_tree, paper_network, random
        ).to_dict(),
        "paper_sigmoid_threshold_probes": evaluate_paper_exact_agreement(
            evaluated_tree, paper_network, thresholds
        ).to_dict(),
        "paper_sigmoid_combined": evaluate_paper_exact_agreement(
            evaluated_tree, paper_network, combined
        ).to_dict(),
        "hard_indicator_reference": evaluate_equivalence(
            evaluated_tree, hard_network, combined
        ).to_dict(),
    }
