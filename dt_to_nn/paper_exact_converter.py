"""Editable XAI parser following the paper's equations exactly.

This module intentionally differs from ``converter.py``. The original sparse
converter uses hard branch indicators to preserve exact decision-tree
predictions. Here, the first layer follows Editable XAI Eq. (1) with a smooth
logistic sigmoid and unscaled weights:

    true branch:  sigmoid(+x_j - threshold)
    false branch: sigmoid(-x_j + threshold)

Subsequent layers use the paper's ReLU conjunction, pass-through propagation,
and clipped-ReLU output disjunction.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dt_to_nn.trainable import TrainableParsedNetwork
from dt_to_nn.tree import TreeNode, class_labels, required_n_features


class PaperExactParsedNetwork(TrainableParsedNetwork):
    """Dense parsed network initialized exactly from Editable XAI Eq. (1)-(5)."""

    @classmethod
    def from_tree(
        cls,
        tree: TreeNode,
        *,
        classes: Sequence[Any] | None = None,
        n_features: int | None = None,
        zero_padding_width: int = 0,
        zero_padding_layers: int = 0,
    ) -> "PaperExactParsedNetwork":
        labels = tuple(classes if classes is not None else class_labels(tree))
        feature_count = (
            required_n_features(tree) if n_features is None else n_features
        )
        return super().from_tree(
            tree,
            classes=labels,
            n_features=feature_count,
            condition_temperature=1.0,
            zero_padding_width=zero_padding_width,
            zero_padding_layers=zero_padding_layers,
        )

    def outputs_one(self, x: Sequence[float]) -> dict[Any, float]:
        """Return class activations for one sample."""

        values = self.outputs(np.asarray([x], dtype=float))[0]
        return {
            label: float(values[index])
            for index, label in enumerate(self.classes)
        }

    def predict_one(self, x: Sequence[float]) -> Any:
        """Return the argmax class for one sample."""

        return self.predict(np.asarray([x], dtype=float))[0]

    def summary(self) -> dict[str, Any]:
        """Return the dense architecture and activation functions."""

        return {
            "classes": list(self.classes),
            "layer_shapes": [
                {
                    "weights": list(weight.shape),
                    "biases": list(bias.shape),
                    "activation": activation,
                }
                for weight, bias, activation in zip(
                    self.weights, self.biases, self.activations
                )
            ],
            "first_layer_formula": "sigmoid(Wx+b), W in {+1,-1}, b in {-tau,+tau}",
        }


def convert_tree_to_paper_exact_network(
    tree: TreeNode,
    *,
    classes: Sequence[Any] | None = None,
    n_features: int | None = None,
    zero_padding_width: int = 0,
    zero_padding_layers: int = 0,
) -> PaperExactParsedNetwork:
    """Convert a tree using the exact smooth-sigmoid equations in the paper."""

    return PaperExactParsedNetwork.from_tree(
        tree,
        classes=classes,
        n_features=n_features,
        zero_padding_width=zero_padding_width,
        zero_padding_layers=zero_padding_layers,
    )
