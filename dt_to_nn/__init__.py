"""Decision-tree to neural-network conversion utilities."""

from dt_to_nn.converter import TreeToNNConverter, convert_tree_to_network
from dt_to_nn.evaluation import (
    EvaluationResult,
    evaluate_equivalence,
    make_grid,
    random_samples,
    threshold_probe_samples,
)
from dt_to_nn.network import NeuralDecisionNetwork
from dt_to_nn.tree import DecisionNode, Leaf

__all__ = [
    "DecisionNode",
    "EvaluationResult",
    "Leaf",
    "NeuralDecisionNetwork",
    "TreeToNNConverter",
    "convert_tree_to_network",
    "evaluate_equivalence",
    "make_grid",
    "random_samples",
    "threshold_probe_samples",
]
