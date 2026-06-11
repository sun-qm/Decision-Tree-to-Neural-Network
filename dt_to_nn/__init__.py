"""Decision-tree to neural-network conversion utilities."""

from dt_to_nn.converter import TreeToNNConverter, convert_tree_to_network
from dt_to_nn.direct_path_converter import (
    DirectPathTreeToNNConverter,
    convert_tree_to_direct_path_network,
)
from dt_to_nn.evaluation import (
    EvaluationResult,
    evaluate_equivalence,
    make_grid,
    random_samples,
    threshold_probe_samples,
)
from dt_to_nn.network import NeuralDecisionNetwork
from dt_to_nn.paper_exact_converter import (
    PaperExactParsedNetwork,
    convert_tree_to_paper_exact_network,
)
from dt_to_nn.paper_exact_evaluation import (
    PaperAgreementResult,
    evaluate_paper_exact_agreement,
    run_paper_exact_evaluation,
)
from dt_to_nn.paper_evaluation import run_paper_style_evaluation
from dt_to_nn.trainable import TrainableParsedNetwork
from dt_to_nn.tree import DecisionNode, Leaf
from dt_to_nn.visualization import (
    render_network_svg,
    render_tree_svg,
    save_network_svg,
    save_tree_and_network_svg,
    save_tree_svg,
)

__all__ = [
    "DecisionNode",
    "EvaluationResult",
    "Leaf",
    "NeuralDecisionNetwork",
    "PaperAgreementResult",
    "PaperExactParsedNetwork",
    "TrainableParsedNetwork",
    "DirectPathTreeToNNConverter",
    "TreeToNNConverter",
    "convert_tree_to_direct_path_network",
    "convert_tree_to_network",
    "convert_tree_to_paper_exact_network",
    "evaluate_equivalence",
    "evaluate_paper_exact_agreement",
    "make_grid",
    "random_samples",
    "render_network_svg",
    "render_tree_svg",
    "run_paper_style_evaluation",
    "run_paper_exact_evaluation",
    "save_network_svg",
    "save_tree_and_network_svg",
    "save_tree_svg",
    "threshold_probe_samples",
]
