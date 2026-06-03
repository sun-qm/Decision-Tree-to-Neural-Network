"""Evaluation for direct root-to-leaf path conversion."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.direct_path_converter import (
    convert_tree_to_direct_path_network,
    summarize_direct_path_network,
)
from dt_to_nn.evaluation import evaluate_equivalence, random_samples, threshold_probe_samples
from dt_to_nn.experimental_variants import extract_tree_paths, original_structure, path_matrix_structure
from dt_to_nn.paper_evaluation import build_editable_xai_demo_tree, make_nonlinear_classification_data
from dt_to_nn.torch_trainable import TorchParsedNetwork, train_torch_model
from dt_to_nn.tree import DecisionNode, Leaf, TreeNode, predict_batch


def run_direct_path_comparison(
    *,
    epochs: int = 180,
    learning_rate: float = 0.02,
    seed: int = 29,
) -> dict[str, Any]:
    classes = (0, 1)
    demo_tree = build_editable_xai_demo_tree()
    deep_tree = build_deep_unbalanced_tree()
    x, y = make_nonlinear_classification_data(n_samples=1000, seed=seed)

    structural = {
        "demo_tree": _structure_block(demo_tree, classes=classes, n_features=3),
        "deep_unbalanced_tree": _structure_block(deep_tree, classes=classes, n_features=3),
    }

    direct_network = convert_tree_to_direct_path_network(demo_tree, classes=classes)
    original_network = convert_tree_to_network(demo_tree, classes=classes)
    probe_samples = random_samples(3, 1000, low=0.0, high=1.0, seed=seed)
    probe_samples.extend(threshold_probe_samples(demo_tree, epsilon=1e-12))

    exactness = {
        "original": evaluate_equivalence(demo_tree, original_network, probe_samples).to_dict(),
        "direct_path": evaluate_equivalence(demo_tree, direct_network, probe_samples).to_dict(),
    }

    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=0.2, random_state=seed, stratify=y
    )
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_oh = _one_hot(y_train, classes)
    y_test_oh = _one_hot(y_test, classes)

    original_model = TorchParsedNetwork.from_tree(
        demo_tree,
        classes=classes,
        n_features=3,
        zero_padding_width=4,
        zero_padding_layers=1,
    )
    train_torch_model(
        original_model,
        x_train,
        y_train_oh,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=64,
        seed=seed,
    )

    # Replace adjacent-layer training comparison with the exact three-layer
    # direct path matrix equivalent.
    paths = extract_tree_paths(demo_tree, classes=classes)
    from dt_to_nn.experimental_variants import PathMatrixNetwork, model_loss, train_model

    direct_path_model = PathMatrixNetwork(paths, n_features=3, and_kind="lukasiewicz")
    train_model(
        direct_path_model,
        x_train,
        y_train_oh,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=64,
        seed=seed,
    )

    training = {
        "tree_before_training": _metrics(y_test, predict_batch(demo_tree, x_test), classes),
        "original_adjacent_after_training": _metrics(
            y_test,
            original_model.predict_numpy(x_test),
            classes,
            loss=original_model.loss_numpy(x_test, y_test_oh),
        ),
        "direct_path_after_training": _metrics(
            y_test,
            direct_path_model.predict_numpy(x_test),
            classes,
            loss=model_loss(direct_path_model, x_test, y_test_oh),
        ),
    }

    return {
        "claim": "One path neuron per root-to-leaf path produces a fixed three-layer network.",
        "structural_comparison": structural,
        "exact_equivalence": exactness,
        "training_comparison": training,
        "recommendation": _recommendation(structural, training),
    }


def build_deep_unbalanced_tree() -> TreeNode:
    return DecisionNode(
        feature_index=0,
        threshold=0.5,
        true_child=Leaf(1),
        false_child=DecisionNode(
            feature_index=1,
            threshold=0.5,
            true_child=Leaf(0),
            false_child=DecisionNode(
                feature_index=2,
                threshold=0.5,
                true_child=Leaf(1),
                false_child=DecisionNode(
                    feature_index=0,
                    threshold=0.2,
                    true_child=Leaf(0),
                    false_child=Leaf(1),
                    name="deep_3",
                ),
                name="deep_2",
            ),
            name="deep_1",
        ),
        name="deep_root",
    )


def _structure_block(tree: TreeNode, *, classes: tuple[int, int], n_features: int) -> dict[str, Any]:
    original = original_structure(tree, classes=classes)
    direct = summarize_direct_path_network(
        convert_tree_to_direct_path_network(tree, classes=classes)
    )
    matrix = path_matrix_structure(extract_tree_paths(tree, classes=classes), n_features=n_features)
    return {
        "original_adjacent": asdict(original),
        "direct_path_sparse": asdict(direct) | {"total_neurons": direct.total_neurons},
        "path_matrix_equivalent": asdict(matrix),
    }


def _metrics(
    y_true: np.ndarray,
    y_pred: list[int] | np.ndarray,
    classes: tuple[int, int],
    *,
    loss: float | None = None,
) -> dict[str, float | None]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)
        ),
        "loss": loss,
    }


def _one_hot(labels: np.ndarray, classes: tuple[int, int]) -> np.ndarray:
    index = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        y[row, index[int(label)]] = 1.0
    return y


def _recommendation(
    structural: dict[str, Any],
    training: dict[str, dict[str, float | None]],
) -> list[str]:
    deep_original = structural["deep_unbalanced_tree"]["original_adjacent"]
    deep_direct = structural["deep_unbalanced_tree"]["direct_path_sparse"]
    direct_acc = training["direct_path_after_training"]["accuracy"] or 0.0
    original_acc = training["original_adjacent_after_training"]["accuracy"] or 0.0
    return [
        "Use direct-path conversion when the goal is exact DT-to-NN equivalence, compactness, or deployment.",
        "Use the original adjacent-layer parser when you need to preserve the paper's layer-by-layer trace topology or inspect partial path neurons.",
        (
            "For deep or unbalanced trees, direct-path is structurally better: "
            f"{deep_direct['total_neurons']} neurons / {deep_direct['edges']} edges / "
            f"{deep_direct['layers']} layers vs original {deep_original['neurons']} neurons / "
            f"{deep_original['edges']} edges / {deep_original['layers']} layers."
        ),
        (
            "For training, compare on the target dataset: in this run direct-path accuracy "
            f"was {direct_acc:.3f} vs original adjacent {original_acc:.3f}."
        ),
    ]


if __name__ == "__main__":
    from pprint import pprint

    pprint(run_direct_path_comparison(), sort_dicts=False)
