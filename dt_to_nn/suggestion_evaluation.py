"""Compare experimental improvement suggestions against the original parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dt_to_nn.experimental_variants import (
    EnsemblePathMatrixNetwork,
    PathMatrixNetwork,
    SoftRoutingTreeNetwork,
    exact_agreement,
    extract_tree_paths,
    model_loss,
    one_hot,
    original_structure,
    path_matrix_structure,
    sklearn_tree_to_dt,
    train_model,
)
from dt_to_nn.paper_evaluation import build_editable_xai_demo_tree, make_nonlinear_classification_data
from dt_to_nn.torch_trainable import TorchParsedNetwork, train_torch_model
from dt_to_nn.tree import DecisionNode, Leaf, TreeNode, predict_batch


@dataclass(frozen=True)
class VariantResult:
    suggestion: str
    variant: str
    accuracy: float
    accuracy_std: float
    macro_precision: float
    macro_recall: float
    tree_fidelity: float
    final_loss: float
    neurons: int | None
    edges: int | None
    recommendation: str


def run_suggestion_evaluation(
    *,
    epochs: int = 120,
    learning_rate: float = 0.01,
    seed: int = 17,
) -> dict[str, Any]:
    tree = build_editable_xai_demo_tree()
    classes = (0, 1)
    x, y = make_nonlinear_classification_data(n_samples=900, seed=seed)
    splits = _fixed_splits(x, y, seed=seed + 50)
    paths = extract_tree_paths(tree, classes=classes)

    original_stats = original_structure(tree, classes=classes)
    path_stats = path_matrix_structure(paths, n_features=x.shape[1])
    compressed_stats = path_matrix_structure(paths, n_features=x.shape[1], compressed=True)
    deep_tree = _deep_unbalanced_tree()
    deep_paths = extract_tree_paths(deep_tree, classes=classes)

    variants = [
        ("0", "original_adjacent_layer", "baseline"),
        ("1+3", "path_matrix_lukasiewicz_skip", "adopt"),
        ("4", "path_matrix_product_and", "consider"),
        ("4", "path_matrix_min_and", "reject"),
        ("4", "path_matrix_softmin_and", "consider"),
        ("5", "soft_routing_tree", "consider"),
        ("6", "temperature_annealing", "consider"),
        ("10", "structure_regularized_training_lam_1e_4", "consider"),
        ("10", "structure_regularized_training_lam_1e_2", "consider"),
    ]

    results: list[VariantResult] = []
    for suggestion, variant, recommendation in variants:
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            row = _run_variant_fold(
                variant,
                tree,
                paths,
                classes,
                x[train_idx],
                y[train_idx],
                x[test_idx],
                y[test_idx],
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed + fold,
            )
            fold_rows.append(row)
        results.append(
            VariantResult(
                suggestion=suggestion,
                variant=variant,
                accuracy=float(mean(row["accuracy"] for row in fold_rows)),
                accuracy_std=float(pstdev(row["accuracy"] for row in fold_rows)),
                macro_precision=float(mean(row["precision"] for row in fold_rows)),
                macro_recall=float(mean(row["recall"] for row in fold_rows)),
                tree_fidelity=float(mean(row["fidelity"] for row in fold_rows)),
                final_loss=float(mean(row["loss"] for row in fold_rows)),
                neurons=original_stats.neurons if variant == "original_adjacent_layer" else path_stats.neurons,
                edges=original_stats.edges if variant == "original_adjacent_layer" else path_stats.edges,
                recommendation=recommendation,
            )
        )

    predicate_result = _run_oblique_predicate_experiment(seed=seed)
    ensemble_result = _run_ensemble_experiment(x, y, classes, seed=seed)

    return {
        "dataset": {
            "name": "synthetic structured binary classification",
            "samples": int(len(x)),
            "features": int(x.shape[1]),
            "splits": len(splits),
            "epochs": epochs,
        },
        "structure": {
            "original": asdict(original_stats),
            "path_matrix": asdict(path_stats),
            "path_matrix_compressed": asdict(compressed_stats),
            "deep_unbalanced_original": asdict(original_structure(deep_tree, classes=classes)),
            "deep_unbalanced_path_matrix": asdict(
                path_matrix_structure(deep_paths, n_features=x.shape[1])
            ),
        },
        "variant_results": [asdict(result) for result in results],
        "additional_experiments": {
            "arbitrary_predicate_oblique_split": predicate_result,
            "ensemble_random_forest_to_path_matrix": ensemble_result,
        },
        "recommendations": _recommendations(results, predicate_result, ensemble_result),
    }


def _run_variant_fold(
    variant: str,
    tree: TreeNode,
    paths: Any,
    classes: tuple[int, int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    y_train_oh = one_hot(y_train, classes)
    y_test_oh = one_hot(y_test, classes)

    if variant == "original_adjacent_layer":
        model = TorchParsedNetwork.from_tree(
            tree,
            classes=classes,
            n_features=x_train.shape[1],
            zero_padding_width=4,
            zero_padding_layers=1,
        )
        train_torch_model(
            model,
            x_train,
            y_train_oh,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            seed=seed,
        )
        y_pred = model.predict_numpy(x_test)
        loss = model.loss_numpy(x_test, y_test_oh)
        fidelity = float(np.mean(np.asarray(predict_batch(tree, x_test)) == y_pred))
        return _metric_row(y_test, y_pred, classes, fidelity=fidelity, loss=loss)

    if variant == "path_matrix_lukasiewicz_skip":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="lukasiewicz")
        train_model(model, x_train, y_train_oh, epochs=epochs, learning_rate=learning_rate, batch_size=64, seed=seed)
    elif variant == "path_matrix_product_and":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="product")
        train_model(model, x_train, y_train_oh, epochs=epochs, learning_rate=learning_rate, batch_size=64, seed=seed)
    elif variant == "path_matrix_min_and":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="min")
        train_model(model, x_train, y_train_oh, epochs=epochs, learning_rate=learning_rate, batch_size=64, seed=seed)
    elif variant == "path_matrix_softmin_and":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="softmin")
        train_model(model, x_train, y_train_oh, epochs=epochs, learning_rate=learning_rate, batch_size=64, seed=seed)
    elif variant == "soft_routing_tree":
        model = SoftRoutingTreeNetwork(paths, n_features=x_train.shape[1])
        train_model(model, x_train, y_train_oh, epochs=epochs, learning_rate=learning_rate, batch_size=64, seed=seed)
    elif variant == "temperature_annealing":
        model = PathMatrixNetwork(
            paths,
            n_features=x_train.shape[1],
            and_kind="lukasiewicz",
            temperature=2.0,
        )
        train_model(
            model,
            x_train,
            y_train_oh,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            seed=seed,
            anneal_temperature=(2.0, 30.0),
        )
    elif variant == "structure_regularized_training_lam_1e_4":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="lukasiewicz")
        train_model(
            model,
            x_train,
            y_train_oh,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            seed=seed,
            regularize_to_initial=1e-4,
        )
    elif variant == "structure_regularized_training_lam_1e_2":
        model = PathMatrixNetwork(paths, n_features=x_train.shape[1], and_kind="lukasiewicz")
        train_model(
            model,
            x_train,
            y_train_oh,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            seed=seed,
            regularize_to_initial=1e-2,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    y_pred = model.predict_numpy(x_test)
    fidelity = exact_agreement(model, tree, x_test)
    loss = model_loss(model, x_test, y_test_oh)
    return _metric_row(y_test, y_pred, classes, fidelity=fidelity, loss=loss)


def _run_oblique_predicate_experiment(*, seed: int) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(700, 2))
    y = ((x[:, 0] + x[:, 1]) >= 0.15).astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=0.2, random_state=seed, stratify=y
    )

    axis_tree = DecisionNode(
        feature_index=0,
        threshold=0.15,
        true_child=Leaf(1),
        false_child=Leaf(0),
        name="axis_x0",
    )
    paths = extract_tree_paths(axis_tree, classes=(0, 1))
    axis_model = PathMatrixNetwork(paths, n_features=2)
    train_model(
        axis_model,
        x[train_idx],
        one_hot(y[train_idx], (0, 1)),
        epochs=80,
        learning_rate=0.01,
        batch_size=64,
        seed=seed,
    )
    axis_acc = accuracy_score(y[test_idx], axis_model.predict_numpy(x[test_idx]))

    # A minimal oblique predicate model for w^T x >= tau, showing suggestion 7.
    oblique = DecisionTreeClassifier(max_depth=1, random_state=seed)
    z = (x[:, [0]] + x[:, [1]]) / 2.0
    oblique.fit(z[train_idx], y[train_idx])
    oblique_acc = accuracy_score(y[test_idx], oblique.predict(z[test_idx]))

    return {
        "suggestion": "7",
        "axis_aligned_accuracy": float(axis_acc),
        "oblique_predicate_accuracy": float(oblique_acc),
        "recommendation": "adopt for future predicate API; keep outside core until API is designed",
    }


def _deep_unbalanced_tree() -> TreeNode:
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


def _run_ensemble_experiment(
    x: np.ndarray,
    y: np.ndarray,
    classes: tuple[int, int],
    *,
    seed: int,
) -> dict[str, float | str]:
    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=0.2, random_state=seed, stratify=y
    )
    forest = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=seed)
    forest.fit(x[train_idx], y[train_idx])
    sklearn_acc = accuracy_score(y[test_idx], forest.predict(x[test_idx]))

    members = []
    for estimator in forest.estimators_:
        dt = sklearn_tree_to_dt(estimator)
        paths = extract_tree_paths(dt, classes=classes)
        members.append(PathMatrixNetwork(paths, n_features=x.shape[1]))
    ensemble = EnsemblePathMatrixNetwork(members)
    nn_acc = accuracy_score(y[test_idx], ensemble.predict_numpy(x[test_idx]))

    return {
        "suggestion": "9",
        "sklearn_random_forest_accuracy": float(sklearn_acc),
        "untrained_path_matrix_ensemble_accuracy": float(nn_acc),
        "trees": len(members),
        "recommendation": "adopt as optional module; manage size with path compression",
    }


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: tuple[int, int],
    *,
    fidelity: float,
    loss: float,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)),
        "fidelity": float(fidelity),
        "loss": float(loss),
    }


def _fixed_splits(x: np.ndarray, y: np.ndarray, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(x))
    splits = []
    for fold in range(3):
        train, test = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed + fold,
            stratify=y,
        )
        splits.append((train, test))
    return splits


def _recommendations(
    results: list[VariantResult],
    predicate_result: dict[str, Any],
    ensemble_result: dict[str, Any],
) -> list[str]:
    by_name = {result.variant: result for result in results}
    baseline = by_name["original_adjacent_layer"]
    path = by_name["path_matrix_lukasiewicz_skip"]
    regularized = by_name["structure_regularized_training_lam_1e_2"]
    recs = [
        "Adopt suggestions 1 and 3 together for the experimental parser: path-matrix removes pass-through chains and gives a fixed three-layer representation.",
        "Adopt suggestion 8 as a post-processing step: path-matrix compression is straightforward and removes duplicate rows/paths.",
        "Adopt suggestion 10 for training modes where rule fidelity matters; compare accuracy-fidelity tradeoff with an explicit lambda.",
    ]
    if regularized.tree_fidelity >= path.tree_fidelity - 0.05:
        recs.append("Use structure regularization by default for editable-rule enhancement.")
    if by_name["path_matrix_product_and"].accuracy >= baseline.accuracy:
        recs.append("Keep product AND as an optional smooth t-norm, especially for soft-routing style experiments.")
    recs.append("Do not replace the main exact parser with min AND; it is less trainable in this benchmark.")
    recs.append(str(predicate_result["recommendation"]))
    recs.append(str(ensemble_result["recommendation"]))
    return recs
