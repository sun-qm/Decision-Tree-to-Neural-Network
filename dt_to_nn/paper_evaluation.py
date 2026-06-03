"""Paper-style evaluation for the Editable XAI tree-to-NN parser."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.evaluation import evaluate_equivalence, random_samples, threshold_probe_samples
from dt_to_nn.trainable import TrainableParsedNetwork, one_hot
from dt_to_nn.tree import DecisionNode, Leaf, predict_batch


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    loss: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "loss": self.loss,
        }


def build_editable_xai_demo_tree() -> DecisionNode:
    """A compact user-authored rule tree for the benchmark."""

    return DecisionNode(
        feature_index=0,
        threshold=0.45,
        name="root_x0",
        true_child=DecisionNode(
            feature_index=1,
            threshold=0.55,
            name="x1_when_high_x0",
            true_child=Leaf(1),
            false_child=Leaf(0),
        ),
        false_child=DecisionNode(
            feature_index=2,
            threshold=0.65,
            name="x2_when_low_x0",
            true_child=Leaf(1),
            false_child=Leaf(0),
        ),
    )


def make_nonlinear_classification_data(
    *,
    n_samples: int = 1200,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a structured binary task where the rule tree is useful but imperfect."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n_samples, 3))
    score = (
        1.8 * (x[:, 0] - 0.45)
        + 1.3 * (x[:, 1] - 0.50)
        - 1.1 * (x[:, 2] - 0.45)
        + 1.4 * ((x[:, 0] > 0.72) & (x[:, 2] < 0.35))
        + 0.8 * ((x[:, 0] < 0.35) & (x[:, 1] > 0.75))
    )
    y = (score > 0.25).astype(int)
    return x, y


def fixed_splits(
    n_samples: int,
    *,
    folds: int = 5,
    test_fraction: float = 0.2,
    seed: int = 11,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Five fixed 80/20 splits, following the DJINN evaluation pattern."""

    rng = np.random.default_rng(seed)
    test_size = int(round(n_samples * test_fraction))
    splits = []
    for _ in range(folds):
        order = rng.permutation(n_samples)
        test = order[:test_size]
        train = order[test_size:]
        splits.append((train, test))
    return splits


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    classes: Sequence[Any],
    *,
    loss: float | None = None,
) -> Metrics:
    y_true = list(y_true)
    y_pred = list(y_pred)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    precisions = []
    recalls = []
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
    return Metrics(
        accuracy=float(accuracy),
        macro_precision=float(mean(precisions)),
        macro_recall=float(mean(recalls)),
        loss=loss,
    )


def run_paper_style_evaluation(
    *,
    epochs: int = 250,
    learning_rate: float = 0.03,
    zero_padding_width: int = 4,
    zero_padding_layers: int = 1,
    seed: int = 3,
) -> dict[str, Any]:
    """Evaluate exact parsing plus zero-padded training and random baselines."""

    tree = build_editable_xai_demo_tree()
    classes = (0, 1)
    x, y = make_nonlinear_classification_data(seed=seed)
    labels_from_tree = predict_batch(tree, x)
    exact_network = convert_tree_to_network(tree, classes=classes)

    probe_samples = random_samples(3, 1000, low=0.0, high=1.0, seed=seed)
    probe_samples.extend(threshold_probe_samples(tree, epsilon=1e-12))
    equivalence = evaluate_equivalence(tree, exact_network, probe_samples).to_dict()

    results: dict[str, list[Metrics]] = {
        "tree_or_exact_parsed_nn": [],
        "zero_padded_parsed_before_training": [],
        "zero_padded_parsed_after_training": [],
        "random_dense_after_training": [],
        "random_sparse_after_training": [],
    }
    losses: dict[str, list[list[float]]] = {
        "zero_padded_parsed_after_training": [],
        "random_dense_after_training": [],
        "random_sparse_after_training": [],
    }

    splits = fixed_splits(len(x), seed=seed + 100)
    for fold, (train_idx, test_idx) in enumerate(splits):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_onehot = one_hot(y_train, classes)
        y_test_onehot = one_hot(y_test, classes)

        tree_pred = [exact_network.predict(row) for row in x_test]
        results["tree_or_exact_parsed_nn"].append(
            classification_metrics(y_test, tree_pred, classes)
        )

        parsed = TrainableParsedNetwork.from_tree(
            tree,
            classes=classes,
            n_features=3,
            zero_padding_width=zero_padding_width,
            zero_padding_layers=zero_padding_layers,
        )
        parsed_before_pred = parsed.predict(x_test)
        results["zero_padded_parsed_before_training"].append(
            classification_metrics(
                y_test,
                parsed_before_pred,
                classes,
                loss=parsed.loss(x_test, y_test_onehot),
            )
        )
        history = parsed.fit(
            x_train,
            y_train_onehot,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            seed=seed + fold,
        )
        losses["zero_padded_parsed_after_training"].append(history.losses)
        results["zero_padded_parsed_after_training"].append(
            classification_metrics(
                y_test,
                parsed.predict(x_test),
                classes,
                loss=parsed.loss(x_test, y_test_onehot),
            )
        )

        for mode in ("dense", "sparse"):
            model_name = f"random_{mode}_after_training"
            baseline = parsed.random_like(seed=seed + 1000 + fold, mode=mode)
            history = baseline.fit(
                x_train,
                y_train_onehot,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=64,
                seed=seed + 2000 + fold,
            )
            losses[model_name].append(history.losses)
            results[model_name].append(
                classification_metrics(
                    y_test,
                    baseline.predict(x_test),
                    classes,
                    loss=baseline.loss(x_test, y_test_onehot),
                )
            )

    return {
        "method_notes": {
            "paper_2_reference": (
                "Uses fixed five-fold 80/20 train/test splits, reports "
                "classification accuracy, macro precision, macro recall, and "
                "compares parsed initialization against random dense/sparse "
                "same-architecture baselines."
            ),
            "editable_xai_reference": (
                "Parser uses paired decision-branch neurons, ReLU(a+b-1) "
                "for trace conjunction, cReLU(sum(paths)) for class disjunction, "
                "and zero padding with zero weights/biases for trainable capacity."
            ),
        },
        "equivalence_to_input_tree": equivalence,
        "tree_accuracy_against_dataset_labels": classification_metrics(
            y, labels_from_tree, classes
        ).to_dict(),
        "cross_validation": _summarize_results(results),
        "training_loss": _summarize_losses(losses),
        "config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "zero_padding_width": zero_padding_width,
            "zero_padding_layers": zero_padding_layers,
            "n_samples": len(x),
            "n_features": x.shape[1],
            "folds": len(splits),
        },
    }


def _summarize_results(results: dict[str, list[Metrics]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, values in results.items():
        summary[name] = {}
        for field in ("accuracy", "macro_precision", "macro_recall", "loss"):
            series = [getattr(item, field) for item in values]
            series = [item for item in series if item is not None]
            if not series:
                continue
            summary[name][field] = {
                "mean": float(mean(series)),
                "std": float(pstdev(series)),
                "folds": [float(item) for item in series],
            }
    return summary


def _summarize_losses(losses: dict[str, list[list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, histories in losses.items():
        if not histories:
            continue
        summary[name] = {
            "initial_mean": float(mean(history[0] for history in histories)),
            "final_mean": float(mean(history[-1] for history in histories)),
            "final_std": float(pstdev(history[-1] for history in histories)),
        }
    return summary
