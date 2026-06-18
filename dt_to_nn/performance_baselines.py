"""Performance-feasibility baselines for soft tree-to-network initialization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from sklearn.datasets import (
        fetch_california_housing,
        load_breast_cancer,
        load_diabetes,
        load_wine,
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        explained_variance_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("performance baselines require scikit-learn") from exc


Task = str
ModelName = str


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "diabetes": {
        "task": "regression",
        "loader": "load_diabetes",
        "tree_depth": 5,
        "epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 16,
    },
    "california_housing": {
        "task": "regression",
        "loader": "fetch_california_housing",
        "tree_depth": 5,
        "epochs": 200,
        "learning_rate": 6e-3,
        "batch_size": 512,
        "max_samples": 2500,
    },
    "wine": {
        "task": "classification",
        "loader": "load_wine",
        "tree_depth": 3,
        "epochs": 50,
        "learning_rate": 4e-3,
        "batch_size": 8,
    },
    "breast_cancer": {
        "task": "classification",
        "loader": "load_breast_cancer",
        "tree_depth": 4,
        "epochs": 100,
        "learning_rate": 6e-3,
        "batch_size": 16,
    },
}


@dataclass(frozen=True)
class SklearnTreePaths:
    """Root-to-leaf path metadata extracted from a sklearn binary tree."""

    task: Task
    n_features: int
    output_dim: int
    node_features: np.ndarray
    node_thresholds: np.ndarray
    paths: tuple[tuple[tuple[int, int], ...], ...]
    leaf_outputs: np.ndarray
    max_depth: int
    node_count: int
    leaf_count: int


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(
    name: str,
    *,
    seed: int = 0,
    max_samples: int | None = None,
    use_config_max_samples: bool = True,
) -> tuple[np.ndarray, np.ndarray, Task]:
    config = DATASET_CONFIGS[name]
    if config["loader"] == "load_diabetes":
        data = load_diabetes()
    elif config["loader"] == "load_breast_cancer":
        data = load_breast_cancer()
    elif config["loader"] == "load_wine":
        data = load_wine()
    elif config["loader"] == "fetch_california_housing":
        data_home = Path.cwd() / ".cache" / "sklearn_data"
        data_home.mkdir(parents=True, exist_ok=True)
        data = fetch_california_housing(data_home=str(data_home))
    else:  # pragma: no cover
        raise ValueError(f"unknown loader: {config['loader']}")

    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target)
    if max_samples is None and use_config_max_samples:
        max_samples = config.get("max_samples")
    if max_samples is not None and len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y, config["task"]


def extract_sklearn_paths(
    estimator: Any,
    *,
    task: Task,
    output_dim: int,
) -> SklearnTreePaths:
    tree = estimator.tree_
    node_to_internal: dict[int, int] = {}
    features: list[int] = []
    thresholds: list[float] = []
    paths: list[tuple[tuple[int, int], ...]] = []
    leaf_outputs: list[np.ndarray] = []

    def internal_id(node_id: int) -> int:
        if node_id not in node_to_internal:
            node_to_internal[node_id] = len(features)
            features.append(int(tree.feature[node_id]))
            thresholds.append(float(tree.threshold[node_id]))
        return node_to_internal[node_id]

    def leaf_output(node_id: int) -> np.ndarray:
        raw = np.asarray(tree.value[node_id]).reshape(-1)
        if task == "classification":
            counts = raw[:output_dim].astype(np.float32)
            total = float(counts.sum())
            probs = counts / total if total else np.ones(output_dim, dtype=np.float32) / output_dim
            return np.log(np.clip(probs, 1e-6, 1.0)).astype(np.float32)
        return np.asarray([float(raw[0])], dtype=np.float32)

    def walk(node_id: int, selected: list[tuple[int, int]], depth: int) -> None:
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        if left == right:
            paths.append(tuple(selected))
            leaf_outputs.append(leaf_output(node_id))
            return
        idx = internal_id(node_id)
        walk(left, selected + [(idx, 0)], depth + 1)
        walk(right, selected + [(idx, 1)], depth + 1)

    walk(0, [], 0)
    return SklearnTreePaths(
        task=task,
        n_features=int(estimator.n_features_in_),
        output_dim=output_dim,
        node_features=np.asarray(features, dtype=np.int64),
        node_thresholds=np.asarray(thresholds, dtype=np.float32),
        paths=tuple(paths),
        leaf_outputs=np.vstack(leaf_outputs).astype(np.float32),
        max_depth=int(tree.max_depth),
        node_count=int(tree.node_count),
        leaf_count=int(sum(tree.children_left == tree.children_right)),
    )


class SoftTreeNetwork(nn.Module):
    """Soft Editable-XAI recursive tree network or EntropyNet-like path expansion."""

    def __init__(
        self,
        paths: SklearnTreePaths,
        *,
        alpha: float,
        mode: str,
        random_init: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in {"editable", "path_expansion"}:
            raise ValueError("mode must be 'editable' or 'path_expansion'")
        set_all_seeds(seed)
        self.paths = paths
        self.mode = mode
        self.task = paths.task
        n_nodes = len(paths.node_features)
        self.condition = nn.Linear(paths.n_features, n_nodes)
        self.leaf_outputs = nn.Parameter(torch.as_tensor(paths.leaf_outputs, dtype=torch.float32))
        max_path_len = max((len(path) for path in paths.paths), default=0)
        path_indices = np.zeros((len(paths.paths), max_path_len), dtype=np.int64)
        path_branches = np.zeros((len(paths.paths), max_path_len), dtype=np.int64)
        path_mask = np.zeros((len(paths.paths), max_path_len), dtype=bool)
        for row, path in enumerate(paths.paths):
            for col, (node_idx, branch) in enumerate(path):
                path_indices[row, col] = node_idx
                path_branches[row, col] = branch
                path_mask[row, col] = True
        self.register_buffer("path_indices", torch.as_tensor(path_indices, dtype=torch.long))
        self.register_buffer("path_branches", torch.as_tensor(path_branches, dtype=torch.bool))
        self.register_buffer("path_mask", torch.as_tensor(path_mask, dtype=torch.bool))
        self.register_buffer("path_lengths", torch.as_tensor(path_mask.sum(axis=1), dtype=torch.float32))
        with torch.no_grad():
            if random_init:
                nn.init.xavier_uniform_(self.condition.weight)
                self.condition.bias.zero_()
                nn.init.xavier_uniform_(self.leaf_outputs)
            else:
                self.condition.weight.zero_()
                self.condition.bias.zero_()
                for row, (feature, threshold) in enumerate(
                    zip(paths.node_features, paths.node_thresholds)
                ):
                    self.condition.weight[row, int(feature)] = -float(alpha)
                    self.condition.bias[row] = float(alpha * threshold)

    def branch_probabilities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.sigmoid(self.condition(x))
        right = 1.0 - left
        return left, right

    def path_activations(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.branch_probabilities(x)
        if self.path_indices.shape[1] == 0:
            return torch.ones((len(x), len(self.paths.paths)), dtype=x.dtype, device=x.device)
        selected_left = left[:, self.path_indices]
        selected_right = right[:, self.path_indices]
        selected = torch.where(self.path_branches.unsqueeze(0), selected_right, selected_left)
        mask = self.path_mask.unsqueeze(0)
        if self.mode == "path_expansion":
            return torch.where(mask, selected, torch.ones_like(selected)).prod(dim=2)
        sums = torch.where(mask, selected, torch.zeros_like(selected)).sum(dim=2)
        return F.relu(sums - (self.path_lengths.to(x.device).unsqueeze(0) - 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path_values = self.path_activations(x)
        return path_values @ self.leaf_outputs


class DJINNLikeNetwork(nn.Module):
    """PyTorch implementation of the LLNL DJINN tree-to-architecture mapping."""

    def __init__(
        self,
        estimator: Any,
        *,
        task: Task,
        output_dim: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        set_all_seeds(seed)
        self.task = task
        self.output_dim = output_dim
        self.layers, self.architecture = self._build_layers(estimator, output_dim, seed)

    @staticmethod
    def _xavier(nin: int, nout: int) -> float:
        return float(np.random.normal(0.0, np.sqrt(3.0 / max(1, nin + nout))))

    def _build_layers(
        self,
        estimator: Any,
        output_dim: int,
        seed: int,
    ) -> tuple[nn.ModuleList, list[int]]:
        np.random.seed(seed)
        tree = estimator.tree_
        n_features = int(estimator.n_features_in_)
        children_left = tree.children_left
        children_right = tree.children_right
        features = tree.feature

        node_depth = np.zeros(tree.node_count, dtype=np.int64)
        stack = [(0, -1)]
        while stack:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((int(children_left[node_id]), parent_depth + 1))
                stack.append((int(children_right[node_id]), parent_depth + 1))

        num_layers = int(node_depth.max()) + 1
        nodes_per_depth = np.zeros(num_layers, dtype=np.int64)
        for depth in range(num_layers):
            ids = np.where(node_depth == depth)[0]
            nodes_per_depth[depth] = int(np.sum(features[ids] >= 0))

        max_depth_feature = np.zeros(n_features, dtype=np.int64)
        for f in range(n_features):
            ids = np.where(features == f)[0]
            if len(ids):
                max_depth_feature[f] = int(np.max(node_depth[ids]))

        arch = [n_features]
        for depth in range(1, num_layers - 1):
            arch.append(int(arch[-1] + nodes_per_depth[depth]))
        arch.append(output_dim)
        if len(arch) < 3:
            arch = [n_features, max(n_features, 2), output_dim]

        weights = [
            np.zeros((arch[i + 1], arch[i]), dtype=np.float32)
            for i in range(len(arch) - 1)
        ]

        offsets: dict[int, int] = {}
        for depth in range(1, len(arch) - 1):
            offsets[depth] = arch[depth - 1]
        next_slot = {depth: 0 for depth in range(1, len(arch) - 1)}
        node_repr: dict[int, tuple[int, int]] = {}

        for layer in range(len(arch) - 2):
            for f in range(n_features):
                if layer < max_depth_feature[f] - 1 and f < arch[layer] and f < arch[layer + 1]:
                    weights[layer][f, f] = 1.0

        def allocate(node_id: int) -> tuple[int, int]:
            depth = int(node_depth[node_id])
            if depth == 0:
                return 0, int(features[node_id])
            if node_id not in node_repr:
                layer = min(depth, len(arch) - 2)
                idx = offsets[layer] + next_slot[layer]
                next_slot[layer] += 1
                node_repr[node_id] = (layer, idx)
            return node_repr[node_id]

        def carry_to_final(layer: int, idx: int) -> int:
            current_idx = idx
            for w_layer in range(layer, len(arch) - 2):
                if current_idx < weights[w_layer].shape[0] and current_idx < weights[w_layer].shape[1]:
                    weights[w_layer][current_idx, current_idx] = 1.0
            return current_idx

        for node_id in range(tree.node_count):
            if features[node_id] < 0:
                continue
            parent_layer, parent_idx = allocate(node_id)
            for child in (int(children_left[node_id]), int(children_right[node_id])):
                if features[child] >= 0:
                    child_layer, child_idx = allocate(child)
                    w_layer = max(0, child_layer - 1)
                    if parent_idx < weights[w_layer].shape[1]:
                        weights[w_layer][child_idx, parent_idx] = self._xavier(
                            weights[w_layer].shape[1], weights[w_layer].shape[0]
                        )
                    feature = int(features[child])
                    if feature < weights[w_layer].shape[1]:
                        weights[w_layer][child_idx, feature] = self._xavier(
                            weights[w_layer].shape[1], weights[w_layer].shape[0]
                        )
                else:
                    final_idx = carry_to_final(parent_layer, parent_idx)
                    if final_idx < weights[-1].shape[1]:
                        weights[-1][:, final_idx] = np.asarray(
                            [
                                self._xavier(weights[-1].shape[1], weights[-1].shape[0])
                                for _ in range(output_dim)
                            ],
                            dtype=np.float32,
                        )

        layers = nn.ModuleList()
        for w in weights:
            layer = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(w, dtype=torch.float32))
                scale = math.sqrt(3.0 / max(1, w.shape[0] + w.shape[1]))
                layer.bias.normal_(0.0, scale)
            layers.append(layer)
        return layers, arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
        return self.layers[-1](out)


class MLPBaseline(nn.Module):
    """Simple parameter-matched MLP baseline."""

    def __init__(self, n_features: int, output_dim: int, *, width: int, seed: int) -> None:
        super().__init__()
        set_all_seeds(seed)
        self.layers = nn.Sequential(
            nn.Linear(n_features, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NeuralEnsemble(nn.Module):
    """Average an ensemble of neural models."""

    def __init__(self, members: Sequence[nn.Module], *, task: Task) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([member(x) for member in self.members], dim=0)
        if self.task == "classification":
            probs = torch.softmax(outputs, dim=2).mean(dim=0)
            return torch.log(torch.clamp(probs, min=1e-8))
        return outputs.mean(dim=0)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for parameter in model.parameters():
        total += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter.detach()).item())
    return total, nonzero


def count_neurons(model: nn.Module) -> int:
    if isinstance(model, SoftTreeNetwork):
        return len(model.paths.node_features) + model.paths.leaf_count + model.paths.output_dim
    if isinstance(model, NeuralEnsemble):
        return sum(count_neurons(member) for member in model.members)
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.out_features
    return total


def count_layers(model: nn.Module) -> int:
    if isinstance(model, SoftTreeNetwork):
        return 3
    if isinstance(model, NeuralEnsemble):
        return sum(count_layers(member) for member in model.members)
    return sum(1 for module in model.modules() if isinstance(module, nn.Linear))


def train_model(
    model: nn.Module,
    *,
    task: Task,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> list[dict[str, float]]:
    set_all_seeds(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    xtr = torch.as_tensor(x_train, dtype=torch.float32)
    xva = torch.as_tensor(x_val, dtype=torch.float32)
    if task == "classification":
        ytr = torch.as_tensor(y_train, dtype=torch.long)
        yva = torch.as_tensor(y_val, dtype=torch.long)
    else:
        ytr = torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        yva = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    rows: list[dict[str, float]] = []
    n = len(xtr)
    batch_size = max(1, min(batch_size, n))
    start_time = time.perf_counter()
    for epoch in range(epochs):
        order = torch.randperm(n)
        model.train()
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            pred = model(xtr[batch])
            loss = F.cross_entropy(pred, ytr[batch]) if task == "classification" else F.mse_loss(pred, ytr[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(xtr)
            val_pred = model(xva)
            train_loss = (
                F.cross_entropy(train_pred, ytr).item()
                if task == "classification"
                else F.mse_loss(train_pred, ytr).item()
            )
            val_loss = (
                F.cross_entropy(val_pred, yva).item()
                if task == "classification"
                else F.mse_loss(val_pred, yva).item()
            )
            if task == "classification":
                train_metric = float((train_pred.argmax(dim=1) == ytr).float().mean().item())
                val_metric = float((val_pred.argmax(dim=1) == yva).float().mean().item())
            else:
                train_metric = -train_loss
                val_metric = -val_loss
        rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_metric_primary": float(train_metric),
                "val_metric_primary": float(val_metric),
                "elapsed_time": float(time.perf_counter() - start_time),
            }
        )
    return rows


def predict_model(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).detach().cpu().numpy()


def classification_summary_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    return classification_summary_from_probs(y_true, probs, pred=pred)


def classification_summary_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    pred: np.ndarray | None = None,
) -> dict[str, float]:
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    if pred is None:
        pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "cross_entropy": float(log_loss(y_true, probs, labels=np.arange(probs.shape[1]))),
    }


def regression_summary(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = pred.reshape(-1)
    return {
        "mse": float(mean_squared_error(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "explained_variance": float(explained_variance_score(y_true, pred)),
    }


def primary_metric_name(task: Task) -> str:
    return "accuracy" if task == "classification" else "mse"


def inverse_regression(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def source_predictions(source: Any, x: np.ndarray, task: Task) -> np.ndarray:
    if task == "classification":
        return source.predict(x)
    return source.predict(x).reshape(-1)


def model_fidelity(
    *,
    task: Task,
    model_output: np.ndarray,
    source_pred: np.ndarray,
    y_scaler: StandardScaler | None,
) -> float:
    if task == "classification":
        return float(np.mean(model_output.argmax(axis=1) == source_pred))
    assert y_scaler is not None
    pred = inverse_regression(model_output.reshape(-1), y_scaler)
    return float(mean_squared_error(source_pred, pred))


def make_model(
    model_name: ModelName,
    *,
    task: Task,
    estimator: Any,
    output_dim: int,
    alpha: float | None,
    seed: int,
    param_target: int | None = None,
) -> nn.Module:
    paths = extract_sklearn_paths(estimator, task=task, output_dim=output_dim)
    if model_name == "coexplain_soft":
        return SoftTreeNetwork(paths, alpha=float(alpha), mode="editable", seed=seed)
    if model_name == "same_arch_random":
        return SoftTreeNetwork(paths, alpha=float(alpha or 1.0), mode="editable", random_init=True, seed=seed)
    if model_name == "path_expansion":
        return SoftTreeNetwork(paths, alpha=float(alpha), mode="path_expansion", seed=seed)
    if model_name == "djinn":
        return DJINNLikeNetwork(estimator, task=task, output_dim=output_dim, seed=seed)
    if model_name == "mlp":
        width = choose_mlp_width(paths.n_features, output_dim, param_target or 0)
        return MLPBaseline(paths.n_features, output_dim, width=width, seed=seed)
    raise ValueError(f"unknown neural model: {model_name}")


def choose_mlp_width(n_features: int, output_dim: int, target_params: int) -> int:
    best_width = 8
    best_gap = float("inf")
    for width in range(2, 257):
        params = (n_features + 1) * width + (width + 1) * width + (width + 1) * output_dim
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_width = width
    return best_width


def build_source_model(
    *,
    task: Task,
    setting: str,
    depth: int,
    seed: int,
) -> Any:
    if task == "classification":
        if setting == "ensemble_10":
            return RandomForestClassifier(n_estimators=10, max_depth=depth, random_state=seed)
        return DecisionTreeClassifier(max_depth=depth, random_state=seed)
    if setting == "ensemble_10":
        return RandomForestRegressor(n_estimators=10, max_depth=depth, random_state=seed)
    return DecisionTreeRegressor(max_depth=depth, random_state=seed)


def source_estimators(source: Any) -> list[Any]:
    return list(getattr(source, "estimators_", [source]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def summarize_runs(rows: list[dict[str, Any]], *, task: Task) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["setting"]), str(row["model_name"]))
        groups.setdefault(key, []).append(row)
    metric = "final_test_accuracy" if task == "classification" else "final_test_mse"
    summaries = []
    for (dataset, setting, model), values in groups.items():
        vals = [float(v[metric]) for v in values if v.get(metric) not in ("", None)]
        if not vals:
            continue
        summaries.append(
            {
                "dataset": dataset,
                "setting": setting,
                "model_name": model,
                f"{metric}_mean": mean(vals),
                f"{metric}_std": pstdev(vals) if len(vals) > 1 else 0.0,
                "runs": len(vals),
            }
        )
    return summaries


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    tables_dir = results_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_epoch: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    model_size: list[dict[str, Any]] = []

    def maybe_flush() -> None:
        if getattr(args, "stream_results", False):
            write_outputs(raw_dir, tables_dir, per_epoch, per_run, model_size)

    for dataset in args.datasets:
        if getattr(args, "verbose", False):
            print(f"[dataset] {dataset}", flush=True)
        x, y, task = load_dataset(
            dataset,
            seed=args.split_seeds[0],
            max_samples=args.max_samples,
            use_config_max_samples=not args.no_dataset_cap,
        )
        config = DATASET_CONFIGS[dataset]
        output_dim = int(len(np.unique(y))) if task == "classification" else 1
        stratify_full = y if task == "classification" else None
        for split_seed in args.split_seeds:
            if getattr(args, "verbose", False):
                print(f"  [split] seed={split_seed}", flush=True)
            train_idx, test_idx = train_test_split(
                np.arange(len(x)),
                test_size=0.2,
                random_state=split_seed,
                stratify=stratify_full,
            )
            x_train_full, x_test = x[train_idx], x[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]
            stratify_train = y_train_full if task == "classification" else None
            train_sub_idx, val_sub_idx = train_test_split(
                np.arange(len(x_train_full)),
                test_size=0.2,
                random_state=split_seed + 100,
                stratify=stratify_train,
            )
            x_train, x_val = x_train_full[train_sub_idx], x_train_full[val_sub_idx]
            y_train, y_val = y_train_full[train_sub_idx], y_train_full[val_sub_idx]

            x_scaler = MinMaxScaler().fit(x_train)
            x_train_s = x_scaler.transform(x_train).astype(np.float32)
            x_val_s = x_scaler.transform(x_val).astype(np.float32)
            x_test_s = x_scaler.transform(x_test).astype(np.float32)
            x_train_full_s = x_scaler.transform(x_train_full).astype(np.float32)

            y_scaler: StandardScaler | None = None
            if task == "regression":
                y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
                y_train_nn = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
                y_val_nn = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)
                y_train_full_tree = y_scaler.transform(y_train_full.reshape(-1, 1)).reshape(-1)
            else:
                y_train_nn = y_train.astype(np.int64)
                y_val_nn = y_val.astype(np.int64)
                y_train_full_tree = y_train_full

            for setting in args.settings:
                if getattr(args, "verbose", False):
                    print(f"    [setting] {setting}", flush=True)
                source = build_source_model(
                    task=task,
                    setting=setting,
                    depth=int(config["tree_depth"]),
                    seed=split_seed,
                )
                source.fit(x_train_full_s, y_train_full_tree)
                source_test_pred_scaled = source_predictions(source, x_test_s, task)
                source_test_pred = (
                    inverse_regression(source_test_pred_scaled, y_scaler)
                    if task == "regression" and y_scaler is not None
                    else source_test_pred_scaled
                )

                if "source_tree" in args.models:
                    metrics = (
                        classification_summary_from_probs(y_test, source.predict_proba(x_test_s))
                        if task == "classification"
                        else regression_summary(y_test, source_test_pred)
                    )
                    per_run.append(
                        base_run_row(
                            dataset, task, split_seed, setting, "source_tree", "", 0,
                            metrics, metrics, None, None, 0.0,
                        )
                    )
                    add_source_size(model_size, dataset, setting, source)
                    maybe_flush()

                estimators = source_estimators(source)
                for model_name in args.models:
                    if model_name == "source_tree":
                        continue
                    alpha_values: list[float | None] = (
                        list(args.alphas)
                        if model_name in {"coexplain_soft", "same_arch_random", "path_expansion"}
                        else [None]
                    )
                    for alpha in alpha_values:
                        members = []
                        param_target = None
                        if model_name == "mlp":
                            ref = make_model(
                                "coexplain_soft",
                                task=task,
                                estimator=estimators[0],
                                output_dim=output_dim,
                                alpha=float(args.alphas[0]),
                                seed=split_seed,
                            )
                            param_target = count_parameters(ref)[0]
                        if getattr(args, "verbose", False):
                            alpha_text = "" if alpha is None else f" alpha={alpha}"
                            print(f"      [model] {model_name}{alpha_text}", flush=True)
                        for tree_id, estimator in enumerate(estimators):
                            members.append(
                                make_model(
                                    model_name,
                                    task=task,
                                    estimator=estimator,
                                    output_dim=output_dim,
                                    alpha=alpha,
                                    seed=split_seed * 1000 + tree_id,
                                    param_target=param_target,
                                )
                            )
                        model: nn.Module = (
                            members[0]
                            if setting == "single_tree"
                            else NeuralEnsemble(members, task=task)
                        )
                        initial_output = predict_model(model, x_test_s)
                        initial_metrics = eval_output(task, y_test, initial_output, y_scaler)
                        initial_fidelity = model_fidelity(
                            task=task,
                            model_output=initial_output,
                            source_pred=source_test_pred,
                            y_scaler=y_scaler,
                        )
                        initial_train_loss = compute_loss(model, task, x_train_s, y_train_nn)
                        initial_val_loss = compute_loss(model, task, x_val_s, y_val_nn)

                        start = time.perf_counter()
                        history = train_model(
                            model,
                            task=task,
                            x_train=x_train_s,
                            y_train=y_train_nn,
                            x_val=x_val_s,
                            y_val=y_val_nn,
                            epochs=int(config["epochs"] if args.epochs is None else args.epochs),
                            learning_rate=float(config["learning_rate"] if args.learning_rate is None else args.learning_rate),
                            batch_size=int(config["batch_size"] if args.batch_size is None else args.batch_size),
                            seed=split_seed,
                        )
                        training_time = time.perf_counter() - start
                        for item in history:
                            per_epoch.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "split_seed": split_seed,
                                    "setting": setting,
                                    "model_name": model_name,
                                    "ensemble_size": len(members),
                                    "tree_id": "ensemble" if len(members) > 1 else 0,
                                    "alpha": "" if alpha is None else alpha,
                                    "init_seed": split_seed,
                                    **item,
                                }
                            )

                        final_output = predict_model(model, x_test_s)
                        final_metrics = eval_output(task, y_test, final_output, y_scaler)
                        final_fidelity = model_fidelity(
                            task=task,
                            model_output=final_output,
                            source_pred=source_test_pred,
                            y_scaler=y_scaler,
                        )
                        best_epoch, best_val = best_epoch_loss(history)
                        row = base_run_row(
                            dataset,
                            task,
                            split_seed,
                            setting,
                            model_name,
                            "" if alpha is None else alpha,
                            len(members),
                            initial_metrics,
                            final_metrics,
                            initial_fidelity,
                            final_fidelity,
                            training_time,
                        )
                        row["initial_train_loss"] = initial_train_loss
                        row["initial_val_loss"] = initial_val_loss
                        row["best_val_loss"] = best_val
                        row["best_epoch"] = best_epoch
                        row["area_under_train_loss_curve"] = float(sum(h["train_loss"] for h in history))
                        row["area_under_val_loss_curve"] = float(sum(h["val_loss"] for h in history))
                        per_run.append(row)
                        total_params, nonzero_params = count_parameters(model)
                        model_size.append(
                            {
                                "dataset": dataset,
                                "setting": setting,
                                "model_name": model_name,
                                "alpha": "" if alpha is None else alpha,
                                "num_layers": count_layers(model),
                                "num_neurons": count_neurons(model),
                                "num_trainable_parameters": total_params,
                                "num_nonzero_parameters": nonzero_params,
                                "source_tree_depth": np.mean([e.tree_.max_depth for e in estimators]),
                                "source_tree_num_nodes": np.mean([e.tree_.node_count for e in estimators]),
                                "source_tree_num_leaves": np.mean(
                                    [sum(e.tree_.children_left == e.tree_.children_right) for e in estimators]
                                ),
                                "average_per_model_size": total_params / len(members),
                                "total_ensemble_size": total_params,
                            }
                        )
                        maybe_flush()

    write_outputs(raw_dir, tables_dir, per_epoch, per_run, model_size)
    metadata = {
        "python": f"{tuple(__import__('sys').version_info[:3])}",
        "torch": torch.__version__,
        "sklearn_datasets": args.datasets,
        "settings": args.settings,
        "models": args.models,
        "alphas": args.alphas,
        "split_seeds": args.split_seeds,
        "max_samples": args.max_samples,
        "no_dataset_cap": args.no_dataset_cap,
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"per_run": per_run, "model_size": model_size, "per_epoch": per_epoch}


def compute_loss(model: nn.Module, task: Task, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32))
        if task == "classification":
            target = torch.as_tensor(y, dtype=torch.long)
            return float(F.cross_entropy(pred, target).item())
        target = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
        return float(F.mse_loss(pred, target).item())


def eval_output(
    task: Task,
    y_true: np.ndarray,
    output: np.ndarray,
    y_scaler: StandardScaler | None,
) -> dict[str, float]:
    if task == "classification":
        return classification_summary_from_logits(y_true, output)
    assert y_scaler is not None
    return regression_summary(y_true, inverse_regression(output.reshape(-1), y_scaler))


def best_epoch_loss(history: list[dict[str, float]]) -> tuple[int, float]:
    best = min(history, key=lambda row: row["val_loss"])
    return int(best["epoch"]), float(best["val_loss"])


def base_run_row(
    dataset: str,
    task: Task,
    split_seed: int,
    setting: str,
    model_name: str,
    alpha: Any,
    ensemble_size: int,
    initial_metrics: dict[str, float],
    final_metrics: dict[str, float],
    initial_fidelity: float | None,
    final_fidelity: float | None,
    training_time: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "split_seed": split_seed,
        "setting": setting,
        "model_name": model_name,
        "ensemble_size": ensemble_size,
        "alpha": alpha,
        "init_seed": split_seed,
        "fidelity_initial": "" if initial_fidelity is None else initial_fidelity,
        "fidelity_final": "" if final_fidelity is None else final_fidelity,
        "training_time_total": training_time,
    }
    for key, value in initial_metrics.items():
        row[f"initial_test_{key}"] = value
    for key, value in final_metrics.items():
        row[f"final_test_{key}"] = value
    return row


def add_source_size(rows: list[dict[str, Any]], dataset: str, setting: str, source: Any) -> None:
    estimators = source_estimators(source)
    rows.append(
        {
            "dataset": dataset,
            "setting": setting,
            "model_name": "source_tree",
            "alpha": "",
            "num_layers": 0,
            "num_neurons": np.mean([e.tree_.node_count for e in estimators]),
            "num_trainable_parameters": 0,
            "num_nonzero_parameters": 0,
            "source_tree_depth": np.mean([e.tree_.max_depth for e in estimators]),
            "source_tree_num_nodes": np.mean([e.tree_.node_count for e in estimators]),
            "source_tree_num_leaves": np.mean(
                [sum(e.tree_.children_left == e.tree_.children_right) for e in estimators]
            ),
            "average_per_model_size": 0,
            "total_ensemble_size": 0,
        }
    )


def write_outputs(
    raw_dir: Path,
    tables_dir: Path,
    per_epoch: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
    model_size: list[dict[str, Any]],
) -> None:
    epoch_fields = [
        "dataset", "task", "split_seed", "setting", "model_name", "ensemble_size",
        "tree_id", "alpha", "init_seed", "epoch", "train_loss", "val_loss",
        "train_metric_primary", "val_metric_primary", "elapsed_time",
    ]
    run_fields = sorted({key for row in per_run for key in row})
    size_fields = sorted({key for row in model_size for key in row})
    write_csv(raw_dir / "per_epoch_metrics.csv", per_epoch, epoch_fields)
    write_csv(raw_dir / "per_run_summary.csv", per_run, run_fields)
    write_csv(raw_dir / "model_size_summary.csv", model_size, size_fields)

    write_csv(tables_dir / "table_final_performance.csv", per_run, run_fields)
    write_csv(tables_dir / "table_initialization_quality.csv", per_run, run_fields)
    write_csv(tables_dir / "table_fidelity.csv", per_run, run_fields)
    write_csv(tables_dir / "table_model_size.csv", model_size, size_fields)
    write_csv(tables_dir / "table_alpha_sensitivity.csv", per_run, run_fields)
    selected_alpha = select_alpha_rows(per_run)
    if selected_alpha:
        selected_fields = sorted({key for row in selected_alpha for key in row})
        write_csv(tables_dir / "table_validation_selected_alpha.csv", selected_alpha, selected_fields)


def select_alpha_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alpha_models = {"coexplain_soft", "same_arch_random", "path_expansion"}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_run:
        if row.get("model_name") not in alpha_models:
            continue
        if row.get("best_val_loss") in ("", None):
            continue
        key = (
            row.get("dataset"),
            row.get("task"),
            row.get("split_seed"),
            row.get("setting"),
            row.get("model_name"),
        )
        groups.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for values in groups.values():
        best = min(values, key=lambda row: float(row["best_val_loss"]))
        chosen = dict(best)
        chosen["selection_rule"] = "min_best_val_loss"
        selected.append(chosen)
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run soft parser feasibility baselines.")
    parser.add_argument("--datasets", nargs="+", default=["diabetes", "breast_cancer"])
    parser.add_argument("--settings", nargs="+", default=["single_tree"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["source_tree", "coexplain_soft", "same_arch_random", "djinn", "path_expansion", "mlp"],
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[5.0, 20.0])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--results-dir", default="results/performance_feasibility")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-dataset-cap", action="store_true")
    parser.add_argument("--stream-results", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
