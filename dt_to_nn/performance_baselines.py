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


class CoExplainZeroPaddedNetwork(nn.Module):
    """Dense zero-padded soft parser following the Editable-XAI enhancement idea.

    The tree-derived connections are initialized explicitly, while all other
    dense matrix entries start at zero and are trainable. During optimization,
    those zero entries may become nonzero, which is the "zero padding" capacity
    used by the enhanced parsed network.
    """

    def __init__(
        self,
        paths: SklearnTreePaths,
        *,
        alpha: float,
        zero_padding_width: int = 8,
        zero_padding_layers: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        set_all_seeds(seed)
        self.paths = paths
        self.task = paths.task
        self.zero_padding_width = zero_padding_width
        self.zero_padding_layers = zero_padding_layers
        weights, biases, activations = self._initial_matrices(
            paths,
            alpha=alpha,
            zero_padding_width=zero_padding_width,
            zero_padding_layers=zero_padding_layers,
        )
        self.activations = activations
        self.layers = nn.ModuleList()
        for weight, bias in zip(weights, biases):
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
            self.layers.append(layer)

    @staticmethod
    def _initial_matrices(
        paths: SklearnTreePaths,
        *,
        alpha: float,
        zero_padding_width: int,
        zero_padding_layers: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
        n_internal = len(paths.node_features)
        condition_width = 2 * n_internal + zero_padding_width
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        activations: list[str] = []

        w_condition = np.zeros((condition_width, paths.n_features), dtype=np.float32)
        b_condition = np.zeros(condition_width, dtype=np.float32)
        for idx, (feature, threshold) in enumerate(
            zip(paths.node_features, paths.node_thresholds)
        ):
            left_row = 2 * idx
            right_row = 2 * idx + 1
            w_condition[left_row, int(feature)] = -float(alpha)
            b_condition[left_row] = float(alpha * threshold)
            w_condition[right_row, int(feature)] = float(alpha)
            b_condition[right_row] = float(-alpha * threshold)
        weights.append(w_condition)
        biases.append(b_condition)
        activations.append("sigmoid")

        max_depth = max((len(path) for path in paths.paths), default=0)
        total_path_layers = max_depth + zero_padding_layers
        if total_path_layers == 0:
            w_out = np.zeros((paths.output_dim, condition_width), dtype=np.float32)
            b_out = np.asarray(paths.leaf_outputs[0], dtype=np.float32)
            weights.append(w_out)
            biases.append(b_out)
            activations.append("identity")
            return weights, biases, activations

        path_slots_by_depth: list[dict[tuple[tuple[int, int], ...], int]] = []
        for depth in range(1, total_path_layers + 1):
            prefixes: list[tuple[tuple[int, int], ...]] = []
            for path in paths.paths:
                if not path:
                    prefixes.append(())
                elif depth <= len(path):
                    prefixes.append(path[:depth])
                else:
                    prefixes.append(path)
            ordered = sorted(set(prefixes), key=lambda item: (len(item), item))
            path_slots_by_depth.append({prefix: idx for idx, prefix in enumerate(ordered)})

        prev_width = condition_width
        for depth, slots in enumerate(path_slots_by_depth, start=1):
            path_offset = condition_width
            width = condition_width + len(slots) + zero_padding_width
            w = np.zeros((width, prev_width), dtype=np.float32)
            b = np.zeros(width, dtype=np.float32)
            for row in range(condition_width):
                if row < prev_width:
                    w[row, row] = 1.0
            for prefix, row in slots.items():
                row = path_offset + row
                if not prefix:
                    b[row] = 1.0
                    continue
                if depth == 1:
                    node_idx, branch = prefix[-1]
                    w[row, 2 * node_idx + branch] = 1.0
                elif depth <= len(prefix):
                    parent = prefix[:-1]
                    parent_col = path_offset + path_slots_by_depth[depth - 2][parent]
                    node_idx, branch = prefix[-1]
                    w[row, parent_col] = 1.0
                    w[row, 2 * node_idx + branch] = 1.0
                    b[row] = -1.0
                else:
                    parent_col = path_offset + path_slots_by_depth[depth - 2][prefix]
                    w[row, parent_col] = 1.0
            weights.append(w)
            biases.append(b)
            activations.append("relu")
            prev_width = width

        final_slots = path_slots_by_depth[-1]
        w_out = np.zeros((paths.output_dim, prev_width), dtype=np.float32)
        b_out = np.zeros(paths.output_dim, dtype=np.float32)
        for path, leaf_output in zip(paths.paths, paths.leaf_outputs):
            col = condition_width + final_slots[path]
            w_out[:, col] += leaf_output
        weights.append(w_out)
        biases.append(b_out)
        activations.append("identity")
        return weights, biases, activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, activation in zip(self.layers, self.activations):
            out = layer(out)
            if activation == "sigmoid":
                out = torch.sigmoid(out)
            elif activation == "relu":
                out = F.relu(out)
            elif activation == "identity":
                pass
            else:  # pragma: no cover
                raise ValueError(f"unknown activation: {activation}")
        return out


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


class DJINNSparseFixedNetwork(DJINNLikeNetwork):
    """DJINN-like model where initially zero weights are constrained to stay zero."""

    def __init__(
        self,
        estimator: Any,
        *,
        task: Task,
        output_dim: int,
        seed: int = 0,
    ) -> None:
        super().__init__(estimator, task=task, output_dim=output_dim, seed=seed)
        self.weight_masks: list[torch.Tensor] = []
        for layer in self.layers:
            mask = layer.weight.detach().ne(0.0).float()
            self.register_buffer(f"_weight_mask_{len(self.weight_masks)}", mask)
            self.weight_masks.append(mask)
            layer.weight.register_hook(lambda grad, m=mask: grad * m)
        self.apply_constraints()

    def apply_constraints(self) -> None:
        with torch.no_grad():
            for layer, mask in zip(self.layers, self.weight_masks):
                layer.weight.mul_(mask)


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


class LSUVNetwork(nn.Module):
    """Parameter-matched MLP with layer-sequential unit-variance initialization."""

    def __init__(
        self,
        n_features: int,
        output_dim: int,
        *,
        width: int,
        calibration_x: np.ndarray,
        variance_tolerance: float = 0.1,
        max_trials: int = 10,
        calibration_batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(calibration_x) == 0:
            raise ValueError("LSUV requires a non-empty calibration batch")
        if variance_tolerance <= 0.0 or max_trials <= 0:
            raise ValueError("LSUV tolerance and max_trials must be positive")
        set_all_seeds(seed)
        self.layers = nn.ModuleList(
            [
                nn.Linear(n_features, width),
                nn.Linear(width, width),
                nn.Linear(width, output_dim),
            ]
        )
        with torch.no_grad():
            for layer in self.layers:
                nn.init.orthogonal_(layer.weight)
                layer.bias.zero_()

        rng = np.random.default_rng(seed)
        batch_size = min(calibration_batch_size, len(calibration_x))
        indices = rng.choice(len(calibration_x), size=batch_size, replace=False)
        batch = torch.as_tensor(calibration_x[indices], dtype=torch.float32)
        self.initial_variances = self._unit_variance_initialize(
            batch,
            tolerance=variance_tolerance,
            max_trials=max_trials,
        )

    def _output_at_layer(self, x: torch.Tensor, target_layer: int) -> torch.Tensor:
        out = x
        for layer_id, layer in enumerate(self.layers):
            out = layer(out)
            if layer_id == target_layer:
                return out
            out = F.relu(out)
        raise IndexError(target_layer)

    def _unit_variance_initialize(
        self,
        batch: torch.Tensor,
        *,
        tolerance: float,
        max_trials: int,
    ) -> list[float]:
        variances: list[float] = []
        with torch.no_grad():
            for layer_id, layer in enumerate(self.layers):
                for _ in range(max_trials):
                    output = self._output_at_layer(batch, layer_id)
                    variance = float(output.var(unbiased=False).item())
                    if not math.isfinite(variance) or variance <= 1e-12:
                        break
                    if abs(variance - 1.0) < tolerance:
                        break
                    layer.weight.div_(math.sqrt(variance))
                final_output = self._output_at_layer(batch, layer_id)
                variances.append(float(final_output.var(unbiased=False).item()))
        return variances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
        return self.layers[-1](out)


def make_lsuv(
    n_features: int,
    output_dim: int,
    *,
    width: int,
    calibration_x: np.ndarray,
    seed: int = 0,
) -> LSUVNetwork:
    """Build a parameter-matched MLP and initialize it with LSUV."""
    return LSUVNetwork(
        n_features,
        output_dim,
        width=width,
        calibration_x=calibration_x,
        seed=seed,
    )


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

    def apply_constraints(self) -> None:
        for member in self.members:
            apply_constraints = getattr(member, "apply_constraints", None)
            if callable(apply_constraints):
                apply_constraints()


class TBNNNetwork(nn.Module):
    """Ivanova-Kubat tree-based neural network (TBNN) initialization.

    This implements the three-layer interval/AND/OR construction and the
    initialization in Equations (3), (4), (6), and (7) of the attached paper.
    TBNN is a classification method; the paper does not define a regression
    analogue.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        output_dim: int,
        activation_level: float = 0.99,
        sigmoid_slope: float = 10.0,
        epsilon: float = 1e-4,
        perturbation: float = 1e-3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(estimator, DecisionTreeClassifier):
            raise TypeError("TBNN requires a fitted DecisionTreeClassifier")
        if not 0.5 < activation_level < 1.0:
            raise ValueError("activation_level must lie strictly between 0.5 and 1")
        if sigmoid_slope <= 0.0:
            raise ValueError("sigmoid_slope must be positive")
        if epsilon < 0.0 or perturbation < 0.0:
            raise ValueError("epsilon and perturbation must be non-negative")

        set_all_seeds(seed)
        self.task = "classification"
        self.output_dim = output_dim
        self.activation_level = float(activation_level)
        self.sigmoid_slope = float(sigmoid_slope)
        self.epsilon = float(epsilon)

        paths = extract_sklearn_paths(estimator, task="classification", output_dim=output_dim)
        interval_features, interval_centers, interval_widths, feature_intervals = (
            self._make_intervals(paths)
        )
        regular_rules = self._make_leaf_rules(paths, interval_centers, feature_intervals)
        leaf_classes = paths.leaf_outputs.argmax(axis=1)

        self.register_buffer(
            "interval_features", torch.as_tensor(interval_features, dtype=torch.long)
        )
        self.register_buffer(
            "interval_centers", torch.as_tensor(interval_centers, dtype=torch.float32)
        )
        self.register_buffer(
            "interval_widths", torch.as_tensor(interval_widths, dtype=torch.float32)
        )
        self.and_layer = nn.Linear(len(interval_features), paths.leaf_count)
        self.or_layer = nn.Linear(paths.leaf_count, output_dim)
        self._initialize_layers(regular_rules, leaf_classes, perturbation, seed)

    @staticmethod
    def _make_intervals(
        paths: SklearnTreePaths,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]]]:
        """Partition each used feature's scaled [0, 1] range at tree cuts."""
        interval_features: list[int] = []
        centers: list[float] = []
        widths: list[float] = []
        feature_intervals: dict[int, list[int]] = {}
        for feature in sorted(set(int(f) for f in paths.node_features)):
            cuts = sorted(
                set(
                    float(np.clip(t, 0.0, 1.0))
                    for f, t in zip(paths.node_features, paths.node_thresholds)
                    if int(f) == feature and 0.0 < float(t) < 1.0
                )
            )
            bounds = [0.0, *cuts, 1.0]
            indices: list[int] = []
            for low, high in zip(bounds[:-1], bounds[1:]):
                if high <= low:
                    continue
                indices.append(len(interval_features))
                interval_features.append(feature)
                centers.append((low + high) / 2.0)
                widths.append(high - low)
            feature_intervals[feature] = indices
        return (
            np.asarray(interval_features, dtype=np.int64),
            np.asarray(centers, dtype=np.float32),
            np.asarray(widths, dtype=np.float32),
            feature_intervals,
        )

    @staticmethod
    def _make_leaf_rules(
        paths: SklearnTreePaths,
        interval_centers: np.ndarray,
        feature_intervals: dict[int, list[int]],
    ) -> list[tuple[list[int], list[int]]]:
        """Reduce every root-to-leaf path to positive/negated interval literals."""
        rules: list[tuple[list[int], list[int]]] = []
        for path in paths.paths:
            bounds = {feature: [-np.inf, np.inf] for feature in feature_intervals}
            for node_idx, branch in path:
                feature = int(paths.node_features[node_idx])
                threshold = float(paths.node_thresholds[node_idx])
                if branch == 0:
                    bounds[feature][1] = min(bounds[feature][1], threshold)
                else:
                    bounds[feature][0] = max(bounds[feature][0], threshold)

            positive: list[int] = []
            negative: list[int] = []
            for feature, indices in feature_intervals.items():
                low, high = bounds[feature]
                if np.isneginf(low) and np.isposinf(high):
                    continue
                allowed = [
                    idx for idx in indices
                    if interval_centers[idx] > low and interval_centers[idx] <= high
                ]
                if len(allowed) == 1:
                    positive.append(allowed[0])
                else:
                    negative.extend(idx for idx in indices if idx not in allowed)
            rules.append((positive, negative))
        return rules

    def _initialize_layers(
        self,
        rules: list[tuple[list[int], list[int]]],
        leaf_classes: np.ndarray,
        perturbation: float,
        seed: int,
    ) -> None:
        a = self.activation_level
        h = self.sigmoid_slope
        eps = self.epsilon
        psi = math.log(a / (1.0 - a)) / h

        with torch.no_grad():
            self.and_layer.weight.fill_(eps)
            self.or_layer.weight.fill_(eps)

            for leaf, (positive, negative) in enumerate(rules):
                n, m = len(positive), len(negative)
                regular = n + m
                k = self.and_layer.in_features - regular
                if regular == 0:
                    self.and_layer.bias[leaf] = psi
                    continue
                denominator = a * (regular + 1) - regular
                if denominator <= 0.0:
                    raise ValueError(
                        "activation_level is too small for this tree; increase it "
                        f"above {regular / (regular + 1):.6f}"
                    )
                weight = 2.0 * (psi + k * eps) / denominator
                threshold = (psi + k * eps) * (
                    a * (regular - 1) + (n - m)
                ) / denominator
                self.and_layer.weight[leaf, positive] = weight
                self.and_layer.weight[leaf, negative] = -weight
                self.and_layer.bias[leaf] = -threshold

            for output in range(self.output_dim):
                regular = np.flatnonzero(leaf_classes == output).tolist()
                n = len(regular)
                k = self.or_layer.in_features - n
                if n == 0:
                    self.or_layer.bias[output] = -psi
                    continue
                denominator = a * (n + 1) - n
                if denominator <= 0.0:
                    raise ValueError(
                        "activation_level is too small for the number of leaves in a class"
                    )
                weight = 2.0 * (psi + k * eps) / denominator
                threshold = (psi + k * eps) * (n - a * (n - 1)) / denominator
                self.or_layer.weight[output, regular] = weight
                self.or_layer.bias[output] = -threshold

            if perturbation:
                generator = torch.Generator(device=self.and_layer.weight.device)
                generator.manual_seed(seed)
                for weight in (self.and_layer.weight, self.or_layer.weight):
                    noise = torch.empty_like(weight).uniform_(
                        -perturbation, perturbation, generator=generator
                    )
                    weight.add_(noise)

    def interval_memberships(self, x: torch.Tensor) -> torch.Tensor:
        selected = x[:, self.interval_features]
        closeness = (
            self.interval_widths - 2.0 * torch.abs(selected - self.interval_centers)
        ) / (2.0 * self.interval_widths)
        return torch.sigmoid(self.sigmoid_slope * closeness)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intervals = self.interval_memberships(x)
        conjunctions = torch.sigmoid(self.sigmoid_slope * self.and_layer(intervals))
        return torch.sigmoid(self.sigmoid_slope * self.or_layer(conjunctions))


def make_tbnn(
    estimator: Any,
    *,
    output_dim: int,
    activation_level: float = 0.99,
    sigmoid_slope: float = 10.0,
    epsilon: float = 1e-4,
    perturbation: float = 1e-3,
    seed: int = 0,
) -> TBNNNetwork:
    """Build a paper-method TBNN baseline from a fitted classification tree."""
    return TBNNNetwork(
        estimator,
        output_dim=output_dim,
        activation_level=activation_level,
        sigmoid_slope=sigmoid_slope,
        epsilon=epsilon,
        perturbation=perturbation,
        seed=seed,
    )


class KBANNNetwork(nn.Module):
    """Towell-Shavlik knowledge-based ANN initialized from tree-derived rules.

    Each decision-tree branch test is treated as a propositional supporting
    fact, each root-to-leaf rule becomes a conjunctive hidden unit, and class
    units implement disjunction. Rule links use the paper's strong weight and
    all additional links begin near zero so backpropagation can refine them.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        output_dim: int,
        omega: float = 4.0,
        condition_strength: float = 20.0,
        perturbation: float = 1e-3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(estimator, DecisionTreeClassifier):
            raise TypeError("KBANN requires a fitted DecisionTreeClassifier")
        if omega <= 0.0 or condition_strength <= 0.0:
            raise ValueError("omega and condition_strength must be positive")
        if perturbation < 0.0:
            raise ValueError("perturbation must be non-negative")

        set_all_seeds(seed)
        self.task = "classification"
        self.output_dim = output_dim
        self.paths = extract_sklearn_paths(
            estimator, task="classification", output_dim=output_dim
        )
        self.condition_strength = float(condition_strength)
        self.num_branch_facts = 2 * len(self.paths.node_features)
        self.num_fact_units = self.num_branch_facts + self.paths.n_features
        self.rule_layer = nn.Linear(self.num_fact_units, self.paths.leaf_count)
        self.output_layer = nn.Linear(self.paths.leaf_count, output_dim)
        self._initialize_from_rules(float(omega), perturbation, seed)

    def _initialize_from_rules(
        self,
        omega: float,
        perturbation: float,
        seed: int,
    ) -> None:
        leaf_classes = self.paths.leaf_outputs.argmax(axis=1)
        with torch.no_grad():
            self.rule_layer.weight.zero_()
            self.rule_layer.bias.zero_()
            self.output_layer.weight.zero_()
            self.output_layer.bias.fill_(-0.5 * omega)

            for leaf, path in enumerate(self.paths.paths):
                if path:
                    facts = [2 * node_idx + branch for node_idx, branch in path]
                    self.rule_layer.weight[leaf, facts] = omega
                    self.rule_layer.bias[leaf] = -(len(path) - 0.5) * omega
                else:
                    self.rule_layer.bias[leaf] = 0.5 * omega

            for output in range(self.output_dim):
                rules = np.flatnonzero(leaf_classes == output).tolist()
                self.output_layer.weight[output, rules] = omega

            if perturbation:
                generator = torch.Generator(device=self.rule_layer.weight.device)
                generator.manual_seed(seed)
                for parameter in self.parameters():
                    noise = torch.empty_like(parameter).uniform_(
                        -perturbation, perturbation, generator=generator
                    )
                    parameter.add_(noise)

    def supporting_facts(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.paths.node_features) == 0:
            return x
        selected = x[:, torch.as_tensor(self.paths.node_features, device=x.device)]
        thresholds = torch.as_tensor(
            self.paths.node_thresholds, dtype=x.dtype, device=x.device
        )
        left = torch.sigmoid(self.condition_strength * (thresholds - selected))
        right = 1.0 - left
        branch_facts = torch.stack((left, right), dim=2).reshape(len(x), -1)
        return torch.cat((branch_facts, x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        facts = self.supporting_facts(x)
        rules = torch.sigmoid(self.rule_layer(facts))
        return torch.sigmoid(self.output_layer(rules))


def make_kbann(
    estimator: Any,
    *,
    output_dim: int,
    seed: int = 0,
) -> KBANNNetwork:
    """Build a KBANN from the propositional rules of a fitted tree."""
    return KBANNNetwork(estimator, output_dim=output_dim, seed=seed)


class PBNNNetwork(nn.Module):
    """Setiono-Leow pruning-based neural network initialized from a tree.

    The source tree defines mutually exclusive attribute intervals and labels
    every valid binary interval pattern. A fully-connected one-hidden-layer
    network is fitted to those patterns with L-BFGS (a quasi-Newton method),
    then connections and redundant units are removed with a saturating
    penalty from the N2P2F family of pruning methods.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        output_dim: int,
        max_binary_patterns: int = 8192,
        pretrain_steps: int = 75,
        penalty_steps: int = 50,
        penalty_strength: float = 1e-4,
        penalty_sharpness: float = 10.0,
        ridge_strength: float = 1e-6,
        prune_threshold: float = 0.05,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(estimator, DecisionTreeClassifier):
            raise TypeError("PBNN requires a fitted DecisionTreeClassifier")
        if max_binary_patterns <= 0:
            raise ValueError("max_binary_patterns must be positive")
        if pretrain_steps < 0 or penalty_steps < 0:
            raise ValueError("pretraining step counts must be non-negative")
        if penalty_strength < 0.0 or ridge_strength < 0.0:
            raise ValueError("penalty strengths must be non-negative")
        if penalty_sharpness <= 0.0 or prune_threshold < 0.0:
            raise ValueError("penalty_sharpness must be positive and prune_threshold non-negative")

        set_all_seeds(seed)
        self.task = "classification"
        self.output_dim = output_dim
        paths = extract_sklearn_paths(estimator, task="classification", output_dim=output_dim)
        interval_features, interval_centers, interval_widths, feature_intervals = (
            TBNNNetwork._make_intervals(paths)
        )
        if not feature_intervals:
            raise ValueError("PBNN requires a tree with at least one decision node")

        self.register_buffer(
            "interval_features", torch.as_tensor(interval_features, dtype=torch.long)
        )
        self.register_buffer(
            "interval_centers", torch.as_tensor(interval_centers, dtype=torch.float32)
        )
        self.register_buffer(
            "interval_widths", torch.as_tensor(interval_widths, dtype=torch.float32)
        )
        self._group_features = sorted(feature_intervals)
        self._group_ranges: list[tuple[int, int]] = []
        for group_id, feature in enumerate(self._group_features):
            indices = feature_intervals[feature]
            start, end = indices[0], indices[-1] + 1
            self._group_ranges.append((start, end))
            cuts = interval_centers[start : end - 1] + interval_widths[start : end - 1] / 2.0
            self.register_buffer(
                f"_interval_cuts_{group_id}", torch.as_tensor(cuts, dtype=torch.float32)
            )

        hidden_width = max(1, paths.leaf_count)
        self.hidden_layer = nn.Linear(len(interval_features), hidden_width)
        self.output_layer = nn.Linear(hidden_width, output_dim)
        self._reset_parameters(seed)

        patterns, targets = self._binary_training_patterns(
            estimator,
            feature_intervals,
            interval_centers,
            max_binary_patterns=max_binary_patterns,
            seed=seed,
        )
        self.num_binary_patterns = len(patterns)
        self.total_binary_patterns = int(
            math.prod(len(feature_intervals[feature]) for feature in self._group_features)
        )
        pattern_tensor = torch.as_tensor(patterns, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
        self._fit_patterns(
            pattern_tensor,
            target_tensor,
            steps=pretrain_steps,
            penalty_strength=0.0,
            penalty_sharpness=penalty_sharpness,
            ridge_strength=0.0,
        )
        self._fit_patterns(
            pattern_tensor,
            target_tensor,
            steps=penalty_steps,
            penalty_strength=penalty_strength,
            penalty_sharpness=penalty_sharpness,
            ridge_strength=ridge_strength,
        )
        self._compress_and_mask(prune_threshold)

    def _reset_parameters(self, seed: int) -> None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        with torch.no_grad():
            for layer in (self.hidden_layer, self.output_layer):
                bound = math.sqrt(6.0 / max(1, layer.in_features + layer.out_features))
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.zero_()

    @staticmethod
    def _binary_training_patterns(
        estimator: Any,
        feature_intervals: dict[int, list[int]],
        interval_centers: np.ndarray,
        *,
        max_binary_patterns: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = sorted(feature_intervals)
        sizes = [len(feature_intervals[feature]) for feature in features]
        total = int(math.prod(sizes))
        if total <= max_binary_patterns:
            combinations = np.asarray(list(np.ndindex(*sizes)), dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            flat = np.sort(rng.choice(total, size=max_binary_patterns, replace=False))
            combinations = np.empty((len(flat), len(sizes)), dtype=np.int64)
            remainder = flat.copy()
            for col in range(len(sizes) - 1, -1, -1):
                combinations[:, col] = remainder % sizes[col]
                remainder //= sizes[col]

        n_intervals = sum(sizes)
        patterns = np.zeros((len(combinations), n_intervals), dtype=np.float32)
        source_inputs = np.full(
            (len(combinations), int(estimator.n_features_in_)), 0.5, dtype=np.float32
        )
        for group, feature in enumerate(features):
            indices = np.asarray(feature_intervals[feature], dtype=np.int64)
            selected = indices[combinations[:, group]]
            patterns[np.arange(len(patterns)), selected] = 1.0
            source_inputs[:, feature] = interval_centers[selected]

        predicted = estimator.predict(source_inputs)
        class_to_index = {label: idx for idx, label in enumerate(estimator.classes_)}
        targets = np.asarray([class_to_index[label] for label in predicted], dtype=np.int64)
        return patterns, targets

    def _fit_patterns(
        self,
        patterns: torch.Tensor,
        targets: torch.Tensor,
        *,
        steps: int,
        penalty_strength: float,
        penalty_sharpness: float,
        ridge_strength: float,
    ) -> None:
        if steps == 0:
            return
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=1.0,
            max_iter=steps,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = self.output_layer(torch.tanh(self.hidden_layer(patterns)))
            loss = F.cross_entropy(logits, targets)
            if penalty_strength or ridge_strength:
                weights = (self.hidden_layer.weight, self.output_layer.weight)
                compact = sum(
                    (
                        penalty_sharpness
                        * weight.square()
                        / (1.0 + penalty_sharpness * weight.square())
                    ).sum()
                    for weight in weights
                )
                ridge = sum(weight.square().sum() for weight in weights)
                loss = loss + penalty_strength * compact + ridge_strength * ridge
            loss.backward()
            return loss

        optimizer.step(closure)

    def _compress_and_mask(self, threshold: float) -> None:
        with torch.no_grad():
            hidden_mask = self.hidden_layer.weight.abs() >= threshold
            output_mask = self.output_layer.weight.abs() >= threshold
            for output in range(self.output_dim):
                if not output_mask[output].any():
                    strongest = int(self.output_layer.weight[output].abs().argmax().item())
                    output_mask[output, strongest] = True

            hidden_active = output_mask.any(dim=0) & (
                hidden_mask.any(dim=1) | (self.hidden_layer.bias.abs() >= threshold)
            )
            if not hidden_active.any():
                strongest = int(self.output_layer.weight.abs().sum(dim=0).argmax().item())
                hidden_active[strongest] = True
            active_hidden = torch.nonzero(hidden_active, as_tuple=False).flatten()

            input_active = hidden_mask[active_hidden].any(dim=0)
            if not input_active.any():
                strongest = int(
                    self.hidden_layer.weight[active_hidden].abs().sum(dim=0).argmax().item()
                )
                input_active[strongest] = True
            active_inputs = torch.nonzero(input_active, as_tuple=False).flatten()

            new_hidden = nn.Linear(len(active_inputs), len(active_hidden))
            new_output = nn.Linear(len(active_hidden), self.output_dim)
            new_hidden.weight.copy_(self.hidden_layer.weight[active_hidden][:, active_inputs])
            new_hidden.bias.copy_(self.hidden_layer.bias[active_hidden])
            new_output.weight.copy_(self.output_layer.weight[:, active_hidden])
            new_output.bias.copy_(self.output_layer.bias)

            hidden_keep = hidden_mask[active_hidden][:, active_inputs]
            output_keep = output_mask[:, active_hidden]
            new_hidden.weight.mul_(hidden_keep)
            new_output.weight.mul_(output_keep)
            self.hidden_layer = new_hidden
            self.output_layer = new_output
            self.register_buffer("active_interval_indices", active_inputs)
            self.register_buffer("_hidden_weight_mask", hidden_keep.float())
            self.register_buffer("_output_weight_mask", output_keep.float())
            self.hidden_layer.weight.register_hook(
                lambda grad: grad * self._hidden_weight_mask
            )
            self.output_layer.weight.register_hook(
                lambda grad: grad * self._output_weight_mask
            )
            self.apply_constraints()

    def interval_memberships(self, x: torch.Tensor) -> torch.Tensor:
        memberships = torch.zeros(
            (len(x), len(self.interval_features)), dtype=x.dtype, device=x.device
        )
        rows = torch.arange(len(x), device=x.device)
        for group_id, (feature, (start, _)) in enumerate(
            zip(self._group_features, self._group_ranges)
        ):
            cuts = getattr(self, f"_interval_cuts_{group_id}")
            bins = torch.bucketize(x[:, feature].contiguous(), cuts, right=False)
            memberships[rows, start + bins] = 1.0
        return memberships[:, self.active_interval_indices]

    def apply_constraints(self) -> None:
        with torch.no_grad():
            self.hidden_layer.weight.mul_(self._hidden_weight_mask)
            self.output_layer.weight.mul_(self._output_weight_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intervals = self.interval_memberships(x)
        return self.output_layer(torch.tanh(self.hidden_layer(intervals)))


def make_pbnn(
    estimator: Any,
    *,
    output_dim: int,
    seed: int = 0,
) -> PBNNNetwork:
    """Build a PBNN from a fitted classification tree."""
    return PBNNNetwork(estimator, output_dim=output_dim, seed=seed)


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
    if isinstance(model, CoExplainZeroPaddedNetwork):
        return sum(layer.out_features for layer in model.layers)
    if isinstance(model, NeuralEnsemble):
        return sum(count_neurons(member) for member in model.members)
    if isinstance(model, TBNNNetwork):
        return (
            len(model.interval_features)
            + model.and_layer.out_features
            + model.or_layer.out_features
        )
    if isinstance(model, KBANNNetwork):
        return (
            model.num_fact_units
            + model.rule_layer.out_features
            + model.output_layer.out_features
        )
    if isinstance(model, PBNNNetwork):
        return (
            len(model.active_interval_indices)
            + model.hidden_layer.out_features
            + model.output_layer.out_features
        )
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += module.out_features
    return total


def count_layers(model: nn.Module) -> int:
    if isinstance(model, SoftTreeNetwork):
        return 3
    if isinstance(model, CoExplainZeroPaddedNetwork):
        return len(model.layers)
    if isinstance(model, NeuralEnsemble):
        return sum(count_layers(member) for member in model.members)
    if isinstance(model, (TBNNNetwork, KBANNNetwork, PBNNNetwork)):
        return 3
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
            apply_constraints = getattr(model, "apply_constraints", None)
            if callable(apply_constraints):
                apply_constraints()

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
    calibration_x: np.ndarray | None = None,
) -> nn.Module:
    paths = extract_sklearn_paths(estimator, task=task, output_dim=output_dim)
    if model_name == "coexplain_soft":
        return SoftTreeNetwork(paths, alpha=float(alpha), mode="editable", seed=seed)
    if model_name == "coexplain_soft_zero_padded":
        return CoExplainZeroPaddedNetwork(
            paths,
            alpha=float(alpha),
            zero_padding_width=8,
            zero_padding_layers=1,
            seed=seed,
        )
    if model_name == "same_arch_random":
        return SoftTreeNetwork(paths, alpha=float(alpha or 1.0), mode="editable", random_init=True, seed=seed)
    if model_name == "path_expansion":
        return SoftTreeNetwork(paths, alpha=float(alpha), mode="path_expansion", seed=seed)
    if model_name == "djinn":
        return DJINNLikeNetwork(estimator, task=task, output_dim=output_dim, seed=seed)
    if model_name == "djinn_sparse_fixed":
        return DJINNSparseFixedNetwork(estimator, task=task, output_dim=output_dim, seed=seed)
    if model_name == "mlp":
        width = choose_mlp_width(paths.n_features, output_dim, param_target or 0)
        return MLPBaseline(paths.n_features, output_dim, width=width, seed=seed)
    if model_name == "lsuv":
        if calibration_x is None:
            raise ValueError(
                "LSUV requires calibration_x for unit-variance initialization"
            )
        width = choose_mlp_width(paths.n_features, output_dim, param_target or 0)
        return make_lsuv(
            paths.n_features,
            output_dim,
            width=width,
            calibration_x=calibration_x,
            seed=seed,
        )
    if model_name == "tbnn":
        if task != "classification":
            raise ValueError(
                "The Ivanova-Kubat TBNN baseline is classification-only; "
                "the paper does not specify a regression mapping."
            )
        return make_tbnn(estimator, output_dim=output_dim, seed=seed)
    if model_name == "pbnn":
        if task != "classification":
            raise ValueError("The Setiono-Leow PBNN baseline is classification-only.")
        return make_pbnn(estimator, output_dim=output_dim, seed=seed)
    if model_name == "kbann":
        if task != "classification":
            raise ValueError("The Towell-Shavlik KBANN baseline is classification-only.")
        return make_kbann(estimator, output_dim=output_dim, seed=seed)
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    completed: set[tuple[str, str, str, str, str]] = set()
    if getattr(args, "resume", False):
        per_epoch = read_csv_rows(raw_dir / "per_epoch_metrics.csv")
        per_run = read_csv_rows(raw_dir / "per_run_summary.csv")
        model_size = read_csv_rows(raw_dir / "model_size_summary.csv")
        for row in per_run:
            completed.add(
                (
                    str(row.get("dataset", "")),
                    str(row.get("split_seed", "")),
                    str(row.get("setting", "")),
                    str(row.get("model_name", "")),
                    str(row.get("alpha", "")),
                )
            )
        if getattr(args, "verbose", False):
            print(f"[resume] loaded {len(completed)} completed run rows", flush=True)

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
                    run_key = (dataset, str(split_seed), setting, "source_tree", "")
                    if run_key not in completed:
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
                        completed.add(run_key)
                        maybe_flush()

                estimators = source_estimators(source)
                for model_name in args.models:
                    if model_name == "source_tree":
                        continue
                    if (
                        model_name in {"tbnn", "pbnn", "kbann"}
                        and task != "classification"
                    ):
                        if getattr(args, "verbose", False):
                            print(
                                f"      [skip] {model_name} is classification-only",
                                flush=True,
                            )
                        continue
                    alpha_values: list[float | None] = (
                        list(args.alphas)
                        if model_name in {
                            "coexplain_soft",
                            "coexplain_soft_zero_padded",
                            "same_arch_random",
                            "path_expansion",
                        }
                        else [None]
                    )
                    for alpha in alpha_values:
                        alpha_key = "" if alpha is None else str(float(alpha))
                        run_key = (dataset, str(split_seed), setting, model_name, alpha_key)
                        if run_key in completed:
                            if getattr(args, "verbose", False):
                                alpha_text = "" if alpha is None else f" alpha={alpha}"
                                print(f"      [skip] {model_name}{alpha_text}", flush=True)
                            continue
                        members = []
                        param_target = None
                        if model_name in {"mlp", "lsuv"}:
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
                                    calibration_x=x_train_s,
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
                        completed.add(run_key)
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
    alpha_models = {
        "coexplain_soft",
        "coexplain_soft_zero_padded",
        "same_arch_random",
        "path_expansion",
    }
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
        default=[
            "source_tree",
            "coexplain_soft",
            "same_arch_random",
            "djinn",
            "path_expansion",
            "mlp",
            "lsuv",
            "tbnn",
            "pbnn",
            "kbann",
        ],
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
