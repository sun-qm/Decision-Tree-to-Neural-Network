"""Experimental tree-to-network variants.

These classes intentionally live outside the main converter so the original
Editable XAI parser remains available as an unchanged baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.torch_trainable import TorchParsedNetwork
from dt_to_nn.tree import DecisionNode, Leaf, TreeNode, class_labels, iter_internal_nodes, predict_batch


AndKind = Literal["lukasiewicz", "product", "min", "softmin"]


@dataclass(frozen=True)
class ExtractedPaths:
    classes: tuple[Any, ...]
    condition_names: tuple[str, ...]
    condition_features: np.ndarray
    condition_thresholds: np.ndarray
    condition_signs: np.ndarray
    path_matrix: np.ndarray
    path_lengths: np.ndarray
    path_labels: tuple[Any, ...]
    path_to_class: np.ndarray


@dataclass(frozen=True)
class StructureStats:
    neurons: int
    edges: int
    layers: int
    paths: int
    conditions: int
    notes: str


def extract_tree_paths(tree: TreeNode, classes: Sequence[Any] | None = None) -> ExtractedPaths:
    labels = tuple(classes if classes is not None else class_labels(tree))
    internal_nodes = list(iter_internal_nodes(tree))
    condition_index: dict[tuple[int, str], int] = {}
    condition_names: list[str] = []
    condition_features: list[int] = []
    condition_thresholds: list[float] = []
    condition_signs: list[float] = []

    for node_idx, node in enumerate(internal_nodes):
        node_name = node.name or f"n{node_idx}"
        for branch, sign in (("true", 1.0), ("false", -1.0)):
            condition_index[(id(node), branch)] = len(condition_names)
            condition_names.append(f"{branch}_{node_name}")
            condition_features.append(node.feature_index)
            condition_thresholds.append(node.threshold)
            condition_signs.append(sign)

    rows: list[np.ndarray] = []
    path_labels: list[Any] = []

    def walk(node: TreeNode, selected: list[int]) -> None:
        if isinstance(node, Leaf):
            row = np.zeros(len(condition_names), dtype=np.float32)
            row[selected] = 1.0
            rows.append(row)
            path_labels.append(node.label)
            return
        walk(node.true_child, selected + [condition_index[(id(node), "true")]])
        walk(node.false_child, selected + [condition_index[(id(node), "false")]])

    walk(tree, [])

    path_matrix = np.vstack(rows) if rows else np.zeros((1, len(condition_names)), dtype=np.float32)
    path_lengths = path_matrix.sum(axis=1).astype(np.float32)
    path_to_class = np.zeros((len(labels), len(path_labels)), dtype=np.float32)
    class_index = {label: i for i, label in enumerate(labels)}
    for path_idx, label in enumerate(path_labels):
        path_to_class[class_index[label], path_idx] = 1.0

    return ExtractedPaths(
        classes=labels,
        condition_names=tuple(condition_names),
        condition_features=np.asarray(condition_features, dtype=np.int64),
        condition_thresholds=np.asarray(condition_thresholds, dtype=np.float32),
        condition_signs=np.asarray(condition_signs, dtype=np.float32),
        path_matrix=path_matrix,
        path_lengths=path_lengths,
        path_labels=tuple(path_labels),
        path_to_class=path_to_class,
    )


class PathMatrixNetwork(nn.Module):
    """Three-layer path-matrix network: input -> conditions -> paths -> output."""

    def __init__(
        self,
        paths: ExtractedPaths,
        *,
        n_features: int,
        and_kind: AndKind = "lukasiewicz",
        temperature: float = 30.0,
        softmin_beta: float = 20.0,
        trainable_conditions: bool = True,
        trainable_lukasiewicz_layers: bool = True,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.classes = paths.classes
        self.and_kind = and_kind
        self.softmin_beta = softmin_beta
        self.register_buffer("path_matrix", torch.as_tensor(paths.path_matrix, dtype=torch.float32))
        self.register_buffer("path_lengths", torch.as_tensor(paths.path_lengths, dtype=torch.float32))
        self.register_buffer("path_to_class", torch.as_tensor(paths.path_to_class, dtype=torch.float32))

        w = torch.zeros((len(paths.condition_names), n_features), dtype=torch.float32)
        b = torch.zeros(len(paths.condition_names), dtype=torch.float32)
        for row, (feature, threshold, sign) in enumerate(
            zip(paths.condition_features, paths.condition_thresholds, paths.condition_signs)
        ):
            w[row, int(feature)] = float(sign * temperature)
            b[row] = float(-sign * temperature * threshold)
        self.condition = nn.Linear(n_features, len(paths.condition_names))
        with torch.no_grad():
            self.condition.weight.copy_(w)
            self.condition.bias.copy_(b)
        if not trainable_conditions:
            for parameter in self.condition.parameters():
                parameter.requires_grad = False

        self.path_layer: nn.Linear | None = None
        self.output_layer: nn.Linear | None = None
        if and_kind == "lukasiewicz" and trainable_lukasiewicz_layers:
            self.path_layer = nn.Linear(len(paths.condition_names), paths.path_matrix.shape[0])
            self.output_layer = nn.Linear(paths.path_matrix.shape[0], len(paths.classes))
            with torch.no_grad():
                self.path_layer.weight.copy_(torch.as_tensor(paths.path_matrix, dtype=torch.float32))
                self.path_layer.bias.copy_(1.0 - torch.as_tensor(paths.path_lengths, dtype=torch.float32))
                self.output_layer.weight.copy_(torch.as_tensor(paths.path_to_class, dtype=torch.float32))
                self.output_layer.bias.zero_()

    def set_temperature(self, temperature: float) -> None:
        """Reset condition logits to the original tree thresholds at a new temperature."""

        with torch.no_grad():
            self.condition.weight.zero_()
            self.condition.bias.zero_()
            for row, (feature, threshold, sign) in enumerate(
                zip(
                    self.paths.condition_features,
                    self.paths.condition_thresholds,
                    self.paths.condition_signs,
                )
            ):
                self.condition.weight[row, int(feature)] = float(sign * temperature)
                self.condition.bias[row] = float(-sign * temperature * threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conditions = torch.sigmoid(self.condition(x))
        if self.path_layer is not None and self.output_layer is not None:
            path_values = F.relu(self.path_layer(conditions))
            return torch.clamp(self.output_layer(path_values), 0.0, 1.0)
        path_values = self._path_values(conditions)
        outputs = torch.clamp(path_values @ self.path_to_class.T, 0.0, 1.0)
        return outputs

    def _path_values(self, conditions: torch.Tensor) -> torch.Tensor:
        if self.and_kind == "lukasiewicz":
            return F.relu(conditions @ self.path_matrix.T - (self.path_lengths - 1.0))

        mask = self.path_matrix.bool()
        expanded = conditions[:, None, :].expand(-1, mask.shape[0], -1)
        selected = torch.where(mask[None, :, :], expanded, torch.ones_like(expanded))
        if self.and_kind == "product":
            return selected.prod(dim=2)
        if self.and_kind == "min":
            return selected.min(dim=2).values
        if self.and_kind == "softmin":
            masked = torch.where(mask[None, :, :], expanded, torch.full_like(expanded, 1.0))
            return -torch.logsumexp(-self.softmin_beta * masked, dim=2) / self.softmin_beta
        raise ValueError(f"unknown AND kind: {self.and_kind}")

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            scores = self(torch.as_tensor(x, dtype=torch.float32))
            indices = torch.argmax(scores, dim=1).cpu().numpy()
        return np.asarray([self.classes[i] for i in indices])


class SoftRoutingTreeNetwork(nn.Module):
    """Soft decision tree with probabilistic path routing."""

    def __init__(self, paths: ExtractedPaths, *, n_features: int, temperature: float = 30.0) -> None:
        super().__init__()
        self.paths = paths
        self.classes = paths.classes
        n_nodes = len(paths.condition_names) // 2
        w = torch.zeros((n_nodes, n_features), dtype=torch.float32)
        b = torch.zeros(n_nodes, dtype=torch.float32)
        for condition_pair in range(n_nodes):
            true_row = 2 * condition_pair
            feature = int(paths.condition_features[true_row])
            threshold = float(paths.condition_thresholds[true_row])
            w[condition_pair, feature] = temperature
            b[condition_pair] = -temperature * threshold
        self.condition = nn.Linear(n_features, n_nodes)
        with torch.no_grad():
            self.condition.weight.copy_(w)
            self.condition.bias.copy_(b)

        true_false = paths.path_matrix.reshape(paths.path_matrix.shape[0], n_nodes, 2)
        self.register_buffer("true_mask", torch.as_tensor(true_false[:, :, 0], dtype=torch.float32))
        self.register_buffer("false_mask", torch.as_tensor(true_false[:, :, 1], dtype=torch.float32))
        self.register_buffer("path_to_class", torch.as_tensor(paths.path_to_class, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        true_probs = torch.sigmoid(self.condition(x))
        false_probs = 1.0 - true_probs
        log_prob = (
            self.true_mask[None, :, :] * torch.log(true_probs[:, None, :].clamp_min(1e-8))
            + self.false_mask[None, :, :] * torch.log(false_probs[:, None, :].clamp_min(1e-8))
        ).sum(dim=2)
        path_probs = torch.exp(log_prob)
        return torch.clamp(path_probs @ self.path_to_class.T, 0.0, 1.0)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            scores = self(torch.as_tensor(x, dtype=torch.float32))
            indices = torch.argmax(scores, dim=1).cpu().numpy()
        return np.asarray([self.classes[i] for i in indices])


class EnsemblePathMatrixNetwork(nn.Module):
    """Average an ensemble of path-matrix tree networks."""

    def __init__(self, members: Sequence[PathMatrixNetwork]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.classes = members[0].classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(x) for member in self.members], dim=0).mean(dim=0)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            scores = self(torch.as_tensor(x, dtype=torch.float32))
            indices = torch.argmax(scores, dim=1).cpu().numpy()
        return np.asarray([self.classes[i] for i in indices])


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y_onehot: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    regularize_to_initial: float = 0.0,
    anneal_temperature: tuple[float, float] | None = None,
) -> list[float]:
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_onehot, dtype=torch.float32)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    losses: list[float] = []

    for epoch in range(epochs):
        if anneal_temperature and hasattr(model, "set_temperature"):
            start, end = anneal_temperature
            ratio = epoch / max(1, epochs - 1)
            model.set_temperature(start + ratio * (end - start))

        order = torch.randperm(len(x_tensor))
        for start_idx in range(0, len(x_tensor), batch_size):
            batch = order[start_idx : start_idx + batch_size]
            optimizer.zero_grad()
            outputs = model(x_tensor[batch])
            loss = F.binary_cross_entropy(outputs.clamp(1e-6, 1.0 - 1e-6), y_tensor[batch])
            if regularize_to_initial:
                penalty = torch.zeros((), dtype=torch.float32)
                for parameter, original in zip(model.parameters(), initial):
                    penalty = penalty + torch.sum((parameter - original) ** 2)
                loss = loss + regularize_to_initial * penalty
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            outputs = model(x_tensor)
            losses.append(float(F.binary_cross_entropy(outputs.clamp(1e-6, 1.0 - 1e-6), y_tensor).item()))
    return losses


def model_loss(model: nn.Module, x: np.ndarray, y_onehot: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(torch.as_tensor(x, dtype=torch.float32))
        targets = torch.as_tensor(y_onehot, dtype=torch.float32)
        return float(F.binary_cross_entropy(outputs.clamp(1e-6, 1.0 - 1e-6), targets).item())


def path_matrix_structure(paths: ExtractedPaths, *, n_features: int, compressed: bool = False) -> StructureStats:
    matrix = paths.path_matrix
    labels = np.asarray([repr(label) for label in paths.path_labels])[:, None]
    if compressed:
        keyed = np.concatenate([matrix, labels], axis=1)
        path_count = np.unique(keyed, axis=0).shape[0]
    else:
        path_count = matrix.shape[0]
    condition_edges = len(paths.condition_names)
    path_edges = int(matrix.sum())
    output_edges = path_count
    return StructureStats(
        neurons=len(paths.condition_names) + path_count + len(paths.classes),
        edges=condition_edges + path_edges + output_edges,
        layers=3,
        paths=path_count,
        conditions=len(paths.condition_names),
        notes="compressed duplicate path rows" if compressed else "three-layer path matrix",
    )


def original_structure(tree: TreeNode, *, classes: Sequence[Any]) -> StructureStats:
    network = convert_tree_to_network(tree, classes=classes)
    summary = network.summary()
    layer_count = len(summary["layers"])
    return StructureStats(
        neurons=summary["hidden_neurons"] + summary["output_neurons"],
        edges=summary["edges"] + summary["condition_neurons"],
        layers=layer_count,
        paths=sum(len(output.incoming) for output in network.output_neurons.values()),
        conditions=summary["condition_neurons"],
        notes="original adjacent-layer parser with pass-through chains",
    )


def sklearn_tree_to_dt(sklearn_tree: Any, *, feature_names: Sequence[str] | None = None) -> TreeNode:
    tree_ = sklearn_tree.tree_

    def build(node_id: int) -> TreeNode:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == right:
            label = int(np.argmax(tree_.value[node_id][0]))
            return Leaf(label)
        feature = int(tree_.feature[node_id])
        threshold = float(tree_.threshold[node_id])
        name = feature_names[feature] if feature_names else f"sk_{node_id}"
        return DecisionNode(
            feature_index=feature,
            threshold=threshold,
            true_child=build(right),
            false_child=build(left),
            name=name,
        )

    return build(0)


def exact_agreement(model: nn.Module, tree: TreeNode, x: np.ndarray) -> float:
    y_tree = predict_batch(tree, x)
    y_model = model.predict_numpy(x)
    return float(np.mean(np.asarray(y_tree) == y_model))


def one_hot(labels: Sequence[Any], classes: Sequence[Any]) -> np.ndarray:
    index = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        y[row, index[label]] = 1.0
    return y
