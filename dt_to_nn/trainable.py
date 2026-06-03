"""Trainable NumPy networks initialized from parsed decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.network import NeuralDecisionNetwork
from dt_to_nn.tree import TreeNode, class_labels, iter_internal_nodes, max_depth


@dataclass
class TrainingHistory:
    losses: list[float]


def one_hot(labels: Sequence[Any], classes: Sequence[Any]) -> np.ndarray:
    index = {label: i for i, label in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=float)
    for row, label in enumerate(labels):
        y[row, index[label]] = 1.0
    return y


class TrainableParsedNetwork:
    """Dense, trainable form of the Editable XAI parser.

    The sparse exact network is first produced from the decision tree. It is
    then embedded in dense matrices and optionally zero-padded with additional
    neurons and pass-through layers. Layer semantics follow the Editable XAI
    algorithm: sigmoid decision-branch neurons, ReLU conjunction/path neurons,
    and clipped-ReLU output disjunction.
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        activations: list[str],
        classes: Sequence[Any],
    ) -> None:
        self.weights = weights
        self.biases = biases
        self.activations = activations
        self.classes = tuple(classes)

    @classmethod
    def from_tree(
        cls,
        tree: TreeNode,
        *,
        classes: Sequence[Any] | None = None,
        n_features: int | None = None,
        condition_temperature: float = 30.0,
        zero_padding_width: int = 0,
        zero_padding_layers: int = 0,
    ) -> "TrainableParsedNetwork":
        labels = tuple(classes if classes is not None else class_labels(tree))
        padded_depth = max_depth(tree) + zero_padding_layers
        sparse = convert_tree_to_network(tree, classes=labels, max_depth=padded_depth)
        n_inputs = n_features if n_features is not None else _required_features_from_sparse(sparse)
        return cls.from_sparse(
            sparse,
            n_features=n_inputs,
            condition_temperature=condition_temperature,
            min_hidden_width=2 * len(list(iter_internal_nodes(tree))),
            zero_padding_width=zero_padding_width,
        )

    @classmethod
    def from_sparse(
        cls,
        sparse: NeuralDecisionNetwork,
        *,
        n_features: int,
        condition_temperature: float = 30.0,
        min_hidden_width: int = 0,
        zero_padding_width: int = 0,
    ) -> "TrainableParsedNetwork":
        hidden_layers = [
            layer for layer in sorted(sparse.layer_map()) if layer <= sparse.max_depth
        ]
        names_by_layer: dict[int, list[str]] = {}
        widths: dict[int, int] = {}
        for layer in hidden_layers:
            names = sparse.layer_map()[layer]
            names_by_layer[layer] = names
            widths[layer] = max(len(names), min_hidden_width) + zero_padding_width

        name_to_position: dict[str, tuple[int, int]] = {}
        for layer in hidden_layers:
            for index, name in enumerate(names_by_layer[layer]):
                name_to_position[name] = (layer, index)

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        activations: list[str] = []

        if not hidden_layers:
            weights.append(np.zeros((len(sparse.classes), n_features), dtype=float))
            biases.append(np.ones(len(sparse.classes), dtype=float))
            activations.append("crelu")
            return cls(weights, biases, activations, sparse.classes)

        first_layer = hidden_layers[0]
        w1 = np.zeros((widths[first_layer], n_features), dtype=float)
        b1 = np.zeros(widths[first_layer], dtype=float)
        for name in names_by_layer[first_layer]:
            neuron = sparse.condition_neurons.get(name)
            if neuron is None:
                continue
            _, row = name_to_position[name]
            sign = 1.0 if neuron.branch == "true" else -1.0
            w1[row, neuron.feature_index] = sign * condition_temperature
            b1[row] = -sign * condition_temperature * neuron.threshold
        weights.append(w1)
        biases.append(b1)
        activations.append("sigmoid")

        for layer in hidden_layers[1:]:
            prev_layer = layer - 1
            w = np.zeros((widths[layer], widths[prev_layer]), dtype=float)
            b = np.zeros(widths[layer], dtype=float)
            for name in names_by_layer[layer]:
                neuron = sparse.computed_neurons[name]
                _, row = name_to_position[name]
                b[row] = neuron.bias
                for conn in neuron.incoming:
                    source_layer, col = name_to_position[conn.source]
                    if source_layer != prev_layer:
                        raise ValueError(
                            f"connection {conn.source}->{name} skips from layer "
                            f"{source_layer} to {layer}"
                        )
                    w[row, col] = conn.weight
            weights.append(w)
            biases.append(b)
            activations.append("relu")

        output_layer = sparse.max_depth + 1
        prev_width = widths[hidden_layers[-1]]
        w_out = np.zeros((len(sparse.classes), prev_width), dtype=float)
        b_out = np.zeros(len(sparse.classes), dtype=float)
        for row, label in enumerate(sparse.classes):
            output = sparse.output_neurons[label]
            if output.layer != output_layer:
                raise ValueError("unexpected output layer")
            for conn in output.incoming:
                _, col = name_to_position[conn.source]
                w_out[row, col] = conn.weight
        weights.append(w_out)
        biases.append(b_out)
        activations.append("crelu")
        return cls(weights, biases, activations, sparse.classes)

    def copy(self) -> "TrainableParsedNetwork":
        return TrainableParsedNetwork(
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases],
            list(self.activations),
            self.classes,
        )

    def random_like(
        self,
        *,
        seed: int,
        mode: str,
    ) -> "TrainableParsedNetwork":
        rng = np.random.default_rng(seed)
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for base_w, base_b in zip(self.weights, self.biases):
            fan_out, fan_in = base_w.shape
            scale = np.sqrt(3.0 / max(1, fan_in + fan_out))
            if mode == "dense":
                w = rng.normal(0.0, scale, size=base_w.shape)
            elif mode == "sparse":
                nonzero = int(np.count_nonzero(base_w))
                flat_size = base_w.size
                w = np.zeros_like(base_w)
                if nonzero:
                    chosen = rng.choice(flat_size, size=min(nonzero, flat_size), replace=False)
                    w.flat[chosen] = rng.normal(0.0, scale, size=len(chosen))
            else:
                raise ValueError("mode must be 'dense' or 'sparse'")
            weights.append(w)
            biases.append(rng.normal(0.0, scale, size=base_b.shape))
        return TrainableParsedNetwork(weights, biases, self.activations, self.classes)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        activations = [x]
        preactivations: list[np.ndarray] = []
        current = x
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            current = np.nan_to_num(current, nan=0.0, posinf=1.0, neginf=-1.0)
            w = np.nan_to_num(w, nan=0.0, posinf=60.0, neginf=-60.0)
            b = np.nan_to_num(b, nan=0.0, posinf=60.0, neginf=-60.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                z = current @ w.T + b
            z = np.nan_to_num(z, nan=0.0, posinf=60.0, neginf=-60.0)
            z = np.clip(z, -60.0, 60.0)
            preactivations.append(z)
            current = _activate(z, activation)
            activations.append(current)
        return preactivations, activations

    def outputs(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=float))[1][-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.outputs(x)
        return np.asarray([self.classes[i] for i in np.argmax(scores, axis=1)])

    def loss(self, x: np.ndarray, y_onehot: np.ndarray) -> float:
        y_pred = self.outputs(x)
        return float(np.mean((y_pred - y_onehot) ** 2))

    def fit(
        self,
        x: np.ndarray,
        y_onehot: np.ndarray,
        *,
        epochs: int = 200,
        learning_rate: float = 0.01,
        batch_size: int | None = None,
        seed: int = 0,
        gradient_clip: float = 5.0,
    ) -> TrainingHistory:
        rng = np.random.default_rng(seed)
        n = len(x)
        batch_size = n if batch_size is None else batch_size
        losses: list[float] = []

        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = order[start : start + batch_size]
                grad_w, grad_b = self._gradients(x[batch], y_onehot[batch])
                for i in range(len(self.weights)):
                    if gradient_clip > 0:
                        grad_w[i] = np.clip(grad_w[i], -gradient_clip, gradient_clip)
                        grad_b[i] = np.clip(grad_b[i], -gradient_clip, gradient_clip)
                    self.weights[i] -= learning_rate * grad_w[i]
                    self.biases[i] -= learning_rate * grad_b[i]
                    self.weights[i] = np.nan_to_num(
                        self.weights[i], nan=0.0, posinf=gradient_clip, neginf=-gradient_clip
                    )
                    self.biases[i] = np.nan_to_num(
                        self.biases[i], nan=0.0, posinf=gradient_clip, neginf=-gradient_clip
                    )
                    self.weights[i] = np.clip(self.weights[i], -gradient_clip, gradient_clip)
                    self.biases[i] = np.clip(self.biases[i], -gradient_clip, gradient_clip)
            losses.append(self.loss(x, y_onehot))
        return TrainingHistory(losses)

    def _gradients(
        self, x: np.ndarray, y_onehot: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pre, acts = self.forward(x)
        n = max(1, len(x))
        delta = (2.0 / n) * (acts[-1] - y_onehot) * _activation_grad(
            pre[-1], self.activations[-1]
        )
        grad_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grad_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        for layer in reversed(range(len(self.weights))):
            delta = np.nan_to_num(delta, nan=0.0, posinf=5.0, neginf=-5.0)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                grad_w[layer] = delta.T @ acts[layer]
            grad_b[layer] = np.sum(delta, axis=0)
            if layer > 0:
                w = np.clip(
                    np.nan_to_num(
                        self.weights[layer], nan=0.0, posinf=5.0, neginf=-5.0
                    ),
                    -5.0,
                    5.0,
                )
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    delta = (delta @ w) * _activation_grad(
                        pre[layer - 1], self.activations[layer - 1]
                    )
        return grad_w, grad_b


def _required_features_from_sparse(sparse: NeuralDecisionNetwork) -> int:
    max_index = -1
    for neuron in sparse.condition_neurons.values():
        max_index = max(max_index, neuron.feature_index)
    return max_index + 1


def _activate(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "sigmoid":
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))
    if activation == "relu":
        return np.maximum(0.0, z)
    if activation == "crelu":
        return np.clip(z, 0.0, 1.0)
    if activation == "identity":
        return z
    raise ValueError(f"unknown activation: {activation}")


def _activation_grad(z: np.ndarray, activation: str) -> np.ndarray:
    if activation == "sigmoid":
        y = _activate(z, activation)
        return y * (1.0 - y)
    if activation == "relu":
        return (z >= 0.0).astype(float)
    if activation == "crelu":
        return ((z >= 0.0) & (z <= 1.0)).astype(float)
    if activation == "identity":
        return np.ones_like(z)
    raise ValueError(f"unknown activation: {activation}")
