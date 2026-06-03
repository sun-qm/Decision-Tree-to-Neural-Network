"""PyTorch training utilities for parsed tree-to-NN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dt_to_nn.trainable import TrainableParsedNetwork
from dt_to_nn.tree import TreeNode


@dataclass(frozen=True)
class TorchTrainingHistory:
    losses: list[float]


class TorchParsedNetwork(nn.Module):
    """PyTorch module initialized from the Editable XAI parser."""

    def __init__(
        self,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        activations: Sequence[str],
        classes: Sequence[Any],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = list(activations)
        self.classes = tuple(classes)
        for weight, bias in zip(weights, biases):
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
                layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
            self.layers.append(layer)

    @classmethod
    def from_tree(
        cls,
        tree: TreeNode,
        *,
        classes: Sequence[Any],
        n_features: int,
        condition_temperature: float = 30.0,
        zero_padding_width: int = 0,
        zero_padding_layers: int = 0,
    ) -> "TorchParsedNetwork":
        parsed = TrainableParsedNetwork.from_tree(
            tree,
            classes=classes,
            n_features=n_features,
            condition_temperature=condition_temperature,
            zero_padding_width=zero_padding_width,
            zero_padding_layers=zero_padding_layers,
        )
        return cls(parsed.weights, parsed.biases, parsed.activations, parsed.classes)

    def clone_random_like(self, *, seed: int, mode: str) -> "TorchParsedNetwork":
        torch.manual_seed(seed)
        clone = TorchParsedNetwork(
            [layer.weight.detach().cpu().numpy() for layer in self.layers],
            [layer.bias.detach().cpu().numpy() for layer in self.layers],
            self.activations,
            self.classes,
        )
        with torch.no_grad():
            for src, dst in zip(self.layers, clone.layers):
                fan_out, fan_in = dst.weight.shape
                scale = float(np.sqrt(3.0 / max(1, fan_in + fan_out)))
                if mode == "dense":
                    dst.weight.normal_(0.0, scale)
                elif mode == "sparse":
                    mask = src.weight.ne(0.0)
                    random_values = torch.empty_like(dst.weight).normal_(0.0, scale)
                    dst.weight.zero_()
                    dst.weight[mask] = random_values[mask]
                else:
                    raise ValueError("mode must be 'dense' or 'sparse'")
                dst.bias.normal_(0.0, scale)
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, activation in zip(self.layers, self.activations):
            out = layer(out)
            if activation == "sigmoid":
                out = torch.sigmoid(out)
            elif activation == "relu":
                out = F.relu(out)
            elif activation == "crelu":
                out = torch.clamp(out, 0.0, 1.0)
            elif activation == "identity":
                pass
            else:
                raise ValueError(f"unknown activation: {activation}")
        return out

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            scores = self(torch.as_tensor(x, dtype=torch.float32))
            indices = torch.argmax(scores, dim=1).cpu().numpy()
        return np.asarray([self.classes[i] for i in indices])

    def loss_numpy(self, x: np.ndarray, y_onehot: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            outputs = self(torch.as_tensor(x, dtype=torch.float32))
            targets = torch.as_tensor(y_onehot, dtype=torch.float32)
            return float(_editable_xai_cross_entropy(outputs, targets).item())


def train_torch_model(
    model: TorchParsedNetwork,
    x: np.ndarray,
    y_onehot: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    freeze_except_thresholds: bool = False,
) -> TorchTrainingHistory:
    """Train with cross-entropy, following Editable XAI Section 4.3."""

    torch.manual_seed(seed)
    if freeze_except_thresholds:
        for parameter in model.parameters():
            parameter.requires_grad = False
        first_layer = model.layers[0]
        first_layer.bias.requires_grad = True
        parameters = [first_layer.bias]
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        parameters = list(model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_onehot, dtype=torch.float32)
    n = len(x_tensor)
    losses: list[float] = []

    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            outputs = model(x_tensor[batch])
            loss = _editable_xai_cross_entropy(outputs, y_tensor[batch])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            losses.append(float(_editable_xai_cross_entropy(model(x_tensor), y_tensor).item()))
    return TorchTrainingHistory(losses)


def _editable_xai_cross_entropy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    outputs = torch.clamp(outputs, 1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(outputs, targets)
