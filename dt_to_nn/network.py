"""Graph-style neural network produced from a decision tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def relu(value: float) -> float:
    return max(0.0, value)


def clipped_relu(value: float) -> float:
    return min(max(0.0, value), 1.0)


@dataclass(frozen=True)
class Connection:
    source: str
    weight: float = 1.0


@dataclass(frozen=True)
class ConditionNeuron:
    """A first-layer branch-condition neuron."""

    name: str
    layer: int
    feature_index: int
    threshold: float
    branch: str
    tree_node_name: str

    def evaluate(self, x: Sequence[float]) -> float:
        value = x[self.feature_index]
        if self.branch == "true":
            return 1.0 if value >= self.threshold else 0.0
        if self.branch == "false":
            return 1.0 if value < self.threshold else 0.0
        raise ValueError(f"unknown branch: {self.branch!r}")


@dataclass(frozen=True)
class ComputedNeuron:
    """A ReLU/identity neuron in a hidden path layer."""

    name: str
    layer: int
    incoming: tuple[Connection, ...]
    bias: float = 0.0
    activation: str = "relu"

    def evaluate(self, values: Mapping[str, float]) -> float:
        z = self.bias + sum(conn.weight * values[conn.source] for conn in self.incoming)
        if self.activation == "relu":
            return relu(z)
        if self.activation == "identity":
            return z
        raise ValueError(f"unknown activation: {self.activation!r}")


@dataclass(frozen=True)
class OutputNeuron:
    """A clipped-ReLU class output that ORs all paths for one label."""

    name: str
    label: Any
    layer: int
    incoming: tuple[Connection, ...] = field(default_factory=tuple)

    def evaluate(self, values: Mapping[str, float]) -> float:
        return clipped_relu(sum(conn.weight * values[conn.source] for conn in self.incoming))


@dataclass
class NeuralDecisionNetwork:
    """Sparse neural network with explicit neurons and weighted edges."""

    max_depth: int
    classes: tuple[Any, ...]
    condition_neurons: dict[str, ConditionNeuron] = field(default_factory=dict)
    computed_neurons: dict[str, ComputedNeuron] = field(default_factory=dict)
    output_neurons: dict[Any, OutputNeuron] = field(default_factory=dict)

    def add_condition(self, neuron: ConditionNeuron) -> None:
        self.condition_neurons[neuron.name] = neuron

    def add_computed(self, neuron: ComputedNeuron) -> None:
        self.computed_neurons[neuron.name] = neuron

    def set_output(self, neuron: OutputNeuron) -> None:
        self.output_neurons[neuron.label] = neuron

    def values(self, x: Sequence[float]) -> dict[str, float]:
        """Evaluate every non-output neuron and return their values."""

        values: dict[str, float] = {}
        for neuron in self.condition_neurons.values():
            values[neuron.name] = neuron.evaluate(x)

        for neuron in sorted(
            self.computed_neurons.values(), key=lambda item: (item.layer, item.name)
        ):
            values[neuron.name] = neuron.evaluate(values)
        return values

    def outputs(self, x: Sequence[float]) -> dict[Any, float]:
        """Return class output activations."""

        values = self.values(x)
        return {
            label: self.output_neurons[label].evaluate(values)
            for label in self.classes
            if label in self.output_neurons
        }

    def predict(self, x: Sequence[float]) -> Any:
        """Return the class with the largest output activation."""

        outputs = self.outputs(x)
        if not outputs:
            raise ValueError("network has no outputs")
        return max(self.classes, key=lambda label: (outputs.get(label, 0.0), repr(label)))

    def layer_map(self) -> dict[int, list[str]]:
        """Return neuron names grouped by layer."""

        layers: dict[int, list[str]] = {}
        for neuron in self.condition_neurons.values():
            layers.setdefault(neuron.layer, []).append(neuron.name)
        for neuron in self.computed_neurons.values():
            layers.setdefault(neuron.layer, []).append(neuron.name)
        for neuron in self.output_neurons.values():
            layers.setdefault(neuron.layer, []).append(neuron.name)
        return {layer: sorted(names) for layer, names in sorted(layers.items())}

    def summary(self) -> dict[str, Any]:
        """Return a compact structural summary."""

        hidden_count = len(self.condition_neurons) + len(self.computed_neurons)
        edge_count = sum(len(n.incoming) for n in self.computed_neurons.values()) + sum(
            len(n.incoming) for n in self.output_neurons.values()
        )
        return {
            "max_depth": self.max_depth,
            "classes": list(self.classes),
            "condition_neurons": len(self.condition_neurons),
            "computed_neurons": len(self.computed_neurons),
            "hidden_neurons": hidden_count,
            "output_neurons": len(self.output_neurons),
            "edges": edge_count,
            "layers": self.layer_map(),
        }
