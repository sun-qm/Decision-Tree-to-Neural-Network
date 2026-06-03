"""Direct root-to-leaf path parser.

This converter is an experimental alternative to the original recursive
Tree-to-NN parser. It keeps the same condition neurons, but creates one path
neuron per leaf path:

    path = ReLU(sum(required_branch_conditions) - (path_length - 1))

The result is a fixed three-stage graph:

    input -> condition neurons -> path neurons -> output neurons

The original converter remains unchanged and should still be used when the
paper's adjacent-layer topology is required.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from dt_to_nn.network import (
    ComputedNeuron,
    ConditionNeuron,
    Connection,
    NeuralDecisionNetwork,
    OutputNeuron,
)
from dt_to_nn.tree import DecisionNode, Leaf, TreeNode, class_labels, iter_internal_nodes


@dataclass(frozen=True)
class DirectPathSummary:
    condition_neurons: int
    path_neurons: int
    output_neurons: int
    edges: int
    layers: int

    @property
    def total_neurons(self) -> int:
        return self.condition_neurons + self.path_neurons + self.output_neurons


class DirectPathTreeToNNConverter:
    """Convert a tree by creating one AND neuron for each leaf path."""

    def __init__(
        self,
        tree: TreeNode,
        *,
        classes: Sequence[Any] | None = None,
    ) -> None:
        self.tree = tree
        self.classes = tuple(classes if classes is not None else class_labels(tree))
        self.network = NeuralDecisionNetwork(max_depth=2, classes=self.classes)
        self._condition_names: dict[tuple[int, str], str] = {}
        self._class_paths: dict[Any, list[str]] = defaultdict(list)
        self._path_counter = 0

    def convert(self) -> NeuralDecisionNetwork:
        self._initialize_condition_neurons()

        if isinstance(self.tree, Leaf):
            path_name = self._new_path_name()
            self.network.add_computed(
                ComputedNeuron(
                    name=path_name,
                    layer=2,
                    incoming=(),
                    bias=1.0,
                    activation="identity",
                )
            )
            self._class_paths[self.tree.label].append(path_name)
        else:
            self._collect_paths(self.tree, [])

        for label in self.classes:
            incoming = tuple(Connection(path_name, 1.0) for path_name in self._class_paths[label])
            self.network.set_output(
                OutputNeuron(
                    name=f"out_{self._safe_label(label)}",
                    label=label,
                    layer=3,
                    incoming=incoming,
                )
            )
        return self.network

    def _initialize_condition_neurons(self) -> None:
        for index, node in enumerate(iter_internal_nodes(self.tree)):
            node_name = node.name or f"n{index}"
            for branch in ("true", "false"):
                name = f"g_{branch}_{node_name}"
                self._condition_names[(id(node), branch)] = name
                self.network.add_condition(
                    ConditionNeuron(
                        name=name,
                        layer=1,
                        feature_index=node.feature_index,
                        threshold=node.threshold,
                        branch=branch,
                        tree_node_name=node_name,
                    )
                )

    def _collect_paths(self, node: TreeNode, conditions: list[str]) -> None:
        if isinstance(node, Leaf):
            path_name = self._new_path_name()
            path_len = len(conditions)
            self.network.add_computed(
                ComputedNeuron(
                    name=path_name,
                    layer=2,
                    incoming=tuple(Connection(name, 1.0) for name in conditions),
                    bias=-(path_len - 1),
                    activation="relu",
                )
            )
            self._class_paths[node.label].append(path_name)
            return

        self._collect_paths(
            node.true_child,
            conditions + [self._condition_names[(id(node), "true")]],
        )
        self._collect_paths(
            node.false_child,
            conditions + [self._condition_names[(id(node), "false")]],
        )

    def _new_path_name(self) -> str:
        name = f"path_{self._path_counter}"
        self._path_counter += 1
        return name

    @staticmethod
    def _safe_label(label: Any) -> str:
        return "".join(char if char.isalnum() else "_" for char in str(label))


def convert_tree_to_direct_path_network(
    tree: TreeNode,
    *,
    classes: Sequence[Any] | None = None,
) -> NeuralDecisionNetwork:
    """Build a fixed-depth direct-path network from a decision tree."""

    return DirectPathTreeToNNConverter(tree, classes=classes).convert()


def summarize_direct_path_network(network: NeuralDecisionNetwork) -> DirectPathSummary:
    """Return structural counts including condition-input edges."""

    condition_edges = len(network.condition_neurons)
    path_edges = sum(len(neuron.incoming) for neuron in network.computed_neurons.values())
    output_edges = sum(len(neuron.incoming) for neuron in network.output_neurons.values())
    return DirectPathSummary(
        condition_neurons=len(network.condition_neurons),
        path_neurons=len(network.computed_neurons),
        output_neurons=len(network.output_neurons),
        edges=condition_edges + path_edges + output_edges,
        layers=len(network.layer_map()),
    )
