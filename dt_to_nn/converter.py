"""Decision-tree to neural-network conversion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from dt_to_nn.network import (
    ComputedNeuron,
    ConditionNeuron,
    Connection,
    NeuralDecisionNetwork,
    OutputNeuron,
)
from dt_to_nn.tree import (
    DecisionNode,
    Leaf,
    TreeNode,
    class_labels,
    iter_internal_nodes,
    max_depth as tree_max_depth,
)


class TreeToNNConverter:
    """Convert a binary decision tree into a topologically equivalent network."""

    def __init__(
        self,
        tree: TreeNode,
        *,
        classes: list[Any] | tuple[Any, ...] | None = None,
        max_depth: int | None = None,
    ) -> None:
        self.tree = tree
        self.classes = tuple(classes if classes is not None else class_labels(tree))
        self.max_depth = tree_max_depth(tree) if max_depth is None else max_depth
        if self.max_depth < tree_max_depth(tree):
            raise ValueError("max_depth cannot be smaller than the tree depth")

        self.network = NeuralDecisionNetwork(self.max_depth, self.classes)
        self._node_names: dict[int, str] = {}
        self._condition_names: dict[tuple[int, str], str] = {}
        self._path_counter = 0
        self._class_paths: dict[Any, list[str]] = defaultdict(list)

    def convert(self) -> NeuralDecisionNetwork:
        """Build and return the neural network."""

        self._initialize_condition_neurons()
        if isinstance(self.tree, Leaf):
            root_name = self._new_name("const_root")
            self.network.add_computed(
                ComputedNeuron(
                    name=root_name,
                    layer=0,
                    incoming=(),
                    bias=1.0,
                    activation="identity",
                )
            )
            final = self._pass_through(root_name, 0, self.max_depth)
            self._class_paths[self.tree.label].append(final)
        else:
            self._build_from_tree(self.tree, current_path=None, current_layer=0)

        output_layer = self.max_depth + 1
        for label in self.classes:
            paths = tuple(Connection(path, 1.0) for path in self._class_paths[label])
            self.network.set_output(
                OutputNeuron(
                    name=f"out_{self._safe_label(label)}",
                    label=label,
                    layer=output_layer,
                    incoming=paths,
                )
            )
        return self.network

    def _initialize_condition_neurons(self) -> None:
        for index, node in enumerate(iter_internal_nodes(self.tree)):
            node_name = node.name or f"n{index}"
            self._node_names[id(node)] = node_name
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

    def _build_from_tree(
        self,
        node: DecisionNode,
        *,
        current_path: str | None,
        current_layer: int,
    ) -> None:
        branches = (
            (node.true_child, self._condition_names[(id(node), "true")]),
            (node.false_child, self._condition_names[(id(node), "false")]),
        )

        for child, branch_condition in branches:
            if current_path is None:
                new_path = branch_condition
                new_layer = 1
            else:
                aligned_branch = self._pass_through(branch_condition, 1, current_layer)
                new_path = self._and_neuron(current_path, aligned_branch, current_layer + 1)
                new_layer = current_layer + 1

            if isinstance(child, Leaf):
                final = self._pass_through(new_path, new_layer, self.max_depth)
                self._class_paths[child.label].append(final)
            else:
                self._build_from_tree(
                    child,
                    current_path=new_path,
                    current_layer=new_layer,
                )

    def _pass_through(self, source: str, source_layer: int, target_layer: int) -> str:
        current = source
        layer = source_layer
        while layer < target_layer:
            layer += 1
            name = self._new_name("pass")
            self.network.add_computed(
                ComputedNeuron(
                    name=name,
                    layer=layer,
                    incoming=(Connection(current, 1.0),),
                    bias=0.0,
                    activation="relu",
                )
            )
            current = name
        return current

    def _and_neuron(self, left: str, right: str, layer: int) -> str:
        name = self._new_name("and")
        self.network.add_computed(
            ComputedNeuron(
                name=name,
                layer=layer,
                incoming=(Connection(left, 1.0), Connection(right, 1.0)),
                bias=-1.0,
                activation="relu",
            )
        )
        return name

    def _new_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._path_counter}"
        self._path_counter += 1
        return name

    @staticmethod
    def _safe_label(label: Any) -> str:
        return "".join(char if char.isalnum() else "_" for char in str(label))


def convert_tree_to_network(
    tree: TreeNode,
    *,
    classes: list[Any] | tuple[Any, ...] | None = None,
    max_depth: int | None = None,
) -> NeuralDecisionNetwork:
    """Convenience wrapper for ``TreeToNNConverter``."""

    return TreeToNNConverter(tree, classes=classes, max_depth=max_depth).convert()
