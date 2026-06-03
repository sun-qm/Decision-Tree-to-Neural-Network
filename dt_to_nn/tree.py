"""Small decision-tree representation used by the converter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence, Union


@dataclass(frozen=True)
class Leaf:
    """A terminal decision-tree node."""

    label: Any


@dataclass(frozen=True)
class DecisionNode:
    """A binary decision node with explicit branch semantics.

    The true branch is selected when ``x[feature_index] >= threshold``.
    The false branch is selected when ``x[feature_index] < threshold``.
    """

    feature_index: int
    threshold: float
    true_child: "TreeNode"
    false_child: "TreeNode"
    name: str | None = None


TreeNode = Union[DecisionNode, Leaf]


def predict_one(node: TreeNode, x: Sequence[float]) -> Any:
    """Evaluate a single sample with the decision tree."""

    current = node
    while isinstance(current, DecisionNode):
        if current.feature_index >= len(x):
            raise ValueError(
                f"sample has {len(x)} features, but tree needs feature "
                f"{current.feature_index}"
            )
        current = (
            current.true_child
            if x[current.feature_index] >= current.threshold
            else current.false_child
        )
    return current.label


def predict_batch(node: TreeNode, samples: Sequence[Sequence[float]]) -> list[Any]:
    """Evaluate multiple samples with the decision tree."""

    return [predict_one(node, x) for x in samples]


def iter_internal_nodes(node: TreeNode) -> Iterator[DecisionNode]:
    """Yield decision nodes in pre-order."""

    if isinstance(node, Leaf):
        return
    yield node
    yield from iter_internal_nodes(node.true_child)
    yield from iter_internal_nodes(node.false_child)


def iter_leaves(node: TreeNode) -> Iterator[Leaf]:
    """Yield leaves from left-to-right pre-order."""

    if isinstance(node, Leaf):
        yield node
        return
    yield from iter_leaves(node.true_child)
    yield from iter_leaves(node.false_child)


def class_labels(node: TreeNode) -> list[Any]:
    """Return unique class labels in deterministic order."""

    labels = {leaf.label for leaf in iter_leaves(node)}
    return sorted(labels, key=lambda value: repr(value))


def max_depth(node: TreeNode) -> int:
    """Return the maximum number of decisions on any root-to-leaf path."""

    if isinstance(node, Leaf):
        return 0
    return 1 + max(max_depth(node.true_child), max_depth(node.false_child))


def count_nodes(node: TreeNode) -> int:
    """Return the total number of tree nodes."""

    if isinstance(node, Leaf):
        return 1
    return 1 + count_nodes(node.true_child) + count_nodes(node.false_child)


def required_n_features(node: TreeNode) -> int:
    """Return the minimum feature-vector length required by the tree."""

    if isinstance(node, Leaf):
        return 0
    return max(
        node.feature_index + 1,
        required_n_features(node.true_child),
        required_n_features(node.false_child),
    )
