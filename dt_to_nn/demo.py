"""Runnable demonstration for the tree-to-network conversion."""

from __future__ import annotations

from pprint import pprint

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.evaluation import (
    evaluate_equivalence,
    make_grid,
    random_samples,
    threshold_probe_samples,
)
from dt_to_nn.tree import DecisionNode, Leaf


def build_demo_tree() -> DecisionNode:
    return DecisionNode(
        feature_index=0,
        threshold=0.5,
        name="root",
        true_child=DecisionNode(
            feature_index=1,
            threshold=1.0,
            name="high_x0",
            true_child=Leaf("A"),
            false_child=Leaf("B"),
        ),
        false_child=DecisionNode(
            feature_index=2,
            threshold=-0.2,
            name="low_x0",
            true_child=Leaf("B"),
            false_child=Leaf("C"),
        ),
    )


def main() -> None:
    tree = build_demo_tree()
    network = convert_tree_to_network(tree)

    samples = random_samples(3, 1000, low=-2.0, high=2.0, seed=42)
    samples.extend(threshold_probe_samples(tree))
    random_result = evaluate_equivalence(tree, network, samples)

    grid = make_grid(
        {
            0: [-1.0, 0.5, 2.0],
            1: [0.0, 1.0, 2.0],
            2: [-1.0, -0.2, 1.0],
        }
    )
    grid_result = evaluate_equivalence(tree, network, grid)

    print("Network summary:")
    pprint(network.summary(), sort_dicts=False)
    print("\nRandom + threshold-probe evaluation:")
    pprint(random_result.to_dict(), sort_dicts=False)
    print("\nGrid evaluation:")
    pprint(grid_result.to_dict(), sort_dicts=False)


if __name__ == "__main__":
    main()
